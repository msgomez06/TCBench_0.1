import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob, os
import xarray as xr
import re
from scripts import utils

def flatten(arg):
    # flatten list of any depth into a list of depth 1
    if not isinstance(arg, list): # if not list
        return [arg]
    return [x for sub in arg for x in flatten(sub)] # recurse and collect


class TCData(Dataset):
    def __init__(self, data_path, model_name, mode="deterministic", small=False, data_type='train', normalize=True):
        
        assert data_type in ['train', 'test', 'val'], "data_type must be one of 'train', 'test', or 'val'"
        model_folder = (model_name if model_name != "pangu" else "panguweather") + "/PostProcessing"
        
        self.data_path = data_path
        self.small = small
        self.data_type = data_type

        data = [np.load(f"{self.data_path}/{model_folder}/{model_name}_{mode}_{self.data_type}_{suffix}.npy", mmap_mode='r') for \
                suffix in ["inp_fields", "inp_coords", "inp_ldt", "targets"]]
        data_copy = [arr.copy() for arr in data]
        if self.small:
            data_copy = [arr[:100] for arr in data_copy]
        
        self.data = [torch.from_numpy(arr).float() for arr in data_copy]
        if normalize:
            self.normalize()

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        fields, coords, ldt, target = self.data[0][idx], self.data[1][idx], self.data[2][idx], self.data[3][idx]
        lat, lon = coords
        return fields, lat, lon, ldt, target
    
    def summary(self):
        print(f"Number of samples: {self.data[0].shape[0]}")
        print("Input means: ", [f"{prefix}: {self.inputs_mean[i].reshape(torch.numel(self.inputs_mean[i]))}" for \
            i, prefix in enumerate(['fields', 'coords', 'ldt'])])
        print("Input stds: ", [f"{prefix}: {self.inputs_std[i].reshape(torch.numel(self.inputs_std[i]))}" for \
            i, prefix in enumerate(['fields', 'coords', 'ldt'])])
        print(f"Target means: {self.targets_mean} (Lat, Lon, Wnd, Msl)")
        print(f"Target stds: {self.targets_std} (Lat, Lon, Wnd, Msl)")

    
    def normalize(self):
        fields, coords, ldt = self.data[:3]
        targets = self.data[3]
        self.inputs_mean = [fields.mean(axis=(0, 2, 3)).reshape((1,3,1,1)), coords.mean(axis=0).reshape((1,2)), ldt.mean(axis=0)]
        self.inputs_std = [fields.std(axis=(0, 2, 3)).reshape(1,3,1,1), coords.std(axis=0).reshape(1,2), ldt.std(axis=0)]
        
        self.targets_mean = targets.mean(axis=0)
        self.targets_std = targets.std(axis=0)
        
        self.data[0] = (fields - self.inputs_mean[0]) / self.inputs_std[0]
        #self.data[1] = (coords - self.inputs_mean[1]) / self.inputs_std[1]
        #self.data[2] = (ldt - self.inputs_mean[2]) / self.inputs_std[2]
        self.data[3] = (targets - self.targets_mean) / self.targets_std
        print("Data normalized")




def create_dataset(data_folder, model_name, tc_ids, df_tracks, vars, size=241):
    """
    Create a dataset from a list of track ids and a list of variables
    Since there are different pressure levels, vars is expected to be of the form {var1: [plevs], var2: [plevs], ...}
    For vars without pressure level, e.g. 't2m', the dict should be {'t2m': [0]}
    """
    
    tc_cols = ["WMO_", "USA_", "TOKYO_", "HKO_", "NEWDELHI_", "CMA_", "REUNION_", "BOM_",
                 "NADI_", "WELLINGTON_", "DS824_", "TD9636_", "TD9635_", "MLC_", "NEUMANN_"]
    folder_name = model_name if model_name != "pangu" else "panguweather"
    type_ = "deterministic"
    vars_as_list = flatten([[(var+f"{int(vars[var][i])}" if plev!=0 else var) for i, plev in enumerate(vars[var])] for var in list(vars.keys())])
    
    # Load the data
    key = lambda x: (utils.get_start_date_nc(x), utils.get_lead_time(x))
    
    for tc_id in tc_ids:
        tc_tracks = df_tracks[df_tracks['SID'] == tc_id]
        save_name = f"{data_folder}/{folder_name}/PostProcessing/{model_name}_{tc_id}_{'_'.join(var for var in vars_as_list)}_{type_}"
        saved_files = [f"{save_name}_{suffix}.npy" for suffix in ["inp_fields", "inp_coords", "inp_ldt", "inp_start_time", "targets"]]
        if False in [os.path.isfile(saved_file) for saved_file in saved_files]:
            filenames_list = sorted(glob.glob(f"{data_folder}/{folder_name}/{model_name}_*_{tc_id}_small.nc"), key=key)
            inputs = []
            targets = []
            nb_samples = 0
            for filename in filenames_list:
                data_tmp = xr.open_dataset(filename, engine="netcdf4")
                try:
                    var1 = list(data_tmp.data_vars)[0]
                except IndexError:
                    print(f"Empty file: {filename}")
                    raise IndexError
                
                # fourcastnet records also the intial time step
                sample_tmp = data_tmp[f"{var1}"].shape[0]-1 if model_name=="fourcastnetv2" else data_tmp[f"{var1}"].shape[0]
                nb_samples += sample_tmp

                target_tmp = np.zeros((sample_tmp, 4))# if deterministic else np.zeros(sample_tmp, 4, 3) # 3 parameters for each distrib in stochastic case
                input_tmp = (np.zeros((sample_tmp, len(vars_as_list), size, size)),
                            np.zeros((sample_tmp, 2)), 
                            np.empty((sample_tmp), dtype=int),
                            np.empty((sample_tmp), dtype='datetime64[h]')) # "image", lat/lon, ldt, start_time
                
                for i, var in enumerate(vars_as_list):
                    if model_name=="fourcastnetv2":
                        idx = list(data_tmp.coords).index(var)
                        val = data_tmp["__xarray_dataarray_variable__"].values[1:, 0, idx, :, :]
                    else:
                        plev_match = re.search(r'\d+', var)
                        plev = plev_match.group() if plev_match is not None else None
                        if plev in data_tmp.isobaricInhPa: # upper-air variable
                            plev_idx = plev_match.start()
                            idx = data_tmp.isobaricInhPa.index(plev)
                            val = data_tmp[f"{var[:plev_idx]}"].values[:, idx, :, :]
                        else:
                            try:
                                val = data_tmp[f"{var}"].values[:, :, :]
                            except KeyError:
                                print(data_tmp)
                                raise KeyError(f"Variable {var} not found in {filename}")
                    
                    input_tmp[0][:, i, :, :] = val
                    
                times = data_tmp["time"].values[1:] if model_name=="fourcastnetv2" else data_tmp["time"].values
                start_iso_time = np.datetime64(times[0]-np.timedelta64(6, 'h'), 'h')
                
                tc_track_start = tc_tracks[tc_tracks['ISO_TIME'].astype("datetime64[ns]") == start_iso_time]
                idx_to_remove = []
                for i in range(sample_tmp):
                    tc_track_tmp = tc_tracks[tc_tracks['ISO_TIME'].astype("datetime64[ns]") == times[i]]
                    
                    ldt = np.timedelta64(times[i] - start_iso_time, 'h').astype(int) if i!=0 else 6
                    input_tmp[1][i, 0] = tc_track_start["LAT"].values[0]
                    lon_start = float(tc_track_start["LON"].values[0])
                    input_tmp[1][i, 1] = lon_start + 180 if lon_start < 0 else lon_start
                    input_tmp[2][i] = ldt
                    input_tmp[3][i] = start_iso_time
                    
                    target_tmp[i, 0] = tc_track_tmp['LAT'].values[0]
                    lon = float(tc_track_tmp['LON'].values[0])
                    target_tmp[i, 1] = lon + 180 if lon < 0 else lon
                    wnd, msl = [], []
                    for col in tc_cols:
                        wind, mslp = tc_track_tmp[f"{col}WIND"].values[0], tc_track_tmp[f"{col}PRES"].values[0]
                        if wind not in [' ', '', "", " "]:
                            wnd.append(wind)
                        if mslp not in [' ', '', "", " "]:
                            msl.append(mslp)
                    if len(wnd) == 0:
                        wnd.append(9999)
                        idx_to_remove.append(i)
                    if len(msl) == 0:
                        msl.append(9999)
                        idx_to_remove.append(i)
                    target_tmp[i, 2] = wnd[0]
                    target_tmp[i, 3] = msl[0]
                idx_to_remove = list(set(idx_to_remove))
                target_tmp = np.delete(target_tmp, idx_to_remove, axis=0)
                input_tmp = tuple([np.delete(inp, idx_to_remove, axis=0) for inp in input_tmp])
                
                inputs.append(input_tmp)
                targets.append(target_tmp)
            final_inputs = [np.zeros((nb_samples, len(vars_as_list), size, size)),
                            np.zeros((nb_samples, 2)), 
                            np.empty((nb_samples), dtype=int),
                            np.empty((nb_samples), dtype='datetime64[h]')] # "image", lat/lon, ldt, start_time
            final_targets = np.concatenate(targets, axis=0)
            for i in range(len(inputs[0])):
                final_inputs[i] = np.concatenate([inp[i] for inp in inputs], axis=0)
            
            np.save(save_name+"_inp_fields.npy", final_inputs[0])
            np.save(save_name+"_inp_coords.npy", final_inputs[1])
            np.save(save_name+"_inp_ldt.npy", final_inputs[2])
            np.save(save_name+"_inp_start_time.npy", final_inputs[3])
            np.save(save_name+"_targets.npy", final_targets)
                    

def split_data(data_path, model_name, mode="deterministic", splits={"train":0.7, "test":0.2, "val":0.1}):
    
    # tdb: make the split on the number of sample rather than the nb of TCs
    assert sum([splits[key] for key in splits]) == 1, "Split values must sum to 1"
    
    assert splits["train"] > 2*(splits["test"] + splits["val"]), "Train split must be at least twice as large than test and val splits"
    
    model_folder = (model_name if model_name != "pangu" else "panguweather") + "/PostProcessing"
    
    key = lambda x: -x.shape[0]
    
    targetsFiles = sorted([np.load(x) for x in glob.glob(f"{data_path}/{model_folder}/{model_name}*_{mode}_targets.npy")], key=key)
    fieldsFiles = sorted([np.load(x) for x in glob.glob(f"{data_path}/{model_folder}/{model_name}*_{mode}_inp_fields.npy")], key=key)
    coordsFiles = sorted([np.load(x) for x in glob.glob(f"{data_path}/{model_folder}/{model_name}*_{mode}_inp_coords.npy")], key=key)
    ldtFiles = sorted([np.load(x) for x in glob.glob(f"{data_path}/{model_folder}/{model_name}*_{mode}_inp_ldt.npy")], key=key)
    
    idx_test, idx_val = splitter(targetsFiles, splits)
    
    names = ["targets", "fields", "coords", "ldt"]
    test_files = {f"{names[i]}": x[:idx_test] for i, x in enumerate([targetsFiles, fieldsFiles, coordsFiles, ldtFiles])}
    val_files = {f"{names[i]}": x[idx_test:idx_val] for i, x in enumerate([targetsFiles, fieldsFiles, coordsFiles, ldtFiles])}
    train_files = {f"{names[i]}": x[idx_val:] for i, x in enumerate([targetsFiles, fieldsFiles, coordsFiles, ldtFiles])}
    
    for name in ["targets", "fields", "coords", "ldt"]:
        savename = name if name=="targets" else f"inp_{name}"
        np.save(f"{data_path}/{model_folder}/{model_name}_{mode}_test_{savename}.npy", np.concatenate(test_files[name], axis=0))
        np.save(f"{data_path}/{model_folder}/{model_name}_{mode}_val_{savename}.npy", np.concatenate(val_files[name], axis=0))
        np.save(f"{data_path}/{model_folder}/{model_name}_{mode}_train_{savename}.npy", np.concatenate(train_files[name], axis=0))
        

def splitter(l:list, splits:dict):
    key = lambda x: -x.shape[0]
    l = sorted(l, key=key)
    length = sum([x.shape[0] for x in l])
    idx_test, idx_val = 0, 0
    test_number = int(length*splits["test"])
    
    samples_test = 0
    while(samples_test < test_number):
        samples_test += l.pop(0).shape[0]
        idx_test += 1
    
    length = sum([x.shape[0] for x in l])
    val_number = int(length*splits["val"])
    idx_val = idx_test
    samples_val = 0
    while(samples_val < val_number):
        samples_val += l.pop(0).shape[0]
        idx_val += 1 
    
    
    return idx_test, idx_val