import glob, os, sys, pickle, torch
import pandas as pd
import numpy as np
import xarray as xr

from time import time
from torch.utils.data import Dataset
from utils.main_utils import get_lead_time, get_start_date_nc, get_tc_id_nc
from multiprocessing import Pool


def get_ibtracs_data(df, seasons=[], pres=True):
    
    if not isinstance(seasons, list):
        seasons = [seasons]
        
    df = df[df["SEASON"].isin([str(season) for season in seasons])]
    
    tc_ids = df["SID"].unique()
    valid_dates = {}
    
    wind_col = "USA_WIND"
    pres_cols = [col for col in df.columns if "_PRES" in col]
    pres_columns = {}
    for tc_id in tc_ids:
        tmp_df = df[df["SID"]==tc_id].loc[1:] # we never predict the start of the TC
        idxs = [idx for idx in tmp_df.index if tmp_df.loc[idx, wind_col]!=" "]
        
        if pres:
            key = lambda x: np.count_nonzero(tmp_df[x].values.astype("string")!=" ")
            pres_col = sorted(pres_cols, key=key)[-1] # the one with the highest number of values reported
            idxs = [idx for idx in idxs if tmp_df.loc[idx, pres_col]!=" "] # remove rows with missing values
            pres_columns[tc_id] = pres_col

        valid_dates[tc_id] = tmp_df.loc[idxs, "ISO_TIME"].values
        
    return df, valid_dates, pres_columns
    


class CNN4PP_Dataset(Dataset):
    
    def __init__(self, data_path, model_name, ibtracs_path, seasons, pres=True, 
                 save_path="/users/lpoulain/louis/plots/cnn/", train_seasons=[]):
        
        if isinstance(seasons, str):
            seasons = [int(seasons)]
        if isinstance(seasons, int):
            seasons = [seasons]
        self.pres = pres
        self.save_path = save_path
        self.seasons = seasons
        self.model_name = "pangu" if model_name in ["pangu", "panguweather"] else model_name
        self.data_folder = "panguweather" if model_name=="pangu" else model_name
        self.train_seasons = train_seasons

        ibtracs_df = pd.read_csv(ibtracs_path, na_filter=False, dtype="string")
        self.ibtracs_df, self.valid_dates, self.pres_columns = get_ibtracs_data(ibtracs_df, seasons, pres=pres)
        
        
        self.data_list = []
        self.data_list_scratch = []
        self.ll = []
        
        for season in seasons:
            if not os.path.isfile(os.path.join(save_path, f"Data/{self.data_folder}/netcdf/Data_list_{season}_p_{self.pres}.pkl")):
                data_list = []
                data_list_scratch = []
                tmp_df = ibtracs_df[ibtracs_df["SEASON"] == str(season)]
                tc_ids = tmp_df["SID"].unique()
                for tc_id in tc_ids:
                    key = lambda x: (get_start_date_nc(x), get_lead_time(x))
                    tmp_paths = sorted(glob.glob(os.path.join(data_path, f"{self.data_folder}/{self.model_name}*{tc_id}*.nc")), key=key)

                    for path in tmp_paths:
                        dates = [get_start_date_nc(path) + np.timedelta64(i, "h") for i in range(6, get_lead_time(path)+6, 6)]
                        valid_idxs = [i for i, date in enumerate(dates) if date in self.valid_dates[tc_id].astype("datetime64[ns]")]
                        if len(valid_idxs)==0:
                            continue
                        data_list.append((path, valid_idxs, len(valid_idxs)))
                        data_list_scratch.append((f"/scratch/lpoulain/{season}/" + os.path.basename(path).replace(".nc", "_wind_pres.pkl"\
                                                if self.pres else "_wind.pkl"), valid_idxs, len(valid_idxs)))
                with open(os.path.join(save_path, f"Data/{self.data_folder}/netcdf/Data_list_{season}_p_{self.pres}.pkl"), "wb") as f:
                    pickle.dump(data_list, f)
                with open(os.path.join(save_path, f"Data/{self.data_folder}/scratch/Data_list_{season}_p_{self.pres}.pkl"), "wb") as f:
                    pickle.dump(data_list_scratch, f)
            
            with open(os.path.join(save_path, f"Data/{self.data_folder}/netcdf/Data_list_{season}_p_{self.pres}.pkl"), "rb") as f:
                self.data_list.extend(pickle.load(f))
            with open(os.path.join(save_path, f"Data/{self.data_folder}/scratch/Data_list_{season}_p_{self.pres}.pkl"), "rb") as f:
                self.data_list_scratch.extend(pickle.load(f))
        
        #self.data_list = self.data_list[:100]
        #self.data_list_scratch = self.data_list_scratch[:100]
        for (_, _, s) in self.data_list:
            self.ll.append(s)
        #for i in range(len(self.data_list)):
        #    self.data_list[i] = (*self.data_list[i], self.ll[i])
        
        self.save_as_np()
        self.get_target_normalisation_cst()
        self.get_input_normalisation_cst()
        
        
        #print("Mean: ", self.mean)
        #print("Std: ", self.std)
        #print("Mean target: ", self.target_mean)
        #print("Std target: ", self.target_std)
        

    
    def get_target_normalisation_cst(self):
        tc_ids_all = list(self.valid_dates.keys())
        
        for season in self.train_seasons:
            if False in [os.path.isfile(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_{x}.npy") for x in\
                         ["target_mean_x", "target_mean_x_sq", "target_counter"]]:
                print(f"Computing normalisation constants for targets ({season}) ... ")
                tmp_ibtracs = self.ibtracs_df[self.ibtracs_df["SEASON"]==str(season)]
                tc_ids = [tc_id for tc_id in tc_ids_all if tc_id in tmp_ibtracs["SID"].values]
                count = 0
                meanX = [0, 0] if self.pres else [0]
                meanX_sq = [0, 0] if self.pres else [0]
                for tc_id in tc_ids:

                    tmp_df = tmp_ibtracs[tmp_ibtracs["SID"]==tc_id]
                    tmp_df = tmp_df[tmp_df["ISO_TIME"].astype("datetime64[ns]").isin(self.valid_dates[tc_id].astype("datetime64[ns]"))]
                    
                    for idx in tmp_df.index:
                        count += 1
                        wind = np.array(tmp_df.loc[idx, "USA_WIND"], dtype="float")
                        meanX[0] += wind
                        meanX_sq[0] += wind**2
                        
                        if self.pres:
                            pres = np.array(tmp_df.loc[idx, self.pres_columns[tc_id]], dtype="float")
                            meanX[1] += pres
                            meanX_sq[1] += pres**2
                            
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_mean_x.npy", np.array(meanX)/count)
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_mean_x_sq.npy", np.array(meanX_sq)/count)
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_counter.npy", np.array(count))
        
        c = 0
        self.target_mean = [0, 0] if self.pres else [0]
        self.target_mean_sq = [0, 0] if self.pres else [0]
        self.target_std = [0, 0] if self.pres else [0]
        
        for season in self.train_seasons:
            counter = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_counter.npy").item()
            c += counter
            
            tmp_mean = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_mean_x.npy")
            tmp_mean_sq = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_mean_x_sq.npy")
            
            self.target_mean[0] += tmp_mean[0]*counter
            self.target_mean_sq[0] += tmp_mean_sq[0]*counter
            
            if self.pres:
                self.target_mean[1] += tmp_mean[1]*counter
                self.target_mean_sq[1] += tmp_mean_sq[1]*counter
            
        self.target_mean[0] /= c
        self.target_mean_sq[0] /= c
        self.target_std[0] = np.sqrt(self.target_mean_sq[0] - self.target_mean[0]**2)
        if self.pres:
            self.target_mean[1] /= c
            self.target_mean_sq[1] /= c
            self.target_std[1] = np.sqrt(self.target_mean_sq[1] - self.target_mean[1]**2)
            
            
                    
    def get_input_normalisation_cst(self):
        # taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        
        
        tc_ids_all = list(self.valid_dates.keys())
        t = time()
        l_tot = 0
        for season in self.train_seasons:
            if False in [os.path.isfile(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_{x}.npy") for x in\
                         ["input_mean_x", "input_mean_x_sq", "input_counter"]]:
                print(f"Computing normalisation constants for inputs ({season}) ... ")
                meanX = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
                meanX_sq = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
                count = 0
                
                tc_ids = [tc_id for tc_id in tc_ids_all if tc_id in self.ibtracs_df[self.ibtracs_df["SEASON"]==str(season)]["SID"].values]
                tmp_paths = [path for path in self.data_list if get_tc_id_nc(path[0]) in tc_ids]
                l = len(tmp_paths)
                
                for i, (path, idxs, _) in enumerate(tmp_paths):
                    if (i+1)%(l//5)==0 or i==1:
                        print(f"Processing file {l_tot+i+1}/{len(self.data_list)} - Running time: {time()-t}s\n"+\
                        f"Expected time left: {(time()-t)/(i+1+l_tot)*(len(self.data_list)-i-1-l_tot):.2f}s (total)\n"+
                        f"                    {(time()-t)/(i+1)*(l-i-1):.2f}s (Season)")
                    
                    ds = xr.load_dataset(path)
            
                    for idx in idxs:
                        count += 1
                        ds_tmp = ds.isel(time=idx)
                        try:
                            wind_sq = ds_tmp.u10.values**2 + ds_tmp.v10.values**2
                        except AttributeError:
                            print(f"Error with file {path}")
                            del ds_tmp
                            continue
                        if np.count_nonzero(np.isnan(wind_sq))>0:
                            print(f"Nans in wind for file {path}")
                            del ds_tmp
                            count -= 1
                            continue
                        meanX[0] += wind_sq**0.5
                        meanX_sq[0] += wind_sq
                        
                        if self.pres:
                            pres = ds_tmp.msl.values
                            if np.count_nonzero(np.isnan(pres))>0:
                                print(f"Nans in pres for file {path}")
                                del ds_tmp
                                count -= 1
                                continue
                            meanX[1] += pres
                            meanX_sq[1] += pres**2
                    del ds
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_mean_x.npy", np.array(meanX)/count)
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_mean_x_sq.npy", np.array(meanX_sq)/count)
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_counter.npy", np.array(count))
                
                l_tot += l
        
        c = 0
        self.mean = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
        self.mean_sq = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
        self.std = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
        
        for season in self.train_seasons:
            counter = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_counter.npy").item()
            c += counter
            
            tmp_mean = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_mean_x.npy")
            tmp_mean_sq = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_mean_x_sq.npy")
            if np.count_nonzero(np.isnan(tmp_mean))>0:
                print(f"Error with season {season}")
                print(f"Counter = {counter}")
                raise ValueError("Nan values in mean")
            if np.count_nonzero(np.isnan(tmp_mean_sq))>0:
                print(f"Error with season {season}")
                print(f"Counter = {counter}")
                raise ValueError("Nan values in mean_sq")
            
            self.mean[0] += tmp_mean[0]*counter
            self.mean_sq[0] += tmp_mean_sq[0]*counter
            
            if self.pres:
                self.mean[1] += tmp_mean[1]*counter
                self.mean_sq[1] += tmp_mean_sq[1]*counter
                
        self.mean[0] /= c
        self.mean_sq[0] /= c
        self.std[0] = np.sqrt(self.mean_sq[0] - self.mean[0]**2)
        
        if self.pres:
            self.mean[1] /= c
            self.mean_sq[1] /= c
            self.std[1] = np.sqrt(self.mean_sq[1] - self.mean[1]**2)
            
            
    def __len__(self):
        return sum(self.ll)
    
    
    def __getitem__(self, idx):
        l = 0
        i = 0
        if idx <= self.data_list_scratch[0][2]:
            with open(self.data_list_scratch[0][0], "rb") as f:
                data = pickle.load(f)
            fields, coords, truth = data
            fields = (fields - np.array(self.mean)) / np.array(self.std)
            truth = (truth - np.array(self.target_mean)) / np.array(self.target_std)
            return torch.tensor(fields), torch.tensor(coords), torch.tensor(truth)
        else:
            while idx > l:
                l += self.data_list_scratch[i][2]
                i += 1
            l -= self.data_list_scratch[i-1][2] # we want the biggest l such that l<=idx and ll[i]+l > idx
            i -= 1
            with open(self.data_list_scratch[i][0], "rb") as f:
                data = pickle.load(f)
            fields, coords, truth = data
            fields = (fields - np.array(self.mean)) / np.array(self.std)
            truth = (truth - np.array(self.target_mean)) / np.array(self.target_std)
            return torch.tensor(fields), torch.tensor(coords), torch.tensor(truth)
            
    
    def save_as_np(self):
        save_folder = "/scratch/lpoulain/"
        for season in self.seasons:
            if not os.path.isdir(save_folder+f"{season}"):
                os.mkdir(save_folder+f"{season}")
        idxs = [i for i, path in enumerate(self.data_list_scratch) if not os.path.isfile(path[0])]
        if len(idxs)!=0:
            print(f"Saving data ({len(idxs)} instances) as numpy arrays ...")
            with Pool(min(os.cpu_count()//2, len(idxs))) as p:
                p.map(self.save_one_as_np, idxs)
        print("All data saved as numpy arrays.")
    
    
    def save_one_as_np(self, idx):
        save_loc = self.data_list_scratch[idx][0]
        if not os.path.isfile(save_loc):
            ds = xr.open_dataset(self.data_list[idx][0])
            time0 = get_start_date_nc(self.data_list[idx][0])
            
            ds = ds.isel(time=self.data_list[idx][1])
            
            tc_id = get_tc_id_nc(self.data_list[idx][0])
            df = self.ibtracs_df[self.ibtracs_df["SID"]==tc_id]
            
            start_lat, start_lon = df[df["ISO_TIME"].astype("datetime64[ns]")==time0][["LAT", "LON"]].values[0]
            start_lat, start_lon = float(start_lat), float(start_lon)
            
            if not isinstance(ds.time.values[0], np.datetime64):
                for val in ds.time.values:
                    print(val, type(val))
                    print(get_start_date_nc(self.data_list[idx][0]), type(get_start_date_nc(self.data_list[idx][0])))
                    print(val+get_start_date_nc(self.data_list[idx][0]))
            
            times = ds.time.values if isinstance(ds.time.values[0], np.datetime64) else\
                    [np.datetime64(val + get_start_date_nc(self.data_list[idx][0])) for val in ds.time.values]
                    
            end_lats = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)]["LAT"].values.astype(float)
            end_lons = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)]["LON"].values.astype(float)
            
            ldts = [np.timedelta64((times[i]-time0), "h").astype(float) for i in range(len(times))]
            
            wind_truth = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)]["USA_WIND"].values.astype(float)
            wind = (ds.u10.values**2 + ds.v10.values**2)**0.5
            
            if self.pres:
                pres = ds.msl.values
                pres_truth = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)][self.pres_columns[tc_id]].values.astype(float)
            
            truth = np.column_stack([wind_truth, pres_truth, [end_lat-start_lat for end_lat in end_lats], 
                                     [end_lon-start_lon for end_lon in end_lons]]) if self.pres else\
                    np.column_stack([wind_truth, [end_lat-start_lat for end_lat in end_lats], [end_lon-start_lon for end_lon in end_lons]])

            ds.close()
            del ds
            if self.pres:
                with open(save_loc, "wb") as f:
                    pickle.dump((np.concatenate([wind.reshape(1, *wind.shape), pres.reshape(1, *pres.shape)], axis=0),
                        [start_lat, start_lon], ldts, truth), f)
            else:
                with open(save_loc, "wb") as f:
                    pickle.dump((wind, [start_lat, start_lon], ldts, truth), f)