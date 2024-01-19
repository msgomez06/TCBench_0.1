import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
import pickle

def remove_consecutive_elements(lst: list, nb_idx:int) -> bool:
    
    if len(lst) == 0:
        return True
    
    if len(lst) > nb_idx//2:
        return False
    
    if lst[0] == 0:
        i = 0
        while i < len(lst) - 1 and lst[i+1] == lst[i] + 1:
            i += 1
        lst = lst[i+1:]
    
    if len(lst) == 0:
        return True
    
    if lst[-1] == nb_idx-1:
        i = -1
        while i > -len(lst) and lst[i-1] == lst[i] - 1:
            i -= 1
        lst = lst[:i]
    
    return len(lst) == 0



def filter_tracks(path_ibtracs="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/ibtracs.ALL.list.v04r00.csv", 
                  path_output='/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/',
                  min_year=1980, hours:list=["00", "06", "12", "18"]):
    
    df_init = pd.read_csv(path_ibtracs, dtype="string", na_filter=False).loc[1:]
    
    df_year = df_init[df_init["SEASON"].astype(int)>=min_year]
    df_tropical = df_year[~df_year["NATURE"].isin([''])]
    
    def get_hours(iso_time):
        return iso_time.split(" ")[1].split(':')[0]
    def get_year(iso_time):
        return iso_time.split(" ")[0].split('-')[0]
    def get_month(iso_time):
        return iso_time.split(" ")[0].split('-')[1]
    def get_day(iso_time):
        return iso_time.split(" ")[0].split('-')[2]
    
    df_hours = df_tropical[df_tropical["ISO_TIME"].apply(get_hours).isin(hours)]
    sids = df_hours["SID"].unique()
    
    sids_remove = []
    print(f"Number of TCs: {len(df_hours['SID'].unique())}")
    for sid in sids:
        df_tmp = df_hours[df_hours["SID"]==sid]
        lst = set(df_tmp[df_tmp["USA_WIND"].isin(["", " "])].reset_index(drop=True).index)
        #lst = lst.union(set(df_tmp[df_tmp["USA_PRES"].isin(["", " "])].reset_index(drop=True).index))
        lst = sorted(list(lst))
        
        if not remove_consecutive_elements(lst, len(df_tmp.index)):
            sids_remove.append(sid)
    
    df_final = df_hours[~df_hours["SID"].isin(sids_remove)]
    sids = df_final["SID"].unique()
    print(f"Number of TCs: {len(sids)}")
    valid_dates = {}
    for sid in df_final["SID"].unique():
        df_tmp = df_final[df_final["SID"]==sid]
        valid_years = df_tmp["ISO_TIME"].apply(get_year).unique().tolist()
        for year in valid_years:
            if not year in valid_dates.keys():
                valid_dates[year] = {}
            df_year_tmp = df_tmp[df_tmp["ISO_TIME"].apply(get_year)==year]
            valid_months = df_year_tmp["ISO_TIME"].apply(get_month).unique().tolist()
            for month in valid_months:
                df_tmp_month = df_year_tmp[df_year_tmp["ISO_TIME"].apply(get_month)==month]
                valid_days = df_tmp_month["ISO_TIME"].apply(get_day).unique().tolist()
                
                if not month in valid_dates[year].keys():
                    valid_dates[year][month] = []
                valid_dates[year][month].append(valid_days)
    for year in valid_dates.keys():
        for month in valid_dates[year].keys():
            valid_dates[year][month] = list(np.unique(sorted(flatten(valid_dates[year][month]))))

    with open(f'{path_output}valid_dates_{min_year}_{"_".join(hour for hour in hours)}.pkl', 'wb') as f:
        pickle.dump(valid_dates, f)
    
    df_final.to_csv(f'{path_output}TC_track_filtered_{min_year}_{"_".join(hour for hour in hours)}.csv', index=False)
    
       
def flatten(arg):
    # flatten list of any depth into a list of depth 1
    if not isinstance(arg, list): # if not list
        return [arg]
    return [x for sub in arg for x in flatten(sub)] # recurse and collect



def max_historical_distance_within_step(df: pd.DataFrame, step: int=6) -> int:
    from params_writers import subtract_ibtracs_iso_times
    from cut_region import haversine
    max_dist = 0
    max_dists = []
    dists = []
    dists_lats, dists_lons = [], []
    
    c = 1
    
    # reduce the dataframe to when teledetction started
    df = pd.concat((df.loc[0].to_frame().T,df[1:][df.loc[1:,"SEASON"].astype(int)>1970]), axis=0)
    
    tc_ids = df.loc[1:]["SID"].unique()
    
    l = len(tc_ids)
    tc_id_longest = []
    index_longest = []
    
    
    for tc_id in tc_ids:
        if c == 1 or c % (l//10) == 0:
            print(f"{c}/{l}")
            
        df_tmp = df[df["SID"]==tc_id]
        if len(df_tmp.index) > 1:
            lat_init, lon_init, iso_time_init = float(df_tmp["LAT"].values[0]), float(df_tmp["LON"].values[0]), df_tmp["ISO_TIME"].values[0]
            time_diff = 0.0
            idx_start = 0
            idx_next = df_tmp.index[1]
            # tout à chnager !!!
            while idx_next != df_tmp.index[-1]:
                time_diff += subtract_ibtracs_iso_times(iso_time_init, df_tmp.loc[idx_next]["ISO_TIME"])
                if time_diff <= step:
                    idx_next += 1
                else:
                    latp, lonp = [float(df_tmp.loc[idx_next]["LAT"])], [float(df_tmp.loc[idx_next]["LON"])]
                    dist = haversine(lat_init, lon_init, latp, lonp).item() * step / time_diff
                    dist_lat = haversine(lat_init, lonp[0], latp, lonp).item() * step / time_diff, 
                    dist_lon = haversine(lat_init, lon_init, [lat_init], lonp).item() * step / time_diff
                    dists.append(dist)
                    dists_lats.append(dist_lat)
                    dists_lons.append(dist_lon)
                    if dist > max_dist:
                        max_dist = dist
                        max_dists.append(max_dist)
                        tc_id_longest.append(tc_id)
                        index_longest.append([idx_start, idx_next])
                        
                    idx_start += 1
                    lat_init, lon_init = float(df_tmp["LAT"].values[idx_start]), float(df_tmp["LON"].values[idx_start])
                    iso_time_init = df_tmp["ISO_TIME"].values[idx_start]
                    time_diff = 0.0
                
        c += 1
        
    with open("./max_distances.txt", "a") as f:
        f.write(f"Max dist {step}h: {max_dist}km (TC {tc_id_longest[-1]}, idx {index_longest[-1]})\n")
    print(f"Max dist: {max_dist}km (TC {tc_id_longest[-1]}, idx {index_longest[-1]}).")
    
    np.save(f"./{step}h_maxs.npy", np.array(max_dists))
    np.save(f"./{step}h_idxs.npy", np.array(index_longest))
    np.save(f"./{step}h_tc_ids.npy", np.array(tc_id_longest))
    np.save(f"./{step}h_dists.npy", np.array(dists))
    np.save(f"./{step}h_dists_lats.npy", np.array(dists_lats))
    np.save(f"./{step}h_dists_lons.npy", np.array(dists_lons))
    return max_dist
        
        
            



        


                    
            
        
def combine_in_series(output_path, model_name, remove_old=False, start_in="",
                      ibtracs_path="/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/ibtracs.ALL.list.v04r00.csv"):
    
    model_folder = "panguweather" if model_name=="pangu" else model_name
    key = lambda x: (get_date_time(x), get_lead_time(x))
    
    predictions_list = sorted(glob.glob(output_path + f"{model_folder}/{model_name}_d_*h.grib"), key=key)
    
    dates_times = []
    for pred in predictions_list:
        date = get_date_time(pred)[:8]
        time = get_date_time(pred)[-4:]
        if (date, time) not in dates_times and np.datetime64(date_time_nn_to_netcdf(date, time), 'D') >= np.datetime64(start_in, 'D'):
            dates_times.append((date, time))

    for date_time in dates_times:
        date, time = date_time
        
        combine_ai_models_pred(output_path, model_name, date, time,
                                        remove_old=remove_old, ibtracs_path=ibtracs_path)
                


def date_time_nn_to_netcdf(date: str, time:str, ldt:int=0) -> np.datetime64:
    # date is of the form yyyymmdd
    # time of the form hhmm
    # the output will be of the form yyyy-mm-ddThh:mm:ss.000000000
    
    if len(time) > 2:
        time = time[:2]
    out = date[:4] + "-" + date[4:6] + "-" + date[6:]
    out = np.datetime64(out, 'h') + np.timedelta64(int(time), 'h') + np.timedelta64(ldt, 'h')
    return np.datetime64(out)
    
    
def get_date_time(filename: str) -> str:
    filename = os.path.basename(filename)
    date, time = filename.split("_")[2], filename.split("_")[4]
    
    date_time = date + "_" + time
    
    return date_time



def get_lead_time(filename: str) -> int:
    filename = os.path.basename(filename)
    start, end = get_start_date_nc(filename), get_end_date_nc(filename)
    
    ldt = np.timedelta64(np.datetime64(end)-np.datetime64(start), 'h').astype(int)
    return ldt


def get_start_date_nc(filename: str) -> str:
    filename = os.path.basename(filename)
    
    date = filename.split("_")[1]
    
    return np.datetime64(date)


def get_end_date_nc(filename: str) -> str:
    filename = os.path.basename(filename)
    
    date = filename.split("_")[3]
    
    return np.datetime64(date)


def get_tc_id_nc(filename: str) -> str:
    filename = os.path.basename(filename)
    filesplit = filename.split("_")
    if len(filesplit) == 8: # we have 'small' in the file
        sid = filesplit[6]
    else: # len=7 but last element is sid.nc
        sid = filesplit[6].split(".")[0]
    
    return sid


def grib2xarray_compat(filepath):
    import subprocess
        
    if filepath.endswith(".grib"): # remove the extension
        filepath = filepath[:-5]
    subprocess.run(["bash", "-c", f"module load gcc proj cdo && cdo -R remapcon,r1440x721 -setgridtype,regular {filepath+'.grib'} {filepath+'_r.grib'}"])
    subprocess.run(["bash", "-c", f"module load gcc proj cdo && cdo -f nc copy {filepath+'_r.grib'} {filepath+'.nc'}"])
    ds = xr.open_dataset(filepath+'.nc', engine="netcdf4")
    print(np.count_nonzero(ds.to_array()=='nan'), ds.to_array().size)
    ds = ds.reindex(lat=list(reversed(ds.lat)))
    print(np.count_nonzero(ds.to_array()=='nan'), ds.to_array().size)
    ds.to_netcdf(filepath+'.nc')
    #subprocess.run(["bash", "-c", f"module load gcc proj cdo && cdo -f grb copy {filepath+'.nc'} {filepath+'.grib'}"])
    #subprocess.run(["bash", "-c", f"rm {filepath+'_old.grib'} {filepath+'.nc'}"])
    
    
def fast_rename(path, model_name, years):
    
    #model_folder = "panguweather" if model_name=="pangu" else model_name
    #
    #complete_path = path + model_folder + "/"
    #
    #files = []
    #for year in years:
    #    files.extend(glob.glob(complete_path + f"{model_name}_{year}*.nc"))
    #        
    #for file in files:
    #    if os.path.basename(file).find("_6.nc") != -1:
    #        #ldt = get_lead_time(file)
    #        new_name = os.path.basename(file).replace("_6.nc", f".nc")
    #        subprocess.run(["mv", file, complete_path + new_name])
    pass
    
    
    #### OLD ####
    
    """
    def combine_ai_models_pred(output_path, model_name, start_date, start_time, remove_old=False, remove_waiting:int=30,
                           ibtracs_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    
    
    # only to convert ai-models outputs in nc by regrouping all predictions
    df_ibtracs = pd.read_csv(ibtracs_path, dtype="string", na_filter=False)
    
    model_folder = "panguweather" if model_name=="pangu" else model_name
    start_time = str(int(start_time)*100).zfill(4)[:4] if len(start_time) < 4 else start_time
    key = lambda x: (get_date_time(x), get_lead_time(x))
    
    predictions_list = sorted(glob.glob(output_path + f"{model_folder}/{model_name}_d_{start_date}_t_{start_time}_*h.grib"), key=key)

    end_date, end_time = get_date_time(predictions_list[-1])[:8], get_date_time(predictions_list[-1])[-4:]
    ldt = get_lead_time(predictions_list[-1])
    
    start_iso_time = date_time_nn_to_netcdf(start_date, start_time, 0)
    end_iso_time = date_time_nn_to_netcdf(end_date, end_time, ldt)
    
    tc_ids = []
    df_track_tmp = df_ibtracs[df_ibtracs["ISO_TIME"].astype("datetime64[ns]")==start_iso_time]
    possible_ids = df_track_tmp["SID"].unique()
    for id in possible_ids:
        df_track_tmp2 = df_ibtracs[df_ibtracs["SID"]==id]
        if df_track_tmp2[df_track_tmp2["ISO_TIME"].astype("datetime64[ns]") == end_iso_time]["ISO_TIME"].values.shape[0] > 0:
            tc_ids.append(id)
    
    output_names = [f"{model_name}_{start_iso_time}_to_{end_iso_time}_ldt_{ldt}_{tc_id}.nc" for tc_id in tc_ids]
    
    if False in [os.path.isfile(output_path + model_folder + "/"+ output_name) for output_name in output_names]:
        print(start_date, start_time, "\n")
        
        xarr_list = [xr.open_dataset(prediction, engine="cfgrib") for prediction in predictions_list]
        if model_name=="graphcast":
            xarr_list_u10 = [xr.open_dataset(prediction, engine="cfgrib", backend_kwargs={'filter_by_keys': {'shortName': '10u'}}) for prediction in predictions_list]
            xarr_list_v10 = [xr.open_dataset(prediction, engine="cfgrib", backend_kwargs={'filter_by_keys': {'shortName': '10v'}}) for prediction in predictions_list]
        
        if model_name=="pangu":
            xarr_list_t2m = [xr.open_dataset(prediction, engine="cfgrib", backend_kwargs={'filter_by_keys': {'shortName': '2t'}}) for prediction in predictions_list]
        
        for i, xarr in enumerate(xarr_list):
            if "heightAboveGround" in list(xarr.coords):
                xarr = xarr.drop_vars("heightAboveGround")
            if "meanSea" in list(xarr.coords):
                xarr = xarr.drop_vars("meanSea")
                
            xarr["time"] = xarr["valid_time"]
            xarr = xarr.drop_vars(["step", "valid_time"])
            
            if model_name=="graphcast":
                try:
                    xarr_list_u10[i]["time"], xarr_list_v10[i]["time"] = xarr_list_u10[i]["valid_time"], xarr_list_v10[i]["valid_time"]
                except KeyError:
                    print(predictions_list[i])
                    raise KeyError
                xarr_list_u10[i], xarr_list_v10[i] = xarr_list_u10[i].drop_vars(["step", "valid_time"]), xarr_list_v10[i].drop_vars(["step", "valid_time"])
            if model_name=="pangu":
                xarr_list_t2m[i]["time"] = xarr_list_t2m[i]["valid_time"]
                xarr_list_t2m[i] = xarr_list_t2m[i].drop_vars(["step", "valid_time"])
                
            xarr_list[i] = xarr
            if model_name=="graphcast":
                xarr_list[i]["u10"] = xarr_list_u10[i]["u10"]
                xarr_list[i]["v10"] = xarr_list_v10[i]["v10"]

                xarr_list_u10[i].close()
                xarr_list_v10[i].close()
            
            if model_name=="pangu":
                xarr_list[i]["t2m"] = xarr_list_t2m[i]["t2m"]
                xarr_list_t2m[i].close()
        
        xarr_final = xr.concat(xarr_list, dim="time", coords='minimal')
        
        for output_name in output_names:
            if not os.path.isfile(output_path + model_folder + "/"+ output_name):
                xarr_final.to_netcdf(output_path + model_folder + "/"+ output_name)
        
        for xarr in xarr_list:
            xarr.close()
            del xarr
                
    if remove_old:
        print(f"!!! Removing old files !!! {remove_waiting}s to STOP if this is a mistake")
        import time
        time.sleep(remove_waiting)
        subprocess.run(["rm", "-r", *predictions_list])
        idx_list = glob.glob(os.path.dirname(predictions_list[0]) + "/*.idx")
        subprocess.run(["rm", "-r", *idx_list])
    """