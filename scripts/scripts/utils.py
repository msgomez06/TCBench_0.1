import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
import random
from multiprocessing import Pool, cpu_count
import pickle
import subprocess, time

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



def subtract_ibtracs_iso_times(iso_time1:str, iso_time2:str) -> float:
    # returns the time difference in hours
    
    nb_hours = (np.datetime64(iso_time2, 'm') - np.datetime64(iso_time1, 'm')).astype(float)/60
    
    return nb_hours



def max_historical_distance_within_step(df: pd.DataFrame, step: int=6) -> int:
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
        
        

def haversine(latp, lonp, lat_list, lon_list, **kwargs):
    """──────────────────────────────────────────────────────────────────────────┐
      Haversine formula for calculating distance between target point (latp,
      lonp) and series of points (lat_list, lon_list). This function can handle
      2D lat/lon lists, but has been used with flattened data

      Based on:
      https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97


      Inputs:
          latp - latitude of target point

          lonp - longitude of target point

          lat_list - list of latitudess (lat_p-1, lat_p-2 ... lon_p-n)

          lon_list - list of longitudess (lon_p-1, lon_p-2 ... lon_p-n)

      Outputs:

    └──────────────────────────────────────────────────────────────────────────"""
    kwargs.get("epsilon", 1e-6)

    latp = np.radians(latp)
    lonp = np.radians(lonp)
    lat_list = np.radians(lat_list)
    lon_list = np.radians(lon_list)

    dlon = lonp - lon_list
    dlat = latp - lat_list
    a = np.power(np.sin(dlat / 2), 2) + np.cos(lat_list) * np.cos(latp) * np.power(
        np.sin(dlon / 2), 2
    )

    # Assert that sqrt(a) is within machine precision of 1
    # assert np.all(np.sqrt(a) <= 1 + epsilon), 'Invalid argument for arcsin'

    # Check if square root of a is a valid argument for arcsin within machine precision
    # If not, set to 1 or -1 depending on sign of a
    a = np.where(np.sqrt(a) <= 1, a, np.sign(a))

    return 2 * 6371 * np.arcsin(np.sqrt(a))



def get_rectmask(point, grid, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        # https://stackoverflow.com/a/50140132 for an explanation of unravel_index
        grid = np.meshgrid(*grid)
        distance_calculator = kwargs.get("distance_calculator", haversine)
        circum_points = kwargs.get("circum_points", 4)

        distances = distance_calculator(
            point[0],
            point[1],
            grid[0],
            grid[1],
        ).T[np.newaxis, :, :]

        min_idx = np.unravel_index(distances.argmin(), distances.shape)

        output = np.zeros_like(distances)
        output[
            min_idx[0],
            min_idx[1] - circum_points : min_idx[1] + circum_points + 1,
            min_idx[2] - circum_points : min_idx[2] + circum_points + 1,
        ] = 1
        print(output)
        return output


def closest_longitude(lat, lon, lon_list, **kwargs):

    lat, lon, lon_list = float(lat), float(lon), np.array(lon_list).astype(float)
    distance_calculator = kwargs.get("distance_calculator", haversine)
    distances = distance_calculator(lat, lon, [lat], lon_list).T
    
    min_idx = np.unravel_index(distances.argmin(), distances.shape)

    return lon_list[min_idx].item()


def closest_lat(lat, lon, lat_list, **kwargs):

    lat, lon, lat_list = float(lat), float(lon), np.array(lat_list).astype(float)
    distance_calculator = kwargs.get("distance_calculator", haversine)
    distances = distance_calculator(lat, lon, lat_list, [lon]).T
    
    min_idx = np.unravel_index(distances.argmin(), distances.shape)

    return lat_list[min_idx].item()
    

def cut_rectangle(ds: xr.Dataset, df_tracks: pd.DataFrame, tc_id, tropics=True) -> xr.Dataset:
    # ds: dataset to cut
    # df_tracks: dataframe containing the tracks
    # tc_id: id of the TC to cut

    tc_track = df_tracks[(df_tracks["SID"]==tc_id)]# & (df_tracks["ISO_TIME"]==iso_time)
    start_lat, start_lon = tc_track["LAT"].values[0], float(tc_track["LON"].values[0])
    if float(start_lon) < 0:
        start_lon = float(start_lon)+360
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    
    closest_lon = closest_longitude(start_lat, start_lon, ds[lon_name].values)
    if not tropics:
        closest_lat = closest_lat(start_lat, closest_lon, ds[lat_name].values)
        if ds[lat_name].values[0]>ds[lat_name].values[-1]:
            lats = np.arange(closest_lat+30, closest_lat-30.25, -0.25)
        else:
            lats = np.arange(closest_lat-30, closest_lat+30.25, 0.25)
    else:
        if ds[lat_name].values[0]>ds[lat_name].values[-1]:
            lats = np.arange(30, -30.25, -0.25)
        else:
            lats = np.arange(-30, 30.25, 0.25)
        
    if lon_name=="lon":
        #if ds.lat.values[0] < ds.lat.values[-1]:
        #    ds = ds.reindex(lat=list(reversed(ds.lat.values)))
        if closest_lon-30>0 and closest_lon+30<360:
            ds_new = ds.sel(lat=lats, lon=slice(closest_lon-30, closest_lon+30))
        elif closest_lon-30<0:
            ds_new = ds.sel(lat=lats, lon=((ds["lon"]>=0) & (ds["lon"]<=closest_lon+30)) | \
                                                    ((ds["lon"]>=360+closest_lon-30) & (ds["lon"]<=360)))
        else:
            ds_new = ds.sel(lat=lats, lon=((ds["lon"]>=closest_lon-30) & (ds["lon"]<=360)) | \
                                                    ((ds["lon"]>=0) & (ds["lon"]<=closest_lon+30-360)))
    else:
        #if ds.latitude.values[0] < ds.latitude.values[-1]:
        #    ds = ds.reindex(lat=list(reversed(ds.latitude.values)))
        if closest_lon-30>0 and closest_lon+30<360:
            ds_new = ds.sel(latitude=lats, longitude=slice(closest_lon-30, closest_lon+30))
        elif closest_lon-30<0:
            ds_new = ds.sel(latitude=lats, longitude=((ds["longitude"]>=0) & (ds["longitude"]<=closest_lon+30)) | \
                                                    ((ds["longitude"]>=360+closest_lon-30) & (ds["longitude"]<=360)))
        else:
            ds_new = ds.sel(latitude=lats, longitude=((ds["longitude"]>=closest_lon-30) & (ds["longitude"]<=360)) | \
                                                    ((ds["longitude"]>=0) & (ds["longitude"]<=closest_lon+30-360)))
    
    return ds_new


def cut_and_save_rect(ds_folder, models, df_tracks:pd.DataFrame, date_start, date_end, lead_time, tc_id, output_path, l=None, idx=None):

    assert set(models).issubset(["pangu", "graphcast", "fourcastnetv2"]), f"models must be in ['pangu', 'graphcast', 'fourcastnetv2']"
    
    folder_names = {"pangu":"panguweather", "graphcast":"graphcast", "fourcastnetv2":"fourcastnetv2"}
    
    for i, model in enumerate(models):
            save_name = output_path + f"{folder_names[model]}/{model}_{date_start}_to_{date_end}_ldt_{lead_time}_{tc_id}_small.nc"
            if not os.path.isfile(save_name):
                msg = f"{date_start} to {date_end} ({lead_time}h)" + f" - {idx+1}/{l}" if (l is not None and idx is not None) else ""
                print(msg)
                ds = xr.load_dataset(ds_folder + f"{folder_names[model]}/{model}_{date_start}_to_{date_end}_ldt_{lead_time}_{tc_id}.nc", 
                                    engine="netcdf4")
                try:
                    ds_new = cut_rectangle(ds, df_tracks, tc_id)
                    ds_new.to_netcdf(save_name, engine="netcdf4")
                    
                    del ds, ds_new
                except KeyError:
                    del ds
                    continue

            
def cut_save_in_series(ds_folder, models, year, output_path, parallel=False, remove=False, remove_waiting=15,
                       df_tracks_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    
    assert set(models).issubset(["pangu", "graphcast", "fourcastnetv2"]), "models must be in ['pangu', 'graphcast', 'fourcastnetv2']"
    
    folder_names = {"pangu":"panguweather", "graphcast":"graphcast", "fourcastnetv2":"fourcastnetv2"}
    
    key = lambda x: (get_start_date_nc(x), get_lead_time(x))
    
    params = {model : [(get_start_date_nc(file), get_end_date_nc(file), get_lead_time(file), get_tc_id_nc(file), file) for file in \
        sorted(list(set([f for f in glob.glob(ds_folder+f"{folder_names[model]}/{model}_{year}*_ldt_*_*.nc") if \
            len(os.path.basename(f).split("_"))==7])\
                    -set(glob.glob(ds_folder+f"{folder_names[model]}/{model}_{year}_*small.nc"))), key=key)] for model in models}
    
    df_tracks = pd.read_csv(df_tracks_path, dtype="string", na_filter=False)
    for model in models:
        l = len(params[model])
        done = []
        if not parallel:
            for i in range(len(params[model])):
                date_start, date_end, ldt, tc_id, file = params[model][i]
                cut_and_save_rect(ds_folder, [model], df_tracks, date_start, date_end, ldt, tc_id, output_path, l=l, idx=i)
                done.append(file)
                if len(done) == 50:
                    if remove:
                        print(f"!!! Removing 50 old files !!! {remove_waiting}s to STOP if this is a mistake", flush=True)
                        import time
                        time.sleep(remove_waiting)
                        subprocess.run(["rm", "-r", *done])
                        done = []
                if i == len(params[model])-1:
                    if remove:
                        print(f"!!! Removing {len(done)} old files !!! {remove_waiting}s to STOP if this is a mistake", flush=True)
                        import time
                        time.sleep(remove_waiting)
                        subprocess.run(["rm", "-r", *done])
                        done = []
        else:
            nb_cpus = cpu_count()
            with Pool(nb_cpus//2) as p:
                print(f"Using {nb_cpus//2} cpus")
                p.starmap(cut_and_save_rect, [(ds_folder, [model], df_tracks, date_start, date_end, ldt,\
                    tc_id, output_path, l, i) for i, (date_start, date_end, ldt, tc_id, file) in enumerate(params[model])])
        
        
            
def renaming(folder_name, model="fourcastnetv2", year=2000, remove_old=False, remove_waiting:int=15, cut=False,
            ibtracs_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    print(year, model, flush=True)
    df_ibtracs = pd.read_csv(ibtracs_path, dtype="string", na_filter=False)
    model_folder = "panguweather" if model=="pangu" else model
    
    key = lambda x: (get_start_date_nc(x), get_lead_time(x))
    
    ldts = [l for l in range(6,174,6)]
    filelist = []
    for ldt in ldts:
        filelist.extend(glob.glob(folder_name + f"{model_folder}/{model}_{year}*_ldt_{ldt}.nc"))
    fileList = sorted(filelist, key=key)
    
    i = 1
    done = []
    for file in fileList:
        start_iso_time, end_iso_time = get_start_date_nc(file), get_end_date_nc(file)
        print(f"{i}/{len(fileList)}: {start_iso_time} to {end_iso_time}", flush=True)
        i += 1
        tc_ids = []
        df_track_tmp = df_ibtracs[df_ibtracs["ISO_TIME"].astype("datetime64[ns]")==start_iso_time]
        possible_ids = df_track_tmp["SID"].unique()
        for id in possible_ids:
            df_track_tmp2 = df_ibtracs[df_ibtracs["SID"]==id]
            if df_track_tmp2[df_track_tmp2["ISO_TIME"].astype("datetime64[ns]")==end_iso_time]["ISO_TIME"].values.shape[0] > 0:
                tc_ids.append(id)
        
        if False in [os.path.isfile(os.path.dirname(file) + f"/{model}_{start_iso_time}_to"+\
            f"_{end_iso_time}_ldt_{get_lead_time(file)}_{tc_id}.nc") for tc_id in tc_ids]:
            try:
                ds = xr.open_dataset(file, engine="netcdf4")
            except OSError: # some files are corrupted
                done.append(file)
                continue
            for tc_id in tc_ids:
                save_name = os.path.dirname(file) + f"/{model}_{start_iso_time}_to_{end_iso_time}_ldt_{get_lead_time(file)}_{tc_id}.nc"
                if not os.path.isfile(save_name):
                    ds.to_netcdf(save_name, engine="netcdf4")
        
        done.append(file)
        if len(done) >= 50:
            if remove_old:
                print(f"!!! Removing 50 old files !!! {remove_waiting}s to STOP if this is a mistake", flush=True)
                import time
                time.sleep(remove_waiting)
                subprocess.run(["rm", "-r", *done])
                done = []
            if cut:
                cut_save_in_series(ds_folder=folder_name, models=[model], year=year, output_path=folder_name, parallel=False, remove=remove_old,
                                   df_tracks_path=ibtracs_path)
        if file==fileList[-1]:
            if remove_old:
                print(f"!!! Removing {len(done)} old files !!! {remove_waiting}s to STOP if this is a mistake", flush=True)
                import time
                time.sleep(remove_waiting)
                subprocess.run(["rm", "-r", *done])
                done = []
            if cut:
                cut_save_in_series(ds_folder=folder_name, models=[model], year=year, output_path=folder_name, parallel=False, remove=remove_old,
                                   df_tracks_path=ibtracs_path)
    print("Done", flush=True)


        

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
                
        
        
    
def date_time_netcdf_to_ibtracs(date_str: str, lead_time: int=0) -> str:
    # Provide a lead_time in case the date is not the initialisation date
    
    return np.datetime64(np.datetime64(date_str) + np.timedelta64(lead_time, 'h'))


def date_time_nn_to_netcdf(date: str, time:str, ldt:int=0) -> np.datetime64:
    # date is of the form yyyymmdd
    # time of the form hhmm
    # the output will be of the form yyyy-mm-ddThh:mm:ss.000000000
    
    if len(time) > 2:
        time = time[:2]
    out = date[:4] + "-" + date[4:6] + "-" + date[6:]
    out = np.datetime64(out, 'h') + np.timedelta64(int(time), 'h') + np.timedelta64(ldt, 'h')
    return np.datetime64(out)
    
    
def date_ibtracs_to_nn(date: str) -> str:
    # in ibtracs dates are of the yyyy-mm-dd
    # NNs take date inputs of the form yyyymmdd
    date_split = date.split("-")
    date_new = str("".join(date_split))
    return date_new



def time_ibtracs_to_nn(time: str) -> str:
    # in ibtracs times are of the form hh:mm:ss
    # NNs take times inputs of the form hhmm
    time_split = time.split(":")
    time_new = str("".join(t for t in time_split[:2]))
    return time_new



def pattern_date(dates: list):
    dates_str = list(map(str, dates))
    pattern = str("|".join([date for date in dates_str]))
    return pattern



def pattern_time(times: list):
    times = list(map(str, times))
    pattern = str("|".join([time for time in times]))
    return pattern



def extract_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    return df[df["SEASON"]==season]



def get_all_iso_times(df: pd.DataFrame, TC_id=None, season=None):
    # to be removed: TC year
    assert season is not None or TC_id is not None, "TC_id or season must be specified"
    
    if TC_id is not None:
        df_TC = df[df["SID"]==TC_id]
    else:
        df_TC_tmp = extract_season(df, season=season)
        TC_id = df_TC_tmp["SID"].values[1]
        print(f"TC id: {TC_id} ({season})")
        df_TC = df[df["SID"]==TC_id]
        
            
    return df_TC["ISO_TIME"].values, TC_id



def write_params_for_tc(output_path: str, df: pd.DataFrame, TC_id=None, season=None, step=6, max_lead=168, **kwargs):

    debug = kwargs.get("debug", False)
    
    iso_times, TC_id = get_all_iso_times(df=df, TC_id=TC_id, season=season)
    end_iso_time = iso_times[-1]
    start_iso_time = iso_times[0]
    
    start_date, start_time = date_ibtracs_to_nn(start_iso_time.split(" ")[0]), time_ibtracs_to_nn(start_iso_time.split(" ")[1])
    dates, times, lead_times = [start_date], [start_time], [min(int(subtract_ibtracs_iso_times(start_iso_time, end_iso_time)), max_lead)]
    
    current_iso = start_iso_time
    i = 1
    if max_lead==6:
        for iso in iso_times[1:-1]:
            dates.append(date_ibtracs_to_nn(iso.split(" ")[0]))
            times.append(time_ibtracs_to_nn(iso.split(" ")[1]))
            lead_times.append(6)
            
    while lead_times[-1] > step:
        
        current_iso = iso_times[i]
        date, time = date_ibtracs_to_nn(current_iso.split(" ")[0]), time_ibtracs_to_nn(current_iso.split(" ")[1])
        
        lead_time = min(int(subtract_ibtracs_iso_times(current_iso, end_iso_time)), max_lead)
        
        dates.append(date)
        times.append(time)
        lead_times.append(lead_time)
        i += 1

    timesteps = [True] + [False for i in range(len(times)-1)]
    prev_t = times[0]
    for i, t in enumerate(times[1:]):
        if abs(int(t)//100-int(prev_t)//100)%step==0:
            prev_t = t
            timesteps[i+1] = True
            
    dates, times, lead_times = ["date"] + [int(d) for i, d in enumerate(dates) if timesteps[i]], \
                                ["time"] + [t for i, t in enumerate(times) if timesteps[i]], \
                                ["lead time"] + [int(lt) for i, lt in enumerate(lead_times) if timesteps[i]]
    ids = ["ArrayTaskID"] + [i for i in range(len(dates)-1)]

    filename = f"input_params_{TC_id}_step_{step}_max_{max_lead}h.txt"
    with open(output_path + filename, "w") as w:
        col_format = "{:<12}" + "{:<9}" + "{:<5}" + "{:<9}" + "\n"
        if debug:
            data = np.column_stack((ids[:10], dates[:10], times[:10], lead_times[:10]))
        else:
            data = np.column_stack((ids, dates, times, lead_times))
        for x in data:
            w.write(col_format.format(*x))
            


def write_params_for_period(output_path, start_date:str="20200301", start_time="0000", end_date:str="20200831", end_time="0000", max_lead:int=168, step:int=12, **kwargs):
    # step: start a new forecast every <step> hours
    # complementary to write_input_params_to_file but this function does not take care if th date/time is in IBTrACS (more general)
    
    debug = kwargs.get("debug", False)
    filename = f"input_params_{start_date}T{str(int(start_time)).zfill(2)}_to_{end_date}T{str(int(end_time)).zfill(2)}_step_{step}_{max_lead}.txt"
    
    dates, times = ["date", start_date], ["time", start_time]
    end_iso_time = np.datetime64(date_time_nn_to_netcdf(end_date, end_time, -step), 'h') # last starting date can only be -step hours before the last known location of the TC
    current_date = date_time_nn_to_netcdf(dates[-1], times[-1])
    
    ids, ldts = ["ArrayTaskID", 0], ["lead time", min(subtract_ibtracs_iso_times(current_date, end_iso_time), max_lead)]
    
    while subtract_ibtracs_iso_times(current_date, end_iso_time) >= max_lead: # >= because last iso correspond to last_time-step
        
        current_date = current_date + np.timedelta64(step, 'h')
        
        ids.append(ids[-1]+1)
        dates.append(str(current_date).split("T")[0].replace("-", ""))
        times.append(str(int(str(current_date).split("T")[1])*100).zfill(4))
        ldts.append(max_lead)
    
    ldt = max_lead
    while subtract_ibtracs_iso_times(current_date, end_iso_time) >= step: # >= because last iso correspond to last_time-step
        
        current_date = current_date + np.timedelta64(step, 'h')
        ldt = ldt-step
        
        ids.append(ids[-1]+1)
        dates.append(str(current_date).split("T")[0].replace("-", ""))
        times.append(str(int(str(current_date).split("T")[1])*100).zfill(4))
        ldts.append(ldt)
        
        
    with open(output_path+filename, 'w') as w:
        col_format = "{:<12}" + "{:<9}" + "{:<5}" + "{:<9}" + "\n"
    
        if debug:
            data = np.column_stack((ids[:10], dates[:10], times[:10], ldts[:10]))
        else:
            data = np.column_stack((ids, dates, times, ldts))
        for x in data:
            w.write(col_format.format(*x))
            
            

def write_several_seasons(output_path:str, seasons:list=[2016,2017,2018,2019,2020], step=6, max_lead=168, 
                          ibtracs_path:str='/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/ibtracs.ALL.list.v04r00.csv',
                          **kwargs):
    
    ibtracs_df = pd.read_csv(ibtracs_path, dtype="string", na_filter=False)
    if ibtracs_path[-6:-4]=="00":
        ibtracs_df = ibtracs_df.loc[1:]
    
    all_tcs = kwargs.get("all_tcs", False)
    as_range = kwargs.get("as_range", False)
    
    if as_range:
        seasons = list(range(int(seasons[0]), int(seasons[-1])+1))
    print(f"Seasons: {seasons}")
    
    inputs = []
    for season in seasons:
        df_year = ibtracs_df[ibtracs_df['SEASON']==str(season)]
        basins = df_year['BASIN'].unique()
        
        for basin in basins:
            sids = df_year[df_year['BASIN']==basin]['SID'].unique()
            if not all_tcs:
                sids = [random.sample(list(sids), 1)[0]] # select one tc for each basin randomly
            
            for sid in sids:
                fname = output_path + f"{season}/input_params_{sid}_step_{step}_max_{max_lead}h.txt"
                if not os.path.isfile(fname):
                    if not os.path.isdir(output_path+f"{season}/"):
                        os.mkdir(output_path+f"{season}/")
                    write_params_for_tc(output_path+f"{season}/", ibtracs_df, TC_id=sid, season=season, step=step, max_lead=max_lead)
                inputs.append(fname)
    return sorted(list(set(inputs))) # some TCs may change basin during their lifetime, so we need to remove duplicates


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
    