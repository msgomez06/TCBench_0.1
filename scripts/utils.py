import xarray as xr
import numpy as np
import os
import glob
import pandas as pd


def subtract_ibtracs_iso_times(iso_time1:str, iso_time2:str) -> float:
    # returns the time difference in hours
    
    date1, time1 = iso_time1.split(" ")
    time1 = int(time1.split(":")[0]) + float(time1.split(":")[1])/60
    date2, time2 = iso_time2.split(" ")
    time2 = int(time2.split(":")[0]) + float(time2.split(":")[1])/60
    
    nb_days = days_spacing(date1, date2, separator="-")
    nb_hours = 24 * nb_days + time2-time1
    
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

        return output
    
    

def cut_rectangle(ds: xr.Dataset, df_tracks: pd.DataFrame, tc_id) -> xr.Dataset:
    
    print(ds["msl"].shape)
    try:
        nb_dates = ds["time"].shape[0]
    except IndexError:
        nb_dates = 1
    
    lats, lons = (ds["latitude"].values[0], ds["longitude"].values[0]) if nb_dates>1 else (ds["latitude"].values, ds["longitude"].values)
    final_mask = np.zeros((nb_dates, lats.shape[0], lons.shape[0]))
    
    for i in range(nb_dates):
        ld = ds["step"] if nb_dates==1 else ds["step"][i]
        lead_time = np.timedelta64(ld.values, 'h').astype(int)
        
        time = ds["time"].values if nb_dates==1 else ds["time"].values[i]
        iso_time = date_time_netcdf_to_ibtracs(time, lead_time=lead_time)
        
        df_tc_id = df_tracks[df_tracks["SID"]==tc_id]
        lat_tc_id, lon_tc_id, isotimes_tc = df_tc_id["LAT"].values, df_tc_id["LON"].values, df_tc_id["ISO_TIME"].values
        point = (*[float(lat_tc_id[i]) for i in range(len(lat_tc_id)) if np.datetime64(isotimes_tc[i])==iso_time], 
                 *[float(lon_tc_id[i]) for i in range(len(lon_tc_id)) if np.datetime64(isotimes_tc[i])==iso_time])
        #point = (float( & (np.datetime64(df_tracks["ISO_TIME"].values)==iso_time)]["LAT"].values[0]), 
        #            float(df_tracks[(df_tracks["SID"]==tc_id) & (np.datetime64(df_tracks["ISO_TIME"].values)==iso_time)]["LON"].values[0]))
        print(point)
        rect_mask = get_rectmask(point, (lats, lons))
        final_mask[i] = rect_mask
    
    print(final_mask.shape)
    ds_new = ds.where(final_mask[0])
    print(ds_new, ds_new["time"].shape)
    raise ValueError("ValueError")
    return ds_new



def cut_and_save_rect(ds_path, ibtracs_df:pd.DataFrame, tc_id, output_path):
    ds = xr.open_dataset(ds_path)
    ds_new = cut_rectangle(ds, ibtracs_df, tc_id)
    ds_new.to_netcdf(output_path)
    ds.close()
    ds_new.close()
    
    
    
def date_time_netcdf_to_ibtracs(date_str: str, lead_time: int=0) -> str:
    # Provide a lead_time in case the date is not the initialisation date
    
    return np.datetime64(np.datetime64(date_str) + np.timedelta64(lead_time, 'h'))
    """date_str = np.datetime_as_string(date_str)#, unit='ns'
    date, time = date_str.split("T")
    time = time[:8] # only keep up to minutes
    hours = int(time[:2]) + lead_time
    if hours >=24:
        days_added = hours // 24
        hours = hours - 24 * days_added
        time = str(hours).zfill(2) + time[2:]
        date = date[:-2] + str(int(date[-2:]) + days_added).zfill(2)
    else:
        time = str(hours).zfill(2) + time[2:]
    iso_time = f"{date} {time}"""
    return iso_time


def date_time_nn_to_netcdf(date: str, time:str) -> np.datetime64:
    # date is of the form yyyymmdd
    # time of the form hhmm
    # the output will be of the form yyyy-mm-ddThh:mm:ss.000000000
    
    out = date[:4] + "-" + date[4:6] + "-" + date[6:]
    out = np.datetime64(out, 'h') + np.timedelta64(int(time), 'h')
    return np.datetime64(out)
    
    
def date_ibtracs_to_nn(date: str) -> str:
    # in ibtracs dates are of the yyyy-mm-dd
    # NNs take date inputs of the form yyyymmdd
    date_split = date.split("-")
    date_new = str("".join(date_split))
    return date_new



def time_ibtracs_to_nn(time: str) -> str:
    # in ibtracs dates are of the hh:mm:ss
    # NNs take date inputs of the form hhmm
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



def extract_data_from_date_pattern_old(df: pd.DataFrame, dates: list) -> pd.DataFrame:
    dates_pattern = pattern_date(dates)
    return pd.concat((df.loc[0].to_frame().T,df.loc[1:][df.loc[1:,'ISO_TIME'].str.match(dates_pattern)]), axis=0)



def extract_data_from_date_pattern(df: pd.DataFrame, season: str) -> pd.DataFrame:
    return pd.concat((df.loc[0].to_frame().T, df[df["SEASON"]==season]), axis=0)



def extract_data_from_time_pattern_old(df: pd.DataFrame, times: list) -> pd.DataFrame:
    times_pattern = pattern_time(times)
    return pd.concat((df.loc[0].to_frame().T,df.loc[1:][df.loc[1:,'ISO_TIME'].str.contains(times_pattern)]), axis=0)



def get_all_iso_times(df: pd.DataFrame, TC_year=None, TC_id=None, season=None):
    # to be removed: TC year
    assert TC_year or season is not None, "TC_date or season must be specified"
    assert TC_year is None or len(TC_year[0])==4, "TC_year must be of the form yyyy"
    
    if season is not None:
        df_TC_tmp = extract_data_from_date_pattern(df, season=season)
        if TC_id is None:
            TC_id = df_TC_tmp["SID"].values[1]
            print(f"TC id: {TC_id} ({season})",)
        df_TC = df_TC_tmp[df_TC_tmp["SID"]==TC_id]
            
    return df_TC["ISO_TIME"].values, TC_id



def days_spacing(date1, date2, separator="-"):
    
    from datetime import datetime
    
    assert separator in date1 and separator in date2, "separator must be in both dates"
    
    date_format = "%Y" + separator + "%m" + separator +"%d"

    a = datetime.strptime(date1, date_format)
    b = datetime.strptime(date2, date_format)

    delta = b - a

    return delta.days



def write_input_params_to_file(output_path: str, df: pd.DataFrame, TC_year=None, TC_id=None, season=None, multiple_6=False, max_lead=168, **kwargs):
    # multiple_6: if True, only keep the times that are multiple of 6
    debug = kwargs.get("debug", False)
    
    iso_times, TC_id = get_all_iso_times(df=df, TC_year=TC_year, TC_id=TC_id, season=season)
    iso_time_last = iso_times[-1]
    
    dates, times, lead_times = [], [], []
    # refaire avec la fonction nb_hours --> wayyyyyy better
    
    for iso_time in iso_times[:-1]:
        date, time = date_ibtracs_to_nn(iso_time.split(" ")[0]), time_ibtracs_to_nn(iso_time.split(" ")[1])
        
        lead_time = subtract_ibtracs_iso_times(iso_time, iso_time_last)
        if lead_time > max_lead: # don't go over 7 days (default)
            lead_time = max_lead
        
        dates.append(date)
        times.append(time)
        lead_times.append(lead_time)

    if multiple_6:
        times_6 = [True] + [False for i in range(len(times)-1)]
        prev_t = times[0]
        for i, t in enumerate(times[1:]):
            if abs(int(t)//100-int(prev_t)//100)%6==0:
                prev_t = t
                times_6[i+1] = True
        dates, times, lead_times = ["date"] + [int(d) for i, d in enumerate(dates) if times_6[i]], \
                                   ["time"] + [t for i, t in enumerate(times) if times_6[i]], \
                                   ["lead time"] + [int(lt) for i, lt in enumerate(lead_times) if times_6[i]]
    ids = ["ArrayTaskID"] + [i for i in range(len(dates)-1)]

    filename = f"input_params_{TC_id}.txt"
    with open(output_path + filename, "w") as w:
        col_format = "{:<12}" + "{:<9}" + "{:<5}" + "{:<9}" + "\n"
        if debug:
            data = np.column_stack((ids[:10], dates[:10], times[:10], lead_times[:10]))
        else:
            data = np.column_stack((ids, dates, times, lead_times))
        for x in data:
            w.write(col_format.format(*x))

def get_date_time(filename: str) -> str:
    model_name = filename[:filename.index("_")]
    model_name_length = len(model_name)
    date_time = filename[model_name_length+1:model_name_length+18]
    return date_time, model_name



def get_lead_time(filename: str) -> int:
    reversed_filename = filename[::-1]
    lead_time = reversed_filename[reversed_filename.index("h")+1:reversed_filename.index("_")][::-1]
    return int(lead_time)



def combine_and_convert_gribs(basepath: str):
    grib_files = glob.glob(basepath + '*.grib')
    key = lambda x: (get_date_time(x)[0], get_lead_time(x))
    filelist = sorted([os.path.basename(filename) for filename in grib_files], key=key)
    xarray_init = xr.load_dataset(basepath + filelist[0], engine="cfgrib")
    
    date_time, model_name = get_date_time(filelist[0])
    lead_time_init = get_lead_time(filelist[0])
    tmp_list = [xarray_init]
    
    for idx in range(1, len(filelist)):
        if (idx+1)%100==0:
            print(f"\n{idx+1}/{len(filelist)}\n")
            
        xarr = xr.load_dataset(basepath + filelist[idx], engine="cfgrib")
        if date_time==get_date_time(filelist[idx])[0]:
            tmp_list.append(xarr)
            lead_time_end = get_lead_time(filelist[idx])
        else:
            combined_ds = xr.concat(tmp_list, dim="time")
            combined_ds.to_netcdf(basepath + f"{model_name}_{date_time}_lt_{lead_time_init}-{lead_time_end}h" + ".nc")
            
            lead_time_init = get_lead_time(filelist[idx])
            date_time = get_date_time(filelist[idx])[0]
            tmp_list = [xarr]
            