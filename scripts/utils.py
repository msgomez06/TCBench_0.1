import xarray as xr
import numpy as np
import os
import glob
import pandas as pd


def select_localisation(ds: xr.Dataset, lat_min: float, lat_max: float, 
                        lon_min: float, lon_max: float) -> xr.Dataset:
    ds_new = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)) # lat organised by decreasing order
    return ds_new



def min_MSLP(ds: xr.Dataset) -> xr.DataArray:
    return ds['msl'].min().values



def min_region_MLSP(ds, precision_percent=1) -> xr.DataArray:
    # return the region where MLSP is min up to precision_percent margin
    m = ds['msl'].min().values
    ds_new = ds.where(ds['msl'] <= (1+precision_percent/100) * m, drop=True)
    return ds_new



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
    
    lats, lons = ds["latitude"].values, ds["longitude"].values
    nb_dates = ds["time"].shape[0]
    final_mask = np.zeros((nb_dates, lats.shape[0], lons.shape[0]))
    
    for i in range(nb_dates):
        lead_time = np.timedelta64(ds["step"][i].values, 'h').astype(int)
        iso_time = date_time_netcdf_to_ibtracs(ds["time"].values[i], lead_time=lead_time)
        try:
            point = (float(df_tracks[(df_tracks["SID"]==tc_id) & (df_tracks["ISO_TIME"]==iso_time)]["LAT"].values[0]), 
                    float(df_tracks[(df_tracks["SID"]==tc_id) & (df_tracks["ISO_TIME"]==iso_time)]["LON"].values[0]))
        except IndexError:
            print(lead_time, ds["step"][i].values, iso_time)
            print(df_tracks[df_tracks["SID"]==tc_id].iloc[[0, -1]])
            raise IndexError("IndexError")
        rect_mask = get_rectmask(point, (lats, lons))
        final_mask[i] = rect_mask
    
    ds_new = ds[final_mask]
    return ds_new



def cut_and_save_rect(ds_path, ibtracs_df:pd.DataFrame, tc_id, output_path):
    ds = xr.open_dataset(ds_path)
    ds_new = cut_rectangle(ds, ibtracs_df, tc_id)
    ds_new.to_netcdf(output_path)
    ds.close()
    ds_new.close()
    
    
    
def date_time_netcdf_to_ibtracs(date_str: str, lead_time: int=0) -> str:
    # Provide a lead_time in case the date is not the initialisation date
    
    date_str = np.datetime_as_string(date_str)#, unit='ns'
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
    iso_time = f"{date} {time}"
    return iso_time
    
    
    
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



def extract_data_from_date_pattern(df: pd.DataFrame, seasons: list) -> pd.DataFrame:
    return pd.concat((df.loc[0].to_frame().T, df[df["SEASON"].isin(seasons)]), axis=0)



def extract_data_from_time_pattern_old(df: pd.DataFrame, times: list) -> pd.DataFrame:
    times_pattern = pattern_time(times)
    return pd.concat((df.loc[0].to_frame().T,df.loc[1:][df.loc[1:,'ISO_TIME'].str.contains(times_pattern)]), axis=0)



def get_all_iso_times(df: pd.DataFrame, TC_year=None, TC_id=None, seasons=None):
    # to be removed: TC year
    assert TC_year or seasons is not None, "TC_date or season must be specified"
    assert TC_year is None or len(TC_year[0])==4, "TC_year must be of the form yyyy"
    
    if seasons is not None:
        df_TC_tmp = extract_data_from_date_pattern(df, seasons=seasons)
        if TC_id is None:
            TC_id = df_TC_tmp["SID"].values[1]
            print(TC_id)
        df_TC = df_TC_tmp[df_TC_tmp["SID"]==TC_id]
            
    return df_TC["ISO_TIME"].values



def days_spacing(date1, date2, separator="-"):
    
    from datetime import datetime
    
    assert separator in date1 and separator in date2, "separator must be in both dates"
    
    date_format = "%Y" + separator + "%m" + separator +"%d"

    a = datetime.strptime(date1, date_format)
    b = datetime.strptime(date2, date_format)

    delta = b - a

    return delta.days



def write_input_params_to_file(filename: str, df: pd.DataFrame, TC_year=None, TC_id=None, seasons=None, multiple_6=False, **kwargs):
    # multiple_6: if True, only keep the times that are multiple of 6
    
    iso_times = get_all_iso_times(df=df, TC_year=TC_year, TC_id=TC_id, seasons=seasons)
    last_date, last_time = date_ibtracs_to_nn(iso_times[-1].split(" ")[0]), time_ibtracs_to_nn(iso_times[-1].split(" ")[1])
    
    dates, times, lead_times = [], [], []
    
    for iso_time in iso_times:
        date, time = date_ibtracs_to_nn(iso_time.split(" ")[0]), time_ibtracs_to_nn(iso_time.split(" ")[1])
        dates.append(date)
        times.append(time)
        
        nb_days = days_spacing(date, last_date, separator="")
        nb_hours = np.abs(-int(time)//100 + int(last_time)//100) % 24
        lead_times.append(24 * nb_days + nb_hours)
    
    if multiple_6:
        times_6 = [True if int(t)%600==0 else False for t in times]
        dates, times, lead_times = [d for i, d in enumerate(dates) if times_6[i]], \
                                   [t for i, t in enumerate(times) if times_6[i]], \
                                   [lt for i, lt in enumerate(lead_times) if times_6[i]]
    with open(filename, 'w') as w:
        counter = 0
        for i, date in enumerate(dates):
            time = times[i]
            max_lead_time = lead_times[i]
            for j in range(6 if multiple_6 else 3, max_lead_time, 6 if multiple_6 else 3):
                lead_time = j
                
                w.write(f"{date} {time} {lead_time}\n")
                if counter > 120 and kwargs["debug"]:
                    return 0
                counter += 1
            

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
            