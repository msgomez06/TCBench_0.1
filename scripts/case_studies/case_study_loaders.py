import numpy as np
import pandas as pd
import xarray as xr
from joblib import load
import xgboost as xgb
import glob, sys

from models.loading_utils import stats_list, stats_fcts
from utils.main_utils import get_start_date_nc, get_lead_time
from case_studies.case_study_utils import remove_duplicates


def get_era5(tc_id, 
             data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/",
             df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    
    
    df = pd.read_csv(df_path, dtype="string", na_filter=False)
    df = df[df["SID"]==tc_id]
    
    wind_col = "USA_WIND"
    pres_cols = [col for col in df.columns if "_PRES" in col and "PRES_" not in col and col!="WMO_PRES"]
    
    idxs = [idx for idx in df.index if df.loc[idx, wind_col]!=" "]
        
    key = lambda x: np.count_nonzero(df[x].values.astype("string")!=" ")
    pres_col = sorted(pres_cols, key=key)[-1] # the one with the highest number of values reported
    idxs = [idx for idx in idxs if df.loc[idx, pres_col]!=" "] # remove rows with missing values
    
    if df.index[0] not in idxs:
        idxs = [df.index[0], *idxs]
    df = df.loc[idxs]

    valid_dates = df["ISO_TIME"].values
    year_month = list(set([date[:7] for date in valid_dates]))[0]
    
    era5 = xr.load_dataset(data_path + f"ERA5_{year_month[:4]}_{year_month[5:7]}_surface.grib")
    
    return era5, valid_dates, df, pres_col



def load_tc_forecast(model_name, tc_id, pp_type, pp_params, lead_time,
                     data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/"):
    
    pp_type = pp_type.lower()
    assert pp_type in ["linear", "xgboost", "cnn"], f"Unknown pp_type: {pp_type}."
    model_name = "pangu" if model_name=="panguweather" else model_name
    data_folder = "panguweather" if model_name=="pangu" else model_name
    
    key = lambda x: (get_start_date_nc(x), get_lead_time(x))
    data_list = sorted(glob.glob(data_path+f"{data_folder}/{model_name}_*_{tc_id}_small.nc"), key=key)
    data_list = [p for p in data_list if get_lead_time(p)>=lead_time]
    data_list = remove_duplicates(data_list)
    
    if pp_type == "linear":
        data = []
        
        for path in data_list:
            ds = xr.load_dataset(path)
            ds = ds.isel(time=lead_time//6-1)
            tmp_time = data.time.values if isinstance(data.time.values, np.datetime64) else\
                            np.datetime64(data.time.values + get_start_date_nc(path))
            max_wind = np.sqrt(ds.u10.values**2+ds.v10.values**2).max()
            min_pres = ds.msl.values.min()
            
            loc_idx = np.unravel_index(ds.msl.values.argmin(), ds.msl.values.shape)
            lat, lon = ds.latitude.values[loc_idx[0]], ds.longitude.values[loc_idx[1]]
            data.append((tmp_time, max_wind, min_pres, lat, lon))
        
        return data

    if pp_type == "xgboost":
        data = []
        
        stats = pp_params.get("stats", [])
        stats_wind = pp_params.get("stats_wind", ["max"])
        stats_pres = pp_params.get("stats_pres", ["min"])
        
        stats_wind = sorted(list(set(stats_wind + stats)), key=lambda x: stats_list.index(x))
        stats_pres = sorted(list(set(stats_pres + stats)), key=lambda x: stats_list.index(x))
        
        for path in data_list:
            winds = []
            press = []
            ds = xr.load_dataset(path)
            ds = ds.isel(time=lead_time//6-1)
            tmp_time = data.time.values if isinstance(data.time.values, np.datetime64) else\
                            np.datetime64(data.time.values + get_start_date_nc(path))
            wind = np.sqrt(ds.u10.values**2+ds.v10.values**2)
            pres = ds.msl.values
            loc_idx = np.unravel_index(pres.argmin(), pres.shape)
            lat, lon = ds.latitude.values[loc_idx[0]], ds.longitude.values[loc_idx[1]]
            for stat in stats_wind:
                winds.append(stats_fcts[stat](wind))
            for stat in stats_pres:
                press.append(stats_fcts[stat](pres))
            data.append((tmp_time, winds, press, lat, lon))
    
    if pp_type == "cnn":
        raise ValueError(f"Not implemented yet: {pp_type}.")
    


def load_pp_model(model_name, pp_type, pp_params):
    
    pp_type = pp_type.lower()
    assert pp_type in ["linear", "xgboost", "cnn"], f"Unknown pp_type: {pp_type}."
    model_name = "pangu" if model_name=="panguweather" else model_name
    train_seasons = pp_params.get("train_seasons", ['2000'])
    
    if pp_type == "linear":
        dim = pp_params.get("dim", 2)
        ldt = pp_params.get("ldt", 6)
        basin = pp_params.get("basin", "NA")
        train_seasons = train_seasons[0] if isinstance(train_seasons, list) else train_seasons
        
        model = load(f"models/{pp_type}_model{dim}d_{model_name}_{ldt}_{train_seasons}_basin_{basin}.joblib")
        return model
    
    if pp_type == "xgboost":
        jsdiv = pp_params.get("jsdiv", False)
        ldt = pp_params.get("ldt", 6)
        depth = pp_params.get("depth", 4)
        epochs = pp_params.get("epochs", 100)
        lr = pp_params.get("lr", 0.1)
        gamma = pp_params.get("gamma", 0.)
        sched = pp_params.get("sched", True)
        stats = pp_params.get("stats", [])
        stats_wind = pp_params.get("stats_wind", ["max"])
        stats_pres = pp_params.get("stats_pres", ["min"])
        basin = pp_params.get("basin", "NA")
        
        stats_wind = sorted(list(set(stats_wind + stats)), key=lambda x: stats_list.index(x))
        stats_pres = sorted(list(set(stats_pres + stats)), key=lambda x: stats_list.index(x))
        
        model_path = "/users/lpoulain/louis/plots/xgboost/Models/"
        model_wind_save_name = f"xgb_wind_{model_name}_{ldt}h{'_jsdiv' if jsdiv else ''}_basin_{basin}_"\
                        + f"{'_'.join(train_seasons)}_depth_{depth}_epoch_{epochs}_lr_{lr}_g_{gamma}"\
                        + (f"_sched" if sched else "")\
                        + f"_{'_'.join(stat for stat in stats_wind)}.json"
        model_pres_save_name = f"xgb_pres_{model_name}_{ldt}h{'_jsdiv' if jsdiv else ''}_basin_{basin}_"\
                        + f"{'_'.join(train_seasons)}_depth_{depth}_epoch_{epochs}_lr_{lr}_g_{gamma}"\
                        + (f"_sched" if sched else "")\
                        + f"_{'_'.join(stat for stat in stats_pres)}.json"
        
        model_wind = xgb.Booster().load_model(model_path + model_wind_save_name)
        model_pres = xgb.Booster().load_model(model_path + model_pres_save_name)
        
        return (model_wind, model_pres)
    
    if pp_type == "cnn":
        raise ValueError(f"Not implemented yet: {pp_type}.")
    
    
