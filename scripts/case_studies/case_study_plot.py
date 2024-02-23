import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os, sys, glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils.main_utils import get_start_date_nc, get_lead_time
from utils.cut_region import haversine, cut_rectangle
from case_studies.case_study_utils import find_trajectory_point, remove_duplicates
from case_studies.case_study_loaders import get_era5, load_tc_forecast, load_pp_model


def trajectory_no_pp(tc_id, model_names, max_lead_time=72, pp_type="linear", pp_params=None,
                     data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                     df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                     plot_path="/users/lpoulain/louis/plots/case_studies/"):
    tc_sid_to_name = {"2005236N23285":"Katrina (2005)", "2000185N15117":"Kai-Tak (2000)"}
    assert tc_id in tc_sid_to_name.keys(), f"tc_id should be one of {list(tc_sid_to_name.keys())}"

    era5, valid_dates, df, pres_col = get_era5(tc_id, data_path+"ERA5/", df_path)
    idxs = [i for i in range(era5.valid_time.shape[0]) if era5.valid_time[i].values.astype("datetime64[ns]") in valid_dates.astype("datetime64[ns]")]
    era5 = era5.isel(step=idxs)
    
    lead_times = np.arange(6, max_lead_time+6, 6)
    
    time0 = df["ISO_TIME"].values[0]
    for lead_time in lead_times:
        print(f"Lead time: {lead_time}h")
        grid = ([f"Traj_{x}" for x in model_names+["ERA5"]], 
                [f"Wind_{x}" for x in model_names+["ERA5"]],
                [f"Pres_{x}" for x in model_names+["ERA5"]],
                [f"km_error_{x}" for x in model_names+["ERA5"]]
                )
        
        
        df_tmp = df.loc[df.index[lead_time//6:]]
        times = df_tmp["ISO_TIME"].values.astype("datetime64[ns]")
        truth_lats, truth_lons = df_tmp["LAT"].values.astype(np.float32), df_tmp["LON"].values.astype(np.float32)
        truth_lons = [t+360 if t<0 else t for t in truth_lons]
        
        truth_wind, truth_pres = df_tmp["USA_WIND"].values.astype(np.float32)*0.514444, df_tmp[pres_col].values.astype(np.float32)*100
        
        lat_extent = [min(truth_lats), max(truth_lats)]
        lon_extent = [min(truth_lons), max(truth_lons)]
        lon_extent = [t-360 if t>180 else t for t in lon_extent]
        
        center = [(lat_extent[0]+lat_extent[1])/2, (lon_extent[0]+lon_extent[1])/2]
        radius = [(lat_extent[1]-lat_extent[0])/2, (lon_extent[1]-lon_extent[0])]
        lat_extent = [center[0]-max(radius)-4, center[0]+max(radius)+4]
        lon_extent = [center[1]-max(radius)-4, center[1]+max(radius)+4]
        center_lon = 0
        
        per_subplot_kw = {"Traj_"+model: dict(projection=ccrs.PlateCarree(center_lon)) for model in model_names+["ERA5"]}
        fig, axs = plt.subplot_mosaic(grid, figsize=((len(model_names)+1)*5, len(grid)*6), per_subplot_kw=per_subplot_kw,
                                      gridspec_kw={"hspace":0.1, "wspace":0.1, "top":0.93, "bottom":0.05, "left":0.05, "right":0.95},
                                      sharey=False, sharex=False)
        pres_extent = [truth_pres.min(), truth_pres.max()]
        wind_extent = [truth_wind.min(), truth_wind.max()]
        err_km_extent = [1e6, 0]
        
        for model in model_names:
            data_folder = "panguweather" if model=="pangu" else model
            key = lambda x: (get_start_date_nc(x), get_lead_time(x))
            data_list = sorted(glob.glob(data_path+f"{data_folder}/{model}_*_{tc_id}_small.nc"), key=key) if model!="fourcastnetv2" else\
                        sorted(glob.glob(data_path+f"{data_folder}/{model}_*_{tc_id}.nc"), key=key)
            data_list = [p for p in data_list if get_lead_time(p)>=lead_time]
            data_list = remove_duplicates(data_list)
            
            lats, lons, winds, press, err_km = [], [], [], [], []
            final_times = []
            
            for path in data_list:
                data = xr.open_dataset(path).isel(time=lead_time//6-1 if not model=="fourcastnetv2" else lead_time//6)
                tmp_time = data.time.values if isinstance(data.time.values, np.datetime64) else\
                            np.datetime64(data.time.values + get_start_date_nc(path))

                if tmp_time in times:
                    max_wind, min_pres, pred_lat, pred_lon = find_trajectory_point(data, 
                                                                                   truth_lats[np.where(times==tmp_time)[0][0]],
                                                                                   truth_lons[np.where(times==tmp_time)[0][0]],
                                                                                   centroid_size=5*(lead_time//6))
                    lats.append(pred_lat)
                    lons.append(pred_lon)
                    winds.append(max_wind)
                    press.append(min_pres)
                    err_km.append(haversine(pred_lat, pred_lon, [truth_lats[np.where(times==tmp_time)[0][0]]], [truth_lons[np.where(times==tmp_time)[0][0]]]))
                    final_times.append(np.datetime64(tmp_time, 'h'))
            
            lats, lons, winds, press, err_km = np.array(lats), np.array(lons), np.array(winds), np.array(press), np.array(err_km)
            lons_plot = np.array([t-360 if t>180 else t for t in lons])
            truth_lons_plot = np.array([t-360 if t>180 else t for t in truth_lons])
            

            axs["Traj_"+model].set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree(center_lon))
            axs["Traj_"+model].plot(lons_plot, lats, linestyle='-', marker='x', c = 'blue', label=model, transform=ccrs.PlateCarree(), markersize=5)
            axs["Traj_"+model].plot(truth_lons_plot, truth_lats, '-x', c = 'red', label="Truth", transform=ccrs.PlateCarree(), markersize=5)
            axs["Traj_"+model].add_feature(cfeature.COASTLINE.with_scale('50m'))
            axs["Traj_"+model].add_feature(cfeature.BORDERS.with_scale('50m'))
            axs["Traj_"+model].legend()
            gridliner = axs["Traj_"+model].gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gridliner.top_labels = False
            gridliner.bottom_labels = True
            gridliner.left_labels = True if model==model_names[0] else False
            gridliner.right_labels = False
            gridliner.ylines = False
            gridliner.xlines = False
            axs["Traj_"+model].set_title(f"Model: {model}", fontsize=18)
            
            
            axs["Wind_"+model].plot(winds, c = 'blue', label=model)
            axs["Wind_"+model].plot(truth_wind, c = 'red', label="Truth")
            axs["Wind_"+model].legend()
            if model==model_names[0]:
                axs["Wind_"+model].set_ylabel("Wind speed (m/s)", fontsize=15)
            if not model==model_names[0]:
                axs["Wind_"+model].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axs["Wind_"+model].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
            if winds.min()<wind_extent[0]:
                wind_extent[0] = winds.min()
            if winds.max()>wind_extent[1]:
                wind_extent[1] = winds.max()
                
            axs["Pres_"+model].plot(press, c = 'blue', label=model)
            axs["Pres_"+model].plot(truth_pres, c = 'red', label="Truth")
            axs["Pres_"+model].legend()
            if model==model_names[0]:
                axs["Pres_"+model].set_ylabel("Pressure (Pa)", fontsize=15)
            if not model==model_names[0]:
                axs["Pres_"+model].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axs["Pres_"+model].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
            if press.min()<pres_extent[0]:
                pres_extent[0] = press.min()
            if press.max()>pres_extent[1]:
                pres_extent[1] = press.max()
            
            
            axs["km_error_"+model].plot(err_km, c = 'blue', label=model)
            axs["km_error_"+model].set_xticks(ticks=range(len(winds))[::len(final_times)//7], 
                                          labels=[str(x) for x in final_times][::len(final_times)//7], 
                                          rotation=45, 
                                          horizontalalignment='right', fontsize='small')
            axs["km_error_"+model].legend()
            axs["km_error_"+model].set_xlabel("Time", fontsize=15)
            if model==model_names[0]:
                axs["km_error_"+model].set_ylabel("Error (km)", fontsize=15)
            if not model==model_names[0]:
                axs["km_error_"+model].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            if err_km.min()<err_km_extent[0]:
                err_km_extent[0] = err_km.min()
            if err_km.max()>err_km_extent[1]:
                err_km_extent[1] = err_km.max()
            
        lats, lons, winds, press, err_km = [], [], [], [], []
        final_times = []
        for i, tmp_time in enumerate(times):
            idx = np.where(era5.valid_time.values==tmp_time)[0][0]

            era5_tmp = era5.isel(step=idx)

            era5_tmp = cut_rectangle(era5_tmp, df, tc_id, tmp_time)
            max_wind, min_pres, pred_lat, pred_lon = find_trajectory_point(era5_tmp,
                                                                           truth_lats[np.where(times==tmp_time)[0][0]],
                                                                           truth_lons[np.where(times==tmp_time)[0][0]],
                                                                           centroid_size=5*(lead_time//6))
            lats.append(pred_lat)
            lons.append(pred_lon)
            winds.append(max_wind)
            press.append(min_pres)
            err_km.append(haversine(pred_lat, pred_lon, [truth_lats[np.where(times==tmp_time)[0][0]]], [truth_lons[np.where(times==tmp_time)[0][0]]]))
            final_times.append(np.datetime64(tmp_time, 'h'))
            
        lats, lons, winds, press, err_km = np.array(lats), np.array(lons), np.array(winds), np.array(press), np.array(err_km)
        lons_plot = np.array([t-360 if t>180 else t for t in lons])
        truth_lons_plot = np.array([t-360 if t>180 else t for t in truth_lons])
        
        
        axs["Traj_ERA5"].set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree(center_lon))
        axs["Traj_ERA5"].add_feature(cfeature.COASTLINE.with_scale('50m'))
        axs["Traj_ERA5"].add_feature(cfeature.BORDERS.with_scale('50m'))
        axs["Traj_ERA5"].plot(lons_plot, lats, '-x', c = 'blue', label="ERA5", transform=ccrs.PlateCarree(), markersize=5)
        axs["Traj_ERA5"].plot(truth_lons_plot, truth_lats, '-x', c = 'red', label="Truth", transform=ccrs.PlateCarree(), markersize=5)
        axs["Traj_ERA5"].legend()
        gridliner = axs["Traj_ERA5"].gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gridliner.top_labels = False
        gridliner.bottom_labels = True
        gridliner.left_labels = False
        gridliner.right_labels = False
        gridliner.ylines = False
        gridliner.xlines = False
        axs["Traj_ERA5"].set_title(f"Model: ERA5", fontsize=18)
        axs["Traj_ERA5"].annotate(f"Trajectory comparison", xy=(0.5, 0.94), xycoords="figure fraction", ha="center", fontsize=20)
        
        
        axs["Wind_ERA5"].plot(winds, c = 'blue', label="ERA5")
        axs["Wind_ERA5"].plot(truth_wind, c = 'red', label="Truth")
        axs["Wind_ERA5"].legend()
        axs["Wind_ERA5"].annotate(f"Evolution of wind speed", xy=(0.5, 0.705), xycoords="figure fraction", ha="center", fontsize=20)
        axs["Wind_ERA5"].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axs["Wind_ERA5"].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
        if winds.min()<wind_extent[0]:
            wind_extent[0] = winds.min()
        if winds.max()>wind_extent[1]:
            wind_extent[1] = winds.max()
        
        
        axs["Pres_ERA5"].plot(press, c = 'blue', label="ERA5")
        axs["Pres_ERA5"].plot(truth_pres, c = 'red', label="Truth")
        axs["Pres_ERA5"].legend()
        axs["Pres_ERA5"].annotate(f"Evolution of pressure", xy=(0.5, 0.48), xycoords="figure fraction", ha="center", fontsize=20)
        axs["Pres_ERA5"].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axs["Pres_ERA5"].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
        if press.min()<pres_extent[0]:
            pres_extent[0] = press.min()
        if press.max()>pres_extent[1]:
            pres_extent[1] = press.max()
        
        
        axs["km_error_ERA5"].plot(err_km, c = 'blue', label="ERA5")
        axs["km_error_ERA5"].set_xticks(ticks=range(len(winds))[::len(final_times)//7], 
                                          labels=[str(x) for x in final_times][::len(final_times)//7], 
                                          rotation=45, 
                                          horizontalalignment='right', fontsize='small')
        axs["km_error_ERA5"].legend()
        axs["km_error_ERA5"].set_xlabel("Time", fontsize=15)
        axs["km_error_ERA5"].annotate(f"Evolution of location error", xy=(0.5, 0.255), xycoords="figure fraction", ha="center", fontsize=20)
        axs["km_error_ERA5"].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if err_km.min()<err_km_extent[0]:
            err_km_extent[0] = err_km.min()
        if err_km.max()>err_km_extent[1]:
            err_km_extent[1] = err_km.max()
        
        
        for m in model_names+["ERA5"]:
            axs["Wind_"+m].set_ylim(wind_extent[0]-1, wind_extent[1]+1)
            axs["Pres_"+m].set_ylim(pres_extent[1]+1000, pres_extent[0]-1000)
            axs["km_error_"+m].set_ylim(err_km_extent[0]-10, err_km_extent[1]+10)
            
        st = fig.suptitle(f"Trajectory comparison for {tc_sid_to_name[tc_id]} - Lead time: {lead_time}h\nNo post-processing")
        st.set_y(0.98)
        fig.savefig(plot_path + f"trajectory_{tc_id}_{lead_time}.png", dpi=500, bbox_inches="tight")
        
        
        
        
def trajectory_with_pp(tc_id, model_names, max_lead_time=72, pp_type="linear", pp_params=None,
                     data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                     df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                     plot_path="/users/lpoulain/louis/plots/case_studies/"):
    
    tc_sid_to_name = {"2005236N23285":"Katrina (2005)", "2000185N15117":"Kai-Tak (2000)"}
    assert tc_id in tc_sid_to_name.keys(), f"tc_id should be one of {list(tc_sid_to_name.keys())}"

    era5, valid_dates, df, pres_col = get_era5(tc_id, data_path+"ERA5/", df_path)
    idxs = [i for i in range(era5.valid_time.shape[0]) if era5.valid_time[i].values.astype("datetime64[ns]") in valid_dates.astype("datetime64[ns]")]
    era5 = era5.isel(step=idxs)
    
    lead_times = np.arange(6, max_lead_time+6, 6)
    
    time0 = df["ISO_TIME"].values[0]
    for lead_time in lead_times:
        print(f"Lead time: {lead_time}h")
        grid = ([f"Traj_{x}" for x in model_names+["ERA5"]], 
                [f"Wind_{x}" for x in model_names+["ERA5"]],
                [f"Pres_{x}" for x in model_names+["ERA5"]],
                [f"km_error_{x}" for x in model_names+["ERA5"]]
                )
        pp_params["ldt"] = lead_time
        
        df_tmp = df.loc[df.index[lead_time//6:]]
        times = df_tmp["ISO_TIME"].values.astype("datetime64[ns]")
        truth_lats, truth_lons = df_tmp["LAT"].values.astype(np.float32), df_tmp["LON"].values.astype(np.float32)
        truth_lons = [t+360 if t<0 else t for t in truth_lons]
        
        truth_wind, truth_pres = df_tmp["USA_WIND"].values.astype(np.float32)*0.514444, df_tmp[pres_col].values.astype(np.float32)*100
        
        lat_extent = [min(truth_lats), max(truth_lats)]
        lon_extent = [min(truth_lons), max(truth_lons)]
        lon_extent = [t-360 if t>180 else t for t in lon_extent]
        
        center = [(lat_extent[0]+lat_extent[1])/2, (lon_extent[0]+lon_extent[1])/2]
        radius = [(lat_extent[1]-lat_extent[0])/2, (lon_extent[1]-lon_extent[0])]
        lat_extent = [center[0]-max(radius)-4, center[0]+max(radius)+4]
        lon_extent = [center[1]-max(radius)-4, center[1]+max(radius)+4]
        center_lon = 0
        
        per_subplot_kw = {"Traj_"+model: dict(projection=ccrs.PlateCarree(center_lon)) for model in model_names+["ERA5"]}
        fig, axs = plt.subplot_mosaic(grid, figsize=((len(model_names)+1)*5, len(grid)*6), per_subplot_kw=per_subplot_kw,
                                      gridspec_kw={"hspace":0.1, "wspace":0.1, "top":0.93, "bottom":0.05, "left":0.05, "right":0.95},
                                      sharey=False, sharex=False)
        pres_extent = [truth_pres.min(), truth_pres.max()]
        wind_extent = [truth_wind.min(), truth_wind.max()]
        err_km_extent = [1e6, 0]
        
        for model in model_names:
            data = load_tc_forecast(model, tc_id, pp_type, pp_params, lead_time)
            pp_models = load_pp_model(model, pp_type, pp_params)
            
            lats, lons, winds, press, err_km = [], [], [], [], []
            final_times = []
            
            for i in range(len(data)):
                tmp_time = data[i][0]

                if tmp_time in times:
                    input_wind, input_pres, pred_lat, pred_lon = data[i][1:]
                    
                    if pp_type=="linear" and pp_params.get("dim", 2)==2:
                        max_wind, min_pres = pp_models[0].predict(np.array([input_pres]).reshape(-1, 1))
                    else:
                        if pp_type == "linear":
                            max_wind = pp_models[0].predict(np.array([input_wind, input_pres]).reshape(1, -1))
                            min_pres = pp_models[1].predict(np.array([input_wind, input_pres]).reshape(1, -1))
                        if pp_type == "xgboost":
                            max_wind = pp_models[0].predict(np.array(input_wind), iteration_range=(0, pp_models[0].best_iteration + 1))
                            min_pres = pp_models[1].predict(np.array(input_wind), iteration_range=(0, pp_models[1].best_iteration + 1))
                        if pp_type == "cnn":
                            #max_wind, min_pres, pred_lat, pred_lon = pp_models[0](torch.tensor([input_wind, input_pres]).reshape(1, 1, 2))
                            raise NotImplementedError("CNN post-processing plot not implemented yet")
                        
                    lats.append(pred_lat)
                    lons.append(pred_lon)
                    winds.append(max_wind)
                    press.append(min_pres)
                    err_km.append(haversine(pred_lat, pred_lon, [truth_lats[np.where(times==tmp_time)[0][0]]], [truth_lons[np.where(times==tmp_time)[0][0]]]))
                    final_times.append(np.datetime64(tmp_time, 'h'))
            
            lats, lons, winds, press, err_km = np.array(lats), np.array(lons), np.array(winds), np.array(press), np.array(err_km)
            lons_plot = np.array([t-360 if t>180 else t for t in lons])
            truth_lons_plot = np.array([t-360 if t>180 else t for t in truth_lons])
            

            axs["Traj_"+model].set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree(center_lon))
            axs["Traj_"+model].plot(lons_plot, lats, linestyle='-', marker='x', c = 'blue', label=model, transform=ccrs.PlateCarree(), markersize=5)
            axs["Traj_"+model].plot(truth_lons_plot, truth_lats, '-x', c = 'red', label="Truth", transform=ccrs.PlateCarree(), markersize=5)
            axs["Traj_"+model].add_feature(cfeature.COASTLINE.with_scale('50m'))
            axs["Traj_"+model].add_feature(cfeature.BORDERS.with_scale('50m'))
            axs["Traj_"+model].legend()
            gridliner = axs["Traj_"+model].gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gridliner.top_labels = False
            gridliner.bottom_labels = True
            gridliner.left_labels = True if model==model_names[0] else False
            gridliner.right_labels = False
            gridliner.ylines = False
            gridliner.xlines = False
            axs["Traj_"+model].set_title(f"Model: {model}", fontsize=18)
            
            
            axs["Wind_"+model].plot(winds, c = 'blue', label=model)
            axs["Wind_"+model].plot(truth_wind, c = 'red', label="Truth")
            axs["Wind_"+model].legend()
            if model==model_names[0]:
                axs["Wind_"+model].set_ylabel("Wind speed (m/s)")
            if not model==model_names[0]:
                axs["Wind_"+model].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axs["Wind_"+model].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
            if winds.min()<wind_extent[0]:
                wind_extent[0] = winds.min()
            if winds.max()>wind_extent[1]:
                wind_extent[1] = winds.max()
                
            axs["Pres_"+model].plot(press, c = 'blue', label=model)
            axs["Pres_"+model].plot(truth_pres, c = 'red', label="Truth")
            axs["Pres_"+model].legend()
            if model==model_names[0]:
                axs["Pres_"+model].set_ylabel("Pressure (Pa)")
            if not model==model_names[0]:
                axs["Pres_"+model].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axs["Pres_"+model].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
            if press.min()<pres_extent[0]:
                pres_extent[0] = press.min()
            if press.max()>pres_extent[1]:
                pres_extent[1] = press.max()
            
            
            axs["km_error_"+model].plot(err_km, c = 'blue', label=model)
            axs["km_error_"+model].set_xticks(ticks=range(len(winds))[::len(final_times)//7], 
                                          labels=[str(x) for x in final_times][::len(final_times)//7], 
                                          rotation=45, 
                                          horizontalalignment='right', fontsize='small')
            axs["km_error_"+model].legend()
            axs["km_error_"+model].set_xlabel("Time")
            if model==model_names[0]:
                axs["km_error_"+model].set_ylabel("Error (km)")
            if not model==model_names[0]:
                axs["km_error_"+model].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            if err_km.min()<err_km_extent[0]:
                err_km_extent[0] = err_km.min()
            if err_km.max()>err_km_extent[1]:
                err_km_extent[1] = err_km.max()
            
        lats, lons, winds, press, err_km = [], [], [], [], []
        final_times = []
        for i, tmp_time in enumerate(times):
            idx = np.where(era5.valid_time.values==tmp_time)[0][0]

            era5_tmp = era5.isel(step=idx)

            era5_tmp = cut_rectangle(era5_tmp, df, tc_id, tmp_time)
            max_wind, min_pres, pred_lat, pred_lon = find_trajectory_point(era5_tmp,
                                                                           truth_lats[np.where(times==tmp_time)[0][0]],
                                                                           truth_lons[np.where(times==tmp_time)[0][0]],
                                                                           centroid_size=5*(lead_time//6))
            lats.append(pred_lat)
            lons.append(pred_lon)
            winds.append(max_wind)
            press.append(min_pres)
            err_km.append(haversine(pred_lat, pred_lon, [truth_lats[np.where(times==tmp_time)[0][0]]], [truth_lons[np.where(times==tmp_time)[0][0]]]))
            final_times.append(np.datetime64(tmp_time, 'h'))
            
        lats, lons, winds, press, err_km = np.array(lats), np.array(lons), np.array(winds), np.array(press), np.array(err_km)
        lons_plot = np.array([t-360 if t>180 else t for t in lons])
        truth_lons_plot = np.array([t-360 if t>180 else t for t in truth_lons])
        
        
        axs["Traj_ERA5"].set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree(center_lon))
        axs["Traj_ERA5"].add_feature(cfeature.COASTLINE.with_scale('50m'))
        axs["Traj_ERA5"].add_feature(cfeature.BORDERS.with_scale('50m'))
        axs["Traj_ERA5"].plot(lons_plot, lats, '-x', c = 'blue', label="ERA5", transform=ccrs.PlateCarree(), markersize=5)
        axs["Traj_ERA5"].plot(truth_lons_plot, truth_lats, '-x', c = 'red', label="Truth", transform=ccrs.PlateCarree(), markersize=5)
        axs["Traj_ERA5"].legend()
        gridliner = axs["Traj_ERA5"].gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gridliner.top_labels = False
        gridliner.bottom_labels = True
        gridliner.left_labels = False
        gridliner.right_labels = False
        gridliner.ylines = False
        gridliner.xlines = False
        axs["Traj_ERA5"].set_title(f"Model: ERA5", fontsize=18)
        axs["Traj_ERA5"].annotate(f"Trajectory comparison", xy=(0.5, 0.94), xycoords="figure fraction", ha="center", fontsize=20)
        
        
        axs["Wind_ERA5"].plot(winds, c = 'blue', label="ERA5")
        axs["Wind_ERA5"].plot(truth_wind, c = 'red', label="Truth")
        axs["Wind_ERA5"].legend()
        axs["Wind_ERA5"].annotate(f"Evolution of wind speed", xy=(0.5, 0.705), xycoords="figure fraction", ha="center", fontsize=20)
        axs["Wind_ERA5"].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axs["Wind_ERA5"].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
        if winds.min()<wind_extent[0]:
            wind_extent[0] = winds.min()
        if winds.max()>wind_extent[1]:
            wind_extent[1] = winds.max()
        
        
        axs["Pres_ERA5"].plot(press, c = 'blue', label="ERA5")
        axs["Pres_ERA5"].plot(truth_pres, c = 'red', label="Truth")
        axs["Pres_ERA5"].legend()
        axs["Pres_ERA5"].annotate(f"Evolution of pressure", xy=(0.5, 0.48), xycoords="figure fraction", ha="center", fontsize=20)
        axs["Pres_ERA5"].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axs["Pres_ERA5"].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
        if press.min()<pres_extent[0]:
            pres_extent[0] = press.min()
        if press.max()>pres_extent[1]:
            pres_extent[1] = press.max()
        
        
        axs["km_error_ERA5"].plot(err_km, c = 'blue', label="ERA5")
        axs["km_error_ERA5"].set_xticks(ticks=range(len(winds))[::len(final_times)//7], 
                                          labels=[str(x) for x in final_times][::len(final_times)//7], 
                                          rotation=45, 
                                          horizontalalignment='right', fontsize='small')
        axs["km_error_ERA5"].legend()
        axs["km_error_ERA5"].set_xlabel("Time")
        axs["km_error_ERA5"].annotate(f"Evolution of location error", xy=(0.5, 0.255), xycoords="figure fraction", ha="center", fontsize=20)
        if err_km.min()<err_km_extent[0]:
            err_km_extent[0] = err_km.min()
        if err_km.max()>err_km_extent[1]:
            err_km_extent[1] = err_km.max()
        
        
        for m in model_names+["ERA5"]:
            axs["Wind_"+m].set_ylim(wind_extent[0]-1, wind_extent[1]+1)
            axs["Pres_"+m].set_ylim(pres_extent[1]+1000, pres_extent[0]-1000)
            axs["km_error_"+m].set_ylim(err_km_extent[0]-10, err_km_extent[1]+10)
            
        st = fig.suptitle(f"Trajectory comparison for {tc_sid_to_name[tc_id]} - Lead time: {lead_time}h\nPost-processing using {pp_type}")
        st.set_y(0.98)
        fig.savefig(plot_path + f"trajectory_{pp_type}_{tc_id}_{lead_time}.png", dpi=500, bbox_inches="tight")
        
       
if __name__ == "__main__":
    # 2000185N15117 - Kai-Tak (2000)
    # 2005236N23285 - Katrina (2005)
    tc_id = "2005236N23285"
    model_names = ["pangu", "graphcast"]
    max_lead_time = 72
    trajectory_no_pp(tc_id, model_names, max_lead_time)