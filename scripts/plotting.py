import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import xarray as xr
import utils as ut
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes

dict_model_names = {"pangu": "panguweather", "graphcast": "graphcast", "fourcast": "fourcastnetv2"}
var_wind = {"u": ("uwnd", "uwnd", "var131"), "v": ("vwnd", "vwnd", "var132")}
var_dict = {"mslp": 0, "wind": 1}

def compare_models(model_names: list, dates: list[int], times: list[int], lead_times: list[int], cmap_mlsp, cmap_wind, var_names, plot_dir, 
                   era5_location="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ECMWF/ERA5/",
                   model_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                   dpi=400):
    
    assert len(dates) == len(times) == len(lead_times), "date, time and lead_time must have the same length"
    if "wind" in var_names:
        assert "u" not in var_names and "v" not in var_names, "u and v must not be in var_names if wind is in var_names"
    if "u" in var_names:
        var_names.append("wind")
        var_names.remove("u")
    if "v" in var_names:
        if "wind" not in var_names:
            var_names.append("wind")
        var_names.remove("v")
    
    var_names = sorted(var_names, key=lambda x: var_dict[x])
    
    
    for i in range(len(dates)):
        date, time, lead_time = dates[i], times[i], lead_times[i]
        netcdf_time = ut.date_time_nn_to_netcdf(str(date), str(time+lead_time))
        #print(netcdf_time)
        year = str(date)[:4]       
        
        era5_data = []
        plot_names = []
        for var_name in var_names:
            if var_name.lower()=="mslp":
                fname1 = f"{era5_location}/{var_name}/{var_name}_{year}.nc"
                era5_data.append(xr.open_dataset(fname1, engine="netcdf4").sel(time=netcdf_time)["var151"])
                plot_names.append(("mslp", "Pa"))
                
            elif var_name.lower()=="wind":
                fname1 = f"{era5_location}/{var_wind['u'][0]}/{var_wind['u'][1]}_{year}.nc"
                fname2 = f"{era5_location}/{var_wind['v'][0]}/{var_wind['v'][1]}_{year}.nc"
                arr1 = xr.open_dataset(fname1, engine="netcdf4").sel(time=netcdf_time, plev=1e5)[var_wind['u'][2]]
                arr2 = xr.open_dataset(fname2, engine="netcdf4").sel(time=netcdf_time, plev=1e5)[var_wind['v'][2]]
                era5_data.append(np.sqrt(arr1 ** 2 + arr2 ** 2))
                plot_names.append(("wind", "m/s"))
                
            else:
                raise ValueError(f"wrong variable name. Expected mslp, u, v or wind but got {var_name}")
        
        lat_list = xr.open_dataset(fname1, engine="netcdf4").sel(time=netcdf_time)["lat"].values
        lon_list = xr.open_dataset(fname1, engine="netcdf4").sel(time=netcdf_time)["lon"].values
        lat_len, lon_len = lat_list.shape[0], lon_list.shape[0]
        
        data = [] #np.empty((len(model_names)+1, len(var_names), lat_len, lon_len))
        
        for j, model_name in enumerate(model_names):
            fpath = f"{model_path}/{dict_model_names[model_name]}/{model_name}_"
            fname = fpath + f"d_{date}_t_{str(time*100).zfill(4)}_{lead_time}h.grib"
            ds = xr.open_dataset(fname, engine="cfgrib")
            data.append([])
            
            for k, var in enumerate(var_names):
                data[j].append([])
                if var == "mslp":
                    data[j][k] = ds[var[:-1]]
                    #data[j, k, :, :] = ds[var[:-1]].values # in grib, recorded as "msl" instead
                else:
                    windu, windv = ds.sel(isobaricInhPa=1e3)["u"], ds.sel(isobaricInhPa=1e3)["v"]
                    data[j][k] = np.sqrt(windu ** 2 + windv ** 2)
                    #data[j, k, :, :] = np.sqrt(windu ** 2 + windv ** 2)
                
        data.append(era5_data)
        
        suptitle = f"Comparison of {len(model_names)} models and ERA5 on {netcdf_time} (prediction lead time: {lead_time}h)"
        pic_name = f"comparison_{'_'.join(model for model in model_names)}_{date}_{time}_{lead_time}h.png"
        col_names = [f"{dict_model_names[model_name]}" for model_name in model_names] + ["ERA5"]
        
        test_canvas_holder(data, suptitle, pic_name, col_names, plot_names, lat_list.min(), lat_list.max(), lon_list.min(), 
                            lon_list.max(), cmap_mlsp, cmap_wind, plot_dir, dpi=dpi)
        
        
def project(lat_min, lat_max, lon_min, lon_max, ax=None):
        if ax is None:
            ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        return ax       
    
    
def test_canvas_holder(data, suptitle, pic_name, col_titles, var_names, lat_min, lat_max, lon_min, lon_max, cmap_mslp, cmap_wind, 
                       plot_dir, contrast=False, dpi=400):
    
    samples, channels = len(data), len(data[0])
    fig=plt.figure(figsize=(8*samples,4*channels), facecolor='white')
    axes={}
    ims={}
    data_np = np.array(data)
    axes_class = (GeoAxes,dict(projection=ccrs.PlateCarree()))
    grid = AxesGrid(fig, 111, axes_class=axes_class,
            nrows_ncols=(channels,samples),
            axes_pad=(0.75, 0.25),
            cbar_pad=0.1,
            cbar_location="right",
            cbar_mode="edge",
            cbar_size="7%",
            label_mode='')
    
    for ind in range(samples):
        
        data_plot=data[ind]
        
        for i, var in enumerate(var_names):
            Var=var[0]
            unit=var[1]
            if Var=='mslp':
                Var = multiline_label("Mean Sea Level Pressure", sep=" ")
                cmap = cmap_mslp
            elif Var=='wind' :
                Var = multiline_label("Wind magnitude", sep=" ")
                cmap = cmap_wind
            else:
                cmap = 'coolwarm'
            axes[Var+str(ind)] = project(lat_min, lat_max, lon_min, lon_max, ax=grid[i*samples+ind])

            if not contrast:
                ims[Var+str(ind)] = data_plot[i].plot(ax=axes[Var+str(ind)], 
                                                    transform=ccrs.PlateCarree(), 
                                                    cmap=cmap, 
                                                    add_colorbar=False,
                                                    alpha=1)
            else :
                ims[Var+str(ind)] = data_plot[i].plot(ax=axes[Var+str(ind)], 
                                                    transform=ccrs.PlateCarree(), 
                                                    cmap=cmap, 
                                                    add_colorbar=False,
                                                    alpha=1,
                                                    vmin=data_np.min(axis=(0,2,3))[i],
                                                    vmax=data_np.max(axis=(0,2,3))[i])
            
            
            axes[Var+str(ind)].add_feature(cfeature.COASTLINE.with_scale('110m')) # adding coastline
            axes[Var+str(ind)].add_feature(cfeature.BORDERS.with_scale('110m')) # adding borders

            if i==0:
                axes[Var+str(ind)].set_title(col_titles[ind], fontsize = 15)#33
            else:
                axes[Var+str(ind)].set_title("")
            if ind==0:
                add_unit = ' ('+unit+')' if unit!='' else ""
                grid.cbar_axes[i].colorbar(ims[Var+str(ind)]).set_label(label=Var + add_unit, size=20)#33
                grid.cbar_axes[i].tick_params(labelsize=12)#32
            
                    
    fig.subplots_adjust(bottom=0.005, top=0.95, left=0.05, right=0.95)
    st=fig.suptitle(suptitle, fontsize='20')#36
    st.set_y(0.98)
    #st.set_y(0.98)
    fig.canvas.draw()
    
    #fig.tight_layout()
    plt.savefig(plot_dir+pic_name, dpi=dpi, bbox_inches='tight')    

        
        
def flatten(arg):
    # flattent list of any depth into a list of depth 1
    if not isinstance(arg, list): # if not list
        return [arg]
    return [x for sub in arg for x in flatten(sub)] # recurse and collect


def multiline_label(label, sep=" "):
    # to write long labels on several lines
    new_l = ''
    s = 0
    if type(label)==list:
        s_max = len(''.join(str(l) for l in label))
        cut = s_max // 2 if s_max > 11 else s_max
        for lab in label:
            s += len(lab)
            if s > cut:
                if lab != label[-1]:
                    new_l += '\n'+lab+sep
                else:
                    new_l += '\n'+lab
                s = 0
            else:
                if lab != label[-1]:
                    new_l += lab + sep
                else:
                    new_l += lab
    if type(label)==str:
        s_max = len(label)+1
        cut = s_max // 2 if s_max > 11 else s_max
        elmts = label.split(sep=sep)
        for elmt in elmts:
            s += len(elmt)+1 # +1 for the comma
            if s > cut:
                if elmt != elmts[-1]:
                    new_l += '\n'+elmt + sep
                else:
                    new_l += '\n'+elmt
                s = 0
            else:
                if elmt != elmts[-1]:
                    new_l += elmt + sep
                else:
                    new_l += elmt
    return new_l        





    
    
    
    
    """
    def test_canvas_holder2(data, suptitle, pic_name, row_titles, var_names, lat_min, lat_max, lon_min, lon_max, cmap_mslp, cmap_wind, 
                       plot_dir, contrast=False):
    
    samples, channels = len(data), len(data[0])
    fig=plt.figure(figsize=(6*channels,6*samples), facecolor='white')
    axes={}
    ims={}
    data_np = np.array(data)
    
    axes_class = (GeoAxes,dict(projection=ccrs.PlateCarree()))
    grid = AxesGrid(fig, 111, axes_class=axes_class,
            nrows_ncols=(samples, channels),
            axes_pad=(0.75, 0.25),
            cbar_pad=0.1,
            cbar_location="top",
            cbar_mode="edge",
            cbar_size="7%",
            label_mode='L')
    
    for ind, var in enumerate(var_names):
        mod_name = row_titles[ind]
        Var = var[0]
        unit = var[1]
        if Var=='mslp':
            Var = multiline_label("Mean Sea Level Pressure", sep=" ")
            cmap = cmap_mslp
        elif Var=='wind' :
            Var = multiline_label("Wind magnitude", sep=" ")
            cmap = cmap_wind
        else:
            cmap = 'coolwarm'
                    
        for i in range(samples):
    
            axes[mod_name+str(ind)] = project(lat_min, lat_max, lon_min, lon_max, ax=grid[i*channels+ind])

            if not contrast:
                ims[mod_name+str(ind)] = data[i][ind].plot(ax=axes[mod_name+str(ind)], 
                                                    transform=ccrs.PlateCarree(), 
                                                    cmap=cmap, 
                                                    add_colorbar=False,
                                                    alpha=1)
            else :
                ims[mod_name+str(ind)] = data[i][ind].plot(ax=axes[mod_name+str(ind)], 
                                                    transform=ccrs.PlateCarree(), 
                                                    cmap=cmap, 
                                                    add_colorbar=False,
                                                    alpha=1,
                                                    vmin=data_np.min(axis=(0,2,3))[ind],
                                                    vmax=data_np.max(axis=(0,2,3))[ind])
            
            
            axes[mod_name+str(ind)].add_feature(cfeature.COASTLINE.with_scale('110m')) # adding coastline
            axes[mod_name+str(ind)].add_feature(cfeature.BORDERS.with_scale('110m')) # adding borders
            axes[mod_name+str(ind)].set_title("")

            if ind==0:
                axes[mod_name+str(ind)].set_ylabel(mod_name, fontsize = 15)#33
            else:
                axes[mod_name+str(ind)].set_ylabel("")
            if i==0:
                add_unit = (' ('+unit+')') if unit!='' else ""
                grid.cbar_axes[ind].colorbar(ims[mod_name+str(ind)]).set_label(label=Var + add_unit, size=20)#33
                grid.cbar_axes[ind].tick_params(labelsize=12)#32
            
            
                    
    fig.subplots_adjust(bottom=0.005, top=0.98, left=0.05, right=0.95)
    st=fig.suptitle(suptitle, fontsize='20')#36
    st.set_y(0.98)
    #st.set_y(0.98)
    fig.canvas.draw()
    
    #fig.tight_layout()
    plt.savefig(plot_dir+pic_name, dpi=400, bbox_inches='tight')
    """