import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scripts import utils

lat_min = -60.
lat_max = 31.1859
lon_min = 20.0026
lon_max = 146.8982
INPUT = './panguweather/pangu_20180110_24h.grib'
INPUT2 = './panguweather/pangu_20180110_30h.grib'

ds = xr.load_dataset(INPUT, engine="cfgrib")
ds2 = xr.load_dataset(INPUT2, engine="cfgrib")
ds3 = xr.concat([ds, ds2], dim="time")
print(ds3)
fig = plt.figure(figsize=(8,4))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines(resolution="10m")

plot = ds['u10'].plot(cmap=plt.cm.viridis, transform=ccrs.PlateCarree())
fig.savefig("./test-im-all.png")


fig = plt.figure(figsize=(8,4))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines(resolution="10m")

ds_reduced = utils.select_localisation(ds, lat_min, lat_max, lon_min, lon_max)

plot = ds_reduced['u10'].plot(cmap=plt.cm.viridis, transform=ccrs.PlateCarree())
fig.savefig("./test-im-reduce.png")


fig = plt.figure(figsize=(8,4))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines(resolution="10m")

ds_min = utils.min_region_MLSP(ds_reduced)

plot = ds_min['u10'].plot(cmap=plt.cm.viridis, transform=ccrs.PlateCarree())
fig.savefig("./test-im-min-mlsp.png")


"""import GPUtil
import torch
import onnxruntime as ort
print(torch.cuda.is_available())
device = "cuda"

t = torch.arange(start=0, end=1e3).to(device)
print(ort.get_device())

if GPUtil.getAvailable():
    print(GPUtil.showUtilization())"""