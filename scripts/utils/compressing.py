import os, sys, gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import argparse, subprocess
import numpy as np
import xarray as xr


parser = argparse.ArgumentParser()

parser.add_argument(
    "--folder",
    type=str,
    help="data folder",
    default=None,
)

args = parser.parse_args()
if args.folder is None:
    folder_to_process = input("Enter the folder to process: \n")
else:
    folder_to_process = args.folder

# get folder contents
folder_contents = os.listdir(folder_to_process)

# %%
# Run subprocesses for each file in the folder
for file in folder_contents:
    # check if file is a python file
    if not file.endswith("small.nc"):
        do_stuff = 1
        date_start, date_end, ldt = (
            file.split("_")[1],
            file.split("_")[3],
            file.split("_")[-1][:3],
        )
        print(date_start, date_end, ldt, flush=True)

        ds = xr.open_dataset(os.path.join(folder_to_process + file), chunks="auto")
        save_name = folder_to_process + file.split(".")[0] + "_small.nc"

        if os.path.exists(save_name):
            print(f"{save_name} already exists", flush=True)
            print("Removing original file", flush=True)
            subprocess.run(["rm", os.path.join(folder_to_process + file)])
            continue
        else:
            try:
                encoding = {}
                for data_var in ds.data_vars:
                    encoding[data_var] = {
                        "original_shape": ds[data_var].shape,
                        "_FillValue": -32767,
                        "dtype": np.int16,
                        "add_offset": ds[data_var].mean().compute().values,
                        "scale_factor": ds[data_var].std().compute().values
                        / 1000,  # save up to mean +- 32 std
                    }
                ds.to_netcdf(
                    save_name,
                    engine="netcdf4",
                    mode="w",
                    encoding=encoding,
                    compute=True,
                )
                print(f"Compression of {save_name} successful", flush=True)
                # remove original file
                subprocess.run(["rm", os.path.join(folder_to_process + file)])
                print(f"Removed {file}", flush=True)
                del encoding
            except:
                print("Compression failed", flush=True)
            del ds
            gc.collect()
