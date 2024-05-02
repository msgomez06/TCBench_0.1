import cdsapi
import numpy as np
import pickle
import argparse
import subprocess
import xarray as xr
import pandas as pd

# %% Load Louis' dates
folder_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/"
with open(folder_path + "valid_dates_1980_00_06_12_18.pkl", "rb") as f:
    valid_dates = pickle.load(f)
# %% Load the original ibtracs data
ibtracs_path = "/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/alpha_bench/tracks/ibtracs/ibtracs.ALL.list.v04r00.csv"
df = pd.read_csv(ibtracs_path, dtype=str, skiprows=[1], na_filter=False)
# %% parse the datetimes
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

# Select the years after 1980
df = df[df["ISO_TIME"].dt.year >= 1980]

# and the "standard" timesteps
hours_to_select = [0, 6, 12, 18]
df = df[df["ISO_TIME"].dt.hour.isin(hours_to_select)]

# Adding the "negative" lead times to be able to handle genesis
max_lead = 168
step = 6
iso_times = df["ISO_TIME"].unique()
# set up a timedelta64 array for up to negative max_lead time with step hours
timedeltas = np.arange(-np.timedelta64(max_lead, "h"), 0, np.timedelta64(step, "h"))
# create a new array with the original iso_times and the negative lead times
iso_copy = iso_times.copy()
for delta in timedeltas:
    iso_copy = np.hstack([iso_copy, iso_times + delta])
# convert to datetime index
iso_times = pd.to_datetime(iso_copy).unique()


# years = df["ISO_TIME"].dt.year.unique()
years = iso_times.year.unique()
# %%
full_dates = {}
for year in years:
    # months = df["ISO_TIME"][df["ISO_TIME"].dt.year == year].dt.month.unique()
    months = iso_times[iso_times.year == year].month.unique()
    month_dict = {}
    for month in months:
        # days = df["ISO_TIME"][df["ISO_TIME"].dt.year == year][
        #     df["ISO_TIME"].dt.month == month
        # ].dt.day.unique()
        days = iso_times[iso_times.year == year][
            iso_times[iso_times.year == year].month == month
        ].day.unique()
        month_dict[month] = days.sort_values()
    full_dates[year] = month_dict
# %%
# generate the dates of all calendar days for the years 1980-2023. Make sure to handle
# leap years correctly.
all_dates = {}
dates_to_dl = {}
for year in range(1980, 2024):
    months = list(range(1, 13))
    month_dict = {}
    month_dl = {}
    for month in months:
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days = list(range(1, 32))
        elif month in [4, 6, 9, 11]:
            days = list(range(1, 31))
        else:
            if year % 4 == 0:
                if year % 100 == 0:
                    if year % 400 == 0:
                        days = list(range(1, 30))
                    else:
                        days = list(range(1, 29))
                else:
                    days = list(range(1, 30))
            else:
                days = list(range(1, 29))
        month_dict[month] = np.array(days)

        try:
            ibtracs_month = full_dates[year][month]
            # print(ibtracs_month)
            # print(np.isin(month_dict[month], ibtracs_month))
            if (~np.isin(month_dict[month], ibtracs_month)).sum() == 0:
                print(f"{year}-{month} is the same (ibtracs vs full dates)")
            else:
                print(f"{year}-{month} is different (ibtracs vs full dates)")
                # print out the day that is not in the full dates
                print(month_dict[month][~np.isin(month_dict[month], ibtracs_month)])
        except:
            print(f"{year}-{month} is different")
            print(f"ibtracs does not have this month")

        try:
            if month < 10:
                month_str = "0" + str(month)
            else:
                month_str = str(month)

            louis_month = np.array(valid_dates[str(year)][month_str]).astype(int)
            # print(f'louis: {louis_month}')
            if (~np.isin(month_dict[month], louis_month)).sum() == 0:
                print(f"{year}-{month} is the same (louis vs full dates)")
            else:
                print(f"{year}-{month} is different (louis vs full dates)")
                # print out the day that is not in the full dates
                print(month_dict[month][~np.isin(month_dict[month], louis_month)])
        except:
            print(f"{year}-{month} is different")
            print(f"Louis does not have this month")

        print("Louis month: ", louis_month)
        print("ibtracs month: ", ibtracs_month)

        try:
            if (~np.isin(ibtracs_month, louis_month)).sum() == 0:
                print(f"{year}-{month} is the same (louis vs ibtracs)")
            else:
                print(f"{year}-{month} is different (louis vs ibtracs)")
                # print out the day that is not in the full dates
                print(ibtracs_month[~np.isin(ibtracs_month, louis_month)])
                # month_dl[month] = ibtracs_month[~np.isin(ibtracs_month, louis_month)]

                if not (year == 2023 and month > 5):
                    month_dl[month] = list(
                        ibtracs_month[~np.isin(ibtracs_month, louis_month)].astype(str)
                    )

        except:
            print(f"{year}-{month} is different")
            print(f"Louis does not have this month")

        print("\n")
    all_dates[year] = month_dict
    dates_to_dl[year] = month_dl

# %%
## save the dates to download using pickle
# with open(
#     "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/ERA5/"
#     + "dates_to_dl_milton.pkl",
#     "wb",
# ) as f:
#     pickle.dump(dates_to_dl, f)
monika_dates = {}

for month, dates in full_dates[2020].items():
    monika_dates[month] = [str(date) for date in dates]

# save the dates to download using pickle
with open(
    "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/ERA5/"
    + "dates_to_dl_Monika.pkl",
    "wb",
) as f:
    pickle.dump({2020: monika_dates}, f)

# %%
