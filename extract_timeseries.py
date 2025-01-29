import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from scipy.interpolate import griddata
import argparse
import sys
import xarray as xr
import time
from scipy.interpolate import RegularGridInterpolator


def get_int_var(lat_target, lon_target, lats, lons, WRF_var):
    interpolator = RegularGridInterpolator((lats[:, 0], lons[0, :]), WRF_var)
    interpolated_value = interpolator((lat_target, lon_target))
    return interpolated_value


# Find the closest grid point with same PFT for each site
def find_nearest_grid(lat_target, lon_target, lats, lons):
    """Find the nearest grid index for a given lat/lon."""
    dist = np.sqrt((lats - lat_target) ** 2 + (lons - lon_target) ** 2)
    return np.unravel_index(np.argmin(dist), lats.shape)


def extract_datetime_from_filename(filename):
    """
    Extract datetime from WRF filename assuming format 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'.
    """
    base_filename = os.path.basename(filename)
    date_str = base_filename.split("_")[-2] + "_" + base_filename.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


def exctract_timeseries(wrf_path, start_date, end_date, method):

    output_dir = "/scratch/c7071034/DATA/WRFOUT/csv"
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    # Define target locations (latitude, longitude)
    locations = [
        {"name": "CH-Oe2_ref", "CO2_ID": "", "lat": 47.2863, "lon": 7.7343},
        {"name": "CH-Dav_ref", "CO2_ID": "", "lat": 46.8153, "lon": 9.8559},
        {"name": "DE-Lkb_ref", "CO2_ID": "", "lat": 49.0996, "lon": 13.3047},
        {"name": "IT-Lav_ref", "CO2_ID": "", "lat": 45.9562, "lon": 11.2813},
        {"name": "IT-Ren_ref", "CO2_ID": "", "lat": 46.5869, "lon": 11.4337},
        {"name": "AT-Neu_ref", "CO2_ID": "", "lat": 47.1167, "lon": 11.3175},
        {"name": "IT-MBo_ref", "CO2_ID": "", "lat": 46.0147, "lon": 11.0458},
        {"name": "IT-Tor_ref", "CO2_ID": "", "lat": 45.8444, "lon": 7.5781},
        {"name": "CH-Lae_ref", "CO2_ID": "", "lat": 47.4781, "lon": 8.3644},
        {"name": "CH-Oe2_std", "CO2_ID": "_REF", "lat": 47.2863, "lon": 7.7343},
        {"name": "CH-Dav_std", "CO2_ID": "_REF", "lat": 46.8153, "lon": 9.8559},
        {"name": "DE-Lkb_std", "CO2_ID": "_REF", "lat": 49.0996, "lon": 13.3047},
        {"name": "IT-Lav_std", "CO2_ID": "_REF", "lat": 45.9562, "lon": 11.2813},
        {"name": "IT-Ren_std", "CO2_ID": "_REF", "lat": 46.5869, "lon": 11.4337},
        {"name": "AT-Neu_std", "CO2_ID": "_REF", "lat": 47.1167, "lon": 11.3175},
        {"name": "IT-MBo_std", "CO2_ID": "_REF", "lat": 46.0147, "lon": 11.0458},
        {"name": "IT-Tor_std", "CO2_ID": "_REF", "lat": 45.8444, "lon": 7.5781},
        {"name": "CH-Lae_std", "CO2_ID": "_REF", "lat": 47.4781, "lon": 8.3644},
        {"name": "CH-Oe2_tune", "CO2_ID": "_2", "lat": 47.2863, "lon": 7.7343},
        {"name": "CH-Dav_tune", "CO2_ID": "_2", "lat": 46.8153, "lon": 9.8559},
        {"name": "DE-Lkb_tune", "CO2_ID": "_3", "lat": 49.0996, "lon": 13.3047},
        {"name": "IT-Lav_tune", "CO2_ID": "_4", "lat": 45.9562, "lon": 11.2813},
        {"name": "IT-Ren_tune", "CO2_ID": "_5", "lat": 46.5869, "lon": 11.4337},
        {"name": "AT-Neu_tune", "CO2_ID": "_4", "lat": 47.1167, "lon": 11.3175},
        {"name": "IT-MBo_tune", "CO2_ID": "_3", "lat": 46.0147, "lon": 11.0458},
        {"name": "IT-Tor_tune", "CO2_ID": "_2", "lat": 45.8444, "lon": 7.5781},
        {"name": "CH-Lae_tune", "CO2_ID": "_2", "lat": 47.4781, "lon": 8.3644},
    ]

    # Initialize an empty DataFrame with time as the index and locations as columns
    columns = (
        [f"{location['name']}_GEE" for location in locations]
        + [f"{location['name']}_RES" for location in locations]
        + [f"{location['name']}_T2" for location in locations]
    )

    df_out = pd.DataFrame(columns=columns)
    file_list = [
        f
        for f in sorted(glob.glob(os.path.join(wrf_path, "wrfout_d01*")))
        if start_date_obj <= extract_datetime_from_filename(f) <= end_date_obj
    ]

    # Process each WRF file (representing one timestep)
    for nc_f1 in file_list:
        start_time = time.time()
        nc_fid1 = nc.Dataset(nc_f1, "r")
        xlat = nc_fid1.variables["XLAT"][0]  # Assuming the first time slice
        xlon = nc_fid1.variables["XLONG"][0]
        WRF_T2 = nc_fid1.variables["T2"][0]

        print(nc_f1)
        # Initialize lists to store data for the current timestep
        data_row = {col: None for col in df_out.columns}  # Map columns to values

        # Extract data for each location
        for location in locations:
            lat_target, lon_target = location["lat"], location["lon"]
            WRF_gee = nc_fid1.variables[f"EBIO_GEE{location["CO2_ID"]}"][0, 0, :, :]
            WRF_res = nc_fid1.variables[f"EBIO_RES{location["CO2_ID"]}"][0, 0, :, :]

            if method == "interpolated":
                # interpolate GEE, RES, and T2 for the current location and append to the row
                gee = get_int_var(lat_target, lon_target, xlat, xlon, WRF_gee) / 3600
                res = get_int_var(lat_target, lon_target, xlat, xlon, WRF_res) / 3600
                t2 = get_int_var(lat_target, lon_target, xlat, xlon, WRF_T2)

                # Assign values to their respective columns
                data_row[f"{location['name']}_GEE"] = gee
                data_row[f"{location['name']}_RES"] = res
                data_row[f"{location['name']}_T2"] = t2
            elif method == "NN":
                # Get nearest neighbour of GEE, RES, and T2 for the current location and append to the row

                grid_idx = find_nearest_grid(lat_target, lon_target, xlat, xlon)

                # Assign values to their respective columns
                data_row[f"{location['name']}_GEE"] = (
                    WRF_gee[grid_idx[0], grid_idx[1]] / 3600
                )
                data_row[f"{location['name']}_RES"] = (
                    WRF_res[grid_idx[0], grid_idx[1]] / 3600
                )
                data_row[f"{location['name']}_T2"] = WRF_T2[grid_idx[0], grid_idx[1]]

        # Append the current timestep data as a new row in the DataFrame
        temp_df_out = pd.DataFrame([data_row])
        df_out = pd.concat([df_out, temp_df_out], ignore_index=True)
        # nc_fid1.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")

    # Set the time as the index of the DataFrame
    df_out.index = [extract_datetime_from_filename(f) for f in file_list]
    # Optionally, save the DataFrame to CSV
    output_filename = f"wrf_FLUXNET_sites_{method}_{wrf_path.split('_')[-1]}_{start_date.split('_')[0]}_{end_date.split('_')[0]}.csv"

    df_out.to_csv(
        os.path.join(
            output_dir,
            output_filename,
        )
    )
    return


def main():

    if len(sys.argv) > 1:  # to run all on cluster with 'submit_jobs_tune_VPRM.sh'
        parser = argparse.ArgumentParser(description="Description of your script")
        parser.add_argument("-w", "--wrf_paths", type=str, help="list of WRF paths")
        parser.add_argument(
            "-s", "--start", type=str, help="Format: 2012-07-01 01:00:00"
        )
        parser.add_argument("-e", "--end", type=str, help="Format: 2012-07-01 01:00:00")
        parser.add_argument(
            "-m", "--method", type=str, help="'NN' nearest neighbour or 'interpolated'"
        )

        args = parser.parse_args()
        wrf_path = args.wrf_path
        start_date = args.start
        end_date = args.end
        method = args.method
    else:  # to run locally for single cases
        start_date = "2012-07-01 01:00:00"
        end_date = "2012-07-30 00:00:00"
        wrf_paths = [
            "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km",
            "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250105_193347_ALPS_9km",
            "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241229_112716_ALPS_27km",
            "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241227_183215_ALPS_54km",
        ]
        method = "interpolated"  # "NN" nearest neighbour or "interpolated"
    for wrf_path in wrf_paths:
        exctract_timeseries(wrf_path, start_date, end_date, method)
    # exctract_timeseries(wrf_path, start_date, end_date,"interpolate")


if __name__ == "__main__":
    main()
