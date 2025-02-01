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
from scipy.interpolate import RegularGridInterpolator


def get_int_var(lat_target, lon_target, lats, lons, WRF_var):
    interpolator = RegularGridInterpolator((lats[:, 0], lons[0, :]), WRF_var)
    interpolated_value = interpolator((lat_target, lon_target))
    return interpolated_value


def find_nearest_grid_hgt(
    lat_target, lon_target, lats, lons, location_pft, IVGTYP_vprm, hgt, hgt_site, radius
):
    """Find the nearest grid index for a given lat/lon with the same PFT and lowest height difference."""

    # Get valid lat/lon values
    valid_mask = IVGTYP_vprm == location_pft
    valid_lats = np.where(valid_mask, lats, np.nan)
    valid_lons = np.where(valid_mask, lons, np.nan)

    # Convert latitude and longitude differences to km
    lat_diff = (
        valid_lats - lat_target
    )  # * 111  # approximate conversion factor for degrees to km
    lon_diff = (
        valid_lons - lon_target  # * 111 * np.cos(np.radians(lat_target))
    )  # adjust for latitude
    dist = np.sqrt(lat_diff**2 + lon_diff**2)
    dist_km = dist * 111  # approximate conversion factor for degrees to km

    # Mask the distance to only consider points within the radius
    within_radius_mask = dist_km <= abs(radius)
    dist_within_radius = np.where(within_radius_mask, dist_km, np.nan)

    # Check if there are valid points within the radius
    if np.all(np.isnan(dist_within_radius)):
        return None, None  # No valid points found within the radius

    # Calculate the height difference for points within the radius
    height_diff_within_radius = np.where(
        within_radius_mask, np.abs(hgt - hgt_site), np.nan
    )

    # Get the index of the minimum height difference within the radius
    min_height_diff_idx = np.unravel_index(
        np.nanargmin(height_diff_within_radius), lats.shape
    )

    # Get the value of the distance with the minimum height
    dist_idx = dist_within_radius[min_height_diff_idx[0], min_height_diff_idx[1]]

    # Return the minimum distance and the index of the minimum height difference
    return dist_idx, min_height_diff_idx


# Find the closest grid point with same PFT for each site
def find_nearest_grid(lat_target, lon_target, lats, lons, location_pft, IVGTYP_vprm):
    """Find the nearest grid index for a given lat/lon with the same PFT."""
    # Mask grid points that do not match the PFT
    valid_mask = IVGTYP_vprm == location_pft

    # Get valid lat/lon values
    valid_lats = np.where(valid_mask, lats, np.nan)
    valid_lons = np.where(valid_mask, lons, np.nan)

    # Convert latitude and longitude differences to km
    lat_diff = (
        valid_lats - lat_target
    )  # * 111  # approximate conversion factor for degrees to km
    lon_diff = (
        valid_lons - lon_target  # * 111 * np.cos(np.radians(lat_target))
    )  # adjust for latitude
    dist = np.sqrt(lat_diff**2 + lon_diff**2)
    dist_km = dist * 111  # approximate conversion factor for degrees to km

    min_index = np.unravel_index(
        np.nanargmin(dist), lats.shape
    )  # Find the index of the minimum valid distance

    return np.nanmin(dist_km), min_index


#     # # Debugging prints
#     # print(f"Target lat/lon: ({lat_target}, {lon_target})")
#     # print(f"Valid lats: min={np.nanmin(valid_lats)}, max={np.nanmax(valid_lats)}")
#     # print(f"Valid lons:  min={np.nanmin(valid_lons)}, max={np.nanmax(valid_lons)}")
#     # print(f"Latitude differences: min={np.nanmin(lat_diff)}, max={np.nanmax(lat_diff)}")
#     # print(f"Longitude differences: min={np.nanmin(lon_diff)}, max={np.nanmax(lon_diff)}")
#     # print(f"Distances: min={np.nanmin(dist)}, max={np.nanmax(dist)}")
#     # print(f"Distances (km): min={np.nanmin(dist_km)}, max={np.nanmax(dist_km)}")


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
        {
            "name": "CH-Oe2_ref",
            "CO2_ID": "",
            "lat": 47.2863,
            "lon": 7.7343,
            "pft": 6,  # "CRO",
            "hgt_site": 452,
        },
        {
            "name": "CH-Dav_ref",
            "CO2_ID": "",
            "lat": 46.8153,
            "lon": 9.8559,
            "pft": 1,  # "ENF",
            "hgt_site": 1639,
        },
        {
            "name": "DE-Lkb_ref",
            "CO2_ID": "",
            "lat": 49.0996,
            "lon": 13.3047,
            "pft": 1,  # "ENF",
            "hgt_site": 1308,
        },
        {
            "name": "IT-Lav_ref",
            "CO2_ID": "",
            "lat": 45.9562,
            "lon": 11.2813,
            "pft": 1,  # "ENF",
            "hgt_site": 1353,
        },
        {
            "name": "IT-Ren_ref",
            "CO2_ID": "",
            "lat": 46.5869,
            "lon": 11.4337,
            "pft": 1,  # "ENF",
            "hgt_site": 1730,
        },
        {
            "name": "AT-Neu_ref",
            "CO2_ID": "",
            "lat": 47.1167,
            "lon": 11.3175,
            "pft": 7,  # "GRA",
            "hgt_site": 970,
        },
        {
            "name": "IT-MBo_ref",
            "CO2_ID": "",
            "lat": 46.0147,
            "lon": 11.0458,
            "pft": 7,  # "GRA",
            "hgt_site": 1550,
        },
        {
            "name": "IT-Tor_ref",
            "CO2_ID": "",
            "lat": 45.8444,
            "lon": 7.5781,
            "pft": 7,  # "GRA",
            "hgt_site": 2160,
        },
        {
            "name": "CH-Lae_ref",
            "CO2_ID": "",
            "lat": 47.4781,
            "lon": 8.3644,
            "pft": 3,  # "MF",
            "hgt_site": 689,
        },
        {
            "name": "CH-Oe2_std",
            "CO2_ID": "_REF",
            "lat": 47.2863,
            "lon": 7.7343,
            "pft": 6,  # "CRO",
            "hgt_site": 452,
        },
        {
            "name": "CH-Dav_std",
            "CO2_ID": "_REF",
            "lat": 46.8153,
            "lon": 9.8559,
            "pft": 1,  # "ENF",
            "hgt_site": 1639,
        },
        {
            "name": "DE-Lkb_std",
            "CO2_ID": "_REF",
            "lat": 49.0996,
            "lon": 13.3047,
            "pft": 1,  # "ENF",
            "hgt_site": 1308,
        },
        {
            "name": "IT-Lav_std",
            "CO2_ID": "_REF",
            "lat": 45.9562,
            "lon": 11.2813,
            "pft": 1,  # "ENF",
            "hgt_site": 1353,
        },
        {
            "name": "IT-Ren_std",
            "CO2_ID": "_REF",
            "lat": 46.5869,
            "lon": 11.4337,
            "pft": 1,  # "ENF",
            "hgt_site": 1730,
        },
        {
            "name": "AT-Neu_std",
            "CO2_ID": "_REF",
            "lat": 47.1167,
            "lon": 11.3175,
            "pft": 7,  # "GRA",
            "hgt_site": 970,
        },
        {
            "name": "IT-MBo_std",
            "CO2_ID": "_REF",
            "lat": 46.0147,
            "lon": 11.0458,
            "pft": 7,  # "GRA",
            "hgt_site": 1550,
        },
        {
            "name": "IT-Tor_std",
            "CO2_ID": "_REF",
            "lat": 45.8444,
            "lon": 7.5781,
            "pft": 7,  # "GRA",
            "hgt_site": 2160,
        },
        {
            "name": "CH-Lae_std",
            "CO2_ID": "_REF",
            "lat": 47.4781,
            "lon": 8.3644,
            "pft": 3,  # "MF",
            "hgt_site": 689,
        },
        {
            "name": "CH-Oe2_tune",
            "CO2_ID": "_2",
            "lat": 47.2863,
            "lon": 7.7343,
            "pft": 6,  # "CRO",
            "hgt_site": 452,
        },
        {
            "name": "CH-Dav_tune",
            "CO2_ID": "_2",
            "lat": 46.8153,
            "lon": 9.8559,
            "pft": 1,  # "ENF",
            "hgt_site": 1639,
        },
        {
            "name": "DE-Lkb_tune",
            "CO2_ID": "_3",
            "lat": 49.0996,
            "lon": 13.3047,
            "pft": 1,  # "ENF",
            "hgt_site": 1308,
        },
        {
            "name": "IT-Lav_tune",
            "CO2_ID": "_4",
            "lat": 45.9562,
            "lon": 11.2813,
            "pft": 1,  # "ENF",
            "hgt_site": 1353,
        },
        {
            "name": "IT-Ren_tune",
            "CO2_ID": "_5",
            "lat": 46.5869,
            "lon": 11.4337,
            "pft": 1,  # "ENF",
            "hgt_site": 1730,
        },
        {
            "name": "AT-Neu_tune",
            "CO2_ID": "_4",
            "lat": 47.1167,
            "lon": 11.3175,
            "pft": 7,  # "GRA",
            "hgt_site": 970,
        },
        {
            "name": "IT-MBo_tune",
            "CO2_ID": "_3",
            "lat": 46.0147,
            "lon": 11.0458,
            "pft": 7,  # "GRA",
            "hgt_site": 1550,
        },
        {
            "name": "IT-Tor_tune",
            "CO2_ID": "_2",
            "lat": 45.8444,
            "lon": 7.5781,
            "pft": 7,  # "GRA",
            "hgt_site": 2160,
        },
        {
            "name": "CH-Lae_tune",
            "CO2_ID": "_2",
            "lat": 47.4781,
            "lon": 8.3644,
            "pft": 3,  # "MF",
            "hgt_site": 689,
        },
    ]

    # Define the remapping dictionary for CORINE vegetation types
    corine_to_vprm = {
        24: 1,  # Coniferous Forest (Evergreen)
        23: 2,  # Broad-leaved Forest (Deciduous)
        25: 3,
        29: 3,  # Mixed Forest and Transitional Woodland-Shrub
        27: 4,
        28: 4,  # Moors and Heathland, Sclerophyllous Vegetation (Shrubland)
        35: 5,
        36: 5,
        37: 5,  # Wetlands: Inland Marshes, Peat Bogs, Salt Marshes
        12: 6,
        13: 6,
        14: 6,
        15: 6,
        16: 6,
        17: 6,
        19: 6,
        20: 6,
        21: 6,
        22: 6,  # Cropland
        18: 7,
        26: 7,  # Grassland: Pastures, Natural Grasslands
        # Others mapped to 8 (gray)
        1: 8,
        2: 8,
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 8,
        8: 8,
        9: 8,
        10: 8,
        11: 8,
        30: 8,
        31: 8,
        32: 8,
        33: 8,
        34: 8,
        38: 8,
        39: 8,
        40: 8,
        41: 8,
        42: 8,
        43: 8,
        44: 8,
    }

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
        nc_fid1 = nc.Dataset(nc_f1, "r")
        xlat = nc_fid1.variables["XLAT"][0]  # Assuming the first time slice
        xlon = nc_fid1.variables["XLONG"][0]
        WRF_T2 = nc_fid1.variables["T2"][0]
        hgt = nc_fid1.variables["HGT"][0]
        IVGTYP = nc_fid1.variables["IVGTYP"][0]
        IVGTYP_vprm = np.vectorize(corine_to_vprm.get)(
            IVGTYP[:, :]
        )  # Create a new array for the simplified vegetation categories
        dx = (xlat[0, 0] - xlat[1, 0]) * 111
        radius = dx * 10

        print(nc_f1)
        # Initialize lists to store data for the current timestep
        data_row = {col: None for col in df_out.columns}  # Map columns to values

        # Extract data for each location
        for location in locations:
            lat_target, lon_target = location["lat"], location["lon"]
            WRF_gee = nc_fid1.variables[f"EBIO_GEE{location['CO2_ID']}"][0, 0, :, :]
            WRF_res = nc_fid1.variables[f"EBIO_RES{location['CO2_ID']}"][0, 0, :, :]

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

                dist_km, grid_idx = find_nearest_grid(
                    lat_target,
                    lon_target,
                    xlat,
                    xlon,
                    location["pft"],
                    IVGTYP_vprm,
                    hgt,
                    location["hgt_site"],
                    radius,
                )
                # print(location, dist_km)
                # add dist to the large locations dict which contains all the locations
                for loc in locations:
                    if loc["name"] == location["name"]:
                        loc["dist"] = dist_km
                        loc["hgt_wrf"] = hgt[grid_idx[0], grid_idx[1]]
                        loc["lat_wrf"] = xlat[grid_idx[0], grid_idx[1]]
                        loc["lon_wrf"] = xlon[grid_idx[0], grid_idx[1]]
                        break

                # Assign values to their respective columns
                data_row[f"{location['name']}_GEE"] = (
                    WRF_gee[grid_idx[0], grid_idx[1]] / 3600
                )
                data_row[f"{location['name']}_RES"] = (
                    WRF_res[grid_idx[0], grid_idx[1]] / 3600
                )
                data_row[f"{location['name']}_T2"] = WRF_T2[grid_idx[0], grid_idx[1]]

            elif method == "NNhgt":
                # Get nearest neighbour of GEE, RES, and T2 for the current location and append to the row

                dist_km, grid_idx = find_nearest_grid_hgt(
                    lat_target,
                    lon_target,
                    xlat,
                    xlon,
                    location["pft"],
                    IVGTYP_vprm,
                    hgt,
                    location["hgt_site"],
                    radius,
                )
                # print(location, dist_km)
                # add dist to the large locations dict which contains all the locations
                for loc in locations:
                    if loc["name"] == location["name"]:
                        loc["dist"] = dist_km
                        loc["hgt_wrf"] = hgt[grid_idx[0], grid_idx[1]]
                        loc["lat_wrf"] = xlat[grid_idx[0], grid_idx[1]]
                        loc["lon_wrf"] = xlon[grid_idx[0], grid_idx[1]]
                        break

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
    # write another csv file with the distances but use only the _ref sites
    df_out_dist = pd.DataFrame(
        columns=["name", "dist", "hgt_wrf", "lat_wrf", "lon_wrf", "pft"]
    )
    dist_rows = []
    for loc in locations:
        if "ref" in loc["name"]:
            dist_rows.append(
                {
                    "name": loc["name"],
                    "dist": loc["dist"],
                    "hgt_wrf": loc["hgt_wrf"],
                    "lat_wrf": loc["lat_wrf"],
                    "lon_wrf": loc["lon_wrf"],
                    "pft": loc["pft"],
                }
            )
    df_out_dist = pd.concat([df_out_dist, pd.DataFrame(dist_rows)], ignore_index=True)
    df_out_dist.to_csv(
        os.path.join(
            output_dir,
            f"distances_{method}_{wrf_path.split('_')[-1]}_{start_date.split('_')[0]}_{end_date.split('_')[0]}.csv",
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
            "-m",
            "--method",
            type=str,
            help="'NN' nearest neighbour, NNhgt for NN on same height or 'interpolated'",
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
        method = "NNhgt"  # "NN" nearest neighbour, NNhgt or "interpolated"
    for wrf_path in wrf_paths:
        exctract_timeseries(wrf_path, start_date, end_date, method)
    # exctract_timeseries(wrf_path, start_date, end_date,"interpolate")


if __name__ == "__main__":
    main()
