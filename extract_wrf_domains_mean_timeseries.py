#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 11:36:35 2021

@author: madse
"""

import netCDF4 as nc
import glob
import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def proj_on_finer_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_3km
):
    proj_var = griddata(
        (lats_coarse.flatten(), lons_coarse.flatten()),
        var_coarse.flatten(),
        (lats_fine, lons_fine),
        method="linear",
    ).reshape(WRF_var_3km.shape)
    return proj_var


def proj_CAMS_on_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_3km
):
    # Corrected meshgrid order
    lats_coarse_2d, lons_coarse_2d = np.meshgrid(lats_coarse, lons_coarse)
    # Reverse the order of latitude coordinates
    lat_CAMS_2d_reversed = lats_coarse_2d[::-1]
    # Reverse the order of the variable values
    var_coarse_reversed = np.flipud(var_coarse)
    # Flatten the coordinates
    points_coarse = np.column_stack(
        (lat_CAMS_2d_reversed.flatten(), lons_coarse_2d.flatten())
    )
    points_fine = np.column_stack((lats_fine.flatten(), lons_fine.flatten()))
    # Perform interpolation
    proj_var = griddata(
        points_coarse, var_coarse_reversed.flatten(), points_fine, method="nearest"
    ).reshape(WRF_var_3km.shape)

    return proj_var


def proj_on_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_3km
):
    # Corrected meshgrid order
    lats_coarse_2d, lons_coarse_2d = np.meshgrid(lats_coarse, lons_coarse)
    # Flatten the coordinates
    points_coarse = np.column_stack(
        (lats_coarse_2d.flatten(), lons_coarse_2d.flatten())
    )
    points_fine = np.column_stack((lats_fine.flatten(), lons_fine.flatten()))
    # Perform interpolation
    proj_var = griddata(
        points_coarse, var_coarse.flatten(), points_fine, method="nearest"
    ).reshape(WRF_var_3km.shape)

    return proj_var


def extract_datetime_from_filename(filename):
    """
    Extract datetime from WRF filename assuming format 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'.
    """
    base_filename = os.path.basename(filename)
    date_str = base_filename.split("_")[-2] + "_" + base_filename.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


################################# INPUT ##############################################

start_date = "2012-07-02 00:00:00"
end_date = "2012-07-30 00:00:00"
wrf_paths = [
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250105_193347_ALPS_9km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241229_112716_ALPS_27km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241227_183215_ALPS_54km",
]

outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"
subdaily = "_subdailyC3"  # "_subdailyC3" or "" to use subdaily GPP

#######################################################################################
# load CAMS data
CAMS_path = (
    "/scratch/c7071034/DATA/CAMS/ghg-reanalysis_surface_2012-07-01_2012-07-31.nc"
)

CAMS_data = nc.Dataset(CAMS_path)
times_CAMS = CAMS_data.variables["valid_time"]

CAMS_vars = ["fco2gpp", "fco2rec", "t2m"]
factor_kgC = 1000 / 44.01 * 1000000  # conversion from kgCO2/m2/s to  mumol/m2/s
CAMS_factors = [factor_kgC, -factor_kgC, 273.15]


# Use glob to list all files in the directory
# file_list_27km = sorted(glob.glob(os.path.join(WRF_path, "wrfout_d01*")))
WRF_vars = ["EBIO_GEE", "EBIO_RES", "T2"]
units = [" [mmol m² s⁻¹]", " [mmol m² s⁻¹]", " [K]"]
name_vars = {"EBIO_GEE": "WRF GPP", "EBIO_RES": "WRF RECO", "T2": "WRF T2M"}
WRF_factors = [-1 / 3600, 1 / 3600, 273.15]

# Initialize an empty DataFrame with time as the index and locations as columns
start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
end_date_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
file_list = [
    os.path.basename(f)  # Extract only the filename
    for f in sorted(glob.glob(os.path.join(wrf_paths[0], "wrfout_d01*")))
    if start_date_obj <= extract_datetime_from_filename(f) <= end_date_obj
]
columns = ["GPP", "RECO", "T2"]
timestamps = [extract_datetime_from_filename(f) for f in file_list]
time_index = pd.to_datetime(timestamps)

# set standard deviation of topography
STD_TOPOs = [100]
STD_TOPO_flags = ["gt", "lt"]  # "lt" lower than or "gt" greater than STD_TOPO
for STD_TOPO in STD_TOPOs:
    for STD_TOPO_flag in STD_TOPO_flags:

        # Initialize empty DataFrames with time as the index
        df_out_3km = pd.DataFrame(index=time_index, columns=columns)
        df_out_9km = pd.DataFrame(index=time_index, columns=columns)
        df_out_27km = pd.DataFrame(index=time_index, columns=columns)
        df_out_54km = pd.DataFrame(index=time_index, columns=columns)
        df_out_cams = pd.DataFrame(index=time_index, columns=columns)
        data_row_3km = {col: 0 for col in df_out_3km.columns}
        data_row_9km = {col: 0 for col in df_out_3km.columns}
        data_row_27km = {col: 0 for col in df_out_3km.columns}
        data_row_54km = {col: 0 for col in df_out_3km.columns}
        data_row_cams = {col: 0 for col in df_out_3km.columns}
        # define cold only for GPP
        df_out_P_3km = pd.DataFrame(index=time_index, columns=["GPP"])
        df_out_P_9km = pd.DataFrame(index=time_index, columns=["GPP"])
        df_out_P_27km = pd.DataFrame(index=time_index, columns=["GPP"])
        df_out_P_54km = pd.DataFrame(index=time_index, columns=["GPP"])
        data_row_P_3km = {col: 0 for col in df_out_P_3km.columns}
        data_row_P_9km = {col: 0 for col in df_out_P_3km.columns}
        data_row_P_27km = {col: 0 for col in df_out_P_3km.columns}
        data_row_P_54km = {col: 0 for col in df_out_P_3km.columns}

        for wrf_file in file_list:
            time = extract_datetime_from_filename(wrf_file)
            print("processing ", time)
            for (
                WRF_var,
                CAMS_var,
                factor,
                unit,
                column,
                WRF_factor,
            ) in zip(
                WRF_vars,
                CAMS_vars,
                CAMS_factors,
                units,
                columns,
                WRF_factors,
            ):
                # WRF
                i = 0
                # Loop through the files for the timestep
                # for nc_f1 in file_list_27km:
                nc_fid54km = nc.Dataset(os.path.join(wrf_paths[3], wrf_file), "r")
                nc_fid27km = nc.Dataset(os.path.join(wrf_paths[2], wrf_file), "r")
                nc_fid9km = nc.Dataset(os.path.join(wrf_paths[1], wrf_file), "r")
                nc_fid3km = nc.Dataset(os.path.join(wrf_paths[0], wrf_file), "r")

                times_variable = nc_fid3km.variables["Times"]
                start_date_bytes = times_variable[0, :].tobytes()
                start_date_str = start_date_bytes.decode("utf-8")
                lats_fine = nc_fid3km.variables["XLAT"][0, :, :]
                lons_fine = nc_fid3km.variables["XLONG"][0, :, :]
                landmask = nc_fid3km.variables["LANDMASK"][0, :, :]
                land_mask = landmask == 1

                if WRF_var == "T2":
                    WRF_var_3km = nc_fid3km.variables[WRF_var][0, :, :] - WRF_factor
                    WRF_var_9km = nc_fid9km.variables[WRF_var][0, :, :] - WRF_factor
                    WRF_var_27km = nc_fid27km.variables[WRF_var][0, :, :] - WRF_factor
                    WRF_var_54km = nc_fid54km.variables[WRF_var][0, :, :] - WRF_factor
                else:
                    WRF_var_3km = nc_fid3km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    WRF_var_9km = nc_fid9km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    WRF_var_27km = (
                        nc_fid27km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    )
                    WRF_var_54km = (
                        nc_fid54km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    )

                lats_9km = nc_fid9km.variables["XLAT"][0, :, :]
                lons_9km = nc_fid9km.variables["XLONG"][0, :, :]
                landmask_2 = nc_fid9km.variables["LANDMASK"][0, :, :]
                WRF_var_9km[landmask_2 == 0] = np.nan
                proj_WRF_var_9km = proj_on_finer_WRF_grid(
                    lats_9km, lons_9km, WRF_var_9km, lats_fine, lons_fine, WRF_var_3km
                )

                lats_27km = nc_fid27km.variables["XLAT"][0, :, :]
                lons_27km = nc_fid27km.variables["XLONG"][0, :, :]
                landmask_1 = nc_fid27km.variables["LANDMASK"][0, :, :]
                WRF_var_27km[landmask_1 == 0] = np.nan
                proj_WRF_var_27km = proj_on_finer_WRF_grid(
                    lats_27km,
                    lons_27km,
                    WRF_var_27km,
                    lats_fine,
                    lons_fine,
                    WRF_var_3km,
                )

                lats_54km = nc_fid54km.variables["XLAT"][0, :, :]
                lons_54km = nc_fid54km.variables["XLONG"][0, :, :]
                landmask_1 = nc_fid54km.variables["LANDMASK"][0, :, :]
                stdh_topo_54km = nc_fid54km.variables["VAR"][0, :, :]
                WRF_var_54km[landmask_1 == 0] = np.nan
                proj_WRF_var_54km = proj_on_finer_WRF_grid(
                    lats_54km,
                    lons_54km,
                    WRF_var_54km,
                    lats_fine,
                    lons_fine,
                    WRF_var_3km,
                )
                proj_stdh_topo_54km = proj_on_finer_WRF_grid(
                    lats_54km,
                    lons_54km,
                    stdh_topo_54km,
                    lats_fine,
                    lons_fine,
                    WRF_var_3km,
                )

                if STD_TOPO_flag == "gt":
                    stdh_mask = proj_stdh_topo_54km >= STD_TOPO
                elif STD_TOPO_flag == "lt":
                    stdh_mask = proj_stdh_topo_54km < STD_TOPO
                mask = land_mask * stdh_mask

                WRF_var_3km_topo_m = np.nanmean(WRF_var_3km[mask])
                WRF_var_9km_topo = np.nanmean(proj_WRF_var_9km[mask])
                WRF_var_27km_topo = np.nanmean(proj_WRF_var_27km[mask])
                WRF_var_54km_topo = np.nanmean(proj_WRF_var_54km[mask])
                # process CAMS data if times fit
                start_date_nc_f1 = datetime.strptime(
                    start_date_str, "%Y-%m-%d_%H:%M:%S"
                )
                new_time = start_date_nc_f1
                j = 0
                CAMS_topo = np.nan
                for time_CAMS in times_CAMS:
                    date_CAMS = (
                        datetime(1970, 1, 1)
                        + timedelta(seconds=int(time_CAMS))
                        # - timedelta(hours=1)
                    )
                    j = j + 1
                    if new_time == date_CAMS:
                        lat_CAMS = CAMS_data.variables["latitude"][:]
                        lon_CAMS = CAMS_data.variables["longitude"][:]
                        if WRF_var == "T2":
                            var_CAMS = (
                                CAMS_data.variables[CAMS_var][j - 1, :, :].data - factor
                            )  # convert unit to mmol m-2 s-1
                        else:
                            var_CAMS = (
                                CAMS_data.variables[CAMS_var][j - 1, :, :].data * factor
                            )  # convert unit to mmol m-2 s-1
                        CAMS_proj = proj_CAMS_on_WRF_grid(
                            lat_CAMS,
                            lon_CAMS,
                            var_CAMS,
                            lats_fine,
                            lons_fine,
                            WRF_var_3km,
                        )

                        CAMS_topo = np.mean(CAMS_proj[mask])

                data_row_3km[column] = WRF_var_3km_topo_m
                data_row_9km[column] = WRF_var_9km_topo
                data_row_27km[column] = WRF_var_27km_topo
                data_row_54km[column] = WRF_var_54km_topo
                data_row_cams[column] = CAMS_topo

                if column == "GPP":
                    time_str = time.strftime("%Y-%m-%d_%H:%M:%S")

                    nc_fid3km = nc.Dataset(
                        f"/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gpp_pmodel/gpp_pmodel{subdaily}_3km_"
                        + time_str
                        + ".nc",
                        "r",
                    )
                    gpp_P_3km = nc_fid3km.variables["GPP_Pmodel"][:, :].copy()
                    nc_fid9km = nc.Dataset(
                        f"/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gpp_pmodel/gpp_pmodel{subdaily}_9km_"
                        + time_str
                        + ".nc",
                        "r",
                    )
                    gpp_P_9km = nc_fid9km.variables["GPP_Pmodel"][:, :]
                    nc_fid27km = nc.Dataset(
                        f"/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gpp_pmodel/gpp_pmodel{subdaily}_27km_"
                        + time_str
                        + ".nc",
                        "r",
                    )
                    gpp_P_27km = nc_fid27km.variables["GPP_Pmodel"][:, :]
                    nc_fid54km = nc.Dataset(
                        f"/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gpp_pmodel/gpp_pmodel{subdaily}_54km_"
                        + time_str
                        + ".nc",
                        "r",
                    )
                    gpp_P_54km = nc_fid54km.variables["GPP_Pmodel"][:, :]

                    proj_WRF_P_var_9km = proj_on_finer_WRF_grid(
                        lats_9km, lons_9km, gpp_P_9km, lats_fine, lons_fine, WRF_var_3km
                    )
                    proj_WRF_P_var_27km = proj_on_finer_WRF_grid(
                        lats_27km,
                        lons_27km,
                        gpp_P_27km,
                        lats_fine,
                        lons_fine,
                        WRF_var_3km,
                    )
                    proj_WRF_P_var_54km = proj_on_finer_WRF_grid(
                        lats_54km,
                        lons_54km,
                        gpp_P_54km,
                        lats_fine,
                        lons_fine,
                        WRF_var_3km,
                    )

                    data_row_P_3km[column] = np.nanmean(gpp_P_3km[mask])
                    data_row_P_9km[column] = np.nanmean(proj_WRF_P_var_9km[mask])
                    data_row_P_27km[column] = np.nanmean(proj_WRF_P_var_27km[mask])
                    data_row_P_54km[column] = np.nanmean(proj_WRF_P_var_54km[mask])

                i += 1

            df_out_3km.loc[time, :] = data_row_3km
            df_out_9km.loc[time, :] = data_row_9km
            df_out_27km.loc[time, :] = data_row_27km
            df_out_54km.loc[time, :] = data_row_54km
            df_out_cams.loc[time, :] = data_row_cams
            df_out_P_3km.loc[time, :] = data_row_P_3km
            df_out_P_9km.loc[time, :] = data_row_P_9km
            df_out_P_27km.loc[time, :] = data_row_P_27km
            df_out_P_54km.loc[time, :] = data_row_P_54km

        # Add suffixes to columns
        df_out_3km = df_out_3km.add_suffix("_3km")
        df_out_9km = df_out_9km.add_suffix("_9km")
        df_out_27km = df_out_27km.add_suffix("_27km")
        df_out_54km = df_out_54km.add_suffix("_54km")
        df_out_cams = df_out_cams.add_suffix("_CAMS")
        df_out_P_3km = df_out_P_3km.add_suffix("_pmodel_3km")
        df_out_P_9km = df_out_P_9km.add_suffix("_pmodel_9km")
        df_out_P_27km = df_out_P_27km.add_suffix("_pmodel_27km")
        df_out_P_54km = df_out_P_54km.add_suffix("_pmodel_54km")

        # Merge all DataFrames horizontally
        merged_df = pd.concat(
            [
                df_out_3km,
                df_out_9km,
                df_out_27km,
                df_out_54km,
                df_out_cams,
                df_out_P_3km,
                df_out_P_9km,
                df_out_P_27km,
                df_out_P_54km,
            ],
            axis=1,
        )

        # Save to CSV
        merged_df.to_csv(
            f"{outfolder}timeseries_domain_averaged{subdaily}_std_topo_{STD_TOPO_flag}_{STD_TOPO}_{start_date}_{end_date}.csv",
            index=True,
        )
