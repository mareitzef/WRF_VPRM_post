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
        method="nearest",
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
plotting_scatter = False
T_bin_flag = True
plotting_scatter_all = True
start_date = "2012-07-01 01:00:00"
end_date = "2012-07-31 00:00:00"
T_bin_size = 1
hour_start = 5
hour_end = 18
STD_TOPOs = [100]
STD_TOPO_flags = ["gt"]  # "lt" lower than or "gt" greater than STD_TOPO

wrf_paths = [
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250105_193347_ALPS_9km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241229_112716_ALPS_27km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241227_183215_ALPS_54km",
]
wrf_files = "wrfout_d01*"
wrf_file_00 = "wrfout_d01_2012-07-01_00:00:00"

outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

#######################################################################################
# load CAMS data
CAMS_path = (
    "/scratch/c7071034/DATA/CAMS/ghg-reanalysis_surface_2012-07-01_2012-07-31.nc"
)

CAMS_data = nc.Dataset(CAMS_path)
times_CAMS = CAMS_data.variables["valid_time"]

CAMS_vars = ["fco2gpp", "fco2rec", "t2m"]
factor_kgC = -1000 / 44 * 1000000  # conversion from kgCO2/m2/s to  mumol/m2/s
CAMS_factors = [factor_kgC, factor_kgC, 273.15]


WRF_vars = ["EBIO_GEE", "EBIO_RES", "NEE", "T2"]
units = [" [mmol m² s⁻¹]", " [mmol m² s⁻¹]", " [mmol m² s⁻¹]", " [K]"]
name_vars = {
    "EBIO_GEE": "WRF GPP",
    "EBIO_RES": "WRF RECO",
    "NEE": "WRF NEE",
    "T2": "WRF T2M",
}
WRF_factors = [-1 / 3600, 1 / 3600, 1 / 3600, 273.15]

# Initialize an empty DataFrame with time as the index and locations as columns
start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
end_date_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

file_list = [
    os.path.basename(f)  # Extract only the filename
    for f in sorted(glob.glob(os.path.join(wrf_paths[0], wrf_files)))
    if start_date_obj <= extract_datetime_from_filename(f) <= end_date_obj
]

file_list = [
    f
    for f in file_list
    if hour_start <= extract_datetime_from_filename(f).hour <= hour_end
]

columns = ["GPP", "RECO", "NEE", "T2"]
timestamps = [extract_datetime_from_filename(f) for f in file_list]
time_index = pd.to_datetime(timestamps)


diff_27_3km_4D = []
T2_27km_4D = []

# set standard deviation of topography
for STD_TOPO in STD_TOPOs:
    for STD_TOPO_flag in STD_TOPO_flags:

        for wrf_file in file_list:
            ini_switch = True
            time = extract_datetime_from_filename(wrf_file)
            print("processing ", time)
            for (
                WRF_var,
                unit,
                column,
                WRF_factor,
            ) in zip(
                WRF_vars,
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
                # hgt_3km = nc_fid3km.variables["HGT"][0, :, :]
                land_mask = landmask == 1

                if WRF_var == "T2":
                    WRF_var_3km = nc_fid3km.variables[WRF_var][0, :, :] - WRF_factor
                    WRF_var_9km = nc_fid9km.variables[WRF_var][0, :, :] - WRF_factor
                    WRF_var_27km = nc_fid27km.variables[WRF_var][0, :, :] - WRF_factor
                    # WRF_var_54km = nc_fid54km.variables[WRF_var][0, :, :] - WRF_factor
                elif WRF_var == "NEE":
                    WRF_var_3km = (
                        nc_fid3km.variables["EBIO_GEE"][0, 0, :, :]
                        + nc_fid3km.variables["EBIO_RES"][0, 0, :, :]
                    ) * WRF_factor
                    WRF_var_9km = (
                        nc_fid9km.variables["EBIO_GEE"][0, 0, :, :]
                        + nc_fid9km.variables["EBIO_RES"][0, 0, :, :]
                    ) * WRF_factor
                    WRF_var_27km = (
                        nc_fid27km.variables["EBIO_GEE"][0, 0, :, :]
                        + nc_fid27km.variables["EBIO_RES"][0, 0, :, :]
                    ) * WRF_factor
                    # WRF_var_54km = (
                    #     nc_fid54km.variables["EBIO_GEE"][0, 0, :, :]
                    #     + nc_fid54km.variables["EBIO_RES"][0, 0, :, :]
                    # ) * WRF_factor
                else:
                    WRF_var_3km = nc_fid3km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    WRF_var_9km = nc_fid9km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    WRF_var_27km = (
                        nc_fid27km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    )
                    # WRF_var_54km = (
                    #     nc_fid54km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    # )

                lats_27km = nc_fid27km.variables["XLAT"][0, :, :]
                lons_27km = nc_fid27km.variables["XLONG"][0, :, :]
                landmask_4 = nc_fid27km.variables["LANDMASK"][0, :, :]
                hgt_27km = nc_fid27km.variables["HGT"][0, :, :]
                stdh_topo_27km = nc_fid27km.variables["VAR"][0, :, :]
                WRF_var_27km[landmask_4 == 0] = np.nan
                proj_WRF_var_27km = proj_on_finer_WRF_grid(
                    lats_27km,
                    lons_27km,
                    WRF_var_27km,
                    lats_fine,
                    lons_fine,
                    WRF_var_3km,
                )
                proj_hgt_27km = proj_on_finer_WRF_grid(
                    lats_27km, lons_27km, hgt_27km, lats_fine, lons_fine, WRF_var_3km
                )
                proj_stdh_topo_27km = proj_on_finer_WRF_grid(
                    lats_27km,
                    lons_27km,
                    stdh_topo_27km,
                    lats_fine,
                    lons_fine,
                    WRF_var_3km,
                )

                if STD_TOPO_flag == "gt":
                    stdh_mask = proj_stdh_topo_27km >= STD_TOPO
                elif STD_TOPO_flag == "lt":
                    stdh_mask = proj_stdh_topo_27km < STD_TOPO
                mask = land_mask * stdh_mask

                WRF_var_diff_27_3km_2D = np.where(
                    mask, proj_WRF_var_27km - WRF_var_3km, np.nan
                )
                diff_27_3km_4D.append(
                    {"time": time.hour, WRF_var: WRF_var_diff_27_3km_2D}
                )
                if WRF_var == "T2":
                    WRF_T2_2d = np.where(mask, proj_WRF_var_27km, np.nan)
                    T2_27km_4D.append({"time": time.hour, "T2": WRF_T2_2d})

        # Convert lists to 3D arrays
        var_data = {}
        hours = [entry["time"] for entry in diff_27_3km_4D]

        for entry in diff_27_3km_4D:
            for var_name, data in entry.items():
                if var_name != "time":  # Skip the 'time' key
                    if var_name not in var_data:
                        var_data[var_name] = []
                    var_data[var_name].append(data)
        diff_27_3km_3D = {
            var: np.stack(data_list, axis=0) for var, data_list in var_data.items()
        }
        var_data = {}
        for entry in T2_27km_4D:
            for var_name, data in entry.items():
                if var_name != "time":  # Skip the 'time' key
                    if var_name not in var_data:
                        var_data[var_name] = []
                    var_data[var_name].append(data)
        T2_27km_3D = {
            var: np.stack(data_list, axis=0) for var, data_list in var_data.items()
        }

        T_ref_values = range(
            int(np.nanmin(T2_27km_3D["T2"]) + 1),
            int(np.nanmax(T2_27km_3D["T2"]) - 1),
            T_bin_size,
        )
        if ini_switch == True:
            df_coeff = pd.DataFrame(index=T_ref_values)
            ini_switch = False

        # correlate T2 and WRF_var for each step in T_ref_values
        for WRF_var in WRF_vars[:-1]:
            coeff_all_T = []
            for T_ref in T_ref_values:
                try:
                    temp_mask = (T2_27km_3D["T2"] >= T_ref) & (
                        T2_27km_3D["T2"] <= T_ref + T_bin_size
                    )
                    masked_diff_T2 = diff_27_3km_3D["T2"][temp_mask]
                    masked_diff_var = diff_27_3km_3D[WRF_var][temp_mask]

                    idx = np.isfinite(masked_diff_var) & np.isfinite(masked_diff_T2)
                    diff_T2_t = masked_diff_T2[idx]
                    diff_var_t = masked_diff_var[idx]
                    diff_T2_t = np.array(diff_T2_t)
                    diff_var_t = np.array(diff_var_t)
                    coeff = np.polyfit(masked_diff_T2[idx], masked_diff_var[idx], deg=1)
                    a, b = coeff
                    if plotting_scatter_all:
                        fig, ax = plt.subplots()
                        ax.scatter(
                            masked_diff_T2[idx],
                            masked_diff_var[idx],
                            s=0.1,
                            c="red",
                        )
                        x_poly = np.linspace(
                            masked_diff_T2[idx].min(),
                            masked_diff_T2[idx].max(),
                        )
                        y_poly = np.polyval(coeff, x_poly)
                        ax.plot(
                            x_poly,
                            y_poly,
                            color="b",
                            lw=1.5,
                            linestyle="--",
                            label=f"y_all = {a:.2f} * x + {b:.2f}",
                        )
                        ax.legend()
                        ax.xaxis.grid(True, which="major")
                        ax.yaxis.grid(True, which="major")
                        ax.set_xlabel("T2 diff [°C]")
                        ax.set_ylabel(f"{name_vars[WRF_var]} diff")
                        plt.title(
                            "WRF _27km - 3km T2 and %s correlation at %s °C T_ref"
                            % (name_vars[WRF_var], T_ref)
                        )
                        figname = (
                            outfolder
                            + "WRF_27km_%s_corr_STD_%s_T_ref_%s_%s.png"
                            % (
                                WRF_var,
                                STD_TOPO,
                                T_ref,
                                time,
                            )
                        )
                        plt.savefig(figname)
                        plt.close()
                    coeff_all_T.append(a)
                except:
                    print("Not enough Data for T_ref=%s" % T_ref)
                    coeff_all_T.append(np.nan)
            df_coeff[name_vars[WRF_var]] = coeff_all_T

        if T_bin_flag:
            ax = df_coeff.plot(marker="o", linestyle="-", figsize=(10, 6), grid=True)
            ax.set_xlabel("T_ref")
            ax.set_ylabel("Coefficients [mmol CO2 m² s⁻¹ °C⁻¹]")
            ax.set_title("Coefficient Values for NEE, GPP, and RECO")
            figname = (
                outfolder
                + f"WRF_27km_coefficients_STD_{STD_TOPO}_{hour_start}-{hour_end}h_till_{end_date}.png"
            )
            plt.savefig(figname)
            plt.close()
