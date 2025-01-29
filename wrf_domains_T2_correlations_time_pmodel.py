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
T_bin_flag = True
plotting_scatter_all = False
start_date = "2012-07-01 01:00:00"
end_date = "2012-07-30 00:00:00"
T_bin_size = 0.5
hour_start = 5
hour_end = 18
STD_TOPOs = [100]
STD_TOPO_flags = ["gt"]  # "lt" lower than or "gt" greater than STD_TOPO
subdaily = "_subdailyC3"  # "_subdailyC3" or ""

wrf_paths = [
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241227_183215_ALPS_54km",
]
wrf_files = "wrfout_d01*"

outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

#######################################################################################
pmodel_path = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gpp_pmodel/"


WRF_vars = ["GPP_pmodel", "EBIO_GEE", "EBIO_RES", "NEE", "T2"]
units = [" [mmol m² s⁻¹]", " [mmol m² s⁻¹]", " [mmol m² s⁻¹]", " [mmol m² s⁻¹]", " [K]"]
name_vars = {
    "GPP_pmodel": "GPP P-Model",
    "EBIO_GEE": "WRF GPP",
    "EBIO_RES": "WRF RECO",
    "NEE": "WRF NEE",
    "T2": "WRF T2M",
}
WRF_factors = [1, -1 / 3600, 1 / 3600, 1 / 3600, 273.15]

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

timestamps = [extract_datetime_from_filename(f) for f in file_list]
time_index = pd.to_datetime(timestamps)


diff_54_3km_4D = []
T2_54km_4D = []

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
                WRF_factor,
            ) in zip(
                WRF_vars,
                units,
                WRF_factors,
            ):
                # WRF
                i = 0
                # Loop through the files for the timestep
                # for nc_f1 in file_list_27km:
                nc_fid54km = nc.Dataset(os.path.join(wrf_paths[1], wrf_file), "r")
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
                    WRF_var_54km = nc_fid54km.variables[WRF_var][0, :, :] - WRF_factor
                elif WRF_var == "NEE":
                    WRF_var_3km = (
                        nc_fid3km.variables["EBIO_GEE"][0, 0, :, :]
                        + nc_fid3km.variables["EBIO_RES"][0, 0, :, :]
                    ) * WRF_factor
                    WRF_var_54km = (
                        nc_fid54km.variables["EBIO_GEE"][0, 0, :, :]
                        + nc_fid54km.variables["EBIO_RES"][0, 0, :, :]
                    ) * WRF_factor
                elif WRF_var == "GPP_pmodel":
                    # get pmodel gpp
                    time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
                    gpp_pmodel_3km = xr.open_dataset(
                        f"{pmodel_path}gpp_pmodel{subdaily}_3km_{time_str}.nc"
                    )
                    gpp_pmodel_54km = xr.open_dataset(
                        f"{pmodel_path}gpp_pmodel{subdaily}_54km_{time_str}.nc"
                    )
                    WRF_var_3km = gpp_pmodel_3km["GPP_Pmodel"].to_numpy()
                    WRF_var_54km = gpp_pmodel_54km["GPP_Pmodel"].to_numpy()
                else:
                    WRF_var_3km = nc_fid3km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    WRF_var_54km = (
                        nc_fid54km.variables[WRF_var][0, 0, :, :] * WRF_factor
                    )

                lats_54km = nc_fid54km.variables["XLAT"][0, :, :]
                lons_54km = nc_fid54km.variables["XLONG"][0, :, :]
                landmask_4 = nc_fid54km.variables["LANDMASK"][0, :, :]
                hgt_54km = nc_fid54km.variables["HGT"][0, :, :]
                stdh_topo_54km = nc_fid54km.variables["VAR"][0, :, :]
                WRF_var_54km[landmask_4 == 0] = np.nan
                proj_WRF_var_54km = proj_on_finer_WRF_grid(
                    lats_54km,
                    lons_54km,
                    WRF_var_54km,
                    lats_fine,
                    lons_fine,
                    WRF_var_3km,
                )
                proj_hgt_54km = proj_on_finer_WRF_grid(
                    lats_54km, lons_54km, hgt_54km, lats_fine, lons_fine, WRF_var_3km
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

                WRF_var_diff_54_3km_2D = np.where(
                    mask, proj_WRF_var_54km - WRF_var_3km, np.nan
                )
                diff_54_3km_4D.append(
                    {"time": time.hour, WRF_var: WRF_var_diff_54_3km_2D}
                )
                if WRF_var == "T2":
                    WRF_T2_2d = np.where(mask, proj_WRF_var_54km, np.nan)
                    T2_54km_4D.append({"time": time.hour, "T2": WRF_T2_2d})

        # Convert lists to 3D arrays
        var_data = {}
        hours = [entry["time"] for entry in diff_54_3km_4D]

        for entry in diff_54_3km_4D:
            for var_name, data in entry.items():
                if var_name != "time":  # Skip the 'time' key
                    if var_name not in var_data:
                        var_data[var_name] = []
                    var_data[var_name].append(data)
        diff_54_3km_3D = {
            var: np.stack(data_list, axis=0) for var, data_list in var_data.items()
        }
        var_data = {}
        for entry in T2_54km_4D:
            for var_name, data in entry.items():
                if var_name != "time":  # Skip the 'time' key
                    if var_name not in var_data:
                        var_data[var_name] = []
                    var_data[var_name].append(data)
        T2_54km_3D = {
            var: np.stack(data_list, axis=0) for var, data_list in var_data.items()
        }

        T_ref_values = np.arange(
            round(np.nanmin(T2_54km_3D["T2"])) + 1,
            round(np.nanmax(T2_54km_3D["T2"])) - 12,
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
                    temp_mask = (T2_54km_3D["T2"] >= T_ref) & (
                        T2_54km_3D["T2"] <= T_ref + T_bin_size
                    )
                    masked_diff_T2 = diff_54_3km_3D["T2"][temp_mask]
                    masked_diff_var = diff_54_3km_3D[WRF_var][temp_mask]

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
                        formatted_T_ref = "{:.2f}".format(T_ref).replace(".", "_")[:4]

                        plt.title(
                            f"WRF 54km - 3km T2 and {name_vars[WRF_var]} correlation at {formatted_T_ref} °C T_ref"
                        )
                        figname = (
                            outfolder
                            + f"WRF_T2_{WRF_var}_corr{subdaily}_STD_{STD_TOPO}_T_ref_{formatted_T_ref}_{time}.png"
                        )
                        plt.savefig(figname)
                        plt.close()
                    coeff_all_T.append(a)
                except:
                    print("Not enough Data for T_ref=%s" % T_ref)
                    coeff_all_T.append(np.nan)
            df_coeff[name_vars[WRF_var]] = coeff_all_T

        if T_bin_flag:
            ax = df_coeff.plot(linestyle="-", figsize=(10, 6), grid=True)
            ax.set_xlabel("T_ref")
            ax.set_ylabel("Coefficients [mmol CO2 m² s⁻¹ °C⁻¹]")
            ax.set_title("Coefficient Values for NEE, GPP, and RECO")
            figname = (
                outfolder
                + f"WRF_T_ref_coefficients{subdaily}_STD_{STD_TOPO}_{hour_start}-{hour_end}h_till_{end_date}.png"
            )
            plt.savefig(figname)
            plt.close()
