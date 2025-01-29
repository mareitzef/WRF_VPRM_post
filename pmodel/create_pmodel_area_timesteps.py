import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import xarray as xr
from pyrealm.pmodel import PModel, PModelEnvironment
from pyrealm.splash.splash import SplashModel
from pyrealm.core.calendar import Calendar
import pyrealm.pmodel
from pyrealm.core.pressure import calc_patm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import cftime
from scipy.ndimage import generic_filter
import os


# Load WRF dataset
# wrf_path = "/home/madse/Downloads/Fluxnet_Data/wrfout_d01_2012-07-01_12:00:00.nc"  # Replace with your file path
wrf_paths = [
    # "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250105_193347_ALPS_9km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241229_112716_ALPS_27km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241227_183215_ALPS_54km",
]


# wrf_path = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km/"
modis_path = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/"

for wrf_path in wrf_paths:
    wrf_path_dx_str = wrf_path.split("_")[-1]
    # list all files in the wrf_path
    files = os.listdir(wrf_path)
    files = [f for f in files if f.startswith("wrfout")]
    files.sort()

    for file in files:
        day = int(file.split("_")[2].split("-")[2])
        if day > 30:
            continue
        wrf_ds = xr.open_dataset(wrf_path + "/" + file)

        # Load variables from WRF dataset
        temp = wrf_ds["T2"].to_numpy() - 273.15  # Convert to Celsius
        patm = wrf_ds["PSFC"].to_numpy()  # Pa
        co2 = wrf_ds["CO2_BIO"].isel(bottom_top=0).to_numpy()  # ppmv
        qvapor = (
            wrf_ds["QVAPOR"].isel(bottom_top=0).to_numpy()
        )  # Water vapor mixing ratio (kg/kg) at the surface level
        psfc = wrf_ds["PSFC"].isel(Time=0).to_numpy()  # Surface pressure (Pa)
        t2 = wrf_ds["T2"].isel(Time=0).to_numpy()  # Temperature at 2m (K)

        # Calculate actual vapor pressure (ea) in kPa
        ea = (qvapor * psfc) / (0.622 + qvapor)  # Pa
        # Calculate saturation vapor pressure (es) in kPa
        es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3)) * 1000  # convert to Pa
        # Calculate VPD
        vpd = np.maximum(0, es - ea)  # Force non-negative VPD

        # Load required variables from WRF dataset
        vegfra = wrf_ds["VEGFRA"].to_numpy()  # Vegetation fraction (0 to 1)
        albedo = wrf_ds["ALBEDO"].to_numpy()  # Albedo (0 to 1)
        swdown = wrf_ds["SWDOWN"].to_numpy()  # Downward shortwave radiation (W/m^2)
        ppfd = (
            swdown * 2.30785
        )  # Shortwave radiation (W/m²) × 0.505 -> PAR (W/m²) × 4.57 -> 2.3*x ~ PPFD (umol/m²/s)
        xlat = wrf_ds["XLAT"].to_numpy()  # Latitude (degrees)
        xlon = wrf_ds["XLONG"].to_numpy()  # Longitude (degrees)
        xlat = xlat[0, :, :]
        xlon = xlon[0, :, :]
        fapar_wrf = (1 - albedo) * (vegfra / 100)  # Calculate fAPAR

        # get modis fpar
        modis_path_in = f"{modis_path}fpar_interpol/interpolated_fpar_{wrf_path_dx_str}_2012-07-{day:02d}T12:00:00.nc"
        modis_ds = xr.open_dataset(modis_path_in)
        fpar_modis = modis_ds["Fpar_500m"].to_numpy()  # fAPAR from MODIS
        # where modis is nan use values from fapar_wrf
        fpar_modis = np.where(
            np.isnan(fpar_modis), fapar_wrf, fpar_modis
        )  # TODO: how is this handled in literature? There is no fPAR Data around cities, so these areas could also be masked out, but I would habe to modify the landcover maps...

        # Ensure proper dimensions and clean invalid data
        temp[temp < -25] = np.nan  # Mask temperatures below -25°C
        vpd = np.clip(vpd, 0, np.inf)  # Force VPD ≥ 0

        # Run P-model environment
        env = PModelEnvironment(tc=temp, co2=co2, patm=patm, vpd=vpd)
        env.summarize()

        # Estimate productivity
        model = PModel(env)
        model.estimate_productivity(fpar_modis, ppfd)
        gC_to_mumol = 0.0833  # 1 µg C m⁻² s⁻¹ × (1 µmol C / 12.01 µg C) × (1 µmol CO₂ / 1 µmol C) = 0.0833 µmol CO₂ m⁻² s⁻¹
        data = model.gpp[0, :, :] * gC_to_mumol
        # save data in netcdf
        date_time = file.split("_")[2] + "_" + file.split("_")[3]
        modis_path_out = (
            f"{modis_path}gpp_pmodel/gpp_pmodel_{wrf_path_dx_str}_{date_time}.nc"
        )
        xr.DataArray(data, name="GPP_Pmodel").to_netcdf(
            modis_path_out, format="NETCDF4_CLASSIC"
        )
        print(f"Saved GPP data to {modis_path_out}")
