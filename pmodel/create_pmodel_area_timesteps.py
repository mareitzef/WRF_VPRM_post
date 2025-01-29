import pandas as pd
import numpy as np
import xarray as xr
from pyrealm.pmodel import PModel, PModelEnvironment, SubdailyScaler, SubdailyPModel
import os


# Load WRF dataset
# wrf_path = "/home/madse/Downloads/Fluxnet_Data/wrfout_d01_2012-07-01_12:00:00.nc"  # Replace with your file path
wrf_paths = [
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250105_193347_ALPS_9km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241229_112716_ALPS_27km",
    # "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241227_183215_ALPS_54km",
]


# wrf_path = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km/"
modis_path = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/"

for wrf_path in wrf_paths:
    wrf_path_dx_str = wrf_path.split("_")[-1]
    # list all files in the wrf_path
    files = os.listdir(wrf_path)
    files = [f for f in files if f.startswith("wrfout_d01")]
    files.sort()
    timesteps = len(files)
    # get datetime from first file
    datetimestart = files[0].split("_")[2] + " " + files[0].split("_")[3]
    wrf_ds = xr.open_dataset(wrf_path + "/" + files[0])
    temp = wrf_ds["T2"].to_numpy()
    l, m, n = temp.shape
    fpar_modis_arr = np.zeros((timesteps, m, n))
    ppfd_arr = np.zeros((timesteps, m, n))
    tc_arr = np.zeros((timesteps, m, n))  # Store temperature time series
    co2_arr = np.zeros((timesteps, m, n))  # Store CO₂ time series
    patm_arr = np.zeros((timesteps, m, n))  # Store atmospheric pressure
    vpd_arr = np.zeros((timesteps, m, n))  # Store vapor pressure deficit
    t = 0
    for file in files[1:]:
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

        tc_arr[t, :, :] = temp[0, :, :]
        co2_arr[t, :, :] = co2[0, :, :]
        patm_arr[t, :, :] = patm[0, :, :]
        vpd_arr[t, :, :] = vpd[0, :, :]

        fpar_modis_arr[t, :, :] = fpar_modis[0, :, :]
        ppfd_arr[t, :, :] = ppfd[0, :, :]
        t += 1

    env_arr = PModelEnvironment(tc=tc_arr, co2=co2_arr, patm=patm_arr, vpd=vpd_arr)
    env_arr.summarize()

    # calculate GPP with acclimation
    datetimes = pd.date_range(
        start=datetimestart, periods=timesteps, freq="h"
    ).to_numpy()

    fsscaler = SubdailyScaler(datetimes)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(1, "h"),
    )

    subdailyC3 = SubdailyPModel(
        env=env_arr,
        fapar=fpar_modis_arr,
        ppfd=ppfd_arr,
        fs_scaler=fsscaler,
        alpha=1 / 15,
        allow_holdover=True,
    )

    t = 0
    for file in files[1:]:
        subdailyC3_gpp = subdailyC3.gpp[t, :, :] * gC_to_mumol
        # print(t, " ", np.nanmax(subdailyC3_gpp))
        # save data in netcdf
        date_time = file.split("_")[2] + "_" + file.split("_")[3]
        gpp_path_out = f"{modis_path}gpp_pmodel/gpp_pmodel_subdailyC3_{wrf_path_dx_str}_{date_time}.nc"
        xr.DataArray(subdailyC3_gpp, name="GPP_Pmodel").to_netcdf(
            gpp_path_out, format="NETCDF4_CLASSIC"
        )
        t += 1

    print(f"Saved GPP data to {gpp_path_out}")
