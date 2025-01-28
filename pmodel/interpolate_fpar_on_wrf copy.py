import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from pyproj import Proj, Transformer, CRS
import os
from datetime import datetime
import os
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from scipy.interpolate import griddata


def load_modis_files(modis_dir, closest_files):
    """Load MODIS datasets based on the closest files."""
    try:
        modis_before = xr.open_dataset(os.path.join(modis_dir, closest_files[0][0]))
        modis_after = xr.open_dataset(os.path.join(modis_dir, closest_files[1][0]))
        print(f"Loaded MODIS files: {closest_files[0][0]}, {closest_files[1][0]}")
        return modis_before, modis_after
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading MODIS files: {e}")


def georeference_modis_data(modis_before, modis_crs, wrf_crs):
    """Transform MODIS grid coordinates to WRF projection."""
    xdim = modis_before["xdim"].values
    ydim = modis_before["ydim"].values

    # Create a meshgrid for xdim and ydim
    xgrid, ygrid = np.meshgrid(xdim, ydim)

    # Create a transformer from MODIS CRS to WRF CRS
    transformer = Transformer.from_crs(modis_crs, wrf_crs)
    modis_lats, modis_lons = transformer.transform(xgrid, ygrid)

    return modis_lats.flatten(), modis_lons.flatten()


def interpolate_time(fpar_before, fpar_after, time_before, time_after, wrf_time_dt):
    """Perform linear interpolation in time for FPAR data."""
    # Mask invalid FPAR values
    fpar_before.values[fpar_before.values > 1] = np.nan
    fpar_after.values[fpar_after.values > 1] = np.nan

    # Calculate interpolation weights
    weight_before = (time_after - wrf_time_dt).total_seconds() / (
        time_after - time_before
    ).total_seconds()
    weight_after = 1 - weight_before

    # Time interpolation
    fpar_interpolated = fpar_before * weight_before + fpar_after * weight_after
    return fpar_interpolated.values.flatten()


def interpolate_to_wrf_grid(fpar_interpolated_time, modis_lats, modis_lons, wrf_ds):
    """Interpolate MODIS FPAR data to the WRF grid."""
    xlat = wrf_ds["XLAT"].values
    xlon = wrf_ds["XLONG"].values
    wrf_coords = np.column_stack([xlat.flatten(), xlon.flatten()])

    # Perform spatial interpolation
    fpar_on_wrf_grid = griddata(
        np.column_stack([modis_lats, modis_lons]),
        fpar_interpolated_time,
        wrf_coords,
        method="linear",
    )

    return fpar_on_wrf_grid.reshape(xlat.shape)


def save_to_netcdf(fpar_on_wrf_grid, output_path, wrf_path_dx_str, wrf_time_str):
    """Save interpolated FPAR data to a NetCDF file."""
    output_file = f"{output_path}/interpolated_fpar_{wrf_path_dx_str}_{wrf_time_str}.nc"
    xr.DataArray(fpar_on_wrf_grid, name="Fpar_500m").to_netcdf(
        output_file, format="NETCDF4_CLASSIC"
    )
    print(f"Interpolated FPAR saved to: {output_file}")


# File paths
# wrf_path = "/home/madse/Downloads/Fluxnet_Data/wrfout_d01_2012-07-01_12:00:00.nc"
# modis_dir = "/home/madse/Downloads/MODIS_data/split_timesteps"
month = 7
wrf_paths = [
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250105_193347_ALPS_9km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241229_112716_ALPS_27km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241227_183215_ALPS_54km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km",
]

for day in range(1, 32):
    for wrf_path_dx in wrf_paths:
        wrf_path = f"{wrf_path_dx}/wrfout_d01_2012-{month:02d}-{day:02d}_12:00:00"
        modis_dir = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/split_timesteps"
        output_path_base = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/fpar_interpol"

        # Step 1: Extract WRF time and convert to cftime.DatetimeJulian
        wrf_ds = xr.open_dataset(wrf_path)
        wrf_time_str = (
            wrf_ds["Times"].values.tobytes().decode("utf-8").replace("_", "T")
        )
        wrf_time_dt = np.datetime64(wrf_time_str).astype("datetime64[s]").astype(object)

        # List MODIS files and find closest dates
        modis_files = [
            f
            for f in os.listdir(modis_dir)
            if f.startswith("MODIS_fpar_") and f.endswith(".nc")
        ]
        modis_dates = [
            datetime.strptime(f.split("_")[2].split(".")[0], "%Y%m%d")
            for f in modis_files
        ]
        modis_files_dates = list(zip(modis_files, modis_dates))
        modis_files_dates.sort(key=lambda x: abs((x[1] - wrf_time_dt).total_seconds()))
        closest_files = modis_files_dates[:2]
        closest_files.sort(key=lambda x: x[1])  # Ensure chronological order

        # # Load closest MODIS files
        # modis_before = xr.open_dataset(os.path.join(modis_dir, closest_files[0][0]))
        # modis_after = xr.open_dataset(os.path.join(modis_dir, closest_files[1][0]))
        # print(f"Loaded MODIS files: {closest_files[0][0]}, {closest_files[1][0]}")

        # # Step 2: Georeference xdim and ydim
        # xdim = modis_before["xdim"].values
        # ydim = modis_before["ydim"].values

        # Extract projection information `grid_mapping`
        # proj_string = modis_before["Fpar_500m"].attrs.get("grid_mapping", None)
        try:
            # Load MODIS files
            modis_before, modis_after = load_modis_files(modis_dir, closest_files)

            # Define CRS
            modis_crs = CRS.from_proj4(
                "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"
            )
            wrf_crs = CRS.from_proj4(
                "+proj=merc +lat_ts=46.189 +lon_0=10.0 +datum=WGS84"
            )

            # Georeference MODIS data
            modis_lats, modis_lons = georeference_modis_data(
                modis_before, modis_crs, wrf_crs
            )

            # Time interpolation
            fpar_interpolated_time = interpolate_time(
                modis_before["Fpar_500m"],
                modis_after["Fpar_500m"],
                closest_files[0][1],
                closest_files[1][1],
                wrf_time_dt,
            )

            # Spatial interpolation to WRF grid
            fpar_on_wrf_grid = interpolate_to_wrf_grid(
                fpar_interpolated_time, modis_lats, modis_lons, wrf_ds
            )

            # Save to NetCDF
            wrf_path_dx_str = wrf_path_dx.split("_")[-1]
            save_to_netcdf(
                fpar_on_wrf_grid, output_path_base, wrf_path_dx_str, wrf_time_str
            )

        except Exception as e:
            print(f"An error occurred: {e}")
