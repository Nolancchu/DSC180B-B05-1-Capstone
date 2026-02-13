"""
Convert regional_rise NetCDF SLR files to pyCIAM-compatible zarr format.

Reads slr_ssp126.nc, slr_ssp245.nc, slr_ssp370.nc, slr_ssp585.nc from the
regional_rise folder, flattens (lat, lon) to site_id, and writes a single
zarr store with dimensions (site_id, scenario, quantile, year) and variables
lsl_msl05, lsl_ncc_msl05, lat, lon. pyCIAM will map SLIIDERS segment
coordinates to the nearest site_id via spherical nearest neighbor.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Default paths relative to this file
REGIONAL_RISE_DIR = Path(__file__).resolve().parent
SSP_FILES = {
    "SSP126": "slr_ssp126.nc",
    "SSP245": "slr_ssp245.nc",
    "SSP370": "slr_ssp370.nc",
    "SSP585": "slr_ssp585.nc",
}
SLR_VAR = "slr"
TIME_DIMS = ("time", "years", "year")  # try in order
LAT_DIM = "lat"
LON_DIM = "lon"
SLR_0_YEAR = 2005
QUANTILE_DEFAULT = 0.5


def _time_dim_name(slr: xr.DataArray) -> str:
    for d in TIME_DIMS:
        if d in slr.dims:
            return d
    raise ValueError(f"No time-like dimension in slr. Dims: {slr.dims}")


def _time_to_year(da: xr.DataArray, time_dim: str) -> np.ndarray:
    """Convert time coordinate to integer years."""
    t = da[time_dim]
    if np.issubdtype(t.dtype, np.floating) or np.issubdtype(t.dtype, np.integer):
        return np.asarray(t, dtype=int)
    # datetime-like
    return pd.to_datetime(t.values).year.values


def load_one_nc(path: Path, scenario: str) -> xr.Dataset:
    """Load one NetCDF and return dataset with (site_id, year) and scenario."""
    ds = xr.open_dataset(path)
    if SLR_VAR not in ds:
        raise KeyError(f"Variable '{SLR_VAR}' not in {path}. Available: {list(ds.data_vars)}")
    slr = ds[SLR_VAR]
    time_dim = _time_dim_name(slr)
    for d in (LAT_DIM, LON_DIM):
        if d not in slr.dims:
            raise ValueError(f"Expected dimension '{d}' in slr. Dims: {slr.dims}")
    years = _time_to_year(slr, time_dim)
    # Flatten lat, lon to site_id
    slr_stacked = slr.stack(site_id=(LAT_DIM, LON_DIM))
    # Relative to SLR_0_YEAR
    if SLR_0_YEAR in years:
        ref = slr_stacked.sel({time_dim: SLR_0_YEAR}).rename({time_dim: "year"})
        ref["year"] = SLR_0_YEAR
    else:
        ref = slr_stacked.isel({time_dim: 0}).rename({time_dim: "year"})
        ref["year"] = years[0]
    slr_rel = slr_stacked.rename({time_dim: "year"})
    slr_rel["year"] = years
    slr_rel = slr_rel - ref.reindex(year=slr_rel.year.values, fill_value=0)
    # Use integer site_id so we can add lat/lon as separate coords (MultiIndex would block that)
    n_sites = slr_rel.sizes["site_id"]
    n_years = len(years)
    lat = slr_stacked[LAT_DIM].values
    lon = slr_stacked[LON_DIM].values
    # Ensure array is (site_id, year); NetCDF may have (time, lat, lon) -> stacked (year, site_id)
    vals = slr_rel.values
    if slr_rel.dims[0] != "site_id":
        vals = vals.T
    assert vals.shape == (n_sites, n_years), f"expected ({n_sites}, {n_years}), got {vals.shape}"
    slr_plain = xr.DataArray(
        vals,
        dims=("site_id", "year"),
        coords={"site_id": np.arange(n_sites), "year": years},
    )
    out = xr.Dataset(
        {"lsl_msl05": slr_plain},
        coords={"lat": ("site_id", lat), "lon": ("site_id", lon)},
    )
    out["scenario"] = scenario
    ds.close()
    return out


def build_slr_zarr(
    input_dir: Path = REGIONAL_RISE_DIR,
    output_path: Path | None = None,
    quantiles: list[float] | None = None,
    ncc_from_scenario: str | None = "SSP126",
    slr_0_year: int = SLR_0_YEAR,
) -> xr.Dataset:
    """
    Build a single pyCIAM-format SLR zarr from regional_rise NetCDFs.

    Parameters
    ----------
    input_dir : Path
        Directory containing slr_ssp*.nc files.
    output_path : Path, optional
        If set, write the dataset to this path as zarr.
    quantiles : list of float, optional
        Quantile dimension values. Default [0.5].
    ncc_from_scenario : str or None
        If set, use this scenario's trajectory as lsl_ncc_msl05 (no climate change).
        If None, lsl_ncc_msl05 is zeros.
    slr_0_year : int
        Reference year for SLR (values set to 0 in this year).

    Returns
    -------
    xr.Dataset
        Dataset with dims (site_id, scenario, quantile, year) and
        data vars lsl_msl05, lsl_ncc_msl05, plus coords lat, lon.
    """
    if quantiles is None:
        quantiles = [QUANTILE_DEFAULT]
    all_ds = []
    for scenario, fname in SSP_FILES.items():
        path = input_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Expected {path}")
        all_ds.append(load_one_nc(path, scenario))
    # Concatenate over scenario (assumes all NetCDFs share the same lat/lon grid)
    combined = xr.concat(all_ds, dim="scenario")
    # Align years across scenarios (use union and reindex)
    all_years = np.unique(np.concatenate([d.year.values for d in all_ds]))
    all_years = np.sort(all_years)
    combined = combined.reindex(year=all_years)
    # Ensure reference year is 0
    if slr_0_year in combined.year.values:
        ref = combined.sel(year=slr_0_year).lsl_msl05
        combined["lsl_msl05"] = combined.lsl_msl05 - ref
    # Add quantile dimension (single value; pyCIAM can use one quantile)
    combined = combined.expand_dims(quantile=quantiles)
    # No-climate-change: (site_id, quantile, year) only - pyCIAM adds scenario via expand_dims
    if ncc_from_scenario is not None and ncc_from_scenario in combined.scenario.values:
        ncc_da = combined.sel(scenario=ncc_from_scenario).lsl_msl05
        if "scenario" in ncc_da.dims:
            ncc_da = ncc_da.isel(scenario=0)
        ncc_da = ncc_da.drop_vars("scenario", errors="ignore")
    else:
        ncc_da = combined.lsl_msl05.isel(scenario=0) * 0
        ncc_da = ncc_da.drop_vars("scenario", errors="ignore")
    combined["lsl_ncc_msl05"] = ncc_da
    # site_id as integer index (pyCIAM get_nearest_slrs expects index alignment)
    n_sites = combined.dims["site_id"]
    combined["site_id"] = np.arange(n_sites)
    # Longitude convention: use -180..180 if needed for pyCIAM
    combined["lon"] = combined.lon.where(combined.lon <= 180, combined.lon - 360)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write main dataset without lsl_ncc_msl05 first, then append NCC so it has no scenario dim in store
        main_ds = combined.drop_vars("lsl_ncc_msl05")
        main_ds.chunk({"site_id": 100}).to_zarr(str(output_path), mode="w")
        ncc_da.chunk({"site_id": 100}).to_dataset(name="lsl_ncc_msl05").to_zarr(
            str(output_path), mode="a"
        )
    return combined


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert regional rise NetCDFs to pyCIAM zarr")
    parser.add_argument("--input-dir", type=Path, default=REGIONAL_RISE_DIR)
    parser.add_argument("--output", type=Path, default=REGIONAL_RISE_DIR / "slr_regional_rise.zarr")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.5])
    parser.add_argument("--ncc-from", type=str, default="SSP126")
    args = parser.parse_args()
    build_slr_zarr(
        input_dir=args.input_dir,
        output_path=args.output,
        quantiles=args.quantiles,
        ncc_from_scenario=args.ncc_from,
    )
    print(f"Wrote {args.output}")
