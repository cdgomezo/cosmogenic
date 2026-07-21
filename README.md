# cosmo14C

Monthly, global, 1°×1° cosmogenic ¹⁴C isoflux fields for the TM5 atmospheric transport model.

**Period:** 2014–2025 | **Resolution:** 1°×1°, 34 vertical layers | **Output:** NetCDF4, CF-compliant

---

## Table of Contents

1. [Scientific Overview](#1-scientific-overview)
2. [Repository Structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Configuration Reference](#5-configuration-reference)
6. [Module Reference](#6-module-reference)
7. [Data Files Reference](#7-data-files-reference)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Output Format](#9-output-format)
10. [Validation](#10-validation)
11. [Testing](#11-testing)
12. [Updating Input Data](#12-updating-input-data)
13. [Physics Notes](#13-physics-notes)
14. [Troubleshooting](#14-troubleshooting)
15. [References](#15-references)

---

## 1. Scientific Overview

Cosmogenic ¹⁴C is produced continuously in the upper atmosphere by nuclear spallation reactions between galactic cosmic rays (GCRs) and atmospheric nitrogen and oxygen. Its production rate varies with:

- **Solar activity** — higher solar modulation potential φ deflects lower-energy GCRs, reducing production.
- **Geomagnetic field** — the dipole cutoff rigidity *P*c shields equatorial regions; polar regions are unshielded.
- **Altitude** — most production occurs in the stratosphere and upper troposphere, with a vertical profile described by Masarik & Beer (2009).

This package computes monthly **isoflux** fields [TgC ‰ yr⁻¹] — the product of the ¹⁴C production rate and the isotopic disequilibrium — on a 1°×1° global grid distributed over 34 TM5 pressure-hybrid layers. These fields are designed to be used as a source term in TM5 atmospheric transport simulations.

### Pipeline at a glance

```
Solar φ(t) [MV]               Geomagnetic Pc(lat,lon) [GV]
     │ (Oulu NM + IGRF)              │ (IGRF dipole, updated yearly)
     ▼                               ▼
  local_Q(φ, Pc)  ─── lookup ───► columnar ¹⁴C production
  [atoms cm⁻² s⁻¹]                  q(lat,lon,month)
                                      │
                      Masarik & Beer  │  vertical shape
                      (2009) profile  ▼
                               3D isoflux field
                               (12, 34, 180, 360)
                                      │
                               NetCDF4 output
                               cosmo14C_{year}.nc
```

### Key numbers

| Quantity | Value | Source |
|---|---|---|
| Global Q at φ = 650 MV | 1.66 atoms cm⁻² s⁻¹ | Kovaltsov et al. (2012) |
| Global Q at φ = 650 MV | 1.64 atoms cm⁻² s⁻¹ | Miller et al. (2025) |
| Dipole moment *M* | 7.8 × 10²² A m² | Current IGRF |
| Alpha/proton ratio | 0.30 | AMS/PAMELA (Kovaltsov 2012) |
| Solar minimum φ (2020) | ~270–320 MV | Oulu/Usoskin 2017 |
| Solar maximum φ (2025) | ~950–1160 MV | Oulu/Usoskin 2017 |

---

## 2. Repository Structure

```
cosmo14C/                       # Main Python package
├── __init__.py
├── config.py                   # Physical constants
├── gcr_spectrum.py             # LIS formula + force-field modulation
├── yield_function.py           # Kovaltsov (2012) ¹⁴C yield Y(E)
├── production.py               # rigidity_cutoff, global_Q, local_Q
├── geomagnetic.py              # IGRF dipole coefficients, geomag lat grid
├── solar_modulation.py         # φ(t) from Oulu NM + regression
├── grid.py                     # 1°×1° lat/lon, Pc grid, cell areas
├── vertical_profile.py         # TM5 hybrid layers + M&B shape weights
├── isoflux.py                  # Unit conversion → TgC ‰ yr⁻¹
├── build_lookup.py             # Q(φ, Pc) lookup table builder/loader
├── output.py                   # NetCDF4 writer (CF-compliant)
└── run.py                      # Pipeline orchestration + CLI entry point

tests/                          # Pytest test suite (114 tests)
├── test_config.py
├── test_gcr_spectrum.py
├── test_yield_function.py
├── test_production.py
├── test_production_local.py
├── test_geomagnetic.py
├── test_solar_modulation.py
├── test_grid.py
├── test_vertical_profile.py
├── test_isoflux.py
├── test_build_lookup.py
├── test_output.py
└── test_integration.py         # Full pipeline smoke test

config/
└── run.yaml                    # Runtime configuration (edit this to change runs)

data/
├── solar_phi/
│   ├── oulu_phi.csv            # Monthly φ [MV], 2014–2025 (Usoskin 2017)
│   └── oulu_nm_rates.csv       # Oulu NM count rates [cpm], 2010–2025 (NMDB)
├── masarik_beer_2009/
│   └── table1.csv              # M&B vertical production profile (Pc × depth)
├── tm5/
│   └── tropo34_coeffs.csv      # TM5 34-layer hybrid σ-pressure coefficients
└── lookup/
    ├── local_Q_table.npz       # Precomputed Q(φ, Pc) lookup table (auto-built)
    └── local_Q_table.npz.hash  # Config hash for cache invalidation (auto-built)

output/                         # Pipeline writes cosmo14C_{year}.nc here

validate_Q.py                   # Standalone acceptance test (global Q at φ=650 MV)
requirements.txt
```

---

## 3. Installation

### Prerequisites

- Python 3.10+ (tested on 3.14)
- conda (recommended) or pip

### conda environment

```bash
# Create environment (replace "carlos" with your preferred name)
conda create -n carlos python=3.11
conda activate carlos

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
numpy>=2.0
scipy>=1.10
pandas>=1.5
xarray>=0.21
netCDF4>=1.6
matplotlib>=3.6
requests>=2.28
ppigrf>=1.0
pyyaml>=6.0
```

> **Note on NumPy:** The code uses `np.trapezoid` (available from NumPy 2.0). The requirement `numpy>=2.0` is intentional and must not be relaxed.

> **Note on ppigrf:** The package accesses ppigrf's internal module (`ppigrf.ppigrf`) to read IGRF Gauss coefficients directly. This is tested against ppigrf >= 1.0. If ppigrf changes its internal API in a future release, see [Troubleshooting §14.1](#141-ppigrf-internal-api-changes).

### Verify installation

```bash
# From the Cosmogenic/ directory:
python -c "import cosmo14C.config; print('OK')"
python validate_Q.py
```

Expected output from `validate_Q.py`:

```
Computing global Q at phi = 650 MV ...

  Computed Q           : 1.6xxx atoms cm⁻² s⁻¹
  Kovaltsov (2012)     : 1.6600  (+x.x%)
  Miller et al. (2025) : 1.6400  (+x.x%)

  PASS — within 5% of Kovaltsov target
```

---

## 4. Quick Start

```bash
# From the Cosmogenic/ directory, with conda env active:
python -m cosmo14C.run config/run.yaml
```

This will:
1. Build the `local_Q` lookup table (~30 s, one-time, cached).
2. Loop over years 2014–2025, processing 12 months per year.
3. Write `output/cosmo14C_{year}.nc` for each year.

Expected console output per year:

```
--- Processing 2020 ---
  Using cached lookup table: data/lookup/local_Q_table.npz
  Computing vertical shape weights...
  Month 01: phi=294 MV, mean Q=1.204 atoms cm^-2 s^-1
  Month 02: phi=279 MV, mean Q=1.219 atoms cm^-2 s^-1
  ...
  Written: output//cosmo14C_2020.nc
```

Total runtime: ~30–60 minutes for 2014–2025 on a modern laptop.

---

## 5. Configuration Reference

All runtime parameters live in `config/run.yaml`. Edit this file to change the run period, grid, or physics assumptions.

```yaml
period:
  start: 2014          # First year to process (inclusive)
  end:   2025          # Last  year to process (inclusive)
```

```yaml
grid:
  lat_res_deg: 1.0     # Latitude  resolution [degrees] — do not change for TM5 1°×1°
  lon_res_deg: 1.0     # Longitude resolution [degrees]
```

```yaml
vertical:
  grid: tropo34                           # TM5 vertical grid label (informational)
  tm5_coeffs: data/tm5/tropo34_coeffs.csv # Hybrid σ-p coefficients (a_Pa, b per layer)
  masarik_beer_table: data/masarik_beer_2009/table1.csv  # M&B (2009) Table 1
  p_surf_hPa: 1013.25                     # Reference surface pressure [hPa]
```

```yaml
geomagnetic:
  M_1e22: 7.8          # Dipole moment [10²² A m²] — used in Pc dipole formula
  update_frequency: yearly  # IGRF coefficients recalculated once per year
```

```yaml
solar_modulation:
  oulu_phi_file: data/solar_phi/oulu_phi.csv      # Primary φ series
  nm_rate_file:  data/solar_phi/oulu_nm_rates.csv # NM count rates for regression
  regression_coeffs:
    a: -0.5740   # φ = a × rate + b  [MV per cpm]
    b: 4168.0    # calibrated over 2010–2021, R²=0.998, RMSE=5.9 MV
  regression_start_year: 2010   # Informational (coefficients are pre-computed)
```

```yaml
production:
  lookup_table: data/lookup/local_Q_table.npz
  phi_grid_MV:
    min: 10       # Minimum φ in table [MV]
    max: 2000     # Maximum φ in table [MV]
    n:   50       # Number of φ grid points
    scale: log    # Logarithmic spacing (dense at low φ where curvature is high)
  Pc_grid_GV:
    min: 0        # Minimum Pc [GV] (polar region)
    max: 20       # Maximum Pc [GV] (equatorial maximum ≈ 1.9 × M)
    n:   100      # Number of Pc grid points
    scale: linear # Linear spacing
```

```yaml
output:
  directory: output/   # Where to write cosmo14C_{year}.nc files
  compress: true       # zlib compression (complevel=4); set false to skip
```

### When to rebuild the lookup table

The lookup table (`data/lookup/local_Q_table.npz`) is automatically rebuilt if any parameter under `production:` changes. If you change `phi_grid_MV` or `Pc_grid_GV` settings (e.g., finer grids), delete `data/lookup/local_Q_table.npz.hash` to force a rebuild, or the change will be detected automatically on the next run.

---

## 6. Module Reference

### `cosmo14C/config.py` — Physical constants

Single source of truth for all constants. Import from here rather than hardcoding values anywhere.

| Name | Value | Unit | Description |
|---|---|---|---|
| `E_rest` | 0.938 | GeV/nuc | Proton rest mass |
| `R_std` | 1.176 × 10⁻¹² | — | ¹⁴C:C ratio of NBS oxalic acid standard |
| `N_A` | 6.022 × 10²³ | mol⁻¹ | Avogadro constant |
| `M_C` | 12.011 | g/mol | Molar mass of carbon |
| `R_earth` | 6.371 × 10⁸ | cm | Mean Earth radius |
| `s_per_yr` | 31,557,600 | s/yr | Seconds per year (365.25 days) |
| `E_TABLE_MIN` | 0.1 | GeV/nuc | Lowest Kovaltsov yield table node |
| `ALPHA_RATIO` | 0.30 | — | Heavier nuclei flux relative to protons (per nucleon) |
| `Q_KOVALTSOV_PHI650` | 1.66 | atoms cm⁻² s⁻¹ | Benchmark global Q at φ=650 MV |
| `Q_MILLER_PREFERRED` | 1.64 | atoms cm⁻² s⁻¹ | Miller et al. (2025) preferred value |

---

### `cosmo14C/gcr_spectrum.py` — GCR spectrum

**`J_LIS_proton(E)`**
Proton Local Interstellar Spectrum [particles/(m² sr s GeV/nuc)] using the Burger et al. (2000)/Usoskin et al. (2005) formula:

$$J_\text{LIS}(E) = \frac{1.9 \times 10^4 \, P^{-2.78}}{1 + 0.4866 \, P^{-2.51}}$$

where *P* = rigidity [GV].

**`J_modulated(E, phi_MV, species='p')`**
Force-field modulated GCR spectrum at 1 AU. Species `'p'` (proton) or `'a'` (alpha). Returns [particles/(m² sr s GeV/nuc)].

The force-field approximation:

$$J_\text{mod}(E, \phi) = J_\text{LIS}(E + \Phi) \cdot \frac{E(E + 2E_r)}{(E + \Phi)(E + \Phi + 2E_r)}$$

where $\Phi = \phi_\text{MV} \times 10^{-3} \times (Z/A)$ [GeV/nuc].

---

### `cosmo14C/yield_function.py` — ¹⁴C yield function

Implements Kovaltsov et al. (2012) Table 1 by log-log interpolation.

**`Y_proton(E)`** — Proton yield [atoms per incident nucleon], omnidirectional (factor π included). Defined at 13 nodes from 0.1 to 999 GeV/nuc.

**`Y_alpha(E)`** — Alpha particle yield [atoms per incident nucleon], omnidirectional. Alpha particles produce more ¹⁴C per nucleon than protons at low energies, converging near 19 GeV/nuc.

Both functions extrapolate beyond the table range using the boundary slopes.

---

### `cosmo14C/production.py` — Core physics

**`rigidity_cutoff(geomag_lat_deg, M_1e22=7.8)`**
Vertical cutoff rigidity [GV] using the dipole approximation:

$$P_c = 1.9 \, M \, \cos^4\!\lambda_G$$

where *M* is in units of 10²² A m² and *λ*_G is the geomagnetic latitude. Returns 0 at the poles.

**`cutoff_energy(Pc_GV, species='p')`**
Minimum kinetic energy per nucleon to penetrate cutoff rigidity *P*c:

$$E_{ic} = \sqrt{E_r^2 + \left(P_c \cdot \frac{Z}{A}\right)^2} - E_r$$

**`global_Q(phi_MV, M_1e22=7.8)`**
Global columnar ¹⁴C production [atoms cm⁻² s⁻¹] using the Kovaltsov accessible-fraction formula:

$$Q = \int_{E_\min}^{E_\max} Y_p(E)\, J_p(E,\phi)\,[1 - f_p(E)]\, dE + \alpha_r \int_{E_\min}^{E_\max} Y_\alpha(E)\, J_\alpha(E,\phi)\,[1 - f_\alpha(E)]\, dE$$

where *f*(*E*) is the fraction of Earth's surface shielded at rigidity *P*(*E*) under a dipole field. The integration uses `scipy.integrate.quad` with kink points at energies where *f* changes slope.

> **Validation target:** `global_Q(650)` must be within 5% of 1.66 atoms cm⁻² s⁻¹ (see `validate_Q.py`).

**`local_Q(phi_MV, Pc_GV)`**
Local columnar ¹⁴C production [atoms cm⁻² s⁻¹] above a hard cutoff rigidity *P*c (appropriate for a specific geographic location). Integrates from the cutoff energy upward. This is the function used in the operational pipeline (via the lookup table).

---

### `cosmo14C/geomagnetic.py` — Geomagnetic latitude grid

**`get_geomag_lat_grid(year, lat_arr, lon_arr)`**
Returns geomagnetic latitude [degrees] at every grid cell, shape (nlat, nlon).

Internally:
1. Calls `ppigrf.ppigrf.read_shc()` to load the IGRF Gauss coefficient file.
2. Interpolates g₁₀, g₁₁, h₁₁ at the requested year using pandas time-series interpolation.
3. Computes the dipole pole position (geographic lat, lon).
4. Applies the spherical law of cosines to every (lat, lon) cell.

The IGRF is updated every 5 years; ppigrf handles extrapolation outside the model range via linear prediction. Coefficients are recalculated once per year in the pipeline.

---

### `cosmo14C/solar_modulation.py` — Solar modulation potential φ(t)

**`get_phi_series(year_start, year_end, config)`**
Returns `{(year, month): phi_MV}` for every month in the requested period.

**Priority order:**
1. **Published φ values** from `oulu_phi.csv` (Usoskin et al. 2017, Burger 2000 LIS). Used when available.
2. **NM regression** from `oulu_nm_rates.csv` using φ = a × rate + b. Used when published φ is missing.
3. If neither source covers a month, a `UserWarning` is emitted listing the missing months.

**Safeguards:**
- Non-positive regression results are clamped to 10 MV with a `UserWarning`.
- Regression use triggers a `UserWarning` noting the extra uncertainty (~20–30 MV).
- Year range inversion (`year_start > year_end`) raises `ValueError`.

---

### `cosmo14C/grid.py` — Geographic grid

**`make_lat_lon_arrays(lat_res_deg, lon_res_deg)`**
Returns cell-centre arrays:
- `lat`: shape (nlat,), range [−89.5°, +89.5°] for 1° resolution
- `lon`: shape (nlon,), range [−179.5°, +179.5°] for 1° resolution

**`make_Pc_grid(geomag_lat_2d, M_1e22=7.8)`**
Calls `rigidity_cutoff` for every cell. Returns *P*c [GV], shape (nlat, nlon).

**`make_cell_area(lat_arr, lat_res_deg, lon_res_deg)`**
Returns cell area [cm²], shape (nlat,):

$$A_\text{cell}(\lambda) = R_\oplus^2 \cos\lambda \, \Delta\lambda_\text{rad} \, \Delta\varphi_\text{rad}$$

---

### `cosmo14C/vertical_profile.py` — Vertical distribution

**`load_tm5_coeffs(path)`**
Reads TM5 hybrid σ-pressure CSV (columns: `half_level`, `a_Pa`, `b`). Returns arrays `(a_arr, b_arr)` sorted surface-first (descending half-level index). Layer pressure is *p* = *a* + *b* × *p*_surf.

**`get_shape_weights(Pc_GV, a_arr, b_arr, mb_file, p_surf_hPa=1013.25)`**
Returns normalized shape weights summing to 1.0, shape (n_layers,).

Steps:
1. Convert hybrid σ-pressure half-levels to atmospheric depth [g/cm²].
2. Load Masarik & Beer (2009) Table 1: production vs. depth for 8 cutoff rigidity values.
3. Interpolate M&B profile at the given *P*c (linear in *P*c space).
4. Integrate profile over each TM5 layer's depth interval using trapezoidal rule.
5. Normalize so weights sum to 1.

---

### `cosmo14C/isoflux.py` — Unit conversion

**`to_isoflux(q_col_2d, shape_3d, cell_area_1d)`**
Converts columnar production [atoms cm⁻² s⁻¹] to 3D isoflux [TgC ‰ yr⁻¹].

Inputs:
- `q_col_2d`: shape (nlat, nlon) — columnar ¹⁴C production
- `shape_3d`: shape (nlev, nlat, nlon) — normalized vertical weights
- `cell_area_1d`: shape (nlat,) — cell area [cm²]

Conversion chain:
```
q_col [atoms cm⁻² s⁻¹]
  × cell_area [cm²]          → atoms s⁻¹ per cell
  × s_per_yr                 → atoms yr⁻¹ per cell
  ÷ N_A × M_C × 10⁻¹²       → TgC yr⁻¹ per cell
  ÷ R_std                    → TgC ‰ yr⁻¹ per cell
```

The vertical distribution is applied by multiplying each layer's weight before the unit conversion.

---

### `cosmo14C/build_lookup.py` — Lookup table

**`build_local_Q_table(output_path, phi_cfg, Pc_cfg)`**
Builds a 2D array Q[phi_i, Pc_j] by calling `local_Q(phi, Pc)` for every grid point. Saves as `.npz` with keys `phi_grid`, `Pc_grid`, `Q`. Supports log or linear spacing for each axis.

Grid defaults (from `config/run.yaml`): 50 φ points (log, 10–2000 MV) × 100 *P*c points (linear, 0–20 GV). Build time: ~30 s.

**`load_local_Q_table(path)`**
Returns a `scipy.interpolate.RegularGridInterpolator` callable that maps (φ, *P*c) → Q [atoms cm⁻² s⁻¹]. Clips inputs to table bounds before interpolation (avoids `bounds_error`).

---

### `cosmo14C/output.py` — NetCDF4 writer

**`write_year(year, data_4d, lat, lon, aux, config)`**
Writes one year of data to `{output.directory}/cosmo14C_{year}.nc`.

- Dimensions: `time`(12), `lev`(34), `lat`(180), `lon`(360)
- Primary variable: `C14_isoflux` [TgC ‰ yr⁻¹], shape (time, lev, lat, lon)
- Diagnostic variables: `phi` [MV], `Pc` [GV], `q_col` [atoms cm⁻² s⁻¹], `geomag_lat` [degrees]
- Time coordinate: month midpoints as days-since-{year}-01-01 (day 15, 46, 74, …)
- Level coordinate: TM5 layer index 1 (surface) → 34 (top), `positive="up"`
- Compression: zlib, complevel=4 (configurable)
- CF conventions: standard_name on lat/lon/time, axis attributes

---

### `cosmo14C/run.py` — Pipeline orchestration

**`run_pipeline(cfg)`**
Full pipeline execution:

1. **Ensure lookup table** — builds if missing or config hash changed.
2. **Set up grid** — lat/lon arrays, cell areas, TM5 coefficients.
3. **Load φ series** — for entire period at once; fails fast if any month is missing.
4. **Year loop:**
   - Compute geomagnetic latitude grid (IGRF, updated annually).
   - Compute *P*c grid, clipped to table bounds.
   - Build vertical shape volume (cached per unique *P*c rounded to 0.1 GV).
   - **Month loop:** look up Q(φ, *P*c), convert to isoflux.
   - Write NetCDF4 file.

**`main()`**
CLI entry point. Reads config path from `sys.argv[1]` (default: `config/run.yaml`).

---

## 7. Data Files Reference

### `data/solar_phi/oulu_phi.csv`

Monthly solar modulation potential φ [MV] for 2014–2025.

| Column | Description |
|---|---|
| `year` | Calendar year |
| `month` | Month (1–12) |
| `phi_MV` | Solar modulation potential [MV] |

**Source:** Usoskin et al. (2017) `Phi_mon.txt`, continuously updated at [cosmicrays.oulu.fi](https://cosmicrays.oulu.fi/phi/phi.html).

**LIS basis:** Burger et al. (2000) — consistent with the Kovaltsov yield function used here. *Do not mix with the Vaisanen (2023) series, which uses the Vos & Potgieter (2015) LIS.*

**Coverage:** 144 rows (2014-01 through 2025-12). φ range: 269 MV (2020 solar minimum) to 1162 MV (November 2025, Solar Cycle 25 peak).

---

### `data/solar_phi/oulu_nm_rates.csv`

Monthly Oulu neutron monitor (NM) count rates for 2010–2025.

| Column | Description |
|---|---|
| `year` | Calendar year |
| `month` | Month (1–12) |
| `count_rate` | Monthly mean Oulu NM corrected count rate [counts per minute] |

**Source:** NMDB (Neutron Monitor Database), Oulu station, `1HCOR_E` efficiency-corrected data. Raw data in counts/second; converted to cpm (×60) and averaged to monthly means.

**URL template:**
```
https://www.nmdb.eu/nest/draw_graph.php?wget=1&stations[]=OULU
  &output=ascii&tabchoice=ori&dtype=corr_for_efficiency
  &date_choice=bydate
  &start_year=YYYY&start_month=MM&start_day=01&start_hour=00&start_min=00
  &end_year=YYYY&end_month=MM&end_day=31&end_hour=23&end_min=59
  &yunits=0&tresolution=1440
```

**Coverage:** 192 rows (2010-01 through 2025-12). Count rate range: 5497–6813 cpm.

**Purpose:** Extends φ coverage via linear regression φ = a × rate + b when published φ is unavailable. Also provides the calibration dataset.

---

### `data/solar_phi/` — Regression calibration

The regression coefficients in `config/run.yaml` (`a = -0.5740`, `b = 4168.0`) were fitted over the 2010–2021 overlap period between `oulu_phi.csv` and `oulu_nm_rates.csv`:

| Metric | Value |
|---|---|
| R² | 0.998 |
| RMSE | 5.9 MV |
| N (months) | 144 |
| φ at 6700 cpm | ~330 MV |
| φ at 5500 cpm | ~1011 MV |

For 2014–2025 the published φ is complete, so the regression is not exercised in the current run. It is available as a fallback if future months are not yet in `Phi_mon.txt`.

---

### `data/masarik_beer_2009/table1.csv`

Vertical ¹⁴C production profile from Masarik & Beer (2009), Table 1.

| Column | Description |
|---|---|
| `depth_gcm2` | Atmospheric depth [g cm⁻²] |
| `Pc0`, `Pc2`, … `Pc17` | Production rate [atoms g⁻¹ s⁻¹] at each cutoff rigidity |

**Cutoff rigidities:** 0, 2, 4, 6, 8, 10, 14, 17 GV. Interpolated linearly between values.

---

### `data/tm5/tropo34_coeffs.csv`

TM5 hybrid σ-pressure coefficients for 34 layers derived from ECMWF L137.

| Column | Description |
|---|---|
| `half_level` | Half-level index (surface = highest index) |
| `a_Pa` | Coefficient *a* [Pa] |
| `b` | Coefficient *b* [dimensionless] |

Layer pressure at any location: *p* = *a*_Pa + *b* × *p*_surf.

---

### `data/lookup/local_Q_table.npz` (auto-generated)

Precomputed 2D lookup table Q(φ, *P*c). Generated automatically on the first run. Contains:
- `phi_grid` [MV]: shape (50,), log-spaced 10–2000
- `Pc_grid` [GV]: shape (100,), linear 0–20
- `Q` [atoms cm⁻² s⁻¹]: shape (50, 100)

Invalidated and rebuilt if `data/lookup/local_Q_table.npz.hash` does not match the current production config hash.

---

## 8. Running the Pipeline

### Standard run (full period)

```bash
cd /path/to/Cosmogenic
conda activate carlos

python -m cosmo14C.run config/run.yaml
```

### Single year (ad hoc)

```python
import yaml
from cosmo14C.run import run_pipeline

with open('config/run.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['period']['start'] = 2020
cfg['period']['end']   = 2020
run_pipeline(cfg)
```

### Custom output directory

Edit `config/run.yaml`:
```yaml
output:
  directory: /path/to/your/output/
```

### Forcing lookup table rebuild

```bash
rm data/lookup/local_Q_table.npz.hash
python -m cosmo14C.run config/run.yaml
```

### Running with a custom config

```bash
cp config/run.yaml config/run_test.yaml
# Edit config/run_test.yaml as needed
python -m cosmo14C.run config/run_test.yaml
```

---

## 9. Output Format

### File naming

`output/cosmo14C_{year}.nc` — one file per year, e.g. `cosmo14C_2020.nc`.

### Dimensions

| Dimension | Size | Description |
|---|---|---|
| `time` | 12 | Calendar months (month midpoints) |
| `lev` | 34 | TM5 tropo34 hybrid layers |
| `lat` | 180 | Geographic latitude cell centres [−89.5°, +89.5°] |
| `lon` | 360 | Geographic longitude cell centres [−179.5°, +179.5°] |

### Variables

| Variable | Dimensions | Units | Description |
|---|---|---|---|
| `C14_isoflux` | (time, lev, lat, lon) | TgC ‰ yr⁻¹ | Cosmogenic ¹⁴C isoflux (primary output) |
| `time` | (time,) | days since {year}-01-01 | Month midpoints (day 15, 46, 74, …) |
| `lev` | (lev,) | 1 | TM5 layer index; 1 = surface, 34 = top |
| `lat` | (lat,) | degrees_north | Latitude cell centres |
| `lon` | (lon,) | degrees_east | Longitude cell centres |
| `phi` | (time,) | MV | Solar modulation potential per month |
| `Pc` | (lat, lon) | GV | Vertical geomagnetic cutoff rigidity |
| `q_col` | (time, lat, lon) | atoms cm⁻² s⁻¹ | Columnar ¹⁴C production rate |
| `geomag_lat` | (lat, lon) | degrees | Geomagnetic latitude (for year mid-point) |

### Level convention

`lev = 1` is the **surface** layer. Index increases toward the top of the atmosphere (TOA). CF attribute: `positive = "up"`. This matches the TM5 tropo34 convention where layer 1 is the lowest tropospheric layer.

### Reading in Python (xarray)

```python
import xarray as xr

ds = xr.open_dataset('output/cosmo14C_2020.nc')
print(ds)

# Global annual isoflux sum [TgC ‰ yr⁻¹]
total = ds['C14_isoflux'].sum().item()
print(f"Global annual isoflux: {total:.1f} TgC ‰ yr⁻¹")

# Zonal mean over all levels
zonal = ds['C14_isoflux'].sum('lev').mean('lon')
```

---

## 10. Validation

### Physics benchmark (`validate_Q.py`)

The primary physics validation checks that the global columnar production at φ = 650 MV reproduces the Kovaltsov et al. (2012) benchmark:

```bash
python validate_Q.py
```

Pass criterion: computed Q within 5% of 1.66 atoms cm⁻² s⁻¹.

This test exercises the full production physics chain: GCR spectrum, yield function, geomagnetic accessible-fraction averaging, and unit conversion.

### Unit tests

```bash
pytest tests/ -v
```

114 tests across 13 test files. Expected output: `114 passed`.

Specific test categories:

| Command | What it tests |
|---|---|
| `pytest tests/test_production.py -v` | Core physics (rigidity, global Q benchmark) |
| `pytest tests/test_geomagnetic.py -v` | IGRF grid shape, range, antisymmetry |
| `pytest tests/test_solar_modulation.py -v` | φ series, regression, warnings |
| `pytest tests/test_integration.py -v` | Full pipeline smoke test (2020) |
| `pytest tests/test_output.py -v` | NetCDF structure and metadata |

### Integration smoke test

`tests/test_integration.py` runs the full pipeline for year 2020 and checks:

1. Output file `cosmo14C_2020.nc` is created.
2. Dimensions are (time=12, lev=34, lat=180, lon=360).
3. All `C14_isoflux` values are positive (no negative production).
4. The array is not all-zero (production is occurring).
5. Global annual sum is in a physically plausible range.
6. `phi` diagnostic is present and in [0, 2000] MV for all months.

### Sanity checks on output

After a full run, spot-check the output:

```python
import xarray as xr
import numpy as np

# Open one year
ds = xr.open_dataset('output/cosmo14C_2020.nc')

# 1. No negative values
assert (ds['C14_isoflux'] >= 0).all(), "Negative isoflux!"

# 2. φ diagnostic within physical range
assert (ds['phi'] > 0).all() and (ds['phi'] < 2000).all()

# 3. Pc pattern: low at poles, high at equator
assert ds['Pc'].sel(lat=0, method='nearest').mean() > ds['Pc'].sel(lat=80, method='nearest').mean()

# 4. Seasonal variation: solar min months should have higher production
assert ds['q_col'].sel(time=0).mean() != ds['q_col'].sel(time=6).mean()

# 5. Column sum of shape weights = 1 (verify via q_col vs isoflux ratio)
print(f"Global mean Q: {ds['q_col'].mean().item():.3f} atoms cm⁻² s⁻¹")
print(f"φ range: {ds['phi'].min().item():.0f} – {ds['phi'].max().item():.0f} MV")
```

### Comparison against Miller et al. (2025)

The public Miller et al. (2025) dataset is archived at [https://doi.org/10.15138/g26t-j556](https://doi.org/10.15138/g26t-j556). To compare:

```python
import xarray as xr, numpy as np

ds_miller = xr.open_dataset('miller_2025_cosmo14C_2015.nc')  # external reference
ds_here   = xr.open_dataset('output/cosmo14C_2015.nc')

# Compare global annual sums
print(f"Miller: {ds_miller['C14_isoflux'].sum().item():.1f}")
print(f"Here  : {ds_here['C14_isoflux'].sum().item():.1f}")
```

Expected agreement within a few percent, with residual differences attributable to different φ reconstruction methods (Miller uses a different NM station calibration).

---

## 11. Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run with coverage

```bash
pip install pytest-cov
pytest tests/ --cov=cosmo14C --cov-report=term-missing
```

### Test structure

| File | Coverage area | Key test |
|---|---|---|
| `test_config.py` | Constants | Exact values |
| `test_gcr_spectrum.py` | GCR spectrum | φ=0 gives LIS; species difference |
| `test_yield_function.py` | Yield Y(E) | Exact match at Kovaltsov table nodes |
| `test_production.py` | global_Q | Within 5% of 1.66 at φ=650 MV |
| `test_production_local.py` | local_Q | Monotone in φ and Pc |
| `test_geomagnetic.py` | IGRF grid | Shape (180,360), range [−90,90], antisymmetry |
| `test_solar_modulation.py` | φ series | Regression fallback, clamping |
| `test_grid.py` | Grid utilities | Total area = 4πR², Pc(equator) > Pc(pole) |
| `test_vertical_profile.py` | Shape weights | Sum = 1, non-negative |
| `test_isoflux.py` | Unit conversion | Sum over levels recovers columnar |
| `test_build_lookup.py` | Lookup table | <2% error vs direct local_Q |
| `test_output.py` | NetCDF output | Dimensions, variables, metadata |
| `test_integration.py` | Full pipeline | Smoke test for 2020 |

### Adding new tests

Place new test files in `tests/`, following the `test_*.py` naming convention. Pytest discovers them automatically.

---

## 12. Updating Input Data

### 12.1 Updating solar φ values (`oulu_phi.csv`)

The Usoskin et al. (2017) `Phi_mon.txt` file is continuously updated. Download the latest version and merge new months.

**Download:**
```bash
curl -sk "https://cosmicrays.oulu.fi/phi/Phi_mon.txt" -o /tmp/Phi_mon.txt
head -5 /tmp/Phi_mon.txt   # inspect header
```

**Format:** Wide table, one row per year, 12 monthly values + annual mean. Missing months shown as `-`.

**Update procedure:**

```python
import csv, re

# Parse Phi_mon.txt
rows = []
with open('/tmp/Phi_mon.txt') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if not parts[0].isdigit():
            continue
        year = int(parts[0])
        # 12 monthly values + annual mean (13 values after year)
        vals = parts[1:13]
        for month, v in enumerate(vals, start=1):
            if v != '-':
                rows.append((year, month, int(v)))

# Write updated CSV (overwrite only if adding new rows)
existing = {}
with open('data/solar_phi/oulu_phi.csv') as f:
    for row in csv.DictReader(f):
        existing[(int(row['year']), int(row['month']))] = float(row['phi_MV'])

new_rows = [(y, m, v) for y, m, v in rows if (y, m) not in existing]
print(f"Adding {len(new_rows)} new months")

with open('data/solar_phi/oulu_phi.csv', 'a', newline='') as f:
    w = csv.writer(f)
    for y, m, v in sorted(new_rows):
        w.writerow([y, m, v])
```

> **Important — LIS consistency:** `Phi_mon.txt` uses the Burger (2000) LIS, which is consistent with the Kovaltsov (2012) yield function used here. The Vaisanen (2023) dataset at the same URL uses the Vos & Potgieter (2015) LIS and **must not** be mixed with this pipeline.

---

### 12.2 Updating Oulu NM count rates (`oulu_nm_rates.csv`)

Download new monthly means from NMDB and append.

**Download daily data:**
```bash
# Replace YYYY and MM with target year/month range
curl -s "https://www.nmdb.eu/nest/draw_graph.php?wget=1\
&stations[]=OULU&output=ascii&tabchoice=ori\
&dtype=corr_for_efficiency&date_choice=bydate\
&start_year=2026&start_month=01&start_day=01&start_hour=00&start_min=00\
&end_year=2026&end_month=12&end_day=31&end_hour=23&end_min=59\
&yunits=0&tresolution=1440" -o /tmp/oulu_new.txt
```

**Compute monthly means and append:**
```python
from collections import defaultdict
import csv

# Parse daily data (c/s → cpm × 60)
monthly = defaultdict(list)
with open('/tmp/oulu_new.txt') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or ';' not in line:
            continue
        parts = line.split(';')
        if len(parts) < 2:
            continue
        try:
            dt, val = parts[0].strip(), parts[1].strip()
            v = float(val)
            if v <= 0:
                continue
            year, month = int(dt[:4]), int(dt[5:7])
            monthly[(year, month)].append(v * 60)   # c/s → cpm
        except ValueError:
            continue

# Append new months to CSV
with open('data/solar_phi/oulu_nm_rates.csv', 'a', newline='') as f:
    w = csv.writer(f)
    for (y, m) in sorted(monthly):
        mean_cpm = sum(monthly[(y, m)]) / len(monthly[(y, m)])
        w.writerow([y, m, f"{mean_cpm:.1f}"])
```

---

### 12.3 Re-calibrating the regression

If the calibration period is extended significantly (e.g., beyond 2021), recalibrate the regression:

```python
import numpy as np
import pandas as pd

phi_df = pd.read_csv('data/solar_phi/oulu_phi.csv').set_index(['year', 'month'])['phi_MV']
nm_df  = pd.read_csv('data/solar_phi/oulu_nm_rates.csv').set_index(['year', 'month'])['count_rate']

# Use overlap period (both datasets available)
common = phi_df.index.intersection(nm_df.index)
rates = nm_df[common].values
phis  = phi_df[common].values

a, b = np.polyfit(rates, phis, 1)
r2 = np.corrcoef(rates, phis)[0, 1]**2
rmse = np.sqrt(np.mean((a * rates + b - phis)**2))

print(f"a={a:.4f}  b={b:.1f}  R²={r2:.4f}  RMSE={rmse:.1f} MV")
print(f"Update config/run.yaml: regression_coeffs: {{a: {a:.4f}, b: {b:.1f}}}")
```

Then update `config/run.yaml` with the new coefficients.

---

### 12.4 Extending the period

To process additional years (e.g., 2026):

1. Append new rows to `oulu_phi.csv` (see §12.1).
2. Append new rows to `oulu_nm_rates.csv` (see §12.2).
3. Edit `config/run.yaml`: set `period.end: 2026`.
4. Run the pipeline. (The lookup table is already built and cached.)

---

### 12.5 Updating the IGRF model

ppigrf bundles the IGRF coefficient file internally. When IGRF-14 (or a later epoch) is released, update ppigrf:

```bash
pip install --upgrade ppigrf
```

No changes to this codebase are needed unless ppigrf changes its internal API (see §14.1).

---

## 13. Physics Notes

### 13.1 LIS and solar modulation

The force-field approximation (Gleeson & Axford 1968) reduces the full Parker transport equation to a one-parameter model. The modulation potential φ [MV] characterizes the energy loss experienced by GCR particles propagating from the heliopause to 1 AU:

$$J_\text{mod}(E, \phi) = J_\text{LIS}(E + \Phi) \cdot \frac{E(E + 2m_r)}{(E + \Phi)(E + \Phi + 2m_r)}$$

where $\Phi = \phi \times 10^{-3} \cdot (Z/A)$ [GeV/nuc] and *m*_r = 0.938 GeV/nuc.

The LIS parametrization of Burger et al. (2000) / Usoskin et al. (2005) is used, consistent with both the Kovaltsov yield function and the Usoskin (2017) φ reconstruction.

### 13.2 Geomagnetic shielding

Vertical cutoff rigidity from the dipole approximation (Störmer):

$$P_c(\lambda_G) = \frac{1.9 M_{22} \cos^4\!\lambda_G}{r^2}$$

at Earth's surface (*r* = 1). The *M* = 7.8 × 10²² A m² value corresponds to the present-day IGRF dipole moment. The IGRF is used only to determine geomagnetic latitude (i.e., the dipole axis direction) via ppigrf; *M* itself is kept fixed at the configured value. The default 7.8 is appropriate for 2014–2025. If extending to earlier or later periods with significantly different *M*, update `geomagnetic.M_1e22` in `config/run.yaml`.

### 13.3 Vertical profile

Masarik & Beer (2009) provide ¹⁴C production as a function of atmospheric depth (g/cm²) for 8 geomagnetic cutoff rigidities. Here, the profile is:
1. Interpolated to the actual *P*c of each grid cell.
2. Integrated over TM5 layer depth intervals.
3. Normalized to yield dimensionless fractional weights that sum to 1.

The TM5 hybrid σ-pressure coordinates are converted to pressure (and hence depth via hydrostatic balance) using the reference surface pressure *p*_surf = 1013.25 hPa.

### 13.4 Alpha contribution

Kovaltsov et al. (2012) show that heavier GCR nuclei (primarily alpha particles) contribute ~30% additional ¹⁴C production per nucleon relative to protons. This is implemented via `ALPHA_RATIO = 0.30`, applied to a separate yield function integration with the alpha yield table and the alpha force-field modulation (*Z*/*A* = 0.5).

### 13.5 Isoflux definition

The ¹⁴C **isoflux** is the product of the carbon flux and the isotopic ratio relative to standard:

$$F_{^{14}C} = \frac{dN/dt}{N_A} \cdot M_C \cdot \frac{1}{R_\text{std}} \quad [\text{TgC ‰ yr}^{-1}]$$

where *R*_std = 1.176 × 10⁻¹² is the ¹⁴C/C ratio of the NBS oxalic acid standard. This quantity is directly used as a source term in ¹⁴CO₂ tracer transport in TM5.

---

## 14. Troubleshooting

### 14.1 ppigrf internal API changes

**Symptom:** `ImportError: ppigrf internal API not found; tested against ppigrf>=1.0.`

**Cause:** ppigrf changed its internal module structure.

**Fix:** Check the ppigrf package structure:
```python
import ppigrf
print(dir(ppigrf))
```
Then update `cosmo14C/geomagnetic.py` line 5 to import from the correct path, and update `_get_dipole_coeffs` if `read_shc()` was renamed or its signature changed.

---

### 14.2 Lookup table out of date

**Symptom:** The pipeline uses a stale lookup table after changing `phi_grid_MV` or `Pc_grid_GV` parameters.

**Fix:** Delete the hash file to force a rebuild:
```bash
rm data/lookup/local_Q_table.npz.hash
```
The pipeline will then rebuild the table on the next run (takes ~30 s).

---

### 14.3 Missing phi values for a month

**Symptom:** `ValueError: No phi value for (2026, 3)` during pipeline execution.

**Cause:** `oulu_phi.csv` does not cover this month and the NM regression fallback also lacks data in `oulu_nm_rates.csv`.

**Fix:** Add the missing month to one or both files (see §12.1 and §12.2), then re-run.

---

### 14.4 NM regression gives non-physical phi

**Symptom:** `UserWarning: Solar modulation regression produced non-positive phi=-xxx.x MV for (year, month); clamping to 10 MV.`

**Cause:** The regression coefficients do not fit the current NM count rate range. This can happen if the regression was calibrated on older data but the NM rates have drifted due to instrumental changes.

**Fix:** Recalibrate the regression (see §12.3) and update `config/run.yaml`.

---

### 14.5 `conda run` fails to find numpy

**Symptom:** `ModuleNotFoundError: No module named 'numpy'` when running via `conda run`.

**Fix:** Use the conda environment's Python directly:
```bash
/Users/you/anaconda3/envs/carlos/bin/python -m cosmo14C.run config/run.yaml
```
Or activate the environment first:
```bash
conda activate carlos
python -m cosmo14C.run config/run.yaml
```

---

### 14.6 NetCDF write fails

**Symptom:** `PermissionError` or `RuntimeError` when writing output.

**Fix:** Check that the output directory exists and is writable:
```bash
mkdir -p output/
python -m cosmo14C.run config/run.yaml
```
If using a custom output directory on a network drive (e.g., OneDrive), ensure the path is accessible before starting the run.

---

### 14.7 Integration tests still skipping

**Symptom:** `pytest tests/test_integration.py` reports all 6 tests as `SKIP`.

**Cause:** `data/solar_phi/oulu_phi.csv` has fewer than 12 rows. The skip guard in `test_integration.py` requires ≥12 rows.

**Fix:** The file should have 144 rows (2014–2025). Re-populate it following §12.1.

---

## 15. References

**Yield function:**
- Kovaltsov, G. A., Mishev, A., & Usoskin, I. G. (2012). A new model of cosmogenic production of radiocarbon ¹⁴C in the atmosphere. *Earth and Planetary Science Letters*, 337–338, 114–120. https://doi.org/10.1016/j.epsl.2012.05.036

**Solar modulation potential:**
- Usoskin, I. G., Gil, A., Kovaltsov, G. A., Mishev, A. L., & Mikhailov, V. V. (2017). Heliospheric modulation of cosmic rays during the neutron monitor era: Calibration using PAMELA data for 2006–2010. *Journal of Geophysical Research: Space Physics*, 122, 3875–3887. https://doi.org/10.1002/2016JA023819

**GCR local interstellar spectrum:**
- Burger, R. A., Potgieter, M. S., & Heber, B. (2000). Rigidity dependence of cosmic ray proton latitudinal gradients measured by the Ulysses spacecraft: Implications for the diffusion tensor. *Journal of Geophysical Research*, 105(A12), 27447–27455.

**Vertical production profile:**
- Masarik, J., & Beer, J. (2009). An updated simulation of particle fluxes and cosmogenic nuclide production in the Earth's atmosphere. *Journal of Geophysical Research: Atmospheres*, 114, D11103. https://doi.org/10.1029/2008JD010557

**Geomagnetic field:**
- Alken, P., et al. (2021). International Geomagnetic Reference Field: the thirteenth generation. *Earth, Planets and Space*, 73, 49. https://doi.org/10.1186/s40623-020-01288-x

**¹⁴C isoflux application:**
- Miller, J. B., Lehman, S. J., & Lindsay, R. P. (2025). Cosmogenic ¹⁴C production and isoflux fields from 2015 to 2024. *Global Biogeochemical Cycles*. https://doi.org/10.15138/g26t-j556

**Oulu neutron monitor:**
- Oulu Cosmic Ray Station: https://cosmicrays.oulu.fi/

**NMDB — Neutron Monitor Database:**
- Klein, K.-L., et al. (2009). The Neutron Monitor Database (NMDB). *Proceedings of the 31st International Cosmic Ray Conference*. https://www.nmdb.eu/

**Force-field approximation:**
- Gleeson, L. J., & Axford, W. I. (1968). Solar modulation of galactic cosmic rays. *The Astrophysical Journal*, 154, 1011. https://doi.org/10.1086/149822

**ppigrf Python package:**
- https://github.com/klaundal/ppigrf
