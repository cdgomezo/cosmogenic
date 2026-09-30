# cosmo14C

Cosmogenic radiocarbon production fields for atmospheric transport models.

Given a time series of the solar modulation potential, this package integrates
the galactic cosmic ray spectrum against the Kovaltsov et al. (2012) ¹⁴C yield
function, applies the geomagnetic cutoff rigidity from the IGRF dipole, and
writes monthly, column-integrated ¹⁴C production as a NetCDF flux field.

The global production rate reproduces Kovaltsov et al. (2012) to within 5 %.
Everything runs single-threaded in pure Python and needs no network access.

## What it computes

```
phi(t)  solar modulation potential [MV]
   |
   |  GCR spectrum (Burger/Usoskin local interstellar spectrum, modulated)
   |  x Kovaltsov et al. (2012) yield function Y(E)
   v
q_col = Q(phi, Pc)   columnar production [atoms cm-2 s-1]
   ^
   |  Pc = 1.9 M cos^4(geomagnetic latitude)   [GV]
   |  geomagnetic latitude from the IGRF dipole (ppigrf)
   |
   v
flux = q_col / (N_A R_std)   [umol m-2 s-1]
```

The output is the flux of the composite tracer **C_D14C = CO₂ × Δ¹⁴C**, with
Δ¹⁴C carried as a dimensionless fraction (per mil ÷ 1000), the convention of
Basu et al. (2016).

Two properties of the field are worth knowing before using it.

**It is column-integrated, and the production is not at the surface.** Roughly
two thirds of cosmogenic ¹⁴C is made in the stratosphere and most of the rest in
the upper troposphere. No vertical profile is applied here, because a transport
model can distribute the column from its own air mass. Reading the field as a
surface emission puts the source in the wrong part of the atmosphere.

**Production has no Δ¹⁴C of its own.** It injects bare ¹⁴C atoms with no
accompanying carbon, so the ratio Δ is defined from has nothing in its
denominator. Writing Δ = R/R_std − 1 and C_D14C = C × Δ = N₁₄/R_std − C, a
source with dN₁₄ > 0 and dC = 0 contributes d(C_D14C) = dN₁₄/R_std, which is the
whole of the conversion above: one mole of ¹⁴C produced is 1/R_std moles of
C_D14C. There is no signature field to accompany the flux, and there cannot be.

## Install

```bash
conda env create -f environment.yml
conda activate cosmo14C
pytest -q                      # 120 tests, about 20 s
```

Or into an existing environment: `pip install -r requirements.txt`. The IGRF
coefficients ship inside `ppigrf`, so nothing is downloaded at run time.

## Use

### Global, monthly, 1°×1°

```bash
python -m cosmo14C.run config/run.yaml
```

One file per year, `output/flux_c14.cosmogenic.<year>.nc`, holding `c14flux`
(time, lat, lon) in µmol m⁻² s⁻¹, stamped at the start of each month, plus the
diagnostics the field was built from: `phi`, `Pc`, `q_col` and `geomag_lat`.

The first run builds a `Q(phi, Pc)` lookup table under `data/lookup/`, which
takes about 30 s. Later runs reuse it, and rebuild it automatically if the
`production:` block of the config changes.

### A region, at any resolution

```bash
python scripts/make_regional_cosmogenic.py \
    --year 2024 --resolution 0.1 --domain 33 73 -15 35 \
    --out-dir output/regional
```

Writes the flux in both conventions (C_D14C and per mil), the Δ¹⁴C tendency the
production imparts to the air column, and a consistency report.

Nothing is interpolated: `q = Q(phi, Pc)` is analytic in the two things it
depends on, and both are evaluated directly on whatever grid is asked for. The
field is nevertheless smooth on the scale of the geomagnetic dipole, so a fine
grid costs nothing and resolves nothing; it exists to match the grid of whatever
the field is used alongside.

## Configuration

```yaml
period:      {start: 2000, end: 2025}
grid:        {lat_res_deg: 1.0, lon_res_deg: 1.0}
geomagnetic: {M_1e22: 7.8, update_frequency: yearly}    # dipole moment
solar_modulation:
  oulu_phi_file: data/solar_phi/oulu_phi.csv            # published phi
  nm_rate_file:  data/solar_phi/oulu_nm_rates.csv       # fallback, see below
  regression_coeffs: {a: -0.5740, b: 4168.0}
production:
  lookup_table: data/lookup/local_Q_table.npz
  phi_grid_MV:  {min: 10, max: 2000, n: 50,  scale: log}
  Pc_grid_GV:   {min: 0,  max: 20,   n: 100, scale: linear}
output:      {directory: output/, compress: true}
```

`config/run_miller.yaml` is the same thing restricted to 2000–2012, the window
the comparison below covers.

## Input data

| Path | What | Update |
|---|---|---|
| `data/solar_phi/oulu_phi.csv` | published monthly solar modulation potential φ, MV | monthly, from the Oulu series |
| `data/solar_phi/oulu_nm_rates.csv` | Oulu neutron monitor count rates | monthly, from NMDB |
| `data/lookup/local_Q_table.npz` | cached `Q(phi, Pc)` table | rebuilt automatically |

φ is taken from the published series wherever it exists. Months that are not yet
published are filled from the neutron monitor count rate through a linear
regression calibrated on the 2010–2021 overlap (R² = 0.998, RMSE 5.9 MV), and
the run warns when it has had to do that.

## Validation

`pytest` covers each physical step against its source, and the gate is the
global production rate: `global_Q(650 MV)` must sit within 5 % of the 1.66 atoms
cm⁻² s⁻¹ of Kovaltsov et al. (2012). It comes out at 1.64, which is also the
value Miller et al. (2025) prefer.

The comparison against the gridded product of Miller et al. (2025) is in
`scripts/`:

```bash
python scripts/make_miller_nc.py --input-dir output/ \
    --output comparison/cosmo14C_2000_2012_miller_format.nc \
    --start 2000 --end 2012 --reference Cosmo.nc
python scripts/plot_miller_comparison.py    # writes figures/
```

The global totals and their time evolution agree. The spatial patterns do not,
and the reason is known: the cutoff rigidity here uses the dipole approximation
rather than a full trajectory computation, which gives 20 to 60 % more
production poleward of 45° and 30 to 50 % less equatorward of 40°. The
discrepancy is steady in time, so it is a pattern difference and not a drift.
Use the dipole with that in mind for high-latitude work.

## Layout

```
cosmo14C/
  run.py              the global pipeline, python -m cosmo14C.run
  regional.py         arbitrary domain and resolution, plus the Delta14C tendency
  production.py       local and global production rates
  gcr_spectrum.py     modulated GCR spectrum
  yield_function.py   Kovaltsov et al. (2012) yield function
  geomagnetic.py      IGRF dipole -> geomagnetic latitude
  grid.py             lat/lon arrays, cutoff rigidity, cell areas
  solar_modulation.py phi(t), published series and regression fallback
  build_lookup.py     the Q(phi, Pc) table
  isoflux.py          production -> C_D14C flux
  output.py           NetCDF writer
  config.py           physical constants and the calibration targets
scripts/              regional product, and the Miller comparison
config/               run configurations
data/                 solar phi, neutron monitor rates, cached lookup table
tests/                120 tests
```

## References

Kovaltsov, G. A., Mishev, A., & Usoskin, I. G. (2012). A new model of cosmogenic
production of radiocarbon ¹⁴C in the atmosphere. *Earth and Planetary Science
Letters*, 337–338, 114–120. https://doi.org/10.1016/j.epsl.2012.05.036

Usoskin, I. G., Gil, A., Kovaltsov, G. A., Mishev, A. L., & Mikhailov, V. V.
(2017). Heliospheric modulation of cosmic rays during the neutron monitor era.
*Journal of Geophysical Research: Space Physics*, 122, 3875–3887.
https://doi.org/10.1002/2016JA023819

Alken, P., et al. (2021). International Geomagnetic Reference Field: the
thirteenth generation. *Earth, Planets and Space*, 73, 49.
https://doi.org/10.1186/s40623-020-01288-x

Basu, S., Miller, J. B., & Lehman, S. (2016). Separation of biospheric and
fossil fuel fluxes of CO₂ by atmospheric inversion of CO₂ and ¹⁴CO₂
measurements. *Atmospheric Chemistry and Physics*, 16, 5665–5683.
https://doi.org/10.5194/acp-16-5665-2016

Miller, J. B., Lehman, S. J., & Lindsay, C. M. (2025). Numerical representation
of contemporary atmospheric Δ¹⁴CO₂. *Global Biogeochemical Cycles*, 39,
e2025GB008522. https://doi.org/10.1029/2025GB008522

Stuiver, M., & Polach, H. A. (1977). Discussion: Reporting of ¹⁴C data.
*Radiocarbon*, 19(3), 355–363.

## Author

Carlos Gómez-Ortiz, Department of Earth and Environmental Sciences, Lund
University. carlos.gomez@mgeo.lu.se

## License

The code in this repository is MIT licensed, see `LICENSE`.

That covers the implementation and nothing else. The physics is not mine, and
neither is some of what ships beside the code:

- **The method.** The yield function, the modulated cosmic ray spectrum and the
  cutoff rigidity formulation are from the papers cited above. This is an
  implementation of published work, so cite those papers rather than this
  repository if you use the results.
- **Tabulated values inside the code.** `cosmo14C/yield_function.py` carries
  Table 1 of Kovaltsov et al. (2012) as arrays, and `cosmo14C/config.py` carries
  that paper's and Miller et al. (2025)'s calibration targets. Those numbers are
  the authors', quoted for interoperability.
- **Redistributed data.** `data/solar_phi/oulu_phi.csv` is the published solar
  modulation potential series derived from Oulu neutron monitor data
  (Usoskin et al. 2017 and its updates), and `data/solar_phi/oulu_nm_rates.csv`
  holds Oulu count rates obtained through NMDB, https://www.nmdb.eu. Both are
  included so a run is reproducible without a download. They are the
  providers' data under the providers' terms, not MIT, and both ask to be
  acknowledged.
- **The geomagnetic field.** The IGRF coefficients come from the `ppigrf`
  dependency, under its own licence; nothing of IGRF is redistributed here.

The comparison figures under `figures/` are generated from this code against a
published reference dataset, which is not included.
