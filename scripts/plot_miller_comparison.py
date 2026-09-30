"""
Comparison figures between our pipeline output and Miller et al. (2025) Cosmo.nc.

Produces three figures saved to figures/:
  fig1_spatial_mean.png   — time-mean maps (Miller, Ours, % diff)
  fig2_time_series.png    — global monthly totals + scatter
  fig3_zonal_mean.png     — zonal-mean profiles and Hovmöller

Usage:
    python scripts/plot_miller_comparison.py \
        --ours comparison/cosmo14C_2000_2012_miller_format.nc \
        --miller Cosmo.nc \
        --outdir figures/
"""

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ── helpers ──────────────────────────────────────────────────────────────────

def load_both(ours_path, miller_path):
    """Return (lat, lon, decdate, phi, ours, miller) all as numpy arrays.

    Both fields are (lat=180, lon=360, time=156) in TgC permil month-1.
    """
    with nc.Dataset(ours_path) as ds:
        lat  = ds.variables['lat'][:].data.astype('f8')
        lon  = ds.variables['lon'][:].data.astype('f8')
        dd   = ds.variables['decdate'][:].data.astype('f8')
        phi  = ds.variables['phi'][:].data.astype('f8')
        ours = ds.variables['C14_isoflux_col'][:].data.astype('f8')  # (180,360,156)

    with nc.Dataset(miller_path) as ds:
        miller_dd = ds.variables['Decimal Date'][:].data.astype('f8')
        miller    = ds.variables['Cosmo'][:].data.astype('f8')  # (180,360,156)

    # align by nearest decimal date (both should already match month-for-month)
    idx = np.argmin(np.abs(miller_dd[:, None] - dd[None, :]), axis=1)  # (156,)
    miller = miller[:, :, idx]  # reorder if needed (should be identity)

    return lat, lon, dd, phi, ours, miller


def global_sum(field):
    """Sum over (lat, lon) → (time,) in TgC permil month-1."""
    return field.sum(axis=(0, 1))


def zonal_mean(field):
    """Mean over lon → (lat, time)."""
    return field.mean(axis=1)


def add_map_axes(fig, pos, proj=None):
    if proj is None:
        proj = ccrs.Robinson()
    ax = fig.add_subplot(*pos, projection=proj)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color='k')
    ax.set_global()
    return ax


def pcolor_map(ax, lon, lat, data, **kw):
    return ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), **kw)


# ── figure 1: spatial time-mean ───────────────────────────────────────────────

def fig_spatial(lat, lon, ours, miller, outdir):
    tmean_o = ours.mean(axis=2)    # (180, 360)
    tmean_m = miller.mean(axis=2)
    pct_diff = (tmean_o - tmean_m) / tmean_m * 100.0

    # shared log scale for absolute maps — use common vmin/vmax excluding extreme polar cells
    vmin = 5e-4   # exclude the near-zero polar cells in Miller
    vmax = max(tmean_o.max(), tmean_m.max())
    norm_abs = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # cap % diff at ±60% so mid-latitude structure is visible (polar cells saturate)
    pct_cap = 60.0

    fig = plt.figure(figsize=(14, 10))
    proj = ccrs.Robinson()

    # --- (a) Miller time-mean
    ax1 = add_map_axes(fig, (3, 2, 1), proj)
    im1 = pcolor_map(ax1, lon, lat, np.clip(tmean_m, vmin, None), norm=norm_abs, cmap='plasma')
    ax1.set_title('(a) Miller et al. (2025)\ntime-mean 2000–2012')
    plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.03, shrink=0.85,
                 label='TgC ‰ month⁻¹ cell⁻¹')

    # --- (b) Ours time-mean
    ax2 = add_map_axes(fig, (3, 2, 2), proj)
    im2 = pcolor_map(ax2, lon, lat, np.clip(tmean_o, vmin, None), norm=norm_abs, cmap='plasma')
    ax2.set_title('(b) This work\ntime-mean 2000–2012')
    plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.03, shrink=0.85,
                 label='TgC ‰ month⁻¹ cell⁻¹')

    # --- (c) absolute difference (ours - miller)
    diff_lim = np.percentile(np.abs(tmean_o - tmean_m), 97)
    ax3 = add_map_axes(fig, (3, 2, 3), proj)
    im3 = pcolor_map(ax3, lon, lat, tmean_o - tmean_m,
                     vmin=-diff_lim, vmax=diff_lim, cmap='RdBu_r')
    ax3.set_title('(c) Difference (this work − Miller)\ntime-mean 2000–2012')
    plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.03, shrink=0.85,
                 label='TgC ‰ month⁻¹ cell⁻¹')

    # --- (d) % difference — capped at ±60%; polar cells saturate (expected)
    ax4 = add_map_axes(fig, (3, 2, 4), proj)
    im4 = pcolor_map(ax4, lon, lat, np.clip(pct_diff, -pct_cap, pct_cap),
                     vmin=-pct_cap, vmax=pct_cap, cmap='RdBu_r')
    ax4.set_title('(d) Relative difference (this work − Miller) / Miller\n'
                  'time-mean 2000–2012  [capped at ±60%, poles saturate red]')
    plt.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.03, shrink=0.85,
                 label='%')

    # --- (e) Jan 2001 % diff (solar maximum, month 12)
    i_jan01 = 12
    jan_pct = (ours[:, :, i_jan01] - miller[:, :, i_jan01]) / miller[:, :, i_jan01] * 100.0
    ax5 = add_map_axes(fig, (3, 2, 5), proj)
    im5 = pcolor_map(ax5, lon, lat, np.clip(jan_pct, -pct_cap, pct_cap),
                     vmin=-pct_cap, vmax=pct_cap, cmap='RdBu_r')
    ax5.set_title('(e) Relative difference — Jan 2001\n(Solar maximum, φ=987 MV)')
    plt.colorbar(im5, ax=ax5, orientation='horizontal', pad=0.03, shrink=0.85,
                 label='%  [capped ±60%]')

    # --- (f) Jan 2010 % diff (solar minimum, month 120)
    i_smin = 120
    smin_pct = (ours[:, :, i_smin] - miller[:, :, i_smin]) / miller[:, :, i_smin] * 100.0
    ax6 = add_map_axes(fig, (3, 2, 6), proj)
    im6 = pcolor_map(ax6, lon, lat, np.clip(smin_pct, -pct_cap, pct_cap),
                     vmin=-pct_cap, vmax=pct_cap, cmap='RdBu_r')
    ax6.set_title('(f) Relative difference — Jan 2010\n(Solar minimum, φ=255 MV)')
    plt.colorbar(im6, ax=ax6, orientation='horizontal', pad=0.03, shrink=0.85,
                 label='%  [capped ±60%]')

    fig.suptitle('Spatial comparison: this work vs. Miller et al. (2025)', fontsize=13, y=1.01)
    plt.tight_layout()
    out = outdir / 'fig1_spatial_mean.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


# ── figure 2: time series & scatter ─────────────────────────────────────────

def fig_time_series(lat, lon, dd, phi, ours, miller, outdir):
    g_ours   = global_sum(ours)   * 12.0   # → TgC permil yr-1 equivalent
    g_miller = global_sum(miller) * 12.0

    years = dd  # decimal years

    fig, axes = plt.subplots(3, 1, figsize=(12, 11),
                             gridspec_kw={'height_ratios': [2.5, 2.5, 2]})

    # --- top: time series
    ax = axes[0]
    ax.plot(years, g_miller, color='#e07020', lw=1.5, label='Miller et al. (2025)', zorder=2)
    ax.plot(years, g_ours,   color='#1a5fb4', lw=1.5, label='This work', zorder=3, alpha=0.9)
    ax.set_ylabel('Global ¹⁴C isoflux\n[TgC ‰ yr⁻¹ equiv.]')
    ax.set_xlim(years[0], years[-1])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('(a) Monthly global total cosmogenic ¹⁴C isoflux')
    ax.grid(alpha=0.3)

    # secondary axis: phi
    ax2 = ax.twinx()
    ax2.fill_between(years, phi, alpha=0.15, color='gray')
    ax2.plot(years, phi, color='gray', lw=0.8, ls='--', label='φ (this work)')
    ax2.set_ylabel('Solar modulation potential φ [MV]', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, phi.max() * 1.4)
    ax2.legend(loc='upper left', framealpha=0.9)

    # --- middle: difference and ratio
    ax3 = axes[1]
    bias_pct = (g_ours - g_miller) / g_miller * 100.0
    ax3.axhline(0, color='k', lw=0.8)
    ax3.plot(years, bias_pct, color='#1a5fb4', lw=1.2)
    ax3.fill_between(years, bias_pct, alpha=0.2, color='#1a5fb4')
    mean_bias = bias_pct.mean()
    ax3.axhline(mean_bias, color='red', lw=1.2, ls='--',
                label=f'Mean bias = {mean_bias:+.2f}%')
    ax3.set_ylabel('(This work − Miller) / Miller [%]')
    ax3.set_xlim(years[0], years[-1])
    ax3.legend(framealpha=0.9)
    ax3.set_title('(b) Relative difference (monthly global total)')
    ax3.grid(alpha=0.3)

    # --- bottom: scatter
    ax4 = axes[2]
    # colour by year
    year_int = dd.astype(int)
    cmap_sc = plt.cm.viridis
    sc = ax4.scatter(g_miller, g_ours, c=year_int,
                     cmap=cmap_sc, s=22, alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax4, label='Year', pad=0.01)
    vmin_sc = min(g_ours.min(), g_miller.min())
    vmax_sc = max(g_ours.max(), g_miller.max())
    ax4.plot([vmin_sc, vmax_sc], [vmin_sc, vmax_sc], 'k--', lw=0.8, label='1:1 line')
    r = float(np.corrcoef(g_miller, g_ours)[0, 1])
    ax4.set_xlabel('Miller et al. (2025)  [TgC ‰ yr⁻¹ equiv.]')
    ax4.set_ylabel('This work  [TgC ‰ yr⁻¹ equiv.]')
    ax4.set_title(f'(c) Scatter — monthly global total  (r = {r:.3f})')
    ax4.legend(framealpha=0.9)
    ax4.grid(alpha=0.3)

    fig.suptitle('Temporal comparison: this work vs. Miller et al. (2025)', fontsize=13)
    plt.tight_layout()
    out = outdir / 'fig2_time_series.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


# ── figure 3: zonal mean & Hovmöller ─────────────────────────────────────────

def fig_zonal(lat, lon, dd, phi, ours, miller, outdir):
    zm_ours   = zonal_mean(ours)    # (180, 156)
    zm_miller = zonal_mean(miller)
    zm_pct    = (zm_ours - zm_miller) / zm_miller * 100.0  # (180, 156)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    # --- (a) time-mean zonal profile (absolute)
    ax = axes[0]
    zm_m_mean = zm_miller.mean(axis=1)
    zm_o_mean = zm_ours.mean(axis=1)
    ax.plot(zm_m_mean, lat, color='#e07020', lw=1.8, label='Miller et al. (2025)')
    ax.plot(zm_o_mean, lat, color='#1a5fb4', lw=1.8, ls='--', label='This work')
    ax.set_ylabel('Latitude [°]')
    ax.set_xlabel('Zonal-mean isoflux\n[TgC ‰ month⁻¹ cell⁻¹]')
    ax.set_title('(a) Time-mean zonal profiles\n2000–2012')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-90, 90)
    ax.set_yticks(np.arange(-90, 91, 30))

    # --- (b) normalized zonal profiles (peak = 1) to compare shape
    ax_n = axes[1]
    ax_n.plot(zm_m_mean / zm_m_mean.max(), lat, color='#e07020', lw=1.8,
              label='Miller et al. (2025)')
    ax_n.plot(zm_o_mean / zm_o_mean.max(), lat, color='#1a5fb4', lw=1.8, ls='--',
              label='This work')
    ax_n.set_xlabel('Normalized zonal-mean\n(peak = 1)')
    ax_n.set_title('(b) Shape comparison\n(normalized to peak)')
    ax_n.legend(framealpha=0.9)
    ax_n.grid(alpha=0.3)
    ax_n.set_ylim(-90, 90)
    ax_n.set_yticks(np.arange(-90, 91, 30))
    ax_n.set_xlim(-0.05, 1.1)

    # --- (c) Hovmöller: zonal-mean % difference — capped at ±100%
    ax2 = axes[2]
    pct_cap_hov = 100.0
    im = ax2.pcolormesh(dd, lat, np.clip(zm_pct, -pct_cap_hov, pct_cap_hov),
                        vmin=-pct_cap_hov, vmax=pct_cap_hov, cmap='RdBu_r', shading='auto')
    cb = plt.colorbar(im, ax=ax2, label='%', orientation='vertical', pad=0.02)
    cb.ax.set_ylabel('%  [capped ±100%]')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Latitude [°]')
    ax2.set_title('(c) Hovmöller — zonal-mean relative diff\n(this work − Miller) / Miller  [±100% cap]')
    ax2.set_ylim(-90, 90)
    ax2.set_yticks(np.arange(-90, 91, 30))
    # phi as inset line chart above Hovmöller
    ax2_inset = ax2.inset_axes([0.0, 1.02, 1.0, 0.18])
    ax2_inset.plot(dd, phi, color='gray', lw=1.2)
    ax2_inset.fill_between(dd, phi, alpha=0.2, color='gray')
    ax2_inset.set_xlim(dd[0], dd[-1])
    ax2_inset.set_yticks([300, 700, 1100])
    ax2_inset.set_ylabel('φ [MV]', fontsize=7)
    ax2_inset.tick_params(labelbottom=False, labelsize=7)
    ax2_inset.grid(alpha=0.2)

    # --- (d) zonal-mean % diff by latitude (time-mean ± std) — capped at ±100%
    ax3 = axes[3]
    zm_pct_mean = zm_pct.mean(axis=1)
    zm_pct_std  = zm_pct.std(axis=1)
    # clip display range
    x_lo = np.clip(zm_pct_mean - zm_pct_std, -pct_cap_hov, pct_cap_hov)
    x_hi = np.clip(zm_pct_mean + zm_pct_std, -pct_cap_hov, pct_cap_hov)
    ax3.fill_betweenx(lat, x_lo, x_hi, alpha=0.25, color='#1a5fb4')
    ax3.plot(np.clip(zm_pct_mean, -pct_cap_hov, pct_cap_hov), lat,
             color='#1a5fb4', lw=1.8)
    ax3.axvline(0, color='k', lw=0.8, ls='--')
    ax3.set_xlabel('Zonal-mean relative diff [%]\n[capped ±100%]')
    ax3.set_ylabel('Latitude [°]')
    ax3.set_title('(d) Time-mean zonal-mean % diff\n± 1σ across months')
    ax3.grid(alpha=0.3)
    ax3.set_ylim(-90, 90)
    ax3.set_yticks(np.arange(-90, 91, 30))
    ax3.set_xlim(-pct_cap_hov * 1.05, pct_cap_hov * 1.05)
    # annotation at the latitude of zero crossing
    crossing = lat[np.argmin(np.abs(zm_pct_mean))]
    ax3.axhline(crossing, color='k', lw=0.6, ls=':', alpha=0.6)
    ax3.text(55, crossing + 2, f'zero crossing ≈ {crossing:.0f}°',
             fontsize=8, color='k')
    # note about polar saturation
    ax3.text(0.97, 0.01,
             'Polar cells (|lat|>75°)\nsaturate; Miller→0 there',
             transform=ax3.transAxes, ha='right', va='bottom', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))

    fig.suptitle('Zonal comparison: this work vs. Miller et al. (2025)', fontsize=13)
    plt.tight_layout()
    out = outdir / 'fig3_zonal_mean.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


# ── annual cycle figure ───────────────────────────────────────────────────────

def fig_annual_cycle(lat, lon, dd, phi, ours, miller, outdir):
    """Monthly climatology (mean across years) — global total."""
    months = np.round((dd % 1) * 12 + 0.5).astype(int)  # 1..12

    g_ours   = global_sum(ours)   * 12.0
    g_miller = global_sum(miller) * 12.0

    clim_o = np.array([g_ours[months == m].mean()   for m in range(1, 13)])
    clim_m = np.array([g_miller[months == m].mean() for m in range(1, 13)])
    clim_phi = np.array([phi[months == m].mean()    for m in range(1, 13)])

    mon_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec']
    x = np.arange(1, 13)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax.plot(x, clim_m, 'o-', color='#e07020', lw=1.8, label='Miller et al. (2025)')
    ax.plot(x, clim_o, 's--', color='#1a5fb4', lw=1.8, label='This work')
    ax.set_ylabel('Global isoflux [TgC ‰ yr⁻¹ equiv.]')
    ax.set_title('(a) Annual cycle of global total ¹⁴C isoflux\nClimatology 2000–2012')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)

    ax3 = ax.twinx()
    ax3.bar(x, clim_phi, alpha=0.15, color='gray', width=0.4)
    ax3.set_ylabel('Mean φ [MV]', color='gray')
    ax3.tick_params(axis='y', labelcolor='gray')

    pct = (clim_o - clim_m) / clim_m * 100.0
    ax2.bar(x, pct, color=['#c0392b' if p > 0 else '#2980b9' for p in pct], alpha=0.75)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_ylabel('(This work − Miller) / Miller [%]')
    ax2.set_xlabel('Month')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mon_labels)
    ax2.set_title('(b) Relative difference by calendar month')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Seasonal comparison: this work vs. Miller et al. (2025)', fontsize=13)
    plt.tight_layout()
    out = outdir / 'fig4_annual_cycle.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ours',    default='comparison/cosmo14C_2000_2012_miller_format.nc')
    p.add_argument('--miller',  default='Cosmo.nc')
    p.add_argument('--outdir',  default='figures/')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    lat, lon, dd, phi, ours, miller = load_both(args.ours, args.miller)
    print(f"  ours  : {ours.shape}  [{ours.min():.3e}, {ours.max():.3e}] TgC permil month-1")
    print(f"  miller: {miller.shape}  [{miller.min():.3e}, {miller.max():.3e}] TgC permil month-1")

    print("\nFigure 1: spatial mean maps …")
    fig_spatial(lat, lon, ours, miller, outdir)

    print("Figure 2: time series …")
    fig_time_series(lat, lon, dd, phi, ours, miller, outdir)

    print("Figure 3: zonal mean / Hovmöller …")
    fig_zonal(lat, lon, dd, phi, ours, miller, outdir)

    print("Figure 4: annual cycle …")
    fig_annual_cycle(lat, lon, dd, phi, ours, miller, outdir)

    print("\nDone.")


if __name__ == '__main__':
    main()
