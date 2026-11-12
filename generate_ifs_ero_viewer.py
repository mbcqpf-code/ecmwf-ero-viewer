import os
import gc
import json
import shutil
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from scipy.ndimage import maximum_filter
import xarray as xr
from ecmwf.opendata import Client

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & DOMAIN GEOMETRY
# ==========================================
OUTPUT_DIR = Path("output")
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
WORKDIR = Path("./temp_download")
WORKDIR.mkdir(exist_ok=True)
CLIMO_PATH = Path("data/era5_conus_pw_percentiles_1950_2025.nc")

MAX_RUNS_TO_KEEP = 10
MAX_NEW_ENS_RUNS_PER_JOB = 2

LAT_NORTH, LAT_SOUTH = 50.0, 24.0
LON_WEST, LON_EAST = -125.0, -66.0

# 50 km circular neighborhood footprint at 0.25 deg (~25 km) grid spacing -> radius = 2 cells
RADIUS_PTS = 2
y_g, x_g = np.ogrid[-RADIUS_PTS:RADIUS_PTS + 1, -RADIUS_PTS:RADIUS_PTS + 1]
CIRCULAR_FOOTPRINT_3D = ((x_g**2 + y_g**2) <= RADIUS_PTS**2).astype(np.uint8)[np.newaxis, :, :]

PROJ = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)

# ==========================================
# 2. DYNAMIC CYCLE & PERIOD RESOLVER
# ==========================================
def build_synoptic_periods(init_dt):
    """
    Constructs Day 1-5 synoptic periods anchored strictly to 12Z-to-12Z valid windows:
      - 00Z: F012–F132 (Days 1-5)
      - 06Z: F006–F126 (Days 1-5)
      - 12Z: F024–F144 (Days 1-5, adds new Day 5)
      - 18Z: F018–F138 (Days 1-5)
    """
    cycle = init_dt.hour
    first_12z_offset = (12 - cycle) % 24
    if first_12z_offset == 0:
        first_12z_offset = 24

    periods = []
    for day_num in range(1, 6):
        f_s = first_12z_offset + (day_num - 1) * 24
        f_e = f_s + 24
        v_s = init_dt + timedelta(hours=f_s)
        v_e = init_dt + timedelta(hours=f_e)
        end_str = v_e.strftime("%a %m/%d")
        periods.append({
            "id": f"p{day_num}",
            "day_num": day_num,
            "name": f"Ending 12Z {end_str}",
            "f_start": f_s,
            "f_end": f_e,
            "steps": list(range(f_s, f_e + 1, 3)),
            "valid_start": v_s,
            "valid_end": v_e,
            "midpoint_doy": (v_s + timedelta(hours=12)).timetuple().tm_yday,
        })
    return periods


def is_ecmwf_step_ready(client, date_str, cycle_hr, stream, req_type, step_num):
    """Probes ECMWF Open Data to verify whether the cycle has disseminated through step_num."""
    probe_file = WORKDIR / f"probe_{stream}_{date_str}_{cycle_hr:02d}_f{step_num}.grib2"
    try:
        client.retrieve(
            date=date_str,
            time=cycle_hr,
            stream=stream,
            type=req_type,
            levtype="sfc",
            step=step_num,
            param="tp",
            number=1 if req_type == "pf" else None,
            target=str(probe_file),
        )
        ready = probe_file.exists() and probe_file.stat().st_size > 1000
    except Exception:
        ready = False
    finally:
        if probe_file.exists():
            probe_file.unlink()
    return ready


def read_cycle_status(cycle_id):
    flag_file = RUNS_DIR / cycle_id / "complete.json"
    if flag_file.exists() and flag_file.stat().st_size > 0:
        try:
            with open(flag_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"cycle": cycle_id, "ens_complete": False, "det_complete": False}


def write_cycle_status(cycle_id, ens_complete, det_complete):
    run_dir = RUNS_DIR / cycle_id
    run_dir.mkdir(parents=True, exist_ok=True)
    flag_file = run_dir / "complete.json"
    with open(flag_file, "w") as f:
        json.dump({
            "cycle": cycle_id,
            "ens_complete": ens_complete,
            "det_complete": det_complete,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def get_pending_and_available_cycles(client_ens, max_lookback_hours=36):
    """
    Scans the last 36 hours for 00/06/12/18Z cycles where the 0.25° ensemble has
    finished through its Day 5 end step (126-144h) and has not yet been processed.
    Returns chronological list (oldest first).
    """
    now = datetime.now(timezone.utc)
    anchor = now.replace(minute=0, second=0, microsecond=0)
    pending = []

    for h_back in range(0, max_lookback_hours + 1, 6):
        cand = anchor - timedelta(hours=h_back)
        cycle_hour = (cand.hour // 6) * 6
        cand = cand.replace(hour=cycle_hour)
        date_str = cand.strftime("%Y%m%d")
        cycle_id = cand.strftime("%Y%m%d_%H")

        status = read_cycle_status(cycle_id)
        if status.get("ens_complete"):
            continue

        periods = build_synoptic_periods(cand)
        req_end_step = periods[-1]["f_end"]  # 132 (00Z), 126 (06Z), 144 (12Z), 138 (18Z)

        if is_ecmwf_step_ready(client_ens, date_str, cycle_hour, "enfo", "pf", req_end_step):
            pending.append(cand)

    pending.sort()
    return pending


# ==========================================
# 3. COLORMAPS & GRAPHICS RENDERING SYSTEM
# ==========================================
PROB_BOUNDS = [5, 10, 15, 20, 30, 40, 50, 70, 90]
PROB_COLORS = ["#7bed9f", "#2ed573", "#ffeaa7", "#fdcb6e", "#f39c12", "#e17055", "#d63031", "#e84393", "#8e44ad"]
CMAP_PROB = ListedColormap(PROB_COLORS)
NORM_PROB = BoundaryNorm(PROB_BOUNDS, CMAP_PROB.N, extend="max")

QPF_BOUNDS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00, 4.00]
QPF_COLORS = ["#b8e186", "#4dac26", "#6baed6", "#2171b5", "#08306b", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#762a83"]
CMAP_QPF = ListedColormap(QPF_COLORS)
NORM_QPF = BoundaryNorm(QPF_BOUNDS, CMAP_QPF.N, extend="max")

CAPE_BOUNDS = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]
CAPE_COLORS = ["#d9f0a3", "#addd8e", "#78c679", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#67001f"]
CMAP_CAPE = ListedColormap(CAPE_COLORS)
NORM_CAPE = BoundaryNorm(CAPE_BOUNDS, CMAP_CAPE.N, extend="max")

PW_BOUNDS = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50]
PW_COLORS = ["#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#034e7b", "#41ab5d", "#fe9929", "#ce1256"]
CMAP_PW = ListedColormap(PW_COLORS)
NORM_PW = BoundaryNorm(PW_BOUNDS, CMAP_PW.N, extend="max")

RANK_BOUNDS = [75.0, 85.0, 90.0, 95.0, 97.5, 99.0, 99.5, 100.0]
RANK_COLORS = ["#bdd7e7", "#6baed6", "#3182bd", "#41ab5d", "#feb24c", "#f03b20", "#ae017e"]
CMAP_RANK = ListedColormap(RANK_COLORS)
NORM_RANK = BoundaryNorm(RANK_BOUNDS, CMAP_RANK.N, extend="neither")

IMFC_BOUNDS = [2, 5, 10, 15, 20, 30, 40, 50, 65, 80]
IMFC_COLORS = ["#e0f3db", "#ccebc5", "#a8ddb5", "#7bccc4", "#4eb3d3", "#2b8cbe", "#08589e", "#feb24c", "#f03b20", "#bd0026"]
CMAP_IMFC = ListedColormap(IMFC_COLORS)
NORM_IMFC = BoundaryNorm(IMFC_BOUNDS, CMAP_IMFC.N, extend="max")


def save_quantized_png(fig, out_path):
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    with Image.open(out_path) as im:
        im_p = im.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
        im_p.save(out_path, optimize=True)


def render_plot(data, lats, lons, title, subtitle, cbar_label, filename, out_dir, plot_type="prob", placeholder_msg=None):
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=110, subplot_kw={"projection": PROJ})
    ax.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax.set_extent([-120, -72, 22, 50], crs=ccrs.PlateCarree())

    if plot_type == "qpf":
        cmap, norm, bounds, extend = CMAP_QPF, NORM_QPF, QPF_BOUNDS, "max"
        tick_labels = [f'{b:g}"' for b in bounds]
    elif plot_type == "cape":
        cmap, norm, bounds, extend = CMAP_CAPE, NORM_CAPE, CAPE_BOUNDS, "max"
        tick_labels = [f"{int(b)}" for b in bounds]
    elif plot_type == "pw":
        cmap, norm, bounds, extend = CMAP_PW, NORM_PW, PW_BOUNDS, "max"
        tick_labels = [f'{b:.2f}"' for b in bounds]
    elif plot_type == "rank":
        cmap, norm, bounds, extend = CMAP_RANK, NORM_RANK, RANK_BOUNDS, "neither"
        tick_labels = [f"{b:g}%" for b in bounds]
    else:
        cmap, norm, bounds, extend = CMAP_PROB, NORM_PROB, PROB_BOUNDS, "max"
        tick_labels = [f"{int(b)}%" for b in bounds]

    if placeholder_msg:
        ax.text(0.5, 0.5, placeholder_msg, transform=ax.transAxes,
                fontsize=15, color="#475569", ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#f1f5f9", edgecolor="#cbd5e1"))
    elif data is not None and np.any(~np.isnan(data)):
        ax.contourf(lons, lats, data, transform=ccrs.PlateCarree(),
                    levels=bounds, cmap=cmap, norm=norm, extend=extend)
    else:
        ax.text(0.5, 0.5, "NO THREAT DETECTED IN THIS WINDOW", transform=ax.transAxes,
                fontsize=18, color="gray", ha="center", va="center", fontweight="bold", alpha=0.6)

    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", loc="left", linespacing=1.3)

    cbar_ax = fig.add_axes([0.18, 0.06, 0.64, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend=extend)
    cbar.set_ticks(bounds)
    cbar.ax.set_xticklabels(tick_labels, fontsize=9.5)
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold")

    plt.subplots_adjust(bottom=0.12)
    save_quantized_png(fig, out_dir / filename)


def render_mfc850_plot(imfc_data, mean_qu, mean_qv, prob_persist, lats, lons, title, subtitle, cbar_label, filename, out_dir):
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=110, subplot_kw={"projection": PROJ})
    ax.add_feature(cfeature.LAND, facecolor="#f8f9fa")
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#333333")
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#666666", linestyle=":")
    ax.set_extent([-120, -72, 22, 50], crs=ccrs.PlateCarree())

    disp_imfc = np.where(imfc_data >= 2.0, imfc_data, np.nan)
    if np.any(~np.isnan(disp_imfc)):
        ax.contourf(lons, lats, disp_imfc, transform=ccrs.PlateCarree(),
                    levels=IMFC_BOUNDS, cmap=CMAP_IMFC, norm=NORM_IMFC, extend="max")

    skip = 6
    LON_2D, LAT_2D = np.meshgrid(lons, lats)
    mag = np.hypot(mean_qu, mean_qv)
    q_mask = mag[::skip, ::skip] >= 40.0

    quiv = ax.quiver(
        LON_2D[::skip, ::skip][q_mask],
        LAT_2D[::skip, ::skip][q_mask],
        mean_qu[::skip, ::skip][q_mask],
        mean_qv[::skip, ::skip][q_mask],
        color="#111111", scale=2200, width=0.0022, headwidth=3.8, headlength=4.5,
        transform=ccrs.PlateCarree(), zorder=5
    )
    ax.quiverkey(
        quiv, X=0.89, Y=1.02, U=150, label="150 g/kg·m/s", labelpos="E",
        coordinates="axes", fontproperties={"size": 9, "weight": "bold"}
    )

    if prob_persist.max() >= 30.0:
        cs = ax.contour(
            lons, lats, prob_persist, levels=[30, 60],
            colors=["#880e4f", "#4a148c"], linewidths=[1.2, 1.8], linestyles=["--", "-"],
            transform=ccrs.PlateCarree(), zorder=6
        )
        ax.clabel(cs, fmt="%d%% ≥9h", inline=True, fontsize=8)

    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", loc="left", linespacing=1.3)

    cbar_ax = fig.add_axes([0.18, 0.06, 0.64, 0.025])
    sm = plt.cm.ScalarMappable(cmap=CMAP_IMFC, norm=NORM_IMFC)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="max")
    cbar.set_ticks(IMFC_BOUNDS)
    cbar.ax.set_xticklabels([f"{b}" for b in IMFC_BOUNDS], fontsize=9.5)
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold")

    plt.subplots_adjust(bottom=0.12)
    save_quantized_png(fig, out_dir / filename)


# ==========================================
# 4. CLIMATOLOGY PERCENTILE RANK HELPER
# ==========================================
def compute_pw_percentile_rank(pw_fcst_2d, pct_vals, thresh_3d):
    rank = np.full_like(pw_fcst_2d, 50.0, dtype=np.float32)
    for i in range(len(pct_vals) - 1):
        low_v, high_v = thresh_3d[i], thresh_3d[i + 1]
        low_p, high_p = pct_vals[i], pct_vals[i + 1]
        mask = (pw_fcst_2d >= low_v) & (pw_fcst_2d < high_v)
        frac = (pw_fcst_2d[mask] - low_v[mask]) / np.maximum(high_v[mask] - low_v[mask], 1e-6)
        rank[mask] = low_p + frac * (high_p - low_p)
    rank[pw_fcst_2d >= thresh_3d[-1]] = 100.0
    return rank


# ==========================================
# 5. ENSEMBLE (0.25°) & 9KM DETERMINISTIC ENGINES
# ==========================================
def process_ensemble_cycle(client_ens, init_dt, climo_ds):
    cycle_id = init_dt.strftime("%Y%m%d_%H")
    date_str = init_dt.strftime("%Y%m%d")
    cycle_hr = init_dt.hour
    current_run_dir = RUNS_DIR / cycle_id
    current_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"PROCESSING ECMWF 0.25° ENSEMBLE: {cycle_id}")
    print(f"==========================================")
    t0 = time.time()

    periods = build_synoptic_periods(init_dt)
    all_steps = sorted(list({s for p in periods for s in p["steps"]}))
    R_EARTH = 6371000.0

    period_state = {}
    lats, lons, lats_rad, lons_rad, cos_lat = None, None, None, None, None
    tp_prev = None

    for step in all_steps:
        t_s = time.time()
        sfc_file = WORKDIR / f"ens_sfc_{cycle_id}_f{step:03d}.grib2"
        pl_file = WORKDIR / f"ens_pl_{cycle_id}_f{step:03d}.grib2"

        client_ens.retrieve(
            date=date_str, time=cycle_hr, stream="enfo", type="pf",
            levtype="sfc", step=step, param=["tp", "mucape", "tcwv"], target=str(sfc_file)
        )
        client_ens.retrieve(
            date=date_str, time=cycle_hr, stream="enfo", type="pf",
            levtype="pl", levelist=850, step=step, param=["u", "v", "q"], target=str(pl_file)
        )

        with xr.open_dataset(sfc_file, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "tp"}, "indexpath": ""}) as ds_tp:
            w_lon = LON_WEST % 360.0 if ds_tp.longitude.max() > 180 else LON_WEST
            e_lon = LON_EAST % 360.0 if ds_tp.longitude.max() > 180 else LON_EAST
            tp_da = ds_tp["tp"].sel(latitude=slice(LAT_NORTH, LAT_SOUTH), longitude=slice(w_lon, e_lon)).load()
            if tp_da.longitude.max() > 180:
                tp_da = tp_da.assign_coords(longitude=tp_da.longitude - 360.0)

        with xr.open_dataset(sfc_file, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "mucape"}, "indexpath": ""}) as ds_cape:
            cape_da = ds_cape["mucape"].sel(latitude=slice(LAT_NORTH, LAT_SOUTH), longitude=slice(w_lon, e_lon)).load()

        with xr.open_dataset(sfc_file, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "tcwv"}, "indexpath": ""}) as ds_pw:
            pw_da = ds_pw["tcwv"].sel(latitude=slice(LAT_NORTH, LAT_SOUTH), longitude=slice(w_lon, e_lon)).load()

        with xr.open_dataset(pl_file, engine="cfgrib", backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": 850}, "indexpath": ""}) as ds_pl:
            pl_sub = ds_pl[["u", "v", "q"]].sel(latitude=slice(LAT_NORTH, LAT_SOUTH), longitude=slice(w_lon, e_lon)).load()
            q_gkg = pl_sub["q"].values.astype(np.float32) * 1000.0
            u_ms = pl_sub["u"].values.astype(np.float32)
            v_ms = pl_sub["v"].values.astype(np.float32)

        sfc_file.unlink(missing_ok=True)
        pl_file.unlink(missing_ok=True)

        tp_curr = tp_da.values.astype(np.float32)
        cape_curr = cape_da.values.astype(np.float32)
        pw_curr = pw_da.values.astype(np.float32) / 25.4  # inches

        if lats is None:
            lats = tp_da.latitude.values
            lons = tp_da.longitude.values
            lats_rad = np.deg2rad(lats)
            lons_rad = np.deg2rad(lons)
            cos_lat = np.cos(lats_rad)[np.newaxis, :, np.newaxis]
            n_mem, n_lat, n_lon = tp_curr.shape
            for p in periods:
                period_state[p["id"]] = {
                    "max_3h_qpf": np.zeros((n_mem, n_lat, n_lon), dtype=np.float32),
                    "max_cape": np.zeros((n_mem, n_lat, n_lon), dtype=np.float32),
                    "max_pw": np.zeros((n_mem, n_lat, n_lon), dtype=np.float32),
                    "imfc_24h": np.zeros((n_mem, n_lat, n_lon), dtype=np.float32),
                    "sum_qu": np.zeros((n_mem, n_lat, n_lon), dtype=np.float32),
                    "sum_qv": np.zeros((n_mem, n_lat, n_lon), dtype=np.float32),
                    "persist_hrs": np.zeros((n_mem, n_lat, n_lon), dtype=np.float32),
                }

        delta_3h = np.clip((tp_curr - tp_prev) * 39.3701, 0.0, None) if tp_prev is not None else None
        tp_prev = tp_curr

        qu = ndimage.gaussian_filter(q_gkg * u_ms, sigma=(0, 1.2, 1.2))
        qv = ndimage.gaussian_filter(q_gkg * v_ms, sigma=(0, 1.2, 1.2))
        div_qV = (np.gradient(qu, lons_rad, axis=2) + np.gradient(qv * cos_lat, lats_rad, axis=1)) / (R_EARTH * cos_lat)
        mfc_hr = -div_qV * 3600.0
        qV_mag = np.hypot(qu, qv)

        for p in periods:
            st = period_state[p["id"]]
            s0, s1 = p["f_start"], p["f_end"]
            if s0 <= step <= s1:
                st["max_cape"] = np.maximum(st["max_cape"], cape_curr)
                st["max_pw"] = np.maximum(st["max_pw"], pw_curr)

                w = 1.5 if (step == s0 or step == s1) else 3.0
                st["imfc_24h"] += np.maximum(0.0, mfc_hr) * w
                st["sum_qu"] += qu * w
                st["sum_qv"] += qv * w
                st["persist_hrs"] += np.where((qV_mag >= 150.0) & (mfc_hr >= 0.5), w, 0.0)

            if delta_3h is not None and (s0 < step <= s1):
                st["max_3h_qpf"] = np.maximum(st["max_3h_qpf"], delta_3h)

        print(f"  [0.25° ENS] Step F{step:03d} processed in {time.time() - t_s:.1f}s")

    pct_vals = climo_ds["percentile"].values
    init_lbl = init_dt.strftime("%Y%m%d %H") + "z"

    for p in periods:
        p_id = p["id"]
        st = period_state[p_id]
        valid_str = f"Valid: {p['valid_start'].strftime('%Y-%m-%d %H')}Z – {p['valid_end'].strftime('%Y-%m-%d %H')}Z"
        sub_title = f"{p['name']} (Day {p['day_num']}: F{p['f_start']:03d}–F{p['f_end']:03d}) | {valid_str} (50-Member 0.25° IFS)"

        # 1. 24-hr Max 3-hr QPF Neighborhood Probabilities (0.5", 1.0", 1.5", 2.0") & 90th Pct
        neigh_qpf = maximum_filter(st["max_3h_qpf"], footprint=CIRCULAR_FOOTPRINT_3D, mode="nearest")
        for thresh, tag in [(0.5, "050"), (1.0, "100"), (1.5, "150"), (2.0, "200")]:
            raw_prob = (neigh_qpf >= thresh).mean(axis=0) * 100.0
            disp_prob = np.where(raw_prob >= 5.0, raw_prob, np.nan)
            render_plot(
                disp_prob, lats, lons,
                f"ECMWF IFS {init_lbl}: Prob of Max 3-hr QPF ≥ {thresh}\" (50-km Neighborhood)",
                sub_title, f"50-km Neighborhood Probability of 3-hr QPF ≥ {thresh}\" (%)",
                f"{p_id}_qpf_prob_{tag}.png", current_run_dir, plot_type="prob"
            )

        qpf_p90 = np.percentile(neigh_qpf, 90, axis=0)
        disp_qpf_p90 = np.where(qpf_p90 >= 0.10, qpf_p90, np.nan)
        render_plot(
            disp_qpf_p90, lats, lons,
            f"ECMWF IFS {init_lbl}: 90th Percentile Max 3-hr QPF (50-km Neighborhood)",
            sub_title, "90th Percentile 3-hr Rainfall Rate within ~50 km (Inches / 3h)",
            f"{p_id}_qpf_p90.png", current_run_dir, plot_type="qpf"
        )

        # Render placeholder for 9km deterministic QPF until the ~2-hr delayed 0.1° run arrives
        render_plot(
            None, lats, lons,
            f"ECMWF 9km (0.1°) {init_lbl}: 24-hr Maximum 3-Hourly QPF",
            f"{p['name']} (Day {p['day_num']}: F{p['f_start']:03d}–F{p['f_end']:03d}) | {valid_str} (0.1° Deterministic IFS)",
            "9 km Deterministic Peak 3-hr Rainfall Rate (Inches / 3h)",
            f"{p_id}_qpf_det_9km.png", current_run_dir, plot_type="qpf",
            placeholder_msg="9 KM DETERMINISTIC IFS PENDING (~2 HR DISSEMINATION DELAY)"
        )

        # 2. CAPE: 24-hr Max MUCAPE (Ensemble Mean & 90th Percentile)
        cape_mean = st["max_cape"].mean(axis=0)
        cape_p90 = np.percentile(st["max_cape"], 90, axis=0)
        render_plot(
            np.where(cape_mean >= 100.0, cape_mean, np.nan), lats, lons,
            f"ECMWF IFS {init_lbl}: Ensemble Mean 24-hr Max MUCAPE",
            sub_title, "Ensemble Mean 24-hr Max MUCAPE (J/kg)",
            f"{p_id}_cape_mean.png", current_run_dir, plot_type="cape"
        )
        render_plot(
            np.where(cape_p90 >= 100.0, cape_p90, np.nan), lats, lons,
            f"ECMWF IFS {init_lbl}: 90th Percentile 24-hr Max MUCAPE",
            sub_title, "90th Percentile 24-hr Max MUCAPE (J/kg)",
            f"{p_id}_cape_p90.png", current_run_dir, plot_type="cape"
        )

        # 3. PW: 24-hr Max Precipitable Water Raw (Mean & 90th Percentile)
        pw_mean = st["max_pw"].mean(axis=0)
        pw_p90 = np.percentile(st["max_pw"], 90, axis=0)
        render_plot(
            np.where(pw_mean >= 0.50, pw_mean, np.nan), lats, lons,
            f"ECMWF IFS {init_lbl}: Ensemble Mean 24-hr Max Precipitable Water",
            sub_title, "Ensemble Mean 24-hr Max PWAT (Inches)",
            f"{p_id}_pw_mean.png", current_run_dir, plot_type="pw"
        )
        render_plot(
            np.where(pw_p90 >= 0.50, pw_p90, np.nan), lats, lons,
            f"ECMWF IFS {init_lbl}: 90th Percentile 24-hr Max Precipitable Water",
            sub_title, "90th Percentile 24-hr Max PWAT (Inches)",
            f"{p_id}_pw_p90.png", current_run_dir, plot_type="pw"
        )

        # 4. PW Anomaly vs ERA5 1950–2025 Climatology
        climo_slice = climo_ds["pw_pct"].sel(dayofyear=p["midpoint_doy"]).sel(
            latitude=lats, longitude=lons, method="nearest"
        ).values
        thresh_p90 = climo_slice[np.where(pct_vals == 90.0)[0][0]]
        thresh_p99 = climo_slice[np.where(pct_vals == 99.0)[0][0]]

        rank_mean = compute_pw_percentile_rank(pw_mean, pct_vals, climo_slice)
        rank_p90 = compute_pw_percentile_rank(pw_p90, pct_vals, climo_slice)
        render_plot(
            np.where(rank_mean >= 75.0, rank_mean, np.nan), lats, lons,
            f"ECMWF IFS {init_lbl}: Ensemble Mean PWAT Historical Percentile (1950–2025 ERA5)",
            sub_title, f"ERA5 1950–2025 Climatological Percentile Rank (DOY {p['midpoint_doy']})",
            f"{p_id}_pw_anom_mean.png", current_run_dir, plot_type="rank"
        )
        render_plot(
            np.where(rank_p90 >= 75.0, rank_p90, np.nan), lats, lons,
            f"ECMWF IFS {init_lbl}: 90th Percentile PWAT Historical Percentile (1950–2025 ERA5)",
            sub_title, f"ERA5 1950–2025 Climatological Percentile Rank (DOY {p['midpoint_doy']})",
            f"{p_id}_pw_anom_p90.png", current_run_dir, plot_type="rank"
        )

        prob_pw90 = (st["max_pw"] >= thresh_p90[np.newaxis, :, :]).mean(axis=0) * 100.0
        prob_pw99 = (st["max_pw"] >= thresh_p99[np.newaxis, :, :]).mean(axis=0) * 100.0
        render_plot(
            np.where(prob_pw90 >= 5.0, prob_pw90, np.nan), lats, lons,
            f"ECMWF IFS {init_lbl}: Probability of Exceeding 90th Climo Percentile PWAT",
            sub_title, "Ensemble Probability of Max PWAT > 90th Historical Percentile (%)",
            f"{p_id}_pw_prob_p90.png", current_run_dir, plot_type="prob"
        )
        render_plot(
            np.where(prob_pw99 >= 5.0, prob_pw99, np.nan), lats, lons,
            f"ECMWF IFS {init_lbl}: Probability of Exceeding 99th Climo Percentile PWAT",
            sub_title, "Ensemble Probability of Max PWAT > 99th Historical Percentile (%)",
            f"{p_id}_pw_prob_p99.png", current_run_dir, plot_type="prob"
        )

        # 5. 850 hPa Moisture Transport Convergence & Mean qV (Mean & 90th Pct)
        mean_qu = (st["sum_qu"] / 24.0).mean(axis=0)
        mean_qv = (st["sum_qv"] / 24.0).mean(axis=0)
        prob_persist = (st["persist_hrs"] >= 9.0).mean(axis=0) * 100.0

        render_mfc850_plot(
            st["imfc_24h"].mean(axis=0), mean_qu, mean_qv, prob_persist, lats, lons,
            f"ECMWF IFS {init_lbl}: Ensemble Mean 24-hr Integrated 850 hPa MFC & Mean qV",
            sub_title, "24-hr Integrated 850 hPa Moisture Convergence (g/kg) & Vector-Mean qV",
            f"{p_id}_mfc850_mean.png", current_run_dir
        )
        render_mfc850_plot(
            np.percentile(st["imfc_24h"], 90, axis=0), mean_qu, mean_qv, prob_persist, lats, lons,
            f"ECMWF IFS {init_lbl}: 90th Percentile 24-hr Integrated 850 hPa MFC & Mean qV",
            sub_title, "90th Percentile 24-hr Integrated 850 hPa MFC (g/kg) & Vector-Mean qV",
            f"{p_id}_mfc850_p90.png", current_run_dir
        )

    del period_state
    gc.collect()
    write_cycle_status(cycle_id, ens_complete=True, det_complete=False)
    print(f"=== Completed 0.25° Ensemble {cycle_id} in {time.time() - t0:.1f}s ===")


def process_deterministic_9km_cycle(client_det, init_dt):
    cycle_id = init_dt.strftime("%Y%m%d_%H")
    date_str = init_dt.strftime("%Y%m%d")
    cycle_hr = init_dt.hour
    current_run_dir = RUNS_DIR / cycle_id

    print(f"\n==========================================")
    print(f"PROCESSING ECMWF 9KM (0.1°) DET QPF: {cycle_id}")
    print(f"==========================================")
    t0 = time.time()

    periods = build_synoptic_periods(init_dt)
    all_steps = sorted(list({s for p in periods for s in p["steps"]}))
    day_max_qpf = {}
    lats, lons, tp_prev = None, None, None

    for step in all_steps:
        fname = WORKDIR / f"det_tp_{cycle_id}_f{step:03d}.grib2"
        client_det.retrieve(
            date=date_str, time=cycle_hr, stream="oper", type="fc",
            levtype="sfc", step=step, param="tp", target=str(fname)
        )
        with xr.open_dataset(fname, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "tp"}, "indexpath": ""}) as ds:
            w_lon = LON_WEST % 360.0 if ds.longitude.max() > 180 else LON_WEST
            e_lon = LON_EAST % 360.0 if ds.longitude.max() > 180 else LON_EAST
            tp_da = ds["tp"].sel(latitude=slice(LAT_NORTH, LAT_SOUTH), longitude=slice(w_lon, e_lon)).load()
            if tp_da.longitude.max() > 180:
                tp_da = tp_da.assign_coords(longitude=tp_da.longitude - 360.0)
        fname.unlink(missing_ok=True)

        tp_curr = tp_da.values.astype(np.float32)
        if lats is None:
            lats = tp_da.latitude.values
            lons = tp_da.longitude.values
            for p in periods:
                day_max_qpf[p["id"]] = np.zeros((len(lats), len(lons)), dtype=np.float32)

        if tp_prev is not None:
            delta_3h = np.clip((tp_curr - tp_prev) * 39.3701, 0.0, None)
            for p in periods:
                if p["f_start"] < step <= p["f_end"]:
                    day_max_qpf[p["id"]] = np.maximum(day_max_qpf[p["id"]], delta_3h)
        tp_prev = tp_curr

    init_lbl = init_dt.strftime("%Y%m%d %H") + "z"
    for p in periods:
        p_id = p["id"]
        valid_str = f"Valid: {p['valid_start'].strftime('%Y-%m-%d %H')}Z – {p['valid_end'].strftime('%Y-%m-%d %H')}Z"
        sub_title = f"{p['name']} (Day {p['day_num']}: F{p['f_start']:03d}–F{p['f_end']:03d}) | {valid_str} (9 km Deterministic IFS)"
        disp_qpf = np.where(day_max_qpf[p_id] >= 0.10, day_max_qpf[p_id], np.nan)
        render_plot(
            disp_qpf, lats, lons,
            f"ECMWF 9km (0.1°) {init_lbl}: 24-hr Maximum 3-Hourly QPF",
            sub_title, "9 km Deterministic Peak 3-hr Rainfall Rate (Inches / 3h)",
            f"{p_id}_qpf_det_9km.png", current_run_dir, plot_type="qpf"
        )

    write_cycle_status(cycle_id, ens_complete=True, det_complete=True)
    print(f"=== Completed 9km Deterministic {cycle_id} in {time.time() - t0:.1f}s ===")


# ==========================================
# 6. MANIFEST & FRONTEND HTML BUILDER
# ==========================================
def update_manifest_and_html():
    all_runs = sorted([d for d in os.listdir(RUNS_DIR) if (RUNS_DIR / d).is_dir()], reverse=True)
    if len(all_runs) > MAX_RUNS_TO_KEEP:
        for old in all_runs[MAX_RUNS_TO_KEEP:]:
            shutil.rmtree(RUNS_DIR / old, ignore_errors=True)
    all_runs = all_runs[:MAX_RUNS_TO_KEEP]

    runs_meta = []
    for r in all_runs:
        try:
            r_dt = datetime.strptime(r, "%Y%m%d_%H").replace(tzinfo=timezone.utc)
            r_periods = build_synoptic_periods(r_dt)
            status = read_cycle_status(r)
            runs_meta.append({
                "id": r,
                "label": r_dt.strftime("%Y-%m-%d %Hz"),
                "init": r_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                "det_complete": bool(status.get("det_complete", False)),
                "periods": [
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "valid_start": p["valid_start"].strftime("%Y-%m-%dT%H:00:00Z"),
                        "valid_end": p["valid_end"].strftime("%Y-%m-%dT%H:00:00Z"),
                    }
                    for p in r_periods
                ],
            })
        except Exception:
            continue

    manifest = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "latest": all_runs[0] if all_runs else "",
        "runs": runs_meta,
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    build_frontend_html(OUTPUT_DIR / "index.html")


def build_frontend_html(dest_path):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ECMWF IFS Day 1–5 Excessive Rainfall Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #090d16;
      --panel: #131b2e;
      --border: #232f48;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #2563eb;
      --accent-hover: #3b82f6;
    }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex; flex-direction: column; align-items: center;
    }
    .header { width: 100%; max-width: 1320px; padding: 10px 16px 2px 16px; text-align: center; }
    .header h2 { margin: 0 0 2px 0; font-size: 1.25rem; }
    .header p { margin: 0; color: var(--muted); font-size: 0.8rem; }

    .toolbar {
      width: 100%; max-width: 1320px; background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px 12px; margin: 6px 0 8px 0; box-sizing: border-box;
      display: flex; flex-direction: column; gap: 6px;
    }
    .row {
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    }
    .btn-group {
      display: flex; gap: 4px; align-items: center; flex-wrap: wrap;
    }
    .btn-group b {
      font-size: 11.5px; color: #e2e8f0; margin-right: 2px; white-space: nowrap;
    }
    .btn {
      background: #1e293b; color: #cbd5e1; border: 1px solid var(--border);
      padding: 4px 8px; border-radius: 4px; font-size: 11.5px; font-weight: 600; cursor: pointer; transition: 0.15s;
      white-space: nowrap; line-height: 1.3;
    }
    .btn:hover:not(:disabled) { background: #334155; color: #fff; }
    .btn.active { background: var(--accent); color: #fff; border-color: var(--accent-hover); }
    .btn:disabled {
      opacity: 0.35; cursor: not-allowed; border-color: #1e293b; color: #64748b;
    }
    .btn-det { background: #1e3a29; border-color: #2e593e; color: #a7f3d0; }
    .btn-det:hover:not(:disabled) { background: #264e37; color: #fff; }
    .btn-det.active { background: #059669; border-color: #10b981; color: #fff; }
    .btn-help { background: #334155; color: #93c5fd; border-color: #475569; }
    .btn-help:hover { background: #475569; color: #fff; }

    .toolbar-divider {
      border-left: 1px solid var(--border);
      margin: 0 3px;
      height: 16px;
      display: inline-block;
    }

    select {
      background: #0f172a; color: #fff; border: 1px solid var(--border);
      padding: 3px 7px; border-radius: 4px; font-size: 11.5px; cursor: pointer;
    }

    .viewer-container {
      width: 100%; max-width: 1320px; display: flex; gap: 12px; justify-content: center;
    }
    .viewer-card {
      flex: 1; background: #000; border-radius: 8px; overflow: hidden;
      box-shadow: 0 10px 25px rgba(0,0,0,0.6); position: relative; border: 1px solid var(--border);
      display: flex; flex-direction: column;
    }
    .pane-toolbar {
      display: none; background: #0f172a; border-bottom: 1px solid var(--border);
      padding: 6px 10px; font-size: 11.5px; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 6px;
    }
    .viewer-card img { width: 100%; height: auto; display: block; }

    /* Modal Help Styling */
    .modal-overlay {
      display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0, 0, 0, 0.75); z-index: 9999; justify-content: center; align-items: center;
    }
    .modal-overlay.open { display: flex; }
    .modal-content {
      background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
      width: 90%; max-width: 850px; max-height: 85vh; overflow-y: auto; padding: 20px 24px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.8); color: var(--text);
    }
    .modal-header {
      display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border);
      padding-bottom: 10px; margin-bottom: 14px;
    }
    .modal-header h3 { margin: 0; font-size: 1.2rem; color: #60a5fa; }
    .modal-close {
      background: transparent; border: none; font-size: 24px; color: var(--muted); cursor: pointer;
    }
    .modal-close:hover { color: #fff; }
    .help-grid { display: flex; flex-direction: column; gap: 12px; }
    .help-section {
      background: #0b1120; border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;
    }
    .help-section h4 { margin: 0 0 6px 0; font-size: 0.95rem; color: #38bdf8; }
    .help-item { margin-bottom: 6px; font-size: 0.85rem; line-height: 1.45; }
    .help-item:last-child { margin-bottom: 0; }
    .help-item b { color: #e2e8f0; }

    .footer { margin: 10px 0 18px 0; color: var(--muted); font-size: 11.5px; }
    kbd { background: #1e293b; border: 1px solid #334155; padding: 2px 5px; border-radius: 3px; }
  </style>
</head>
<body>
  <div class="header">
    <h2>ECMWF IFS Day 1–5 Excessive Rainfall & Moisture Guidance</h2>
    <p id="subStatus">50-Member 0.25° Ensemble + 9 km (0.1°) Deterministic IFS</p>
  </div>

  <div class="toolbar" id="mainToolbar">
    <!-- ROW 1: Temporal Controls (Model Run on left, Valid Period on right) -->
    <div class="row">
      <div class="btn-group">
        <b>Model Run:</b>
        <button class="btn" id="btnNewer" onclick="stepRun(-1)" title="Step to newer run (Arrow Up)">▲ Newer</button>
        <select id="runSelect" onchange="onMainRunChange()"></select>
        <button class="btn" id="btnOlder" onclick="stepRun(1)" title="Step to older run (Arrow Down)">▼ Older</button>
      </div>

      <div class="btn-group" id="periodBox">
        <b>Valid Period:</b>
        <div id="periodButtons" class="btn-group"></div>
      </div>
    </div>

    <!-- ROW 2: Meteorological Products (Left) & Layout / Help Controls (Right) -->
    <div class="row" style="border-top: 1px solid var(--border); padding-top: 6px;">
      <div class="btn-group" id="productGroup">
        <b>Product:</b>
        <button class="btn" data-prod="qpf_prob_050" onclick="setProduct(this)">0.5" 3h</button>
        <button class="btn active" data-prod="qpf_prob_100" onclick="setProduct(this)">1.0" 3h</button>
        <button class="btn" data-prod="qpf_prob_150" onclick="setProduct(this)">1.5" 3h</button>
        <button class="btn" data-prod="qpf_prob_200" onclick="setProduct(this)">2.0" 3h</button>
        <button class="btn" data-prod="qpf_p90" onclick="setProduct(this)">90th QPF</button>
        <button class="btn btn-det" data-prod="qpf_det_9km" onclick="setProduct(this)">9km 3h</button>
        <span class="toolbar-divider"></span>
        <button class="btn" data-prod="cape_mean" onclick="setProduct(this)">CAPE Mean</button>
        <button class="btn" data-prod="cape_p90" onclick="setProduct(this)">CAPE 90th</button>
        <span class="toolbar-divider"></span>
        <button class="btn" data-prod="pw_mean" onclick="setProduct(this)">PW Mean</button>
        <button class="btn" data-prod="pw_p90" onclick="setProduct(this)">PW 90th</button>
        <span class="toolbar-divider"></span>
        <button class="btn" data-prod="pw_anom_mean" onclick="setProduct(this)">PW Pct (Mean)</button>
        <button class="btn" data-prod="pw_anom_p90" onclick="setProduct(this)">PW Pct (90th)</button>
        <button class="btn" data-prod="pw_prob_p90" onclick="setProduct(this)">Prob &gt;90th</button>
        <button class="btn" data-prod="pw_prob_p99" onclick="setProduct(this)">Prob &gt;99th</button>
        <span class="toolbar-divider"></span>
        <button class="btn" data-prod="mfc850_mean" onclick="setProduct(this)">850 MFC Mean</button>
        <button class="btn" data-prod="mfc850_p90" onclick="setProduct(this)">850 MFC 90th</button>
      </div>

      <div class="btn-group">
        <b>Layout:</b>
        <button class="btn active" id="btnSingle" onclick="setLayout('single')">Single</button>
        <button class="btn" id="btnSplit" onclick="setLayout('split')">Split</button>
        <button class="btn btn-help" onclick="toggleHelpModal(true)" title="View explanation of products (Shortcut: ? or H)">ℹ️ Guide</button>
      </div>
    </div>
  </div>

  <div class="viewer-container" id="viewerContainer">
    <!-- PANE 1 -->
    <div class="viewer-card" id="cardPrimary">
      <div class="pane-toolbar" id="pane1Bar">
        <span style="font-weight:bold; color:#60a5fa;">Pane 1 (Left):</span>
        <div style="display:flex; gap:6px; align-items:center;">
          <select id="pane1Run" onchange="onPane1RunChange()"></select>
          <select id="pane1Period" onchange="updatePane1()"></select>
          <select id="pane1Prod" onchange="updatePane1()"></select>
        </div>
      </div>
      <img id="imgPrimary" src="" alt="Primary IFS Graphic">
    </div>

    <!-- PANE 2 -->
    <div class="viewer-card" id="cardSecondary" style="display:none;">
      <div class="pane-toolbar" id="pane2Bar">
        <span style="font-weight:bold; color:#f59e0b;">Pane 2 (Right):</span>
        <div style="display:flex; gap:6px; align-items:center;">
          <select id="pane2Run" onchange="onPane2RunChange()"></select>
          <select id="pane2Period" onchange="updatePane2()"></select>
          <select id="pane2Prod" onchange="updatePane2()"></select>
        </div>
      </div>
      <img id="imgSecondary" src="" alt="Secondary IFS Graphic">
    </div>
  </div>

  <!-- HELP / PRODUCT GUIDE MODAL -->
  <div id="helpModal" class="modal-overlay" onclick="onModalOverlayClick(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3>ECMWF IFS Day 1–5 ERO Viewer — Product Guide</h3>
        <button class="modal-close" onclick="toggleHelpModal(false)">&times;</button>
      </div>
      <div class="help-grid">
        <div class="help-section">
          <h4>🌧️ 24-hr Max 3-hr QPF (0.25° 50-Member Ensemble & 9 km Deterministic)</h4>
          <div class="help-item">
            <b>0.5", 1.0", 1.5", 2.0" 3h Probabilities:</b> 50-km (~2 grid-cell) neighborhood probability that a member's peak 3-hourly rainfall rate exceeds the threshold during the 12Z–12Z synoptic window.
          </div>
          <div class="help-item">
            <b>90th QPF:</b> 90th percentile of the 50-km neighborhood maximum 3-hourly QPF across all 50 members (reasonable worst-case intensity ceiling).
          </div>
          <div class="help-item">
            <b>9km 3h:</b> High-resolution 0.1° (~9 km) deterministic IFS 24-hour maximum 3-hourly rainfall rate. Disseminated ~2 hours after the 0.25° ensemble and backfilled automatically.
          </div>
        </div>

        <div class="help-section">
          <h4>⚡ Convective Instability (MUCAPE)</h4>
          <div class="help-item">
            <b>CAPE Mean & CAPE 90th:</b> Ensemble mean and 90th percentile of each member's 24-hour maximum Most Unstable CAPE (J/kg), highlighting both surface-based warm sectors and elevated nocturnal instability.
          </div>
        </div>

        <div class="help-section">
          <h4>💧 Precipitable Water & ERA5 1950–2025 Climatological Percentiles</h4>
          <div class="help-item">
            <b>PW Mean & PW 90th:</b> Ensemble mean and 90th percentile 24-hour maximum Precipitable Water (inches).
          </div>
          <div class="help-item">
            <b>PW Pct (Mean) & PW Pct (90th):</b> Exact historical percentile rank (75th–100th) of the forecast PW plume compared against 76 years (1950–2025) of ERA5 daily max PW within a 15-day moving calendar window.
          </div>
          <div class="help-item">
            <b>Prob &gt;90th & Prob &gt;99th:</b> Percentage of ensemble members exceeding the local climatological 90th or 99th percentile PW threshold.
          </div>
        </div>

        <div class="help-section">
          <h4>🌬️ 850 hPa Moisture Transport Convergence & Persistence</h4>
          <div class="help-item">
            <b>850 MFC Mean & 90th:</b> Time-integrated positive 850 hPa Moisture Flux Convergence (g/kg) over 24 hours, overlaid with 24-hour vector-mean moisture transport (qV) arrows and magenta contours where ≥30% / ≥60% of members sustain strong convergent transport (|qV| ≥ 150 g/kg·m/s and MFC ≥ 0.5 g/kg/hr) for ≥9 hours—highlighting anchored LLJ noses favorable for training/backbuilding.
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    Shortcuts: <kbd>←</kbd> / <kbd>→</kbd> Change Product | <kbd>↑</kbd> / <kbd>↓</kbd> Trend Cycles (dprog/dt with Valid Time Lock) | <kbd>S</kbd> Split | <kbd>?</kbd> Help.
  </div>

  <script>
    let manifest = { runs: [] };
    let currentRunId = '';
    let currentProd = 'qpf_prob_100';
    let currentPeriod = 'p3';
    let currentValidEnd = '';
    let layoutMode = 'single';
    const preloadedImages = new Map();

    const PRODUCTS = [
      { id: 'qpf_prob_050', name: '0.5" 3h' },
      { id: 'qpf_prob_100', name: '1.0" 3h' },
      { id: 'qpf_prob_150', name: '1.5" 3h' },
      { id: 'qpf_prob_200', name: '2.0" 3h' },
      { id: 'qpf_p90', name: '90th QPF' },
      { id: 'qpf_det_9km', name: '9km 3h' },
      { id: 'cape_mean', name: 'CAPE Mean' },
      { id: 'cape_p90', name: 'CAPE 90th' },
      { id: 'pw_mean', name: 'PW Mean' },
      { id: 'pw_p90', name: 'PW 90th' },
      { id: 'pw_anom_mean', name: 'PW Pct (Mean)' },
      { id: 'pw_anom_p90', name: 'PW Pct (90th)' },
      { id: 'pw_prob_p90', name: 'Prob >90th' },
      { id: 'pw_prob_p99', name: 'Prob >99th' },
      { id: 'mfc850_mean', name: '850 MFC Mean' },
      { id: 'mfc850_p90', name: '850 MFC 90th' }
    ];

    async function init() {
      try {
        const res = await fetch('manifest.json?t=' + Date.now());
        manifest = await res.json();
        if (!manifest.runs || manifest.runs.length === 0) return;

        if (manifest.updated_utc) {
          document.getElementById('subStatus').textContent =
            `50-Member 0.25° Ensemble + 9 km (0.1°) Deterministic IFS  |  Last Checked: ${manifest.updated_utc}`;
        }

        currentRunId = manifest.latest || manifest.runs[0].id;
        const activeRun = manifest.runs.find(r => r.id === currentRunId) || manifest.runs[0];

        if (activeRun.periods.length > 0) {
          const defIdx = Math.min(2, activeRun.periods.length - 1);
          currentPeriod = activeRun.periods[defIdx].id;
          currentValidEnd = activeRun.periods[defIdx].valid_end;
        }

        const runSel = document.getElementById('runSelect');
        const optionsHtml = manifest.runs.map((r, i) => {
          const detTag = r.det_complete ? '' : ' [9km pending]';
          return `<option value="${r.id}">${r.label}${i === 0 ? ' (Latest)' : ''}${detTag}</option>`;
        }).join('');
        runSel.innerHTML = optionsHtml;
        runSel.value = currentRunId;

        ['pane1', 'pane2'].forEach(p => {
          document.getElementById(`${p}Run`).innerHTML = optionsHtml; document.getElementById(`${p}Prod`).innerHTML = PRODUCTS.map(pr =>
            `<option value="${pr.id}">${pr.name}</option>`
          ).join('');
        });

        renderPeriodButtons(activeRun);
        initSplitScreen(activeRun);
        preloadRunImages(currentRunId);
        updateSingleViewer();
        updateNavButtons();
      } catch (e) {
        console.error("Manifest load failed.", e);
      }
    }

    function renderPeriodButtons(runObj) {
      const pBox = document.getElementById('periodButtons');
      pBox.innerHTML = runObj.periods.map(p =>
        `<button class="btn ${p.id === currentPeriod ? 'active' : ''}" data-period="${p.id}" onclick="setPeriod('${p.id}')">${p.name}</button>` ).join(''); }  function initSplitScreen(activeRun) { document.getElementById('pane1Run').value = currentRunId; populatePeriodDropdown('pane1', activeRun); document.getElementById('pane1Period').value = currentPeriod; document.getElementById('pane1Prod').value = currentProd;  const secondaryRun = manifest.runs.length > 1 ? manifest.runs[1] : activeRun; document.getElementById('pane2Run').value = secondaryRun.id; populatePeriodDropdown('pane2', secondaryRun);  const matchP = secondaryRun.periods.find(p => p.valid_end === currentValidEnd); document.getElementById('pane2Period').value = matchP ? matchP.id : secondaryRun.periods[0].id; document.getElementById('pane2Prod').value = manifest.runs.length > 1 ? currentProd : 'qpf_det_9km'; }  function populatePeriodDropdown(panePrefix, runObj) { document.getElementById(`${panePrefix}Period`).innerHTML = runObj.periods.map(pr =>
        `<option value="${pr.id}">${pr.name}</option>`
      ).join('');
    }

    function updateNavButtons() {
      const btnNewer = document.getElementById('btnNewer');
      const btnOlder = document.getElementById('btnOlder');
      const curIdx = manifest.runs.findIndex(r => r.id === currentRunId);
      if (curIdx === -1) return;

      const hasNewer = (curIdx > 0) && manifest.runs[curIdx - 1].periods.some(p => p.valid_end === currentValidEnd);
      const hasOlder = (curIdx < manifest.runs.length - 1) && manifest.runs[curIdx + 1].periods.some(p => p.valid_end === currentValidEnd);

      btnNewer.disabled = !hasNewer;
      btnOlder.disabled = !hasOlder;
      btnNewer.title = !hasNewer ? "No newer run covers this ending valid time" : "Step to newer run (Arrow Up)";
      btnOlder.title = !hasOlder ? "Reached earliest run covering this ending valid time" : "Step to older run (Arrow Down)";
    }

    function changeRun(newRunId, targetPeriodId) {
      const newRunObj = manifest.runs.find(r => r.id === newRunId);
      if (!newRunObj) return;

      currentRunId = newRunId;
      document.getElementById('runSelect').value = newRunId;

      if (targetPeriodId) {
        currentPeriod = targetPeriodId;
        const pObj = newRunObj.periods.find(p => p.id === targetPeriodId);
        if (pObj) currentValidEnd = pObj.valid_end;
      } else {
        const match = newRunObj.periods.find(p => p.valid_end === currentValidEnd);
        if (match) {
          currentPeriod = match.id;
        } else {
          const fallback = newRunObj.periods[newRunObj.periods.length - 1];
          currentPeriod = fallback.id;
          currentValidEnd = fallback.valid_end;
        }
      }

      renderPeriodButtons(newRunObj);
      preloadRunImages(newRunId);
      updateSingleViewer();
      updateNavButtons();
    }

    function getImageUrl(runId, periodId, prod) {
      return `runs/${runId}/${periodId}_${prod}.png`; }  function preloadRunImages(runId) { const runObj = manifest.runs.find(r => r.id === runId); if (!runObj) return;  PRODUCTS.forEach(p => { runObj.periods.forEach(pr => { const u = getImageUrl(runId, pr.id, p.id); if (!preloadedImages.has(u)) { const im = new Image(); im.src = u; preloadedImages.set(u, im); } }); }); }  function setProduct(btn) { document.querySelectorAll('[data-prod]').forEach(b => b.classList.remove('active')); btn.classList.add('active'); currentProd = btn.getAttribute('data-prod'); updateSingleViewer(); updateNavButtons(); }  function setPeriod(pId) { currentPeriod = pId; const activeRun = manifest.runs.find(r => r.id === currentRunId); if (activeRun) { const pObj = activeRun.periods.find(p => p.id === pId); if (pObj) currentValidEnd = pObj.valid_end; renderPeriodButtons(activeRun); } updateSingleViewer(); updateNavButtons(); }  function onMainRunChange() { changeRun(document.getElementById('runSelect').value, null); }  function stepRun(delta) { const curIdx = manifest.runs.findIndex(r => r.id === currentRunId); if (curIdx === -1) return;  const nextIdx = curIdx + delta; if (nextIdx < 0 \vert{}\vert{} nextIdx >= manifest.runs.length) return;  const candidateRun = manifest.runs[nextIdx]; const matchP = candidateRun.periods.find(p => p.valid_end === currentValidEnd); if (!matchP) return;  changeRun(candidateRun.id, matchP.id); }  function stepProduct(delta) { const prodIds = PRODUCTS.map(p => p.id); let idx = prodIds.indexOf(currentProd) + delta; if (idx < 0) idx = prodIds.length - 1; if (idx >= prodIds.length) idx = 0; const nextBtn = document.querySelector(`[data-prod="${prodIds[idx]}"]`);
      if (nextBtn) setProduct(nextBtn);
    }

    function setLayout(mode) {
      layoutMode = mode;
      document.getElementById('btnSingle').classList.toggle('active', mode === 'single');
      document.getElementById('btnSplit').classList.toggle('active', mode === 'split');

      const isSplit = mode === 'split';
      document.getElementById('cardSecondary').style.display = isSplit ? 'flex' : 'none';
      document.getElementById('pane1Bar').style.display = isSplit ? 'flex' : 'none';
      document.getElementById('pane2Bar').style.display = isSplit ? 'flex' : 'none';

      if (isSplit) {
        document.getElementById('pane1Run').value = currentRunId;
        const activeRun = manifest.runs.find(r => r.id === currentRunId);
        populatePeriodDropdown('pane1', activeRun);
        document.getElementById('pane1Period').value = currentPeriod;
        document.getElementById('pane1Prod').value = currentProd;
        updatePane1();
        updatePane2();
      } else {
        updateSingleViewer();
      }
    }

    function updateSingleViewer() {
      if (layoutMode === 'split') return;
      document.getElementById('imgPrimary').src = getImageUrl(currentRunId, currentPeriod, currentProd);
    }

    function updatePane1() {
      const r = document.getElementById('pane1Run').value;
      const p = document.getElementById('pane1Period').value;
      const prod = document.getElementById('pane1Prod').value;
      document.getElementById('imgPrimary').src = getImageUrl(r, p, prod);
    }

    function updatePane2() {
      const r = document.getElementById('pane2Run').value;
      const p = document.getElementById('pane2Period').value;
      const prod = document.getElementById('pane2Prod').value;
      preloadRunImages(r);
      document.getElementById('imgSecondary').src = getImageUrl(r, p, prod);
    }

    function onPane1RunChange() {
      const rId = document.getElementById('pane1Run').value;
      const runObj = manifest.runs.find(r => r.id === rId);
      populatePeriodDropdown('pane1', runObj);
      updatePane1();
    }

    function onPane2RunChange() {
      const rId = document.getElementById('pane2Run').value;
      const runObj = manifest.runs.find(r => r.id === rId);
      populatePeriodDropdown('pane2', runObj);
      updatePane2();
    }

    function toggleHelpModal(show) {
      document.getElementById('helpModal').classList.toggle('open', show);
    }

    function onModalOverlayClick(e) {
      if (e.target.classList.contains('modal-overlay')) {
        toggleHelpModal(false);
      }
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowUp') { e.preventDefault(); stepRun(-1); }
      if (e.key === 'ArrowDown') { e.preventDefault(); stepRun(1); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); stepProduct(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); stepProduct(1); }
      if (e.key === 's' || e.key === 'S') { setLayout(layoutMode === 'single' ? 'split' : 'single'); }
      if (e.key === '?' || e.key === 'h' || e.key === 'H') {
        const modal = document.getElementById('helpModal');
        toggleHelpModal(!modal.classList.contains('open'));
      }
      if (e.key === 'Escape') { toggleHelpModal(false); }
    });

    init();
  </script>
</body>
</html>
"""
    with open(dest_path, "w") as f:
        f.write(html)


# ==========================================
# 7. MAIN PRODUCTION ENGINE & BACKFILLER
# ==========================================
def run_production():
    print("==========================================")
    print("CHECKING FOR ECMWF IFS CYCLES TO PROCESS...")
    print("==========================================")

    client_ens = Client(source="ecmwf", model="ifs", resol="0p25")
    client_det = Client(source="ecmwf", model="ifs", resol="0p1")

    existing_dirs = sorted([d for d in os.listdir(RUNS_DIR) if (RUNS_DIR / d).is_dir()])
    pending_cycles = get_pending_and_available_cycles(client_ens, max_lookback_hours=36)

    # First-run bootstrap: grab ONLY the single newest available cycle (no backfill)
    if not existing_dirs and pending_cycles:
        pending_cycles = [pending_cycles[-1]]
        print(f"Initial bootstrap run: selecting latest cycle {pending_cycles[0].strftime('%Y%m%d_%H')}")
    elif existing_dirs and pending_cycles:
        # Only backfill cycles newer than the oldest retained cycle
        oldest_kept = datetime.strptime(existing_dirs[0], "%Y%m%d_%H").replace(tzinfo=timezone.utc)
        pending_cycles = [c for c in pending_cycles if c > oldest_kept]
        if pending_cycles:
            print(f"Found {len(pending_cycles)} new/backfill ensemble cycle(s): {[c.strftime('%Y%m%d_%H') for c in pending_cycles]}")

    if pending_cycles:
        climo_ds = xr.open_dataset(CLIMO_PATH)
        for cand_dt in pending_cycles[:MAX_NEW_ENS_RUNS_PER_JOB]:
            process_ensemble_cycle(client_ens, cand_dt, climo_ds)
        climo_ds.close()
    else:
        print("All recent available 0.25° ensemble cycles are already cached.")

    # Second independent pass: check if 9km (0.1°) deterministic is ready for any cached run
    all_runs = sorted([d for d in os.listdir(RUNS_DIR) if (RUNS_DIR / d).is_dir()])
    for cycle_id in all_runs:
        status = read_cycle_status(cycle_id)
        if not status.get("ens_complete") or status.get("det_complete"):
            continue

        init_dt = datetime.strptime(cycle_id, "%Y%m%d_%H").replace(tzinfo=timezone.utc)
        periods = build_synoptic_periods(init_dt)
        req_end_step = periods[-1]["f_end"]

        if is_ecmwf_step_ready(client_det, init_dt.strftime("%Y%m%d"), init_dt.hour, "oper", "fc", req_end_step):
            process_deterministic_9km_cycle(client_det, init_dt)
        else:
            print(f"9km (0.1°) deterministic not yet ready through F{req_end_step:03d} for {cycle_id} (will check next run).")

    update_manifest_and_html()
    print("\nBatch production, manifest, and HTML viewer update complete!")


if __name__ == "__main__":
    run_production()
