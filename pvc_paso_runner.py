"""
Run PVC origin localization on PaSo (.mat) 12-lead recordings.

Reads `*_paso_ai.mat` files from a PaSo folder (read-only), extracts a 400 ms
window centered on the file's WOI (Window Of Interest = the PVC QRS that the
PaSo system already segmented), reorders leads to clinical order, and runs
the localizer in `pvc_12lead_localizer.py`.

Source PaSo files are NEVER modified. If you need a local working copy, set
COPY_LOCALLY=True and files are copied to ./paso_local_copy/.

Usage:
    py -3 pvc_paso_runner.py <folder>                  # process all *_paso_ai.mat in folder
    py -3 pvc_paso_runner.py <folder> IS1              # process one file by stem
    py -3 pvc_paso_runner.py <folder> IS1 PM3 ...      # process several
    py -3 pvc_paso_runner.py <folder> --out <dir>      # custom output directory

If <folder> is omitted the script falls back to the DEFAULT_DATA_FOLDER
constant below (kept for backward compatibility).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks as _find_peaks

from pvc_12lead_localizer import (
    LEAD_INDEX,
    LEAD_ORDER,
    detect_qrs_window_multilead,
    localize_pvc_12lead,
)


# ─────────────────────────────────────────────────────────────────
# RAW (B&W, no interpretation) 12-LEAD PLOT
# ─────────────────────────────────────────────────────────────────
def plot_raw_clinical_12lead(
    signal_12ch: np.ndarray,
    fs: float,
    output_path: Path,
    title: str = "",
) -> None:
    """
    Render the 400 ms window as a clean clinical-style 12-lead grid:
    white background, black traces, no annotations or probabilities.

    Layout (standard clinical):
        Row 1: I    aVR  V1  V4
        Row 2: II   aVL  V2  V5
        Row 3: III  aVF  V3  V6
    """
    layout = [
        ['I',   'aVR', 'V1', 'V4'],
        ['II',  'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
    ]
    t_ms = np.arange(signal_12ch.shape[0]) * 1000.0 / fs

    # Common y-range so leads are visually comparable.
    y_max = float(np.max(np.abs(signal_12ch))) * 1.1
    if y_max == 0:
        y_max = 1.0

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), facecolor='white',
                              sharex=True, sharey=True)
    if title:
        fig.suptitle(title, fontsize=13, color='black', y=0.97)

    for r in range(3):
        for c in range(4):
            ax = axes[r, c]
            lead_name = layout[r][c]
            ch = LEAD_INDEX[lead_name]

            ax.set_facecolor('white')
            # Light ECG-paper style grid (every 40 ms)
            for x in np.arange(0, t_ms[-1] + 1, 40):
                ax.axvline(x, color='#f0c0c0', linewidth=0.4)
            for y in np.arange(-y_max, y_max, y_max / 4):
                ax.axhline(y, color='#f0c0c0', linewidth=0.4)

            ax.plot(t_ms, signal_12ch[:, ch], color='black', linewidth=1.2)
            ax.axhline(0, color='#888888', linewidth=0.4)

            ax.text(0.02, 0.92, lead_name, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', color='black',
                    va='top', ha='left')

            ax.set_ylim(-y_max, y_max)
            ax.set_xlim(0, t_ms[-1])
            for sp in ax.spines.values():
                sp.set_color('#888888')
                sp.set_linewidth(0.6)
            ax.tick_params(colors='#888888', labelsize=8)
            if r < 2:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])

    for c in range(4):
        axes[2, c].set_xlabel('Time (ms)', fontsize=9, color='#444444')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

# Fallback folder used when no folder is given on the command line.
DEFAULT_DATA_FOLDER = Path(
    r"C:\Users\MohammadBeshar\Way2Deep\JJ Projects - Documents"
    r"\PVC_Project\W2D_Format\PASO_AI_9__s42_Study_S42_S1714556920\PaSo"
)

# Sampling rate. Confirmed = 1000 Hz: this is CARTO v7.2 data (see
# StudyMetadata/StudyBackup.xml), where time is stored in PIU units and
# 1 PIU = 1 sample = 1 ms. Cross-check: paso_is_table.WOI_START - INTERVAL_START
# (in PIU) equals the local file's WOI_START (in samples) exactly.
SAMPLING_RATE_HZ = 1000.0

# 400 ms window — matches the InceptionTime model input length when fs=1000 Hz.
WINDOW_MS = 400.0

# Optional: copy each .mat into a local working dir before processing.
# Source files are never modified either way — this is purely if you want
# a separate copy you can experiment on.
COPY_LOCALLY = False
LOCAL_COPY_DIR = Path("paso_local_copy")

# Lead order as stored in the .mat file (note: avL comes before avR).
PASO_LEAD_ORDER = ['I', 'II', 'III', 'avL', 'avR', 'avF',
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ─────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────
def load_paso_mat(mat_path: Path) -> dict:
    """
    Load a PaSo *_paso_ai.mat file. Handles both IS (recorded PVC, has WOI)
    and PM (paced beat, has XYZ_Position + mapping time) formats.

    Returns a dict with:
      signal_12ch:  (T, 12) float array in clinical lead order
                    (I, II, III, aVR, aVL, aVF, V1..V6)
      kind:         'IS' or 'PM' (best guess from filename / contents)
      woi_start:    sample index of QRS onset, or None
      woi_end:      sample index of QRS offset, or None
      is_valid:     PaSo's own validity flag (False if absent)
      xyz_position: 3-tuple pacing site (PM only), or None

    Source file is opened read-only via scipy.io.loadmat — never written to.
    """
    d = sio.loadmat(str(mat_path))

    lead_lengths = [d[name].size for name in PASO_LEAD_ORDER]
    T = min(lead_lengths)

    paso_stack = np.stack(
        [d[name].ravel()[:T].astype(np.float32) for name in PASO_LEAD_ORDER],
        axis=1,
    )
    paso_canonical = [n.upper() for n in PASO_LEAD_ORDER]
    perm = [paso_canonical.index(name.upper()) for name in LEAD_ORDER]
    signal_12ch = paso_stack[:, perm]

    has_woi = 'WOI_START' in d and 'WOI_END' in d
    woi_start = int(np.asarray(d['WOI_START']).ravel()[0]) if has_woi else None
    woi_end = int(np.asarray(d['WOI_END']).ravel()[0]) if has_woi else None

    # PaSo files use 'is valid' (with space); guard against key variants
    is_valid = False
    for _key in ('is valid', 'is_valid', 'isValid'):
        if _key in d:
            is_valid = bool(np.asarray(d[_key]).ravel()[0])
            break

    xyz = None
    if 'XYZ_Position' in d:
        arr = np.asarray(d['XYZ_Position']).ravel()
        if arr.size == 3:
            xyz = tuple(float(x) for x in arr)

    kind = 'PM' if mat_path.stem.startswith('PM') or xyz is not None else 'IS'

    return {
        'signal_12ch': signal_12ch,
        'kind': kind,
        'woi_start': woi_start,
        'woi_end': woi_end,
        'is_valid': is_valid,
        'xyz_position': xyz,
    }


def extract_window(
    signal_12ch: np.ndarray,
    woi_start: int | None,
    woi_end: int | None,
    fs: float,
    window_ms: float,
) -> Tuple[np.ndarray, int]:
    """
    Slice a window_ms window centered on the QRS midpoint.

    If WOI is provided (IS files), use it. Otherwise (PM files) auto-detect
    the QRS via the localizer's multi-lead energy peak detector.

    Returns (windowed_signal (W, 12), window_start_index).
    """
    W = int(round(window_ms * fs / 1000.0))
    T = signal_12ch.shape[0]

    if T <= W:
        return signal_12ch, 0

    if woi_start is not None and woi_end is not None:
        qrs_mid = (woi_start + woi_end) // 2
    else:
        onset, peak, _ = detect_qrs_window_multilead(signal_12ch, fs)
        qrs_mid = peak

    start = qrs_mid - W // 2
    start = max(0, min(start, T - W))
    return signal_12ch[start:start + W], start


# ─────────────────────────────────────────────────────────────────
# SINUS BEAT EXTRACTION
# ─────────────────────────────────────────────────────────────────
def find_sinus_beat(
    signal_12ch: np.ndarray,
    fs: float,
    pvc_woi_start: int,
    pvc_woi_end: int,
    window_ms: float = 400.0,
) -> Optional[np.ndarray]:
    """
    Find one good sinus beat in the full recording to use as reference for the
    Betensky V2 transition ratio and Yoshida TZ index.

    Strategy:
      1. Detect all beats via multi-lead energy peaks.
      2. Compute all RR intervals and find the dominant (modal) RR — the most
         common inter-beat interval, which corresponds to the sinus rate even
         in recordings with frequent PVCs or bigeminy.
      3. Keep only beats whose preceding AND following RR both match the modal
         RR within ±15 % — this ensures the candidate sits inside a stable
         sinus run, not immediately after a compensatory pause or before a PVC.
      4. From those candidates pick the one closest to the PVC (same recording
         context / hemodynamic state).

    Returns a (W, 12) windowed sinus beat, or None if no suitable beat found.
    """
    T = signal_12ch.shape[0]
    W = int(round(window_ms * fs / 1000.0))
    if T <= W:
        return None

    baselines = np.median(signal_12ch, axis=0)
    energy = np.sum(np.abs(signal_12ch - baselines), axis=1)

    min_dist = max(1, int(0.30 * fs))
    threshold = 0.25 * float(np.max(energy))
    peaks, _ = _find_peaks(energy, height=threshold, distance=min_dist)

    if len(peaks) < 3:
        return None

    # ── Step 1: find the dominant (modal) RR interval ──────────────
    rr = np.diff(peaks.astype(np.float32))          # N-1 intervals
    # Bin RR values in 50 ms buckets; the most populated bucket = sinus rate
    bin_ms = 50
    bins = np.arange(0, rr.max() + bin_ms, bin_ms)
    counts, edges = np.histogram(rr, bins=bins)
    modal_rr = float(edges[np.argmax(counts)] + bin_ms / 2)  # bin centre

    # ── Step 2: keep beats with stable RR on both sides ────────────
    tolerance = 0.15 * modal_rr
    pvc_centre = (pvc_woi_start + pvc_woi_end) // 2

    candidates = []
    for i in range(1, len(peaks) - 1):
        rr_before = float(peaks[i] - peaks[i - 1])
        rr_after  = float(peaks[i + 1] - peaks[i])
        if (abs(rr_before - modal_rr) <= tolerance
                and abs(rr_after - modal_rr) <= tolerance
                and abs(peaks[i] - pvc_centre) > modal_rr   # at least 1 RR away
                and (peaks[i] - W // 2) >= 0
                and (peaks[i] + W // 2) < T):
            candidates.append(peaks[i])

    if not candidates:
        return None

    # ── Step 3: pick the candidate closest to the PVC ──────────────
    best = min(candidates, key=lambda p: abs(p - pvc_centre))
    start = max(0, min(best - W // 2, T - W))
    return signal_12ch[start:start + W]


# ─────────────────────────────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────────────────────────────
def _natural_key(p: Path):
    """Sort IS1, IS2, ..., IS10 in numeric order, IS before PM."""
    name = p.stem
    prefix = ''.join(c for c in name if not c.isdigit())
    digits = ''.join(c for c in name if c.isdigit())
    return (prefix, int(digits) if digits else 0)


def discover_files(folder: Path, stems: List[str]) -> List[Path]:
    if stems:
        out = []
        for s in stems:
            cand = folder / f"{s}_paso_ai.mat"
            if not cand.exists():
                print(f"  ! missing: {cand.name}")
                continue
            out.append(cand)
        return out
    return sorted(folder.glob("*_paso_ai.mat"), key=_natural_key)


def maybe_copy(src: Path) -> Path:
    if not COPY_LOCALLY:
        return src
    LOCAL_COPY_DIR.mkdir(exist_ok=True)
    dst = LOCAL_COPY_DIR / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def process_one(mat_path: Path, fs: float, window_ms: float, out_dir: Path) -> dict:
    stem = mat_path.stem.replace("_paso_ai", "")
    info = load_paso_mat(mat_path)

    window, win_start = extract_window(
        info['signal_12ch'], info['woi_start'], info['woi_end'], fs, window_ms,
    )

    # If PaSo gave us a WOI, translate file-coords → window-coords.
    # Use the actual multi-lead energy peak within the WOI bounds as the QRS
    # peak — more accurate than the WOI midpoint for MDI computation.
    qrs_window = None
    if info['woi_start'] is not None and info['woi_end'] is not None:
        ws = max(0, info['woi_start'] - win_start)
        we = min(window.shape[0] - 1, info['woi_end'] - win_start)
        if 0 <= ws < we < window.shape[0]:
            baselines_w = np.median(window, axis=0)
            energy_w = np.sum(np.abs(window - baselines_w), axis=1)
            actual_peak = ws + int(np.argmax(energy_w[ws:we + 1]))
            qrs_window = (ws, actual_peak, we)

    # Extract a neighboring sinus beat from the full file (IS files only) so
    # the localizer can compute the proper Betensky V2 ratio and Yoshida TZ index.
    sinus_window: Optional[np.ndarray] = None
    if info['kind'] == 'IS' and info['woi_start'] is not None:
        sinus_window = find_sinus_beat(
            info['signal_12ch'], fs=fs,
            pvc_woi_start=info['woi_start'],
            pvc_woi_end=info['woi_end'],
            window_ms=window_ms,
        )

    png_path = out_dir / f"{stem}_localization.png"
    raw_png_path = out_dir / f"{stem}_raw.png"

    plot_raw_clinical_12lead(
        window, fs=fs, output_path=raw_png_path,
        title=f"{stem} — 400 ms window (raw 12-lead, no interpretation)",
    )

    result = localize_pvc_12lead(
        window, fs=fs, visualize=True, output_path=str(png_path),
        qrs_window=qrs_window,
        sinus_signal_12ch=sinus_window,
    )

    qrs_dur = None
    if info['woi_start'] is not None and info['woi_end'] is not None:
        qrs_dur = (info['woi_end'] - info['woi_start']) * 1000.0 / fs

    cf = result.cross_features
    return {
        "file": mat_path.name,
        "kind": info['kind'],
        "is_valid": info['is_valid'],
        "xyz_position": info['xyz_position'],
        "woi_start": info['woi_start'],
        "woi_end": info['woi_end'],
        "qrs_duration_ms": qrs_dur,
        "window_start_idx": int(win_start),
        "window_samples": int(window.shape[0]),
        "fs_hz": fs,
        "sinus_beat_found": sinus_window is not None,
        # ─── Classification ───
        "predicted_region": result.most_likely,
        "confidence": result.confidence,
        "sub_localization": result.sub_localization,
        "probabilities": result.probabilities,
        # ─── Epicardial markers (Daniels 2006, Berruezo 2004) ───
        "is_epicardial": result.is_epicardial,
        "epi_marker_count": result.epi_marker_count,
        "avg_mdi": result.avg_mdi,
        "avg_pseudo_delta_ms": result.avg_pseudo_delta_ms,
        "intrinsicoid_v2_ms": result.intrinsicoid_v2_ms,
        # ─── Cross-lead indices ───
        "v1_pattern": cf.v1_pattern,
        "frontal_axis_deg": cf.frontal_axis_deg,
        "precordial_transition": cf.precordial_transition,
        "v2_transition_ratio": cf.v2_transition_ratio,
        "v2_transition_ratio_normalized": cf.v2_transition_ratio_normalized,
        "tz_index": cf.tz_index,
        "v2s_v3r_index": cf.v2s_v3r_index,
        # ─── Output files ───
        "png": str(png_path),
        "raw_png": str(raw_png_path),
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run PVC origin localization on PaSo (*_paso_ai.mat) files."
    )
    parser.add_argument(
        "folder", nargs="?", default=None,
        help="Path to the PaSo folder containing *_paso_ai.mat files. "
             "Defaults to DEFAULT_DATA_FOLDER if omitted.",
    )
    parser.add_argument(
        "stems", nargs="*",
        help="Optional file stems to process (e.g. IS1 PM3). "
             "Processes all *_paso_ai.mat files if omitted.",
    )
    parser.add_argument(
        "--out", default="paso_localization_output",
        help="Output directory for PNGs and summary.json (default: paso_localization_output).",
    )
    args = parser.parse_args(argv[1:])

    data_folder = Path(args.folder) if args.folder else DEFAULT_DATA_FOLDER
    out_dir = Path(args.out)

    if not data_folder.exists():
        print(f"Data folder not found: {data_folder}")
        return 1

    out_dir.mkdir(exist_ok=True)
    files = discover_files(data_folder, args.stems)
    if not files:
        print("No files to process.")
        return 1

    print(f"Processing {len(files)} file(s) at fs={SAMPLING_RATE_HZ:.0f} Hz, "
          f"window={WINDOW_MS:.0f} ms ({int(WINDOW_MS * SAMPLING_RATE_HZ / 1000)} samples)")
    print(f"Source folder (read-only): {data_folder}")
    print(f"Output folder: {out_dir.resolve()}")

    summaries = []
    for src in files:
        path = maybe_copy(src)
        try:
            row = process_one(path, SAMPLING_RATE_HZ, WINDOW_MS, out_dir)
        except Exception as e:
            print(f"  [{src.name}] ERROR: {e}")
            continue
        summaries.append(row)
        probs = ", ".join(f"{k}={v:.0%}" for k, v in row["probabilities"].items())
        print(f"  [{src.stem}] {row['predicted_region']:<5} "
              f"({row['confidence']:.0%})  [{row['sub_localization']}]   {probs}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {len(summaries)} result(s) -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
