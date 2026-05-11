"""
PVC Origin Localization from 12-Lead ECG
=========================================

INPUT:  12-lead ECG signal (T samples × 12 channels) + sampling rate
        Standard lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

OUTPUT: Probability distribution over 4 ventricular regions:
        - RVOT (Right Ventricular Outflow Tract)        ~35% of idiopathic PVCs
        - RV   (Right Ventricle body / inferior / FW)   ~15%
        - LVOT (Left Ventricular Outflow Tract / cusps) ~25%
        - LV   (Left Ventricle body / papillary / post) ~25%

ALGORITHM:
  Implements the clinical 4-question hierarchy:
    Q1: V1 morphology         → RV (LBBB) or LV (RBBB)?
    Q2: II/III/aVF axis       → Superior (OT) or Inferior (body)?
    Q3: Lead I polarity       → Right/septal or Left-lateral?
    Q4: Precordial transition → Anterior or Posterior?

  Plus quantitative indices:
    - V2 transition ratio (Betensky 2011)
    - V2S/V3R index        (Yoshida 2014)
    - MDI                  (Daniels 2006) → epicardial marker
    - Pseudo-delta wave    (Berruezo 2004) → epicardial marker

  Combined via independent-evidence Bayesian update on a 2×2 grid:
                             OT axis(+)        body axis(−)
       RV side(V1 neg)         RVOT                RV
       LV side(V1 pos)         LVOT                LV

REFERENCE: Shahsavari et al. 2022, Aita et al. 2019, Yoshida et al. 2014
"""

import numpy as np
from scipy import signal as scipy_signal
from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
LEAD_ORDER: List[str] = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_INDEX: Dict[str, int] = {name: i for i, name in enumerate(LEAD_ORDER)}

# Population priors (idiopathic PVCs, from Shahsavari + clinical literature)
PRIOR = {'RVOT': 0.35, 'RV': 0.15, 'LVOT': 0.25, 'LV': 0.25}


# ─────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────
@dataclass
class LeadFeatures:
    """Features extracted from one lead."""
    name: str
    r_amp: float = 0.0
    s_amp: float = 0.0
    polarity: str = 'positive'      # 'positive' or 'negative' dominant
    qrs_duration_ms: float = 0.0
    time_to_peak_ms: float = 0.0
    mdi: float = 0.0
    rs_ratio: float = 0.5            # |R| / (|R| + |S|)
    notch_count: int = 0
    pseudo_delta_ms: float = 0.0


@dataclass
class CrossLeadFeatures:
    """Features that span multiple leads — these are the most diagnostic."""
    frontal_axis_deg: float = 0.0                # computed from I and aVF
    precordial_transition: int = 4               # lead # (1-6) where R first ≥ S
    v2_transition_ratio: float = 0.0             # raw R/(R+S) in V2 during PVC
    v2_transition_ratio_normalized: float = 0.0  # Betensky: PVC/sinus (0 = no sinus ref)
    tz_index: float = 0.0                        # Yoshida: PVC TZ − sinus TZ (0 = no sinus ref)
    v2s_v3r_index: float = 0.0                   # |S in V2| / R in V3
    inferior_axis_score: float = 0.0             # +1 = clearly inferior, -1 = clearly superior
    v2_pattern_break: bool = False               # LV summit signature
    v1_pattern: str = 'unknown'                  # QS|rS|LBBB-like|transitional|M/W|qrS|dominant-R|narrow-RBBB
    qrs_duration_ms: float = 0.0                 # max across leads


@dataclass
class LocalizationResult:
    """Final classification result."""
    probabilities: Dict[str, float] = field(default_factory=dict)
    most_likely: str = ''
    confidence: float = 0.0
    sub_localization: str = ''        # finer guess (e.g., "RVOT septal")
    is_epicardial: bool = False
    epi_marker_count: int = 0         # how many of 3 epicardial markers triggered
    avg_mdi: float = 0.0              # mean MDI across V1–V3 (Daniels 2006)
    avg_pseudo_delta_ms: float = 0.0  # mean pseudo-delta across V1–V3 (Berruezo 2004)
    intrinsicoid_v2_ms: float = 0.0   # QRS onset to V2 peak (Berruezo 2004)
    reasoning: List[str] = field(default_factory=list)
    lead_features: List[LeadFeatures] = field(default_factory=list)
    cross_features: Optional[CrossLeadFeatures] = None


# ─────────────────────────────────────────────────────────────────
# STEP 1: SYNCHRONIZED QRS DETECTION ACROSS 12 LEADS
# ─────────────────────────────────────────────────────────────────
def detect_qrs_window_multilead(
    signal_12ch: np.ndarray,
    fs: float,
    threshold_frac: float = 0.35,
    max_half_width_ms: float = 150.0,
) -> Tuple[int, int, int]:
    """
    QRS happens at the SAME time in all 12 leads (single dipole, multiple
    projections). Sum |signal| across leads and walk outward from the peak
    until the energy drops below `threshold_frac` of peak.

    Two safeguards keep the window from running into the T-wave when the
    surrounding signal isn't quiet:
      - threshold_frac default 0.35 (was 0.15 — too lenient for noisy strips)
      - max_half_width_ms hard-caps the walk on each side
        (PVC QRS is typically 120-200 ms; 150 ms each side = 300 ms max)

    Returns sample indices (onset, peak, offset).
    """
    baselines = np.median(signal_12ch, axis=0)
    centered = signal_12ch - baselines
    combined = np.sum(np.abs(centered), axis=1)

    peak_idx = int(np.argmax(combined))
    peak_val = combined[peak_idx]
    threshold = threshold_frac * peak_val

    max_half = int(round(max_half_width_ms * fs / 1000.0))
    left_limit = max(0, peak_idx - max_half)
    right_limit = min(len(combined) - 1, peak_idx + max_half)

    onset_idx = peak_idx
    while onset_idx > left_limit and combined[onset_idx] > threshold:
        onset_idx -= 1
    offset_idx = peak_idx
    while offset_idx < right_limit and combined[offset_idx] > threshold:
        offset_idx += 1

    return onset_idx, peak_idx, offset_idx


# ─────────────────────────────────────────────────────────────────
# STEP 2: PER-LEAD FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────
def extract_lead_features(
    lead_signal: np.ndarray,
    lead_name: str,
    onset_idx: int,
    peak_idx: int,
    offset_idx: int,
    fs: float,
) -> LeadFeatures:
    """Extract morphology features for a single lead."""
    qrs = lead_signal[onset_idx:offset_idx + 1]
    if len(qrs) < 3:
        return LeadFeatures(name=lead_name)

    baseline = np.median(lead_signal)
    qrs_centered = qrs - baseline

    qrs_duration_ms = (offset_idx - onset_idx) * 1000.0 / fs
    r_amp = float(np.max(qrs_centered)) if np.max(qrs_centered) > 0 else 0.0
    s_amp = float(np.min(qrs_centered)) if np.min(qrs_centered) < 0 else 0.0
    polarity = 'positive' if abs(r_amp) >= abs(s_amp) else 'negative'

    abs_qrs = np.abs(qrs_centered)
    local_peak = int(np.argmax(abs_qrs))
    time_to_peak_ms = local_peak * 1000.0 / fs
    mdi = time_to_peak_ms / qrs_duration_ms if qrs_duration_ms > 0 else 0.5

    rs_total = abs(r_amp) + abs(s_amp)
    rs_ratio = abs(r_amp) / rs_total if rs_total > 0 else 0.5

    # Notching: derivative sign changes within a smoothed QRS
    if len(qrs) >= 7:
        # savgol needs an odd window <= signal length
        max_win = min(11, len(qrs))
        win = max_win if max_win % 2 == 1 else max_win - 1
        smooth = scipy_signal.savgol_filter(qrs_centered, win, 3)
        deriv = np.diff(smooth)
        # Only count significant inflections (above noise floor)
        sig_thresh = 0.1 * np.max(np.abs(deriv)) if np.max(np.abs(deriv)) > 0 else 0
        sign_changes = 0
        for i in range(1, len(deriv)):
            if (deriv[i-1] > sig_thresh and deriv[i] < -sig_thresh) or \
               (deriv[i-1] < -sig_thresh and deriv[i] > sig_thresh):
                sign_changes += 1
        notch_count = sign_changes
    else:
        notch_count = 0

    # Pseudo-delta wave: time from QRS onset to where slope reaches 50% of max
    if len(qrs) >= 3:
        deriv = np.gradient(qrs_centered)
        max_slope = np.max(np.abs(deriv))
        if max_slope > 0:
            rapid = np.where(np.abs(deriv) >= 0.5 * max_slope)[0]
            pseudo_delta_ms = (rapid[0] * 1000.0 / fs) if len(rapid) > 0 else 0.0
        else:
            pseudo_delta_ms = 0.0
    else:
        pseudo_delta_ms = 0.0

    return LeadFeatures(
        name=lead_name,
        r_amp=r_amp,
        s_amp=s_amp,
        polarity=polarity,
        qrs_duration_ms=qrs_duration_ms,
        time_to_peak_ms=time_to_peak_ms,
        mdi=mdi,
        rs_ratio=rs_ratio,
        notch_count=notch_count,
        pseudo_delta_ms=pseudo_delta_ms,
    )


# ─────────────────────────────────────────────────────────────────
# STEP 2b: V1 MORPHOLOGY PATTERN CLASSIFICATION
# ─────────────────────────────────────────────────────────────────
def classify_v1_pattern(
    v1_signal: np.ndarray,
    onset_idx: int,
    offset_idx: int,
    fs: float,
) -> str:
    """
    Classify the V1 QRS into named clinical morphology types used in the
    Pattern Atlas (PDF steps 11–16).

    Returns one of:
      'QS'          — monophasic negative, no R wave
      'rS'          — small R then dominant S (classic LBBB-like, RVOT)
      'LBBB-like'   — broad dominant negative, less extreme than rS
      'transitional' — R ≈ S (borderline, septal)
      'M/W'         — multiphasic ≥3 direction changes (LCC signature)
      'qrS'         — small q, small r, deep S (L-R commissure signature)
      'dominant-R'  — dominant R wave (RBBB-like, LV origin)
      'narrow-RBBB' — dominant R, narrow QRS (fascicular VT)
      'unknown'     — too short or featureless
    """
    qrs = v1_signal[onset_idx:offset_idx + 1]
    if len(qrs) < 3:
        return 'unknown'

    pre = v1_signal[max(0, onset_idx - 10):onset_idx]
    baseline = float(pre.mean()) if len(pre) > 0 else float(qrs[0])
    qrs_c = qrs - baseline

    if len(qrs_c) >= 7:
        win = min(11, len(qrs_c))
        win = win if win % 2 == 1 else win - 1
        smooth = scipy_signal.savgol_filter(qrs_c, win, 3)
    else:
        smooth = qrs_c.copy()

    r_amp = float(np.max(smooth))
    s_amp = float(np.min(smooth))  # negative value
    dom = max(abs(r_amp), abs(s_amp))
    if dom < 1e-6:
        return 'unknown'

    thresh = 0.10 * dom  # significance threshold: 10% of dominant amplitude
    qrs_dur_ms = (offset_idx - onset_idx) * 1000.0 / fs

    # --- Segment analysis via sign-change regions ---
    signs = np.sign(smooth)
    signs[np.abs(smooth) < thresh] = 0
    # Forward-fill zeros so crossings are clean
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    crossings = np.where(np.diff(signs) != 0)[0]
    boundaries = [0] + list(crossings + 1) + [len(smooth)]

    # Peak amplitude of each contiguous segment
    segments = []
    for i in range(len(boundaries) - 1):
        seg = smooth[boundaries[i]:boundaries[i + 1]]
        if len(seg) == 0:
            continue
        peak_val = float(seg[np.argmax(np.abs(seg))])
        if abs(peak_val) > thresh:
            segments.append(peak_val)

    n_seg = len(segments)
    rs_total = r_amp + abs(s_amp)
    rs_ratio = r_amp / rs_total if rs_total > 0 else 0.5

    # QS: no meaningful positive component
    if rs_ratio < 0.05:
        return 'QS'

    # M/W: 3+ alternating significant segments (e.g. +−+ or −+−+)
    if n_seg >= 3:
        alternations = sum(
            1 for i in range(len(segments) - 1)
            if segments[i] * segments[i + 1] < 0
        )
        if alternations >= 2:
            return 'M/W'

    # qrS: pattern [neg, pos, neg] where initial neg and middle pos are both
    # small relative to the terminal dominant S
    if n_seg >= 2 and segments[0] < -thresh and segments[-1] < -thresh:
        pos_segs = [s for s in segments if s > thresh]
        if pos_segs:
            max_pos = max(pos_segs)
            first_neg_frac = abs(segments[0]) / abs(s_amp)
            if max_pos < 0.5 * abs(s_amp) and first_neg_frac < 0.30:
                return 'qrS'

    # Simple R vs S ratio classification
    if rs_ratio < 0.20:
        return 'rS'
    elif rs_ratio < 0.45:
        return 'LBBB-like'
    elif rs_ratio < 0.60:
        return 'transitional'
    elif qrs_dur_ms < 135:
        return 'narrow-RBBB'
    else:
        return 'dominant-R'


# ─────────────────────────────────────────────────────────────────
# STEP 3: CROSS-LEAD FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────────
def compute_cross_features(
    lead_feats: List[LeadFeatures],
    v1_pattern: str = 'unknown',
    sinus_feats: Optional[List[LeadFeatures]] = None,
) -> CrossLeadFeatures:
    """Compute features that combine information across leads."""
    f = {lf.name: lf for lf in lead_feats}
    cf = CrossLeadFeatures()
    cf.v1_pattern = v1_pattern

    # ─── Frontal axis from Lead I and aVF amplitudes ───
    I_amp   = f['I'].r_amp + f['I'].s_amp
    aVF_amp = f['aVF'].r_amp + f['aVF'].s_amp
    cf.frontal_axis_deg = float(np.degrees(np.arctan2(aVF_amp, I_amp)))

    # ─── Inferior axis score: positive in II, III, aVF → outflow tract ───
    inf_score = 0.0
    for nm in ['II', 'III', 'aVF']:
        if f[nm].polarity == 'positive':
            inf_score += f[nm].rs_ratio
        else:
            inf_score -= (1.0 - f[nm].rs_ratio)
    cf.inferior_axis_score = inf_score / 3.0

    # ─── Precordial transition zone (first lead where R≥S, sustained) ───
    # Require the *next* lead to also be ≥0.5 so a single noisy spike in
    # V2 (while V1 and V3 are negative) doesn't falsely anchor the transition.
    _prec = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    transition = 7  # default: never transitions within V1–V6
    for i, lead in enumerate(_prec, start=1):
        if f[lead].rs_ratio >= 0.5:
            next_lead = _prec[i] if i < len(_prec) else None  # i is 1-based, _prec is 0-based
            sustained = (next_lead is None) or (f[next_lead].rs_ratio >= 0.5)
            if sustained:
                transition = i
                break
    cf.precordial_transition = transition

    # ─── V2 transition ratio raw (Betensky numerator) ───
    cf.v2_transition_ratio = f['V2'].rs_ratio

    # ─── Sinus-reference indices (require a sinus beat window) ───
    if sinus_feats is not None:
        sf = {lf.name: lf for lf in sinus_feats}
        sinus_v2_rs = sf['V2'].rs_ratio
        # Betensky 2011: PVC R/(R+S) in V2 ÷ sinus R/(R+S) in V2 → ≥0.6 = LVOT
        if sinus_v2_rs > 0.01:
            cf.v2_transition_ratio_normalized = cf.v2_transition_ratio / sinus_v2_rs
        # Yoshida 2011: PVC transition lead − sinus transition lead → <0 = LV side
        sinus_tz = 7
        for i, lead in enumerate(_prec, start=1):
            if sf[lead].rs_ratio >= 0.5:
                next_lead = _prec[i] if i < len(_prec) else None
                if (next_lead is None) or (sf[next_lead].rs_ratio >= 0.5):
                    sinus_tz = i
                    break
        cf.tz_index = float(transition - sinus_tz)

    # ─── V2S/V3R index (Yoshida 2014) ───
    s_v2 = abs(f['V2'].s_amp)
    r_v3 = max(f['V3'].r_amp, 0.01)
    cf.v2s_v3r_index = s_v2 / r_v3

    # ─── V2 pattern break (LV summit signature) ───
    v1_r = f['V1'].r_amp
    v2_r = f['V2'].r_amp
    v3_r = f['V3'].r_amp
    if v1_r > 0.3 and v2_r < 0.2 and v3_r > v2_r * 2 and v3_r > 0.3:
        cf.v2_pattern_break = True

    cf.qrs_duration_ms = float(np.max([lf.qrs_duration_ms for lf in lead_feats]))

    return cf


# ─────────────────────────────────────────────────────────────────
# STEP 4: THE 4-QUESTION ALGORITHM → 4-REGION CLASSIFICATION
# ─────────────────────────────────────────────────────────────────
def classify_4regions(
    lead_feats: List[LeadFeatures],
    cf: CrossLeadFeatures,
) -> LocalizationResult:
    """
    Apply the clinical 4-question hierarchy to compute P(region | ECG).

    Decomposition:
      P(RVOT) = P(RV side) · P(OT axis)
      P(RV)   = P(RV side) · P(body axis)
      P(LVOT) = P(LV side) · P(OT axis)
      P(LV)   = P(LV side) · P(body axis)

    Then refine using transition zone, lead I, and quantitative indices.
    """
    f = {lf.name: lf for lf in lead_feats}
    reasoning = []

    # ═══════ Q1: V1 morphology → RV vs LV ═══════
    v1_r = f['V1'].r_amp
    v1_s = abs(f['V1'].s_amp)
    v1_total = v1_r + v1_s
    if v1_total > 0.05:
        # Sigmoid-like: rs_ratio close to 0 → strong RV; close to 1 → strong LV
        rs = f['V1'].rs_ratio
        # Use a logistic for smoother transition
        p_lv = 1.0 / (1.0 + np.exp(-8 * (rs - 0.5)))
        p_rv = 1.0 - p_lv
    else:
        p_rv, p_lv = 0.5, 0.5

    if v1_r > v1_s:
        reasoning.append(f"Q1: V1 dominant R (R={v1_r:.2f}, S={v1_s:.2f}) → RBBB-like → LV side ({p_lv:.0%})")
    else:
        reasoning.append(f"Q1: V1 dominant S (R={v1_r:.2f}, S={v1_s:.2f}) → LBBB-like → RV side ({p_rv:.0%})")

    # ═══════ Q2: Inferior leads → OT vs body ═══════
    # Inferior axis score is in roughly [-1, +1]
    p_ot   = 1.0 / (1.0 + np.exp(-6 * cf.inferior_axis_score))
    p_body = 1.0 - p_ot

    if cf.inferior_axis_score > 0.3:
        reasoning.append(f"Q2: II/III/aVF positive (score={cf.inferior_axis_score:+.2f}) → "
                          f"INFERIOR axis → outflow tract ({p_ot:.0%})")
    elif cf.inferior_axis_score < -0.3:
        reasoning.append(f"Q2: II/III/aVF negative (score={cf.inferior_axis_score:+.2f}) → "
                          f"SUPERIOR axis → body/apex ({p_body:.0%})")
    else:
        reasoning.append(f"Q2: II/III/aVF mixed (score={cf.inferior_axis_score:+.2f}) → ambiguous")

    # ═══════ Q3: Lead I → right/septal vs left-lateral ═══════
    lead_i_pos = f['I'].polarity == 'positive'
    if lead_i_pos:
        reasoning.append(f"Q3: Lead I positive (R={f['I'].r_amp:.2f}) → right/septal lean")
    else:
        reasoning.append(f"Q3: Lead I negative → left-lateral lean (boosts LV/LV summit)")

    # ═══════ Q4: Precordial transition → anterior vs posterior ═══════
    tz = cf.precordial_transition
    if tz <= 2:
        reasoning.append(f"Q4: Early transition at V{tz} → anterior/LV-leaning")
    elif tz >= 5:
        reasoning.append(f"Q4: Late transition at V{tz} → posterior/RV-leaning")
    else:
        reasoning.append(f"Q4: Normal transition at V{tz} → septal/borderline")

    # ═══════ Combine into 4-region probability ═══════
    p = {
        'RVOT': p_rv * p_ot,
        'RV':   p_rv * p_body,
        'LVOT': p_lv * p_ot,
        'LV':   p_lv * p_body,
    }

    # ═══════ Apply population priors ═══════
    for k in p:
        p[k] *= PRIOR[k]

    # ═══════ Refinements from quantitative indices ═══════

    # Lead I refinement
    if not lead_i_pos:
        p['LV'] *= 1.6
        p['LVOT'] *= 1.2
    else:
        p['RVOT'] *= 1.2
        p['RV'] *= 1.1

    # Transition zone refinement
    if tz <= 2:
        p['LVOT'] *= 1.5
        p['LV']   *= 1.3
    elif tz >= 5:
        p['RVOT'] *= 1.3
        p['RV']   *= 1.4

    # V2 transition ratio (Betensky 2011): prefer normalized vs sinus when available
    betensky = (cf.v2_transition_ratio_normalized
                if cf.v2_transition_ratio_normalized > 0
                else cf.v2_transition_ratio)
    if betensky >= 0.6:
        p['LVOT'] *= 2.0
        src = 'normalized' if cf.v2_transition_ratio_normalized > 0 else 'raw'
        reasoning.append(f"V2 transition ratio ({src}) = {betensky:.2f} ≥ 0.6 → LVOT boost")

    # Yoshida TZ index (2011): <0 means PVC transition earlier than sinus → LV side
    if cf.tz_index < 0:
        p['LVOT'] *= 1.4
        p['LV']   *= 1.2
        reasoning.append(
            f"TZ index = {cf.tz_index:+.0f} (PVC transition earlier than sinus) → LV-side lean"
        )

    # V2S/V3R index (Yoshida 2014): ≤1.5 → LVOT
    if cf.v2s_v3r_index <= 1.5 and f['V3'].r_amp > 0.1:
        p['LVOT'] *= 1.5
        reasoning.append(f"V2S/V3R = {cf.v2s_v3r_index:.2f} ≤ 1.5 → LVOT boost")

    # V1 pattern refinements (Pattern Atlas, PDF steps 11–16)
    if cf.v1_pattern == 'M/W':
        p['LVOT'] *= 1.8
        reasoning.append("V1 M/W multiphasic pattern → LCC boost")
    elif cf.v1_pattern == 'qrS':
        p['LVOT'] *= 1.5
        reasoning.append("V1 qrS pattern → L-R commissure boost")
    elif cf.v1_pattern in ('QS', 'rS'):
        p['RVOT'] *= 1.2
        p['RV']   *= 1.1
    elif cf.v1_pattern == 'narrow-RBBB':
        p['LV'] *= 1.5
        reasoning.append("V1 narrow-RBBB → fascicular signature → LV boost")

    # Epicardial markers: MDI (Daniels 2006), pseudo-delta + intrinsicoid V2 (Berruezo 2004)
    is_epicardial = False
    epi_markers = 0
    avg_mdi = np.mean([f[ld].mdi for ld in ['V1', 'V2', 'V3']])
    avg_pseudo = np.mean([f[ld].pseudo_delta_ms for ld in ['V1', 'V2', 'V3']])
    intrinsicoid_v2 = f['V2'].time_to_peak_ms
    if avg_mdi >= 0.55:
        epi_markers += 1
    if avg_pseudo >= 34:
        epi_markers += 1
    if intrinsicoid_v2 >= 85:
        epi_markers += 1
    if epi_markers >= 2:
        is_epicardial = True
        p['LV'] *= 2.0
        reasoning.append(
            f"Epicardial markers ({epi_markers}/3): "
            f"MDI={avg_mdi:.2f}, pseudo-δ={avg_pseudo:.0f} ms, "
            f"intrinsicoid-V2={intrinsicoid_v2:.0f} ms → LV epicardial boost"
        )

    # V2 pattern break → strong LV summit signature
    if cf.v2_pattern_break:
        p['LV'] *= 3.0
        reasoning.append("V2 pattern break detected → LV summit signature → strong LV boost")

    # ═══════ Normalize ═══════
    total = sum(p.values())
    p = {k: round(v / total, 4) for k, v in p.items()}

    most_likely = max(p, key=p.get)
    confidence = p[most_likely]

    # ═══════ Sub-localization hint (matches PDF page 27 quick-reference table) ═══════
    sub = ''
    if most_likely == 'RVOT':
        if f['I'].polarity == 'positive' and f['V1'].notch_count <= 2:
            sub = 'RVOT septal'
        else:
            sub = 'RVOT free wall'
    elif most_likely == 'LVOT':
        if cf.v1_pattern == 'M/W':
            sub = 'LCC — Left Coronary Cusp (M/W in V1)'
        elif cf.v1_pattern == 'qrS':
            sub = 'L-R Commissure (qrS in V1)'
        elif tz <= 2:
            sub = f'RCC — Right Coronary Cusp (early V{tz} transition)'
        else:
            sub = 'LVOT (cusp, indeterminate)'
    elif most_likely == 'LV':
        if cf.v2_pattern_break:
            sub = 'LV Summit (epicardial — V2 pattern break)'
        elif is_epicardial:
            sub = 'LV epicardial'
        elif cf.v1_pattern == 'narrow-RBBB':
            sub = 'LV fascicular (posterior fascicle VT)'
        elif cf.inferior_axis_score < -0.3:
            sub = 'LV papillary muscle / inferior wall'
        else:
            sub = 'LV body'
    else:  # RV
        if cf.inferior_axis_score < -0.3:
            sub = 'RV apex / inferior wall'
        else:
            sub = 'RV free wall'

    return LocalizationResult(
        probabilities=p,
        most_likely=most_likely,
        confidence=confidence,
        sub_localization=sub,
        is_epicardial=is_epicardial,
        epi_marker_count=epi_markers,
        avg_mdi=float(avg_mdi),
        avg_pseudo_delta_ms=float(avg_pseudo),
        intrinsicoid_v2_ms=float(intrinsicoid_v2),
        reasoning=reasoning,
        lead_features=lead_feats,
        cross_features=cf,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def localize_pvc_12lead(
    signal_12ch: np.ndarray,
    fs: float,
    visualize: bool = True,
    output_path: str = '12lead_localization.png',
    qrs_window: Optional[Tuple[int, int, int]] = None,
    sinus_signal_12ch: Optional[np.ndarray] = None,
) -> LocalizationResult:
    """
    Localize a PVC origin from a 12-lead ECG signal.

    Args:
      signal_12ch:       (T, 12) array in order: I, II, III, aVR, aVL, aVF, V1..V6
      fs:                sampling frequency in Hz
      visualize:         render a 12-panel diagnostic figure
      output_path:       where to save the figure
      qrs_window:        optional (onset, peak, offset) from PaSo WOI — skips
                         auto-detection and uses externally-segmented bounds
      sinus_signal_12ch: optional (W, 12) window of a neighboring sinus beat.
                         When supplied, enables the proper Betensky V2 transition
                         ratio (PVC/sinus) and Yoshida TZ index (PVC TZ − sinus TZ).

    Returns: LocalizationResult with probabilities, most_likely, sub-loc, reasoning.
    """
    if signal_12ch.shape[1] != 12:
        raise ValueError(f"Expected 12 channels, got {signal_12ch.shape[1]}. "
                         f"Channel order: {LEAD_ORDER}")

    # Step 1: QRS window — trust externally-supplied bounds if present
    if qrs_window is not None:
        onset, peak, offset = qrs_window
    else:
        onset, peak, offset = detect_qrs_window_multilead(signal_12ch, fs)

    # Step 2: Per-lead features
    lead_feats = []
    for i, name in enumerate(LEAD_ORDER):
        lf = extract_lead_features(signal_12ch[:, i], name, onset, peak, offset, fs)
        lead_feats.append(lf)

    # Step 2b: V1 morphology pattern (needs raw signal + QRS bounds)
    v1_pattern = classify_v1_pattern(
        signal_12ch[:, LEAD_INDEX['V1']], onset, offset, fs
    )

    # Step 2c: Sinus reference features (for normalized V2 ratio and TZ index)
    sinus_feats: Optional[List[LeadFeatures]] = None
    if sinus_signal_12ch is not None and sinus_signal_12ch.shape[1] == 12:
        s_onset, s_peak, s_offset = detect_qrs_window_multilead(sinus_signal_12ch, fs)
        sinus_feats = [
            extract_lead_features(
                sinus_signal_12ch[:, i], name, s_onset, s_peak, s_offset, fs
            )
            for i, name in enumerate(LEAD_ORDER)
        ]

    # Step 3: Cross-lead features
    cf = compute_cross_features(lead_feats, v1_pattern=v1_pattern, sinus_feats=sinus_feats)

    # Step 4: 4-region classification
    result = classify_4regions(lead_feats, cf)

    # Visualization
    if visualize:
        visualize_12lead(signal_12ch, fs, onset, peak, offset, result, output_path)

    return result


# ─────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────
def visualize_12lead(
    signal_12ch: np.ndarray,
    fs: float,
    onset: int, peak: int, offset: int,
    result: LocalizationResult,
    output_path: str,
):
    """Render a clinical-style 12-lead grid + classification panel."""
    fig = plt.figure(figsize=(16, 13), facecolor='#0d1117')
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1.4, 1.0],
                           hspace=0.45, wspace=0.25)

    t_ms = np.arange(signal_12ch.shape[0]) * 1000.0 / fs

    # Highlight bands for OT (positive in inferior) vs body
    for r in range(3):
        for c in range(4):
            ax = fig.add_subplot(gs[r, c])
            lead_idx = r * 4 + c
            # Re-arrange to clinical layout: row 1 = I, aVR, V1, V4
            #                                row 2 = II, aVL, V2, V5
            #                                row 3 = III, aVF, V3, V6
            clinical_layout = [
                ['I',  'aVR', 'V1', 'V4'],
                ['II', 'aVL', 'V2', 'V5'],
                ['III','aVF', 'V3', 'V6'],
            ]
            lead_name = clinical_layout[r][c]
            channel_idx = LEAD_INDEX[lead_name]

            ax.set_facecolor('#1a0000')
            # Light grid
            for x in np.arange(0, t_ms[-1], 40):
                ax.axvline(x, color='#882222', linewidth=0.3, alpha=0.5)

            ax.plot(t_ms, signal_12ch[:, channel_idx], color='#00FF88', linewidth=1.6)
            ax.axhline(0, color='#666666', linewidth=0.5)

            # Highlight QRS region
            ax.axvspan(t_ms[onset], t_ms[offset], alpha=0.18, color='#FFD700')
            ax.axvline(t_ms[peak], color='#FF8800', linestyle='--', linewidth=1, alpha=0.7)

            # Per-lead annotation
            lf = result.lead_features[channel_idx]
            polarity_color = '#88FF88' if lf.polarity == 'positive' else '#FF8888'
            ax.text(0.03, 0.93,
                     f"{lead_name}\n{lf.polarity[:3].upper()} R={lf.r_amp:.1f}",
                     transform=ax.transAxes, fontsize=8.5, color=polarity_color,
                     fontweight='bold', va='top',
                     bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='#0d1117', edgecolor=polarity_color, alpha=0.8))

            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('#444444')

    fig.suptitle('12-Lead ECG — PVC Localization Analysis',
                 color='white', fontsize=15, fontweight='bold', y=0.97)

    # Bottom row: result panel
    ax_result = fig.add_subplot(gs[3, :2])
    ax_result.set_facecolor('#0d1117')
    ax_result.axis('off')

    cf = result.cross_features
    title = f'PREDICTION:  {result.most_likely}  ({result.confidence:.0%} confidence)'
    if result.sub_localization:
        title += f'\nSub-localization: {result.sub_localization}'
    if result.is_epicardial:
        title += '   [EPICARDIAL markers detected]'

    ax_result.text(0.02, 0.95, title, fontsize=14, color='#FFD700',
                    fontweight='bold', va='top', transform=ax_result.transAxes)

    has_sinus = cf.v2_transition_ratio_normalized > 0
    v2_norm_str = f"{cf.v2_transition_ratio_normalized:.2f}" if has_sinus else "--"
    tz_str      = f"{cf.tz_index:+.0f}" if has_sinus else "--"
    cross_text = (
        f"Cross-lead features:\n"
        f"  V1 pattern:             {cf.v1_pattern}\n"
        f"  Frontal axis:           {cf.frontal_axis_deg:+.0f}°\n"
        f"  Inferior axis score:    {cf.inferior_axis_score:+.2f}\n"
        f"  Precordial transition:  V{cf.precordial_transition}\n"
        f"  V2 ratio raw / norm:    {cf.v2_transition_ratio:.2f} / {v2_norm_str}  (≥0.6 → LVOT)\n"
        f"  TZ index (vs sinus):    {tz_str}  (< 0 → LV side)\n"
        f"  V2S/V3R index:          {cf.v2s_v3r_index:.2f}  (≤1.5 → LVOT)\n"
        f"  MDI / pseudo-δ / V2i:   {result.avg_mdi:.2f} / "
        f"{result.avg_pseudo_delta_ms:.0f} ms / {result.intrinsicoid_v2_ms:.0f} ms\n"
        f"  Max QRS duration:       {cf.qrs_duration_ms:.0f} ms\n"
        f"  V2 pattern break:       {cf.v2_pattern_break}"
    )
    ax_result.text(0.02, 0.7, cross_text, fontsize=10, color='#AAAAFF',
                    fontfamily='monospace', va='top', transform=ax_result.transAxes)

    # Bottom row: probability bars
    ax_probs = fig.add_subplot(gs[3, 2:])
    ax_probs.set_facecolor('#0d1117')
    chambers = ['RVOT', 'RV', 'LVOT', 'LV']
    values = [result.probabilities[c] for c in chambers]
    colors = ['#FF6666', '#FF9966', '#66FFAA', '#9966FF']
    bars = ax_probs.barh(chambers, values, color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax_probs.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                       f'{val:.1%}', va='center', color='white',
                       fontweight='bold', fontsize=12)
    ax_probs.set_xlim(0, 1.0)
    ax_probs.set_xlabel('Probability', color='white', fontsize=11)
    ax_probs.set_title('4-Region Probability Distribution',
                        color='white', fontsize=12, fontweight='bold')
    ax_probs.tick_params(colors='white', labelsize=11)
    for sp in ['top','right']: ax_probs.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax_probs.spines[sp].set_color('#888888')

    # Reasoning panel
    ax_reason = fig.add_subplot(gs[4, :])
    ax_reason.set_facecolor('#0d1117')
    ax_reason.axis('off')
    ax_reason.text(0.02, 0.95, 'Algorithm reasoning chain:',
                    fontsize=11, color='#FFAA00', fontweight='bold',
                    transform=ax_reason.transAxes, va='top')

    for i, line in enumerate(result.reasoning[:7]):
        ax_reason.text(0.04, 0.78 - i*0.13, f"• {line}",
                        fontsize=9.5, color='#DDDDDD',
                        transform=ax_reason.transAxes, va='top')

    plt.savefig(output_path, dpi=140, bbox_inches='tight', facecolor='#0d1117')
    plt.close()


# ─────────────────────────────────────────────────────────────────
# DEMO: SYNTHETIC 12-LEAD PVCs FOR EACH OF THE 4 REGIONS
# ─────────────────────────────────────────────────────────────────
def make_synthetic_12lead_pvc(pattern: str, fs: float = 500.0,
                                duration_sec: float = 0.8) -> np.ndarray:
    """
    Generate a realistic synthetic 12-lead PVC for one of the 4 regions.

    Patterns:
      'RVOT': LBBB in V1, inferior axis (II/III/aVF positive),
              Lead I positive, late transition (V4-V5)
      'RV':   LBBB in V1, superior axis, Lead I positive, late transition
      'LVOT': transitional V1 (R≥S), inferior axis,
              Lead I variable, early transition (V2-V3)
      'LV':   RBBB in V1, superior axis, Lead I negative, early transition
    """
    n_samples = int(fs * duration_sec)
    t = np.linspace(0, duration_sec * 1000, n_samples)
    signal_12ch = np.zeros((n_samples, 12))

    # Per-lead amplitude patterns (mV) — based on the 4-question theory
    if pattern == 'RVOT':
        amps = {'I': 0.6, 'II': 1.8, 'III': 1.6, 'aVR': -1.0, 'aVL': -0.4, 'aVF': 1.9,
                'V1': -1.6, 'V2': -1.4, 'V3': -0.7, 'V4': 0.4, 'V5': 1.4, 'V6': 1.6}
        qrs_start, qrs_end = 200, 360
    elif pattern == 'RV':
        amps = {'I': 0.5, 'II': -0.8, 'III': -1.0, 'aVR': 0.6, 'aVL': 0.3, 'aVF': -0.9,
                'V1': -1.4, 'V2': -1.2, 'V3': -0.6, 'V4': 0.2, 'V5': 1.0, 'V6': 1.2}
        qrs_start, qrs_end = 200, 380
    elif pattern == 'LVOT':
        amps = {'I': 0.3, 'II': 1.7, 'III': 1.5, 'aVR': -0.9, 'aVL': -0.7, 'aVF': 1.7,
                'V1': 0.7, 'V2': -0.4, 'V3': 1.3, 'V4': 1.7, 'V5': 1.5, 'V6': 1.0}
        qrs_start, qrs_end = 200, 350
    elif pattern == 'LV':
        amps = {'I': -0.6, 'II': -0.9, 'III': -1.0, 'aVR': 0.7, 'aVL': 0.4, 'aVF': -1.0,
                'V1': 1.6, 'V2': 1.2, 'V3': 0.8, 'V4': 0.4, 'V5': -0.4, 'V6': -0.8}
        qrs_start, qrs_end = 200, 380
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Generate QRS waveform per lead
    for i, lead in enumerate(LEAD_ORDER):
        amp = amps[lead]
        for j, ti in enumerate(t):
            if qrs_start < ti < qrs_end:
                # Half-sine pulse, sign matches amplitude
                signal_12ch[j, i] = amp * np.sin(np.pi * (ti - qrs_start) / (qrs_end - qrs_start))
            elif qrs_end < ti < qrs_end + 100:
                # Discordant T wave (opposite sign to dominant deflection)
                signal_12ch[j, i] = -0.2 * amp * np.sin(np.pi * (ti - qrs_end) / 100)

    # Add tiny realistic noise
    rng = np.random.default_rng(42)
    signal_12ch += 0.02 * rng.standard_normal(signal_12ch.shape)
    return signal_12ch


def run_demo():
    """Run the 4 synthetic test cases and report accuracy."""
    print("=" * 72)
    print("12-LEAD PVC LOCALIZATION ALGORITHM — DEMO")
    print("=" * 72)
    print("\nGenerating synthetic 12-lead PVCs for each of the 4 regions...\n")

    correct = 0
    total = 0
    for pattern in ['RVOT', 'RV', 'LVOT', 'LV']:
        print(f"\n{'─' * 70}")
        print(f"  TEST: synthetic {pattern} PVC")
        print(f"{'─' * 70}")

        sig = make_synthetic_12lead_pvc(pattern, fs=500, duration_sec=0.8)
        result = localize_pvc_12lead(
            sig, fs=500,
            output_path=f'demo_{pattern.lower()}_12lead.png',
        )

        cf = result.cross_features
        print(f"  Inferior axis score: {cf.inferior_axis_score:+.2f}")
        print(f"  Precordial transition: V{cf.precordial_transition}")
        print(f"  V2 transition ratio: {cf.v2_transition_ratio:.2f}")
        print(f"  V2S/V3R index: {cf.v2s_v3r_index:.2f}")
        print(f"\n  4-region probabilities:")
        for chamber, prob in sorted(result.probabilities.items(), key=lambda x: -x[1]):
            bar = '█' * int(prob * 36)
            mark = ' ← TRUE' if chamber == pattern else ''
            print(f"    {chamber:5s}: {prob:6.1%}  {bar}{mark}")
        print(f"\n  Predicted: {result.most_likely}  (sub: {result.sub_localization})")

        is_correct = result.most_likely == pattern
        if is_correct:
            correct += 1
            print(f"  ✓ CORRECT")
        else:
            print(f"  ✗ WRONG (expected {pattern})")
        total += 1

    print(f"\n{'=' * 72}")
    print(f"OVERALL: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print(f"{'=' * 72}\n")
    return correct, total


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        run_demo()
    elif sys.argv[1] == '--demo':
        run_demo()
    else:
        # Real usage: load a CSV with 12 columns
        path = sys.argv[1]
        fs = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0
        sig = np.loadtxt(path, delimiter=',')
        result = localize_pvc_12lead(sig, fs=fs)

        print(f"\nPVC Origin Localization (12-lead)")
        print(f"=" * 50)
        print(f"\nProbabilities:")
        for chamber, prob in sorted(result.probabilities.items(), key=lambda x: -x[1]):
            print(f"  {chamber:5s}: {prob:.1%}")
        print(f"\nMost likely: {result.most_likely}")
        print(f"Sub-localization: {result.sub_localization}")
        print(f"Epicardial: {result.is_epicardial}")
        print(f"\nReasoning:")
        for line in result.reasoning:
            print(f"  • {line}")
