"""
Dinucleotide periodicity analysis for nucleosome positioning signals.

Nucleosomal DNA wraps ~1.65 turns around the histone octamer, requiring
periodic bending every ~10.5 bp (one helical turn). This bending is
facilitated when WW dinucleotides (W = A or T: AA, AT, TA, TT) appear
where the minor groove faces the histone surface, and SS dinucleotides
(S = G or C: CC, CG, GC, GG) appear in anti-phase.

If UCOE candidate sequences lack this ~10.5 bp periodicity, it provides
a mechanistic explanation for nucleosome exclusion encoded directly in
the DNA sequence — the double helix cannot efficiently wrap around the
histone octamer.

Literature:
    Segal et al. (2006) Nature 442:772 — genomic code for nucleosome positioning
    Ioshikhes et al. (1996) J Mol Biol 262:129 — dinucleotide autocorrelation
    Trifonov & Sussman (1980) PNAS 77:3816 — 10.5 bp periodicity
    Satchwell et al. (1986) J Mol Biol 191:659 — AA/TT periodicity
    Drew & Travers (1985) J Mol Biol 186:773 — DNA bending and nucleosomes
    Widom (2001) Q Rev Biophys 34:269 — sequence determinants of positioning
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── Dinucleotide classification ──────────────────────────────────────────────
# WW (weak–weak): minor-groove facing in nucleosomal DNA
# SS (strong–strong): major-groove facing in nucleosomal DNA
# Segal et al. (2006) showed WW/SS oscillate in anti-phase at ~10 bp period.

WW_DINUCS = {"AA", "AT", "TA", "TT"}
SS_DINUCS = {"CC", "CG", "GC", "GG"}

# Nucleosomal period range for peak detection (bp)
# Theoretical: 10.5 bp (one helical turn of B-DNA)
# Empirical range from crystal structures: 10.0–10.5 bp
# We search 9–12 bp to accommodate sequence-dependent variation
PERIOD_MIN = 9.0
PERIOD_MAX = 12.0
EXPECTED_PERIOD = 10.4  # consensus from Widom (2001)

# Maximum autocorrelation lag (bp)
# 50 bp covers ~5 helical turns — sufficient to detect periodicity
# without excessive noise from distant correlations
MAX_LAG = 50


def _binary_signal(seq: str, dinuc_set: set[str]) -> np.ndarray:
    """Create binary occurrence signal for a dinucleotide class.

    For each position i in [0, len(seq)-2], signal[i] = 1 if
    seq[i:i+2] belongs to dinuc_set, else 0.

    Parameters
    ----------
    seq : uppercase DNA string
    dinuc_set : set of 2-letter dinucleotides (e.g., WW_DINUCS)

    Returns
    -------
    signal : binary array of length len(seq) - 1
    """
    n = len(seq)
    if n < 2:
        return np.array([])
    signal = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        if seq[i:i + 2] in dinuc_set:
            signal[i] = 1.0
    return signal


def autocorrelation(signal: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Compute normalized autocorrelation of a mean-centered signal.

    Uses the unbiased estimator (divided by n-k, not n) to avoid
    attenuating long-lag correlations in finite sequences.

    Following Ioshikhes et al. (1996): R(k) = C(k) / C(0), where
    C(k) = (1/(N-k)) * Σ_{i=0}^{N-k-1} x(i)*x(i+k), x = signal - mean.

    Parameters
    ----------
    signal : 1D array (binary occurrence signal)
    max_lag : maximum lag to compute (default 50 bp)

    Returns
    -------
    acf : array of shape (max_lag+1,) with R(0)=1.0, R(1), ..., R(max_lag)
    """
    n = len(signal)
    if n < max_lag + 2:
        max_lag = n - 2
    if max_lag < 1:
        return np.array([1.0])

    x = signal - np.mean(signal)
    c0 = np.dot(x, x) / n  # variance (biased for C(0) normalization)

    if c0 < 1e-15:
        return np.ones(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for k in range(1, max_lag + 1):
        # Unbiased cross-correlation at lag k
        ck = np.dot(x[:n - k], x[k:]) / (n - k)
        acf[k] = ck / c0

    return acf


def power_spectral_density(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density via FFT of mean-centered signal.

    Following Trifonov & Sussman (1980): FFT of the dinucleotide
    occurrence signal reveals periodicity peaks.

    Parameters
    ----------
    signal : 1D binary occurrence signal

    Returns
    -------
    periods : array of periods in bp (excluding DC component)
    psd : corresponding power spectral density values (normalized)
    """
    n = len(signal)
    if n < 10:
        return np.array([]), np.array([])

    x = signal - np.mean(signal)

    # FFT with zero-padding to next power of 2 for efficiency
    nfft = 2 ** int(np.ceil(np.log2(n)))
    fft_vals = np.fft.rfft(x, n=nfft)
    psd_raw = np.abs(fft_vals) ** 2 / n  # normalized by sequence length

    freqs = np.fft.rfftfreq(nfft, d=1.0)  # d=1 bp

    # Exclude DC component (index 0) and Nyquist
    valid = (freqs > 0) & (freqs < 0.5)
    freqs = freqs[valid]
    psd_raw = psd_raw[valid]

    # Convert frequency to period
    periods = 1.0 / freqs

    return periods, psd_raw


def periodicity_snr(
    periods: np.ndarray,
    psd: np.ndarray,
    period_min: float = PERIOD_MIN,
    period_max: float = PERIOD_MAX,
) -> tuple[float, float]:
    """Compute signal-to-noise ratio at the nucleosomal period.

    SNR = (peak power in target band) / (mean power outside target band)

    This metric is robust to sequence length because both numerator and
    denominator scale identically with N.

    Parameters
    ----------
    periods : array of periods from PSD
    psd : corresponding power values
    period_min, period_max : target period range (default 9–12 bp)

    Returns
    -------
    snr : signal-to-noise ratio (>1 means detectable periodicity)
    peak_period : period of the peak within the target range (bp)
    """
    if len(periods) == 0 or len(psd) == 0:
        return 0.0, 0.0

    # Target band: nucleosomal period (9-12 bp)
    in_band = (periods >= period_min) & (periods <= period_max)
    # Background: periods 3-50 bp, excluding target band
    # Exclude very long periods (>50 bp) which are composition, not periodicity
    background = (~in_band) & (periods >= 3.0) & (periods <= 50.0)

    if not np.any(in_band) or not np.any(background):
        return 0.0, 0.0

    peak_power = np.max(psd[in_band])
    peak_idx = np.argmax(psd[in_band])
    peak_period = periods[in_band][peak_idx]

    bg_mean = np.mean(psd[background])

    if bg_mean < 1e-15:
        return 0.0, peak_period

    snr = peak_power / bg_mean
    return float(snr), float(peak_period)


def ww_ss_phase_difference(
    acf_ww: np.ndarray,
    acf_ss: np.ndarray,
) -> float:
    """Estimate phase difference between WW and SS autocorrelation signals.

    In nucleosomal DNA, WW and SS oscillate in anti-phase (Δφ ≈ π).
    This means that when WW autocorrelation peaks at lag ~10 bp,
    SS should have a trough, and vice versa.

    We estimate the phase difference from the cross-correlation of
    the two ACF signals (lags 5-30 bp, covering ~2-3 periods).

    Parameters
    ----------
    acf_ww : WW autocorrelation array
    acf_ss : SS autocorrelation array

    Returns
    -------
    phase_diff : estimated phase difference in radians (π = anti-phase)
        Returns NaN if estimation fails.
    """
    # Use lags 5-30 bp for phase estimation (avoids lag-0 dominance)
    min_lag, max_lag = 5, 30
    if len(acf_ww) <= max_lag or len(acf_ss) <= max_lag:
        max_lag = min(len(acf_ww), len(acf_ss)) - 1
    if max_lag <= min_lag:
        return np.nan

    ww_segment = acf_ww[min_lag:max_lag + 1]
    ss_segment = acf_ss[min_lag:max_lag + 1]

    # Cross-correlation at lag 0 of the two ACF segments
    # Negative value → anti-phase; positive → in-phase
    ww_c = ww_segment - np.mean(ww_segment)
    ss_c = ss_segment - np.mean(ss_segment)

    norm = np.sqrt(np.dot(ww_c, ww_c) * np.dot(ss_c, ss_c))
    if norm < 1e-15:
        return np.nan

    cross_corr = np.dot(ww_c, ss_c) / norm

    # Convert correlation to phase: cos(Δφ) ≈ cross_corr
    # Anti-phase: cross_corr ≈ -1 → Δφ ≈ π
    # In-phase: cross_corr ≈ +1 → Δφ ≈ 0
    cross_corr = np.clip(cross_corr, -1.0, 1.0)
    phase_diff = np.arccos(cross_corr)

    return float(phase_diff)


def periodicity_metrics(seq: str) -> dict:
    """Compute all dinucleotide periodicity metrics for a DNA sequence.

    Returns
    -------
    dict with keys:
        ww_periodicity_snr : float
            SNR of WW periodicity at ~10.5 bp. Higher = stronger
            nucleosome-compatible periodicity. Values >2 indicate
            clear periodicity (Widom 2001).
        ss_periodicity_snr : float
            Same for SS dinucleotides.
        ww_peak_period : float
            Dominant period of WW signal in the 9–12 bp range.
        ss_peak_period : float
            Dominant period of SS signal in the 9–12 bp range.
        ww_ss_phase_diff : float
            Phase difference between WW and SS oscillations (radians).
            Expected ~π (3.14) for nucleosomal DNA (anti-phase).
        ww_autocorr_10bp : float
            WW autocorrelation value at lag 10 bp (point estimate).
            Positive = periodic at helical repeat.
        ss_autocorr_10bp : float
            SS autocorrelation at lag 10 bp.
        ww_fraction : float
            Overall fraction of WW dinucleotides.
        ss_fraction : float
            Overall fraction of SS dinucleotides.
    """
    seq = seq.upper().replace("N", "")
    n = len(seq)

    if n < 50:  # minimum length for meaningful periodicity analysis
        return {
            "ww_periodicity_snr": np.nan,
            "ss_periodicity_snr": np.nan,
            "ww_peak_period": np.nan,
            "ss_peak_period": np.nan,
            "ww_ss_phase_diff": np.nan,
            "ww_autocorr_10bp": np.nan,
            "ss_autocorr_10bp": np.nan,
            "ww_fraction": np.nan,
            "ss_fraction": np.nan,
        }

    # Binary occurrence signals
    sig_ww = _binary_signal(seq, WW_DINUCS)
    sig_ss = _binary_signal(seq, SS_DINUCS)

    # Autocorrelation (Ioshikhes et al. 1996)
    acf_ww = autocorrelation(sig_ww)
    acf_ss = autocorrelation(sig_ss)

    # Power spectral density (Trifonov & Sussman 1980)
    periods_ww, psd_ww = power_spectral_density(sig_ww)
    periods_ss, psd_ss = power_spectral_density(sig_ss)

    # SNR at nucleosomal period
    snr_ww, peak_ww = periodicity_snr(periods_ww, psd_ww)
    snr_ss, peak_ss = periodicity_snr(periods_ss, psd_ss)

    # Phase difference between WW and SS
    phase_diff = ww_ss_phase_difference(acf_ww, acf_ss)

    # Point estimate: autocorrelation at lag 10 bp
    acf_ww_10 = float(acf_ww[10]) if len(acf_ww) > 10 else np.nan
    acf_ss_10 = float(acf_ss[10]) if len(acf_ss) > 10 else np.nan

    # Global composition
    n_dinucs = len(sig_ww)
    ww_frac = float(np.sum(sig_ww)) / n_dinucs if n_dinucs > 0 else 0.0
    ss_frac = float(np.sum(sig_ss)) / n_dinucs if n_dinucs > 0 else 0.0

    return {
        "ww_periodicity_snr": snr_ww,
        "ss_periodicity_snr": snr_ss,
        "ww_peak_period": peak_ww,
        "ss_peak_period": peak_ss,
        "ww_ss_phase_diff": phase_diff,
        "ww_autocorr_10bp": acf_ww_10,
        "ss_autocorr_10bp": acf_ss_10,
        "ww_fraction": ww_frac,
        "ss_fraction": ss_frac,
    }
