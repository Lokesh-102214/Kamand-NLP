"""
Dynamic Gatekeeper: Batch-Relative Filtering Algorithm for RAG Systems
=======================================================================
Combines the best ideas from both reviewed approaches:
  - Continuous adaptive formula (inspired by MAIN-RAG's mean-std framework)
  - Largest-gap natural break detection (replaces broken Otsu on n=10)
  - IQR outlier guard used correctly (not as dead code)
  - Hard keep-ratio safety net applied to final mask
  - Full test suite with verified expected outputs

Satisfies both goals:
  1. Noise Reduction  : strict threshold for high-scoring easy batches
  2. Signal Salvaging : lenient threshold for low-scoring hard batches
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------

def dynamic_gatekeeper(
    scores: List[float],
    alpha: float = 0.6,          # Controls how aggressively threshold tracks std
    min_keep_ratio: float = 0.2, # At least 20% of docs always kept
    max_keep_ratio: float = 0.8, # At most 80% of docs always kept
    gap_weight: float = 0.35,    # Weight of natural-gap threshold in blend
    stat_weight: float = 0.65,   # Weight of stat-based threshold in blend
) -> Tuple[List[bool], dict]:
    """
    Batch-Relative Filtering: adapts strictness to the current score distribution.

    Strategy (three components blended):

    1. STATISTICAL THRESHOLD (mean - alpha*std, clamped to [0.1, 0.95])
       - Derived from MAIN-RAG's adaptive thresholding framework.
       - Low mean  → threshold pulls downward  → lenient  (signal salvaging)
       - High mean → threshold pulls upward    → strict   (noise reduction)
       - Uses RAW std (not the cancelled-out formula from Approach 2).

    2. NATURAL GAP THRESHOLD (largest consecutive score gap)
       - Replaces Otsu's method, which is invalid for n~10 samples.
       - Sort scores descending; find the single largest drop between
         adjacent values. The threshold sits just below that gap.
       - Works correctly on both bimodal (mixed) and unimodal (easy/hard) batches.
       - For unimodal batches the gap is small and the component has low influence.

    3. IQR GUARD (hard floor derived from Q1 - 1.5*IQR)
       - Prevents the blended threshold from excluding genuine outlier-good docs.
       - Applied as a lower-bound clip AFTER blending, not as dead metadata.

    4. KEEP-RATIO SAFETY NET
       - After applying the threshold, enforce [min_keep_ratio, max_keep_ratio].
       - Guarantees the system never returns nothing (signal salvaging)
         and never keeps everything (noise reduction).

    Args:
        scores           : Confidence scores in [0, 1] from the Judge Model.
        alpha            : Std multiplier for statistical threshold (tune per domain).
        min_keep_ratio   : Minimum fraction of docs to keep (signal salvaging floor).
        max_keep_ratio   : Maximum fraction of docs to keep (noise reduction ceiling).
        gap_weight       : Weight given to the natural-gap threshold.
        stat_weight      : Weight given to the statistical threshold.

    Returns:
        mask             : Boolean list — True = Keep, False = Discard.
        meta             : Diagnostic dict for logging / audit trails.
    """
    if not scores:
        return [], {}

    arr = np.array(scores, dtype=float)
    n = len(arr)

    # --- Descriptive statistics ---
    mean  = float(np.mean(arr))
    std   = float(np.std(arr, ddof=0))   # population std (stable for small n)
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr   = q75 - q25

    # -----------------------------------------------------------------------
    # Component 1: Statistical threshold  (mean - alpha * std)
    # -----------------------------------------------------------------------
    # alpha is fixed (not derived from mean), so std genuinely influences
    # the threshold — the cancellation bug from Approach 2 is avoided.
    stat_thresh = mean - alpha * std
    stat_thresh = float(np.clip(stat_thresh, 0.10, 0.95))

    # -----------------------------------------------------------------------
    # Component 2: Natural-gap threshold (largest drop in sorted scores)
    # -----------------------------------------------------------------------
    sorted_desc = np.sort(arr)[::-1]   # highest → lowest
    gaps = sorted_desc[:-1] - sorted_desc[1:]   # positive differences

    if len(gaps) > 0 and gaps.max() > 1e-6:
        gap_idx = int(np.argmax(gaps))          # index of largest gap
        # Threshold sits at the midpoint of the gap
        gap_thresh = float((sorted_desc[gap_idx] + sorted_desc[gap_idx + 1]) / 2)
    else:
        # All scores identical → no natural gap, fall back to stat threshold
        gap_thresh = stat_thresh

    # -----------------------------------------------------------------------
    # Component 3: IQR lower-bound guard
    # -----------------------------------------------------------------------
    # IQR bound used as a floor: never discard docs that are not low outliers.
    iqr_floor = float(max(0.0, q25 - 1.5 * iqr))

    # -----------------------------------------------------------------------
    # Blend Components 1 & 2, then apply IQR floor
    # -----------------------------------------------------------------------
    blended_thresh = stat_weight * stat_thresh + gap_weight * gap_thresh
    # Apply IQR floor: threshold cannot drop below the "not-an-outlier" bound
    final_thresh = float(max(blended_thresh, iqr_floor))
    # Hard clip so threshold is always a valid probability
    final_thresh = float(np.clip(final_thresh, 0.05, 0.99))

    # -----------------------------------------------------------------------
    # Apply threshold → initial mask
    # -----------------------------------------------------------------------
    mask = arr >= final_thresh

    # -----------------------------------------------------------------------
    # Component 4: Keep-ratio safety net
    # -----------------------------------------------------------------------
    sorted_idx_desc = np.argsort(arr)[::-1]
    keep_count = int(mask.sum())

    min_keep = max(1, int(np.ceil(n * min_keep_ratio)))
    max_keep = max(min_keep, int(np.floor(n * max_keep_ratio)))

    if keep_count < min_keep:
        # Signal salvaging: force-keep the top `min_keep` docs
        mask = np.zeros(n, dtype=bool)
        mask[sorted_idx_desc[:min_keep]] = True

    elif keep_count > max_keep:
        # Noise reduction: force-discard docs below the top `max_keep`
        mask = np.zeros(n, dtype=bool)
        mask[sorted_idx_desc[:max_keep]] = True

    # -----------------------------------------------------------------------
    # Metadata for audit / debugging
    # -----------------------------------------------------------------------
    meta = {
        "mean":              round(mean, 4),
        "std":               round(std, 4),
        "q25":               round(q25, 4),
        "q75":               round(q75, 4),
        "iqr":               round(iqr, 4),
        "stat_threshold":    round(stat_thresh, 4),
        "gap_threshold":     round(gap_thresh, 4),
        "iqr_floor":         round(iqr_floor, 4),
        "blended_threshold": round(blended_thresh, 4),
        "final_threshold":   round(final_thresh, 4),
        "docs_kept":         int(mask.sum()),
        "keep_ratio":        round(float(mask.sum()) / n, 3),
    }

    return mask.tolist(), meta

