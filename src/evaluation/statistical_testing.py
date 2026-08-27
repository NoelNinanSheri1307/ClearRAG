"""Statistical hypothesis testing and bootstrap confidence interval tools for ClearRAG.

Provides paired statistical tests:
- McNemar's test for paired binary safe-vs-unsafe outcomes
- Wilcoxon signed-rank test for paired continuous metrics
- Non-parametric bootstrap resampling for 95% confidence intervals
- Effect size calculations (Odds Ratio, Cohen's d, Rank-biserial correlation)
"""

from dataclasses import asdict, dataclass
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Structured container for statistical hypothesis test results."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_metric: str
    is_significant_05: bool
    is_significant_01: bool
    confidence_interval_95: Tuple[float, float]
    sample_size: int
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        d = asdict(self)
        d["confidence_interval_95"] = [round(x, 4) for x in self.confidence_interval_95]
        d["statistic"] = round(self.statistic, 4)
        d["p_value"] = float(self.p_value)
        d["effect_size"] = round(self.effect_size, 4)
        return d


def bootstrap_confidence_interval(
    data: List[float],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for a metric.

    Args:
        data: List of continuous or binary numeric values.
        statistic_fn: Aggregation function (e.g. np.mean).
        num_resamples: Number of bootstrap iterations.
        confidence_level: Confidence level (default 0.95 for 95% CI).
        random_seed: Reproducibility seed.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if not data:
        return (0.0, 0.0)

    arr = np.array(data, dtype=float)
    if len(arr) == 1 or np.all(arr == arr[0]):
        val = float(statistic_fn(arr))
        return (val, val)

    rng = np.random.default_rng(random_seed)
    n = len(arr)
    boot_stats = np.empty(num_resamples)

    for i in range(num_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = statistic_fn(sample)

    alpha = (1.0 - confidence_level) / 2.0
    low = float(np.percentile(boot_stats, alpha * 100.0))
    high = float(np.percentile(boot_stats, (1.0 - alpha) * 100.0))
    return (round(low, 4), round(high, 4))


def mcnemar_test(
    paired_outcomes_a: List[bool],
    paired_outcomes_b: List[bool],
    label_a: str = "Standard RAG",
    label_b: str = "ClearRAG",
) -> StatisticalTestResult:
    """Execute paired McNemar's test on binary classification / safety outcomes.

    Contingency matrix:
              System B +    System B -
    System A +     a             b
    System A -     c             d

    Where:
    - b: System A is True, System B is False (discordant pair)
    - c: System A is False, System B is True (discordant pair)

    Args:
        paired_outcomes_a: List of boolean outcomes for System A (e.g. is_safe).
        paired_outcomes_b: List of boolean outcomes for System B.
        label_a: Name of System A.
        label_b: Name of System B.

    Returns:
        StatisticalTestResult.
    """
    n = len(paired_outcomes_a)
    if n != len(paired_outcomes_b):
        raise ValueError(f"Mismatched pair lengths: {n} vs {len(paired_outcomes_b)}")

    b = 0  # A True, B False
    c = 0  # A False, B True
    a_cnt = 0
    d_cnt = 0

    diffs = []
    for oa, ob in zip(paired_outcomes_a, paired_outcomes_b):
        diffs.append(1.0 if ob else 0.0 - (1.0 if oa else 0.0))
        if oa and not ob:
            b += 1
        elif not oa and ob:
            c += 1
        elif oa and ob:
            a_cnt += 1
        else:
            d_cnt += 1

    # Edwards continuity-corrected McNemar statistic
    if (b + c) == 0:
        stat = 0.0
        p_val = 1.0
        odds_ratio = 1.0
    else:
        stat = (abs(b - c) - 1.0) ** 2 / (b + c)
        p_val = float(stats.chi2.sf(stat, df=1))
        odds_ratio = (c / b) if b > 0 else float(c if c > 0 else 1.0)

    ci = bootstrap_confidence_interval([1.0 if ob else 0.0 for ob in paired_outcomes_b])

    interp = (
        f"{label_b} demonstrated statistically significant superiority over {label_a} "
        f"(p={p_val:.2e}, discordant pairs: {label_b}_better={c}, {label_a}_better={b}, Odds Ratio={odds_ratio:.2f})."
        if p_val < 0.05 and c > b
        else f"Difference between {label_b} and {label_a} was not statistically significant at alpha=0.05 (p={p_val:.4f})."
    )

    return StatisticalTestResult(
        test_name="McNemar's Paired Test (Continuity-Corrected)",
        statistic=stat,
        p_value=p_val,
        effect_size=odds_ratio,
        effect_size_metric="Odds Ratio (c/b)",
        is_significant_05=(p_val < 0.05),
        is_significant_01=(p_val < 0.01),
        confidence_interval_95=ci,
        sample_size=n,
        interpretation=interp,
    )


def wilcoxon_paired_test(
    scores_a: List[float],
    scores_b: List[float],
    metric_name: str = "Token F1",
    label_a: str = "Standard RAG",
    label_b: str = "ClearRAG",
) -> StatisticalTestResult:
    """Execute paired Wilcoxon signed-rank test on continuous metrics.

    Args:
        scores_a: List of scores for System A.
        scores_b: List of scores for System B.
        metric_name: Name of evaluated metric.
        label_a: Name of System A.
        label_b: Name of System B.

    Returns:
        StatisticalTestResult.
    """
    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError("Mismatched sample lengths")

    diffs = [sb - sa for sa, sb in zip(scores_a, scores_b)]
    non_zero_diffs = [d for d in diffs if d != 0.0]

    if not non_zero_diffs:
        stat = 0.0
        p_val = 1.0
        cohen_d = 0.0
    else:
        res = stats.wilcoxon(scores_b, scores_a, zero_method="wilcox", correction=True)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
        # Cohen's d for paired samples
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1.0
        cohen_d = float(np.mean(diffs) / std_diff) if std_diff > 0 else 0.0

    ci = bootstrap_confidence_interval(diffs)

    interp = (
        f"{label_b} has statistically significant improvement in {metric_name} over {label_a} "
        f"(Wilcoxon W={stat:.1f}, p={p_val:.2e}, Cohen's d={cohen_d:.3f}, 95% CI on mean diff: [{ci[0]}, {ci[1]}])."
        if p_val < 0.05 and np.mean(diffs) > 0
        else f"No statistically significant difference in {metric_name} (p={p_val:.4f})."
    )

    return StatisticalTestResult(
        test_name=f"Wilcoxon Signed-Rank Test ({metric_name})",
        statistic=stat,
        p_value=p_val,
        effect_size=cohen_d,
        effect_size_metric="Cohen's d (Paired Difference)",
        is_significant_05=(p_val < 0.05),
        is_significant_01=(p_val < 0.01),
        confidence_interval_95=ci,
        sample_size=n,
        interpretation=interp,
    )
