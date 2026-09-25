"""Metrics: calibration (ECE/MCE/NLL/Brier), ranking (AUROC/AURC), and per-sample scores."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp, softmax
from sklearn.metrics import roc_auc_score


# ---------------- per-sample scores from logits ----------------

def softmax_probs(logits: np.ndarray) -> np.ndarray:
    return softmax(logits, axis=-1)


def entropy_of_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def energy_score(logits: np.ndarray) -> np.ndarray:
    """energy = -logsumexp(z). Lower energy => more ID-like / more confident."""
    return -logsumexp(logits, axis=-1)


def top2_stats(logits: np.ndarray, probs: np.ndarray):
    """Return p_top1, p_top2, margin (logit), prob_margin, pred_class."""
    order = np.argsort(-logits, axis=-1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1]
    rows = np.arange(logits.shape[0])
    z1, z2 = logits[rows, top1_idx], logits[rows, top2_idx]
    p1, p2 = probs[rows, top1_idx], probs[rows, top2_idx]
    return p1, p2, (z1 - z2), (p1 - p2), top1_idx


def build_per_sample_table(logits: np.ndarray, true_class: np.ndarray) -> dict:
    probs = softmax_probs(logits)
    p1, p2, margin, prob_margin, pred_class = top2_stats(logits, probs)
    ent = entropy_of_probs(probs)
    energy = energy_score(logits)
    return dict(
        true_class=true_class,
        pred_class=pred_class,
        correct=(pred_class == true_class).astype(int),
        logits=logits,
        probs=probs,
        p_top1=p1,
        p_top2=p2,
        margin=margin,
        prob_margin=prob_margin,
        entropy=ent,
        energy=energy,
        logit_norm=np.linalg.norm(logits, axis=-1),
        logit_mean=logits.mean(axis=-1),
    )


# ---------------- calibration metrics ----------------

def nll(logits: np.ndarray, true_class: np.ndarray) -> float:
    logp = log_softmax(logits, axis=-1)
    return float(-logp[np.arange(len(true_class)), true_class].mean())


def brier_score(probs: np.ndarray, true_class: np.ndarray) -> float:
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[true_class]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=-1)))


def _binned_calibration(confidence: np.ndarray, correct: np.ndarray, bin_edges: np.ndarray):
    n = len(confidence)
    bin_ids = np.clip(np.digitize(confidence, bin_edges[1:-1], right=True), 0, len(bin_edges) - 2)
    accs, confs, counts = [], [], []
    for b in range(len(bin_edges) - 1):
        mask = bin_ids == b
        cnt = mask.sum()
        counts.append(cnt)
        if cnt == 0:
            accs.append(0.0)
            confs.append(0.0)
        else:
            accs.append(correct[mask].mean())
            confs.append(confidence[mask].mean())
    return np.array(accs), np.array(confs), np.array(counts), n


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15, equal_mass: bool = False):
    if equal_mass:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(confidence, quantiles)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([0.0, 1.0])
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    accs, confs, counts, n = _binned_calibration(confidence, correct, bin_edges)
    weights = counts / n
    gap = np.abs(accs - confs)
    return float(np.sum(weights * gap)), accs, confs, counts, bin_edges


def mce(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    accs, confs, counts, n = _binned_calibration(confidence, correct, bin_edges)
    nonempty = counts > 0
    if not nonempty.any():
        return 0.0
    return float(np.max(np.abs(accs[nonempty] - confs[nonempty])))


def overconfidence_gap(confidence: np.ndarray, correct: np.ndarray) -> float:
    return float(confidence.mean() - correct.mean())


# ---------------- temperature scaling ----------------

def fit_temperature(logits: np.ndarray, true_class: np.ndarray) -> float:
    """Fit scalar T minimizing NLL on the given (validation) logits. z' = z / T."""

    def loss(log_t):
        t = np.exp(log_t)
        return nll(logits / t, true_class)

    res = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


# ---------------- ranking metrics: AUROC / AURC ----------------

def auroc_misclassification(score_higher_is_correct: np.ndarray, correct: np.ndarray) -> float:
    """AUROC for separating correct (1) from incorrect (0) using `score` (higher => predicted correct)."""
    if len(np.unique(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, score_higher_is_correct))


def auroc_ood(score_higher_is_id: np.ndarray, is_id: np.ndarray) -> float:
    """AUROC for separating ID (1) from OOD (0) using `score` (higher => predicted ID)."""
    if len(np.unique(is_id)) < 2:
        return float("nan")
    return float(roc_auc_score(is_id, score_higher_is_id))


def aurc(score_higher_is_correct: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve. Sort by descending confidence, accumulate risk."""
    order = np.argsort(-score_higher_is_correct)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    cum_errors = np.cumsum(1 - correct_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = cum_errors / np.arange(1, n + 1)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(risk, coverage))
