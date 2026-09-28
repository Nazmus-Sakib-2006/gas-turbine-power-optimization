#!/usr/bin/env python3
"""
Gas Turbine Energy Yield (TEY) - Decision Tree pipeline
=======================================================

Predicts turbine energy yield (``TEY``) from eight process / ambient sensors and
turns the predictions into rule-based operational advice.

Workflow (one function or small group of functions per step)
-----------------------------------------------------------
1. ``load_and_merge`` / ``summarize_data``   - merge gt_2011..gt_2015, audit the result
2. ``clean_data``                            - physical sanity checks, NaN/inf, IQR outliers
3. ``split_data`` / ``make_tiers``           - 80/20 split, quantile-based performance tiers
4. ``regression_metrics`` / ``classification_metrics`` + ``assess_*_overfit``
5. ``run_search``                            - Grid/RandomizedSearchCV (max_depth, min_samples_split,
                                               min_samples_leaf, max_features, ccp_alpha)
6. ``compute_validation_curve`` + ``plot_*`` - diagnostics saved as PNG
7. ``predict_tey`` / ``recommend_actions``   - prediction engine + operational action module
8. ``print_executive_summary``

Usage
-----
    python decision_tree_pipeline.py --data-dir path/to/csvs --output-dir path/to/plots

or, from Python::

    import decision_tree_pipeline as dtp
    art = dtp.run_pipeline(dtp.Config(data_dir=Path("data")))
    dtp.predict_tey({"TIT": 1090, "TAT": 549, "CDP": 12.4, "GTEP": 26, "AFDP": 4.0,
                     "AT": 15, "AP": 1013, "AH": 80}, art)

Design notes
------------
* Trees are scale-invariant, so no feature scaling is applied.
* ``CO`` and ``NOX`` are emissions (outputs of combustion), so they are excluded from the
  predictors. ``year`` is only used for bookkeeping and the temporal robustness check.
* Physical limits: pressures and turbine temperatures must be > 0. Ambient temperature (AT) is
  in degrees Celsius and legitimately goes below zero in winter (min -6.2 in this data), so
  only absolute zero is treated as a hard bound for AT. Humidity may overshoot 100 % slightly
  because of sensor tolerance (see ``PHYSICAL_LIMITS``).
* IQR fence: the default ``k = 5`` only removes *extreme* readings. TAT has a long, genuine lower
  tail (full-load operation); a conventional 1.5 x IQR fence would delete about 15 % of the data
  and a 3 x IQR fence ~3.7 %, mostly real high-load states. ``clean_data`` prints how many rows
  each fence would remove so the trade-off is visible.
"""
from __future__ import annotations

import argparse
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split,
                                     validation_curve)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
FEATURES = ("TIT", "TAT", "CDP", "GTEP", "AFDP", "AT", "AP", "AH")

# column: (lower, upper). A value <= lower or > upper is unphysical; None = unbounded.
PHYSICAL_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "AT": (-273.15, None),   # deg C: sub-zero is legitimate, only absolute zero is impossible
    "AP": (0.0, None),
    "AH": (0.0, 101.0),      # % RH; 1 point of sensor overshoot above 100 tolerated
    "AFDP": (0.0, None),
    "GTEP": (0.0, None),
    "TIT": (0.0, None),
    "TAT": (0.0, None),
    "CDP": (0.0, None),
    "TEY": (0.0, None),
}


@dataclass
class Config:
    """All tunable settings in one place."""
    data_dir: Path = Path(".")
    output_dir: Path = Path(".")
    years: tuple[int, ...] = (2011, 2012, 2013, 2014, 2015)
    file_pattern: str = "gt_{year}.csv"
    target: str = "TEY"
    features: tuple[str, ...] = FEATURES
    excluded: tuple[str, ...] = ("CO", "NOX")
    test_size: float = 0.20
    random_state: int = 42
    iqr_k: float = 5.0                    # IQR fence multiplier (see module docstring)
    tier_labels: tuple[str, ...] = ("Low", "Medium", "High")
    # Overfitting rule: flagged if ANY of these is exceeded
    reg_r2_gap: float = 0.01              # train R2 - test R2
    reg_rmse_ratio: float = 1.25          # test RMSE / train RMSE
    clf_f1_gap: float = 0.02              # train macro-F1 - test macro-F1
    # Hyperparameter search
    search: str = "random"                # "random" or "grid"
    n_iter: int = 80
    cv_folds: int = 5
    n_jobs: int = -1
    force_tune: bool = False              # tune even when no overfitting is detected
    max_depth_curve: int = 25
    # Advisor
    tey_target: float | None = None       # None -> Low/Medium tier boundary of the training set
    band_quantiles: tuple[float, float] = (0.10, 0.90)
    min_severity: float = 0.05            # fraction of band width a value must exceed the edge by
    temporal_check: bool = True


# --------------------------------------------------------------------------------------
# 1. Data loading & merging
# --------------------------------------------------------------------------------------
def load_and_merge(cfg: Config) -> pd.DataFrame:
    """Read every annual CSV, tag rows with their ``year`` and concatenate them."""
    frames = []
    for year in cfg.years:
        path = cfg.data_dir / cfg.file_pattern.format(year=year)
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        frame = pd.read_csv(path)
        frame["year"] = year
        frames.append(frame)

    reference = set(frames[0].columns)
    for year, frame in zip(cfg.years, frames):
        if set(frame.columns) != reference:
            raise ValueError(f"Column mismatch in {year}: {sorted(set(frame.columns) ^ reference)}")
    return pd.concat(frames, ignore_index=True)


def summarize_data(df: pd.DataFrame) -> None:
    """Print rows per year, a missing/infinite value audit and dtypes."""
    banner("1. DATA INTEGRATION SUMMARY")
    print("Rows per year:")
    print(df.groupby("year").size().rename("rows").to_frame().T.to_string())
    print(f"Total rows: {len(df):,}   Columns: {df.shape[1]}\n")

    numeric = df.select_dtypes("number")
    audit = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "infinite": np.isinf(numeric).sum().reindex(df.columns).fillna(0).astype(int),
    })
    print("Missing value / dtype audit:")
    print(audit.to_string())
    print(f"\nExact duplicate rows (all measurement columns): "
          f"{df.duplicated(subset=[c for c in df.columns if c != 'year']).sum()}")


# --------------------------------------------------------------------------------------
# 2. Cleaning & physical sanity checks
# --------------------------------------------------------------------------------------
def physical_violations(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Boolean frame, True where a value breaks ``PHYSICAL_LIMITS``."""
    cols = [c for c in (columns or PHYSICAL_LIMITS) if c in df.columns]
    out = pd.DataFrame(False, index=df.index, columns=cols)
    for col in cols:
        lower, upper = PHYSICAL_LIMITS[col]
        if lower is not None:
            out[col] |= df[col] <= lower
        if upper is not None:
            out[col] |= df[col] > upper
    return out


def _iqr_outliers(df: pd.DataFrame, cols: Sequence[str], k: float) -> pd.DataFrame:
    q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
    iqr = q3 - q1
    return (df[cols] < q1 - k * iqr) | (df[cols] > q3 + k * iqr)


def clean_data(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Duplicates -> NaN/inf -> physical limits -> IQR outliers. Returns (clean_df, steps)."""
    banner("2. CLEANING & PHYSICAL SANITY CHECKS")
    cols = list(cfg.features) + [cfg.target]
    work = df.copy()
    steps: list[tuple[str, int]] = []

    n = len(work)
    work = work.drop_duplicates(subset=[c for c in work.columns if c != "year"])
    steps.append(("Exact duplicate rows", n - len(work)))

    work[cols] = work[cols].replace([np.inf, -np.inf], np.nan)
    n = len(work)
    work = work.dropna(subset=cols)
    steps.append(("Missing / infinite values", n - len(work)))

    viol = physical_violations(work, cols)
    n = len(work)
    if viol.any(axis=None):
        print("Physical-limit violations per column:", viol.sum()[viol.sum() > 0].to_dict())
    work = work.loc[~viol.any(axis=1)]
    steps.append(("Physical-limit violations", n - len(work)))

    print("Rows an IQR fence would remove (sensitivity; only the configured k is applied):")
    for k in (1.5, 3.0, cfg.iqr_k):
        mask = _iqr_outliers(work, cols, k)
        per_col = mask.sum()[mask.sum() > 0].to_dict()
        print(f"  k={k:<4}: {int(mask.any(axis=1).sum()):>5} rows   by column: {per_col}")
    n = len(work)
    work = work.loc[~_iqr_outliers(work, cols, cfg.iqr_k).any(axis=1)]
    steps.append((f"IQR outliers (k={cfg.iqr_k})", n - len(work)))

    print("\nCleaning steps:")
    for name, removed in steps:
        print(f"  {name:<32} -{removed:>5}")
    print(f"  Rows: {len(df):,} -> {len(work):,}  ({len(df) - len(work):,} removed, "
          f"{(len(df) - len(work)) / len(df):.2%})")
    return work.reset_index(drop=True), steps


# --------------------------------------------------------------------------------------
# 3. Splitting & performance tiers
# --------------------------------------------------------------------------------------
def split_data(df: pd.DataFrame, cfg: Config):
    """Single 80/20 split (random_state=42) shared by the regressor and the classifier."""
    X, y = df[list(cfg.features)], df[cfg.target]
    return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)


def make_tiers(y_train: pd.Series, y_test: pd.Series, labels: Sequence[str]):
    """Quantile tiers (terciles for 3 labels). Edges come from the TRAIN set only (no leakage)."""
    qs = np.linspace(0, 1, len(labels) + 1)[1:-1]
    edges = y_train.quantile(qs).to_numpy()
    bins = np.r_[-np.inf, edges, np.inf]

    def to_tier(y: pd.Series) -> pd.Series:
        return pd.cut(y, bins=bins, labels=list(labels)).astype(str)

    return to_tier(y_train), to_tier(y_test), edges


# --------------------------------------------------------------------------------------
# 4. Metrics & overfitting assessment
# --------------------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    kw = dict(zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", **kw),
        "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", **kw),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", **kw),
        "Recall (weighted)": recall_score(y_true, y_pred, average="weighted", **kw),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", **kw),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", **kw),
    }


def print_metrics(title: str, train: dict, test: dict) -> None:
    table = pd.DataFrame({"Train": train, "Test": test})
    table["Gap (Train-Test)"] = table["Train"] - table["Test"]
    print(f"\n{title}")
    print(table.round(4).to_string())


def assess_regression_overfit(train: dict, test: dict, cfg: Config) -> tuple[bool, list[str]]:
    reasons = []
    r2_gap = train["R2"] - test["R2"]
    if r2_gap > cfg.reg_r2_gap:
        reasons.append(f"R2 gap {r2_gap:.4f} > {cfg.reg_r2_gap}")
    ratio = test["RMSE"] / train["RMSE"] if train["RMSE"] > 1e-12 else np.inf
    if ratio > cfg.reg_rmse_ratio:
        shown = "inf (train RMSE ~ 0)" if np.isinf(ratio) else f"{ratio:.2f}"
        reasons.append(f"test/train RMSE ratio {shown} > {cfg.reg_rmse_ratio}")
    return bool(reasons), reasons


def assess_classification_overfit(train: dict, test: dict, cfg: Config) -> tuple[bool, list[str]]:
    gap = train["F1 (macro)"] - test["F1 (macro)"]
    reasons = [f"macro-F1 gap {gap:.4f} > {cfg.clf_f1_gap}"] if gap > cfg.clf_f1_gap else []
    return bool(reasons), reasons


# --------------------------------------------------------------------------------------
# 5. Hyperparameter search
# --------------------------------------------------------------------------------------
def param_space(kind: str) -> dict[str, list]:
    """Search space. ccp_alpha scales differ: MSE impurity (regression) vs Gini (classification)."""
    space = {
        "max_depth": [4, 6, 8, 10, 12, 16, 20, None],
        "min_samples_split": [2, 5, 10, 20, 50],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "max_features": [None, 0.8, 0.6, "sqrt"],
    }
    space["ccp_alpha"] = ([0.0, 1e-4, 1e-3, 1e-2] if kind == "regression"
                          else [0.0, 1e-5, 1e-4, 1e-3])
    return space


def run_search(estimator, space: dict, scoring: str, cv, cfg: Config, X, y, label: str):
    """GridSearchCV, or RandomizedSearchCV when the grid is larger than ``cfg.n_iter``."""
    grid_size = int(np.prod([len(v) for v in space.values()]))
    kwargs = dict(scoring=scoring, cv=cv, n_jobs=cfg.n_jobs, refit=True)
    if cfg.search == "random" and cfg.n_iter < grid_size:
        search = RandomizedSearchCV(estimator, space, n_iter=cfg.n_iter,
                                    random_state=cfg.random_state, **kwargs)
        desc = f"RandomizedSearchCV ({cfg.n_iter} of {grid_size} combinations)"
    else:
        search = GridSearchCV(estimator, space, **kwargs)
        desc = f"GridSearchCV ({grid_size} combinations)"
    print(f"\n[{label}] {desc} x {cv.get_n_splits()} folds, scoring={scoring} ...")
    search.fit(X, y)
    return search


def top_candidates(search, n: int = 5, negate: bool = False) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(n)
    sign = -1 if negate else 1
    out = pd.DataFrame({"mean_cv": sign * res["mean_test_score"], "std_cv": res["std_test_score"],
                        "params": res["params"].astype(str)})
    return out


# --------------------------------------------------------------------------------------
# 6. Validation curves & plots
# --------------------------------------------------------------------------------------
def compute_validation_curve(estimator, X, y, scoring: str, cv, cfg: Config) -> pd.DataFrame:
    """Train vs cross-validated score across tree depths (other params = tuned values)."""
    depths = list(range(1, cfg.max_depth_curve + 1))
    tr, va = validation_curve(clone(estimator), X, y, param_name="max_depth", param_range=depths,
                              cv=cv, scoring=scoring, n_jobs=cfg.n_jobs)
    return pd.DataFrame({"depth": depths, "train_mean": tr.mean(1), "train_std": tr.std(1),
                         "cv_mean": va.mean(1), "cv_std": va.std(1)})


def plot_validation_curves(vc_reg: pd.DataFrame, vc_clf: pd.DataFrame, tuned_depth_reg,
                           tuned_depth_clf, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    panels = [(axes[0], vc_reg, "R$^2$", "Regressor (TEY)", tuned_depth_reg),
              (axes[1], vc_clf, "Macro F1", "Classifier (performance tier)", tuned_depth_clf)]
    for ax, vc, ylabel, title, tuned_depth in panels:
        for col, colour, name in (("train", "tab:blue", "Train"), ("cv", "tab:orange", "CV (validation)")):
            ax.plot(vc["depth"], vc[f"{col}_mean"], "o-", color=colour, label=name, ms=4)
            ax.fill_between(vc["depth"], vc[f"{col}_mean"] - vc[f"{col}_std"],
                            vc[f"{col}_mean"] + vc[f"{col}_std"], color=colour, alpha=0.15)
        best = vc.loc[vc["cv_mean"].idxmax()]
        ax.axvline(best["depth"], color="tab:green", ls="--",
                   label=f"Best CV depth = {int(best['depth'])} ({best['cv_mean']:.4f})")
        if tuned_depth is not None:
            ax.axvline(tuned_depth, color="tab:red", ls=":", label=f"Tuned model depth = {tuned_depth}")
        tuned_txt = "unlimited" if tuned_depth is None else str(tuned_depth)
        ax.set_title(f"{title}\nTuned-model max_depth = {tuned_txt}")
        # zoom on the informative region; the shallow depths (1-3) fall below the axis
        floor = vc.loc[vc["depth"] >= 4, ["train_mean", "cv_mean"]].min().min()
        ax.set_ylim(floor - 0.01, min(1.0, vc[["train_mean", "cv_mean"]].max().max() + 0.005))
        ax.text(0.02, 0.95, "depths 1-3 lie below the axis range", transform=ax.transAxes,
                fontsize=8, color="gray")
        ax.set_xlabel("max_depth")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower right")
    fig.suptitle("Validation curves - performance vs. tree depth (other hyperparameters at tuned values)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, labels: Sequence[str], path: Path) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                ax=axes[0], cbar=False)
    axes[0].set_title("Counts")
    sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".1%", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1], vmin=0, vmax=1, cbar=False)
    axes[1].set_title("Row-normalised (recall per tier)")
    for ax in axes:
        ax.set_xlabel("Predicted tier")
        ax.set_ylabel("Actual tier")
    fig.suptitle("Tuned DecisionTreeClassifier - test-set confusion matrix", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_feature_importance(imp_reg: pd.Series, imp_clf: pd.Series, path: Path) -> None:
    order = imp_reg.sort_values(ascending=False).index
    long = pd.concat([imp_reg.rename("Regressor (TEY)"), imp_clf.rename("Classifier (tier)")], axis=1)
    long = long.loc[order].reset_index(names="Feature").melt("Feature", var_name="Model",
                                                            value_name="Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=long, x="Feature", y="Importance", hue="Model", ax=ax)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=8, padding=2)
    ax.set_title("Relative feature importance (impurity-based, tuned models)")
    ax.set_ylabel("Importance (sums to 1 per model)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 7. Prediction engine & operational recommendations
# --------------------------------------------------------------------------------------
# Rule table: which side of the "normal operation" band is unfavourable and what to do about it.
# NOTE: these directions are engineering priors, not learned by the tree. In particular AFDP is
# POSITIVELY correlated with TEY in the data (it scales with air flow / load), so the AFDP rule
# only fires when AFDP is high in absolute terms, as a filter-fouling alarm.
ACTION_RULES: dict[str, dict[str, str]] = {
    "CDP": {"bad": "low", "kind": "controllable",
            "msg": "Check compressor discharge pressure (CDP): inspect the compressor for fouling or "
                   "erosion, and verify IGV / variable-vane scheduling and bleed-valve positions."},
    "TIT": {"bad": "low", "kind": "controllable",
            "msg": "Optimize turbine inlet temperature (TIT): review fuel flow and firing-temperature "
                   "control, staying within OEM temperature limits."},
    "GTEP": {"bad": "low", "kind": "controllable",
             "msg": "GTEP is low: verify the load set-point and flow through the turbine, and inspect "
                    "the exhaust path for anomalies."},
    "AFDP": {"bad": "high", "kind": "controllable",
             "msg": "Inspect air filters due to elevated AFDP pressure drop; clean or replace filters "
                    "and check the inlet system for fouling."},
    "TAT": {"bad": "high", "kind": "controllable",
            "msg": "TAT is high: check hot-gas-path condition (blade wear, tip clearances) and "
                   "combustion tuning."},
    "AT": {"bad": "high", "kind": "environmental",
           "msg": "High ambient temperature derates output: consider inlet air cooling (evaporative "
                  "cooler / fogging) or shifting high-load operation to cooler hours."},
    "AP": {"bad": "low", "kind": "environmental",
           "msg": "Low ambient pressure reduces air mass flow; not controllable, so adjust load "
                  "expectations."},
    "AH": {"bad": "high", "kind": "environmental",
           "msg": "High ambient humidity: small derating effect; check inlet filter moisture loading "
                  "(it can also raise AFDP)."},
}


@dataclass
class PipelineArtifacts:
    """Everything needed after training: models, reference statistics and results."""
    config: Config
    reg_model: DecisionTreeRegressor
    clf_model: DecisionTreeClassifier
    tier_labels: tuple[str, ...]
    tier_edges: np.ndarray
    tey_target: float
    importances_reg: pd.Series
    importances_clf: pd.Series
    normal_band: pd.DataFrame            # per feature: lo / hi / median of normal-operation rows
    train_ranges: pd.DataFrame           # per feature: min / max seen in training
    results: dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return list(self.config.features)


def build_normal_band(X_train: pd.DataFrame, y_train: pd.Series, target: float,
                      quantiles: tuple[float, float]) -> pd.DataFrame:
    """Reference distribution of sensor values on training rows that met the TEY target."""
    normal = X_train[y_train >= target]
    return pd.DataFrame({"lo": normal.quantile(quantiles[0]), "hi": normal.quantile(quantiles[1]),
                         "median": normal.median()})


def _prepare_inputs(inputs, art: PipelineArtifacts) -> pd.DataFrame:
    """Accept a dict (scalars or equal-length lists), list of dicts, or DataFrame; validate it."""
    if isinstance(inputs, pd.DataFrame):
        frame = inputs.copy()
    elif isinstance(inputs, Mapping):
        scalar = all(np.isscalar(v) for v in inputs.values())
        frame = pd.DataFrame([inputs]) if scalar else pd.DataFrame(inputs)
    elif isinstance(inputs, (list, tuple)):
        frame = pd.DataFrame(list(inputs))
    else:
        raise TypeError("inputs must be a dict, a list of dicts or a pandas DataFrame")

    missing = [f for f in art.features if f not in frame.columns]
    if missing:
        raise ValueError(f"Missing required sensor inputs: {missing}")
    X = frame[art.features].apply(pd.to_numeric, errors="raise")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Inputs contain NaN or infinite values")
    bad = physical_violations(X, art.features)
    if bad.any(axis=None):
        raise ValueError(f"Physically impossible inputs: {bad.sum()[bad.sum() > 0].to_dict()} row(s)")

    rng = art.train_ranges
    outside = [f for f in art.features
               if ((X[f] < rng.loc[f, "min"]) | (X[f] > rng.loc[f, "max"])).any()]
    if outside:
        warnings.warn(f"Inputs outside the training range for {outside}: a decision tree cannot "
                      "extrapolate, so predictions are held at the nearest training leaf.",
                      stacklevel=3)
    return X


def predict_tey(inputs, art: PipelineArtifacts) -> pd.Series:
    """Predict TEY for one or many sensor readings (dict / list of dicts / DataFrame)."""
    X = _prepare_inputs(inputs, art)
    return pd.Series(art.reg_model.predict(X), index=X.index, name="TEY_pred")


def recommend_actions(inputs, art: PipelineArtifacts, max_actions: int = 4) -> list[dict]:
    """Predict TEY and, when it is below target, rank likely drivers and suggest actions.

    A driver is flagged when its value lies beyond the unfavourable edge of the normal-operation
    band (see ``build_normal_band``) by at least ``min_severity`` x band width. Flagged drivers
    are ranked by ``importance x severity``.
    """
    cfg = art.config
    X = _prepare_inputs(inputs, art)
    preds = art.reg_model.predict(X)
    tiers = art.clf_model.predict(X)
    results = []
    for (idx, row), pred, tier in zip(X.iterrows(), preds, tiers):
        res: dict[str, Any] = {"index": idx, "predicted_TEY": float(pred), "predicted_tier": str(tier),
                               "target_TEY": art.tey_target, "below_target": bool(pred < art.tey_target),
                               "findings": [], "advice": []}
        if not res["below_target"]:
            res["advice"] = ["Predicted TEY meets the operational target - no corrective action needed."]
            results.append(res)
            continue

        for feat, rule in ACTION_RULES.items():
            lo, hi = art.normal_band.loc[feat, ["lo", "hi"]]
            width = max(hi - lo, 1e-9)
            value = row[feat]
            edge, excess = (lo, lo - value) if rule["bad"] == "low" else (hi, value - hi)
            severity = excess / width
            if severity >= cfg.min_severity:
                res["findings"].append({
                    "feature": feat, "value": float(value), "edge": float(edge), "bad": rule["bad"],
                    "kind": rule["kind"], "severity": float(severity),
                    "priority": float(art.importances_reg[feat] * severity),
                    "text": f"{feat}={value:.2f} is {'below' if rule['bad'] == 'low' else 'above'} "
                            f"the normal-operation band edge ({edge:.2f}). {rule['msg']}"})
        res["findings"].sort(key=lambda f: f["priority"], reverse=True)
        res["advice"] = [f["text"] for f in res["findings"][:max_actions]]
        if not res["advice"]:
            res["advice"] = ["No single input is outside its normal band; the shortfall looks like a "
                             "combined effect. Consider a performance test, a compressor wash and a "
                             "review of the load set-point."]
        results.append(res)
    return results


def format_advice(res: dict) -> str:
    head = (f"Predicted TEY = {res['predicted_TEY']:.2f}  (tier: {res['predicted_tier']}, "
            f"target >= {res['target_TEY']:.2f}) -> "
            f"{'BELOW target' if res['below_target'] else 'meets target'}")
    body = "\n".join(textwrap.fill(f"{i}. {a}", 100, initial_indent="    ", subsequent_indent="       ")
                     for i, a in enumerate(res["advice"], 1))
    return f"{head}\n{body}"


# --------------------------------------------------------------------------------------
# Robustness check & reporting
# --------------------------------------------------------------------------------------
def temporal_holdout_check(df: pd.DataFrame, cfg: Config, tuned_params: dict) -> dict[str, dict]:
    """Train on all years but the last, test on the last year (readings are hourly time series,
    so a random split can be optimistic)."""
    last = max(cfg.years)
    train, test = df[df["year"] < last], df[df["year"] == last]
    out = {}
    for name, params in (("initial (unpruned)", {}), ("tuned", tuned_params)):
        model = DecisionTreeRegressor(random_state=cfg.random_state, **params)
        model.fit(train[list(cfg.features)], train[cfg.target])
        out[name] = regression_metrics(test[cfg.target], model.predict(test[list(cfg.features)]))
    out["_meta"] = {"train_years": f"{min(cfg.years)}-{last - 1}", "test_year": last,
                    "n_train": len(train), "n_test": len(test)}
    return out


def banner(title: str) -> None:
    print("\n" + "=" * 92 + f"\n{title}\n" + "=" * 92)


def print_executive_summary(art: PipelineArtifacts, cleaning_steps, sample_outputs) -> None:
    r = art.results
    banner("EXECUTIVE SUMMARY")
    print(f"Data: {r['n_raw']:,} hourly records (2011-2015) -> {r['n_clean']:,} after cleaning "
          f"({r['n_raw'] - r['n_clean']:,} removed). Train/test = {r['n_train']:,}/{r['n_test']:,}.")

    print("\nREGRESSION - DecisionTreeRegressor (target TEY)")
    for name in ("initial", "tuned"):
        tr, te = r["reg"][f"{name}_train"], r["reg"][f"{name}_test"]
        print(f"  {name:<8} Train RMSE {tr['RMSE']:.3f} | R2 {tr['R2']:.4f}   ->   "
              f"Test MSE {te['MSE']:.3f} | RMSE {te['RMSE']:.3f} | MAE {te['MAE']:.3f} | R2 {te['R2']:.4f}")
    print(f"  Overfitting detected on initial model: {r['reg']['overfit']} "
          f"({'; '.join(r['reg']['reasons']) or 'no gap above thresholds'})")
    print(f"  Tuning: {r['reg']['tuned_note']}")
    print(f"  Optimal hyperparameters: {r['reg']['best_params']}")
    print(f"  Best CV RMSE: {r['reg']['cv_score']:.3f}   |   residual train-test R2 gap after tuning: "
          f"{r['reg']['residual_gap']:.4f}")

    print("\nCLASSIFICATION - DecisionTreeClassifier (tiers by TEY quantiles)")
    print("  Tier edges (train TEY): " + ", ".join(f"{e:.2f}" for e in art.tier_edges) +
          f"  -> {' / '.join(art.tier_labels)}")
    for name in ("initial", "tuned"):
        te = r["clf"][f"{name}_test"]
        print(f"  {name:<8} Test Acc {te['Accuracy']:.4f} | Precision(m/w) {te['Precision (macro)']:.4f}/"
              f"{te['Precision (weighted)']:.4f} | Recall(m/w) {te['Recall (macro)']:.4f}/"
              f"{te['Recall (weighted)']:.4f} | F1(m/w) {te['F1 (macro)']:.4f}/{te['F1 (weighted)']:.4f}")
    print(f"  Overfitting detected on initial model: {r['clf']['overfit']} "
          f"({'; '.join(r['clf']['reasons']) or 'no gap above thresholds'})")
    print(f"  Tuning: {r['clf']['tuned_note']}")
    print(f"  Optimal hyperparameters: {r['clf']['best_params']}")
    print(f"  Best CV macro-F1: {r['clf']['cv_score']:.4f}   |   residual train-test macro-F1 gap after "
          f"tuning: {r['clf']['residual_gap']:.4f}")
    print("  Confusion matrix (rows = actual, cols = predicted; " + ", ".join(art.tier_labels) + "):")
    for lab, row in zip(art.tier_labels, r["clf"]["confusion"]):
        print(f"    {lab:<7} {row.tolist()}")

    print("\nFEATURE IMPORTANCE (tuned regressor)")
    print("  " + ", ".join(f"{k} {v:.3f}" for k, v in art.importances_reg.sort_values(ascending=False).items()))

    if "temporal" in r:
        t = r["temporal"]
        meta = t["_meta"]
        print(f"\nTEMPORAL ROBUSTNESS (train {meta['train_years']}, test {meta['test_year']})")
        for name in ("initial (unpruned)", "tuned"):
            print(f"  {name:<19} RMSE {t[name]['RMSE']:.3f} | MAE {t[name]['MAE']:.3f} | R2 {t[name]['R2']:.4f}")

    print(f"\nOPERATIONAL ACTION SAMPLES (below-target rule: predicted TEY < {art.tey_target:.2f}, "
          "the Low/Medium tier boundary)")
    for title, res in sample_outputs:
        print(f"\n  >> {title}")
        print(textwrap.indent(format_advice(res), "  "))

    print("\nNOTES")
    print("  * Split is random per the spec; hourly readings are autocorrelated, so random-split scores "
          "\n    are optimistic - compare with the temporal check above.")
    print("  * Impurity importances are shared arbitrarily among correlated inputs (CDP, GTEP, TIT).")
    print("  * Advice rules are engineering priors ranked by importance x deviation; they are not causal.")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_pipeline(cfg: Config) -> PipelineArtifacts:
    sns.set_theme(style="whitegrid", context="notebook")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    feats = list(cfg.features)

    # 1-2. data -------------------------------------------------------------------------
    raw = load_and_merge(cfg)
    summarize_data(raw)
    df, cleaning_steps = clean_data(raw, cfg)
    assert not set(cfg.excluded) & set(feats), "emission variables must not be predictors"

    # 3. split & tiers ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    tier_train, tier_test, edges = make_tiers(y_train, y_test, cfg.tier_labels)
    tey_target = cfg.tey_target if cfg.tey_target is not None else float(edges[0])
    results: dict[str, Any] = {"n_raw": len(raw), "n_clean": len(df),
                               "n_train": len(X_train), "n_test": len(X_test)}

    reg_cv = KFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    clf_cv = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    # 4-5a. regression ---------------------------------------------------------------------
    banner("3-4. DECISION TREE REGRESSOR (TEY)")
    reg0 = DecisionTreeRegressor(random_state=cfg.random_state).fit(X_train, y_train)
    reg0_tr = regression_metrics(y_train, reg0.predict(X_train))
    reg0_te = regression_metrics(y_test, reg0.predict(X_test))
    print(f"Initial tree: depth={reg0.get_depth()}, leaves={reg0.get_n_leaves():,} "
          f"(train samples: {len(X_train):,})")
    print_metrics("Initial DecisionTreeRegressor", reg0_tr, reg0_te)
    over_reg, why_reg = assess_regression_overfit(reg0_tr, reg0_te, cfg)
    print(f"Overfitting check: {'YES - ' + '; '.join(why_reg) if over_reg else 'no significant gap'}")

    reg_final, reg_params, reg_cv_score = reg0, {}, float("nan")
    reg_note = "skipped (no overfitting detected)"
    if over_reg or cfg.force_tune:
        search = run_search(DecisionTreeRegressor(random_state=cfg.random_state), param_space("regression"),
                            "neg_root_mean_squared_error", reg_cv, cfg, X_train, y_train, "Regressor")
        reg_final, reg_params = search.best_estimator_, search.best_params_
        reg_cv_score = -search.best_score_
        reg_note = type(search).__name__ + " (5-fold CV, RMSE)"
        print("Top candidates (CV RMSE):")
        print(top_candidates(search, negate=True).round(4).to_string(index=False))
    reg_tr = regression_metrics(y_train, reg_final.predict(X_train))
    reg_te = regression_metrics(y_test, reg_final.predict(X_test))
    print_metrics("Final DecisionTreeRegressor" + (" (tuned)" if reg_params else ""), reg_tr, reg_te)
    if reg_params:
        still, why = assess_regression_overfit(reg_tr, reg_te, cfg)
        print(f"Overfitting re-check after tuning: R2 gap {reg_tr['R2'] - reg_te['R2']:.4f}, "
              f"test/train RMSE ratio {reg_te['RMSE'] / reg_tr['RMSE']:.2f} -> "
              f"{'still flagged (' + '; '.join(why) + ')' if still else 'no significant gap'}")
    results["reg"] = {"initial_train": reg0_tr, "initial_test": reg0_te, "tuned_train": reg_tr,
                      "tuned_test": reg_te, "overfit": over_reg, "reasons": why_reg,
                      "best_params": reg_params, "cv_score": reg_cv_score, "tuned_note": reg_note,
                      "residual_gap": reg_tr["R2"] - reg_te["R2"]}

    # 3-5b. classification ---------------------------------------------------------------
    banner("3-4. DECISION TREE CLASSIFIER (performance tiers)")
    print("Tier counts (train):", tier_train.value_counts().reindex(cfg.tier_labels).to_dict(),
          "| (test):", tier_test.value_counts().reindex(cfg.tier_labels).to_dict())
    clf0 = DecisionTreeClassifier(random_state=cfg.random_state).fit(X_train, tier_train)
    clf0_tr = classification_metrics(tier_train, clf0.predict(X_train))
    clf0_te = classification_metrics(tier_test, clf0.predict(X_test))
    print_metrics("Initial DecisionTreeClassifier", clf0_tr, clf0_te)
    over_clf, why_clf = assess_classification_overfit(clf0_tr, clf0_te, cfg)
    print(f"Overfitting check: {'YES - ' + '; '.join(why_clf) if over_clf else 'no significant gap'}")

    clf_final, clf_params, clf_cv_score = clf0, {}, float("nan")
    clf_note = "skipped (no overfitting detected)"
    if over_clf or cfg.force_tune:
        search = run_search(DecisionTreeClassifier(random_state=cfg.random_state), param_space("classification"),
                            "f1_macro", clf_cv, cfg, X_train, tier_train, "Classifier")
        clf_final, clf_params = search.best_estimator_, search.best_params_
        clf_cv_score = search.best_score_
        clf_note = type(search).__name__ + " (5-fold stratified CV, macro-F1)"
        print("Top candidates (CV macro-F1):")
        print(top_candidates(search).round(4).to_string(index=False))
    clf_tr = classification_metrics(tier_train, clf_final.predict(X_train))
    clf_te = classification_metrics(tier_test, clf_final.predict(X_test))
    print_metrics("Final DecisionTreeClassifier" + (" (tuned)" if clf_params else ""), clf_tr, clf_te)
    if clf_params:
        still, why = assess_classification_overfit(clf_tr, clf_te, cfg)
        print(f"Overfitting re-check after tuning: macro-F1 gap {clf_tr['F1 (macro)'] - clf_te['F1 (macro)']:.4f} -> "
              f"{'still flagged (' + '; '.join(why) + ')' if still else 'no significant gap'}")
    cm = plot_confusion_matrix(tier_test, clf_final.predict(X_test), cfg.tier_labels,
                               cfg.output_dir / "confusion_matrix.png")
    results["clf"] = {"initial_train": clf0_tr, "initial_test": clf0_te, "tuned_train": clf_tr,
                      "tuned_test": clf_te, "overfit": over_clf, "reasons": why_clf,
                      "best_params": clf_params, "cv_score": clf_cv_score, "tuned_note": clf_note,
                      "confusion": cm, "residual_gap": clf_tr["F1 (macro)"] - clf_te["F1 (macro)"]}

    # 4b. validation curves ---------------------------------------------------------------
    banner("5. VALIDATION CURVES & FEATURE IMPORTANCE")
    print("Computing validation curves ...")
    vc_reg = compute_validation_curve(reg_final, X_train, y_train, "r2", reg_cv, cfg)
    vc_clf = compute_validation_curve(clf_final, X_train, tier_train, "f1_macro", clf_cv, cfg)
    plot_validation_curves(vc_reg, vc_clf, reg_final.get_params()["max_depth"],
                           clf_final.get_params()["max_depth"], cfg.output_dir / "validation_curve.png")

    imp_reg = pd.Series(reg_final.feature_importances_, index=feats).sort_values(ascending=False)
    imp_clf = pd.Series(clf_final.feature_importances_, index=feats).reindex(imp_reg.index)
    plot_feature_importance(imp_reg, imp_clf, cfg.output_dir / "feature_importance.png")
    print("Feature importances (tuned regressor):")
    print(imp_reg.round(4).to_string())
    print(f"Saved plots to: {cfg.output_dir.resolve()}")

    # 6. artifacts ------------------------------------------------------------------------
    art = PipelineArtifacts(
        config=cfg, reg_model=reg_final, clf_model=clf_final, tier_labels=tuple(cfg.tier_labels),
        tier_edges=edges, tey_target=tey_target, importances_reg=imp_reg, importances_clf=imp_clf,
        normal_band=build_normal_band(X_train, y_train, tey_target, cfg.band_quantiles),
        train_ranges=pd.DataFrame({"min": X_train.min(), "max": X_train.max()}), results=results)

    if cfg.temporal_check:
        results["temporal"] = temporal_holdout_check(df, cfg, reg_params)

    # sample operational scenarios: a real Low-tier row, a real High-tier row and a what-if
    def median_row(tier: str) -> pd.DataFrame:
        pool = X_test[tier_test == tier]
        target_val = y_test[tier_test == tier].median()
        pick = (y_test[tier_test == tier] - target_val).abs().idxmin()
        return pool.loc[[pick]]

    what_if = art.normal_band["median"].to_dict()
    what_if.update({"CDP": 10.9, "TIT": 1062.0, "AFDP": 5.6})
    scenarios = [("Real test reading, actual tier Low", median_row(cfg.tier_labels[0])),
                 ("Real test reading, actual tier High", median_row(cfg.tier_labels[-1])),
                 ("What-if: normal-operation medians with low CDP, low TIT and high AFDP", what_if)]
    samples = [(title, recommend_actions(inp, art)[0]) for title, inp in scenarios]
    print_executive_summary(art, cleaning_steps, samples)
    return art


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("."), help="folder with gt_2011..gt_2015.csv")
    p.add_argument("--output-dir", type=Path, default=Path("."), help="where the PNG plots are written")
    p.add_argument("--search", choices=["random", "grid"], default="random")
    p.add_argument("--n-iter", type=int, default=80, help="candidates for RandomizedSearchCV")
    p.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--iqr-k", type=float, default=5.0, help="IQR fence multiplier")
    p.add_argument("--tey-target", type=float, default=None, help="TEY target (default: Low/Medium boundary)")
    p.add_argument("--force-tune", action="store_true", help="tune even if no overfitting is detected")
    p.add_argument("--no-temporal-check", action="store_true", help="skip the train<last-year check")
    a = p.parse_args()
    return Config(data_dir=a.data_dir, output_dir=a.output_dir, search=a.search, n_iter=a.n_iter,
                  cv_folds=a.cv, n_jobs=a.n_jobs, iqr_k=a.iqr_k, tey_target=a.tey_target,
                  force_tune=a.force_tune, temporal_check=not a.no_temporal_check)


def main() -> None:
    plt.switch_backend("Agg")  # headless-safe; plots are saved, not shown
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
