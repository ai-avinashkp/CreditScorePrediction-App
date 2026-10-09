"""Weight of Evidence encoding.

WoE = ln(share of goods in bin / share of bads in bin), with bad = default = 1.
Positive WoE therefore means *lower* risk than average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class WoETransformer:
    def __init__(self, n_bins: int = 5, smoothing: float = 0.5, min_bin_frac: float = 0.05):
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.min_bin_frac = min_bin_frac
        self.numeric_: dict[str, np.ndarray] = {}      # inner bin edges
        self.tables_: dict[str, pd.DataFrame] = {}     # per-feature bin table
        self.columns_: list[str] = []

    # ---- fitting -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoETransformer":
        y = pd.Series(np.asarray(y), index=X.index)
        self.columns_ = list(X.columns)
        for col in self.columns_:
            if pd.api.types.is_numeric_dtype(X[col]):
                edges = self._numeric_edges(X[col])
                self.numeric_[col] = edges
                bins = self._assign_numeric(X[col], edges)
                self.tables_[col] = self._table(bins, y, order=self._labels(edges))
            else:
                self.tables_[col] = self._table(X[col].astype(str), y)
        return self

    def _numeric_edges(self, s: pd.Series) -> np.ndarray:
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(s.to_numpy(float), qs))
        # drop edges that would create near-empty bins
        keep = []
        prev = -np.inf
        for e in edges:
            lo = (s > prev).to_numpy() & (s <= e).to_numpy()
            if lo.mean() >= self.min_bin_frac:
                keep.append(e)
                prev = e
        if keep and (s > keep[-1]).mean() < self.min_bin_frac:
            keep.pop()
        return np.asarray(keep, dtype=float)

    @staticmethod
    def _assign_numeric(s: pd.Series, edges: np.ndarray) -> pd.Series:
        idx = np.searchsorted(edges, s.to_numpy(float), side="left")
        labels = WoETransformer._labels(edges)
        return pd.Series([labels[i] for i in idx], index=s.index)

    @staticmethod
    def _labels(edges: np.ndarray) -> list[str]:
        f = lambda v: f"{v:g}"
        if len(edges) == 0:
            return ["all"]
        out = [f"<= {f(edges[0])}"]
        out += [f"{f(a)} - {f(b)}" for a, b in zip(edges[:-1], edges[1:])]
        out.append(f"> {f(edges[-1])}")
        return out

    def _table(self, bins: pd.Series, y: pd.Series, order: list[str] | None = None) -> pd.DataFrame:
        df = pd.DataFrame({"bin": bins.to_numpy(), "y": y.to_numpy()})
        g = df.groupby("bin")["y"].agg(bad="sum", n="count")
        if order is not None:  # keep numeric bins in ascending order, not alphabetical
            g = g.reindex([b for b in order if b in g.index])
        g["good"] = g["n"] - g["bad"]
        s = self.smoothing
        tot_good, tot_bad = g["good"].sum(), g["bad"].sum()
        dist_good = (g["good"] + s) / (tot_good + s * len(g))
        dist_bad = (g["bad"] + s) / (tot_bad + s * len(g))
        g["woe"] = np.log(dist_good / dist_bad)
        g["iv"] = (dist_good - dist_bad) * g["woe"]
        return g.reset_index()

    # ---- applying ----------------------------------------------------------
    def bins(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.columns_:
            if col in self.numeric_:
                out[col] = self._assign_numeric(pd.to_numeric(X[col]), self.numeric_[col])
            else:
                out[col] = X[col].astype(str)
        return pd.DataFrame(out, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        bins = self.bins(X)
        out = {}
        for col in self.columns_:
            mapping = self.tables_[col].set_index("bin")["woe"]
            out[col] = bins[col].map(mapping).fillna(0.0).astype(float)  # unseen -> neutral
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def iv_summary(self) -> pd.DataFrame:
        rows = [(c, float(t["iv"].sum())) for c, t in self.tables_.items()]
        return pd.DataFrame(rows, columns=["feature", "iv"]).sort_values("iv", ascending=False)
