# src/run_mmsp_phase1_pam.py
import json, math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.metrics import pairwise_distances

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
PROCD = ROOT / "data" / "01_processed"
OUTD  = ROOT / "data" / "02_clusters"
OUTD.mkdir(parents=True, exist_ok=True)

# ---------- Selection & output toggles ----------
STABILITY_TOL     = 0.20   # within 20% of best stability counts as "near-best"
N_INIT_FINAL      = 20     # best-of-N final PAM restarts to avoid bad local minima
EMIT_HIGHMM_BOTH  = True   # emit both K=5 and K=6 for High_MM side-by-side
HIGHMM_BOTH_KS    = [5, 6] # which Ks to emit if EMIT_HIGHMM_BOTH is True

# ---------- Clustering params ----------
SEED        = 42
K_RANGE     = range(2, 9)  # 2..8
BOOTSTRAPS  = 30           # (20–100+) more = slower but stabler stability estimate
SUBSAMPLE   = 0.80         # bootstrap subsample fraction

# ---------- Multimorbidity strata ----------
STRATA_BINS   = [0, 2, 4, math.inf]
STRATA_LABELS = ["Low_MM", "Mid_MM", "High_MM"]


# ============ K-MEDOIDS (PAM) ============
def kpp_init_from_D(D: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-medoids++ style init on precomputed distance matrix D (n x n)."""
    n = D.shape[0]
    medoids = [int(rng.integers(0, n))]
    for _ in range(1, k):
        dist_to_nearest = np.min(D[:, medoids], axis=1)
        probs = dist_to_nearest**2
        s = probs.sum()
        if s == 0:
            # all points identical → pick random new medoid not in medoids
            candidates = np.setdiff1d(np.arange(n), np.array(medoids))
            medoids.append(int(rng.choice(candidates)))
        else:
            probs = probs / s
            medoids.append(int(rng.choice(np.arange(n), p=probs)))
    return np.array(medoids, dtype=int)

def assign_labels_from_D(D: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """Assign each point to the nearest medoid using D."""
    return np.argmin(D[:, medoids], axis=1)

def update_medoids_from_D(D: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """For each cluster, choose the index with minimum total distance to others in the cluster."""
    new_medoids = np.empty(k, dtype=int)
    for i in range(k):
        cluster_idx = np.where(labels == i)[0]
        if len(cluster_idx) == 0:
            new_medoids[i] = -1
            continue
        D_sub = D[np.ix_(cluster_idx, cluster_idx)]
        dsum = D_sub.sum(axis=1)
        new_medoids[i] = cluster_idx[int(np.argmin(dsum))]
    return new_medoids

def pam_kmedoids(D: np.ndarray, k: int, seed: int = SEED, max_iter: int = 100):
    """Classic PAM on precomputed D. Returns (labels, medoids)."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    k = int(k)
    if k < 2 or k > n:
        raise ValueError("k must be in [2, n].")

    medoids = kpp_init_from_D(D, k, rng)
    labels  = assign_labels_from_D(D, medoids)

    for _ in range(max_iter):
        new_medoids = update_medoids_from_D(D, labels, k)

        # handle empties: reseed to farthest point
        empties = np.where(new_medoids < 0)[0]
        if len(empties):
            dist_to_current = np.min(D[:, medoids], axis=1)
            far_idx = int(np.argmax(dist_to_current))
            for i in empties:
                new_medoids[i] = far_idx

        if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
            medoids = new_medoids
            labels  = assign_labels_from_D(D, medoids)
            break
        else:
            medoids = new_medoids
            labels  = assign_labels_from_D(D, medoids)

    # stabilize label order by medoid index
    order   = np.argsort(medoids)
    medoids = medoids[order]
    labels  = np.argmin(D[:, medoids], axis=1)
    return labels.astype(int), medoids

def pam_kmedoids_best_of_n(D: np.ndarray, k: int, n_init: int = N_INIT_FINAL, seed: int = 42):
    """Run PAM multiple times; return solution with lowest assignment cost."""
    best_labels, best_medoids, best_cost = None, None, np.inf
    for t in range(n_init):
        labels, medoids = pam_kmedoids(D, k, seed + t)
        cost = np.min(D[:, medoids], axis=1).sum()
        if cost < best_cost:
            best_labels, best_medoids, best_cost = labels, medoids, cost
    return best_labels, best_medoids


# ============ Pipeline helpers ============
def load_views():
    C = pd.read_csv(PROCD / "C_view.csv").set_index("eid")
    P = pd.read_csv(PROCD / "P_view_scaled.csv").set_index("eid")

    ids = C.index.intersection(P.index)

    Y_path = PROCD / "Y_validation.csv"
    if Y_path.exists():
        Y_raw = pd.read_csv(Y_path)
        if "eid" in Y_raw.columns:
            Y = Y_raw.set_index("eid").loc[ids]
        else:
            print("[WARN] Y_validation.csv has no 'eid'; proceeding without Y for now.")
            Y = pd.DataFrame(index=ids)
    else:
        Y = pd.DataFrame(index=ids)

    return C.loc[ids], P.loc[ids], Y

def make_strata(C: pd.DataFrame) -> pd.Series:
    if "num.co" not in C.columns:
        raise ValueError("C_view.csv must contain 'num.co'.")
    s = pd.cut(
        C["num.co"],
        bins=[0, 2, 4, float("inf")],
        labels=["Low_MM", "Mid_MM", "High_MM"],
        right=False,
        include_lowest=True,
    )
    s = s.cat.add_categories(["Unknown"]).fillna("Unknown")
    return s

def internal_metrics(Z: np.ndarray, labels: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": float(silhouette_score(Z, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
        "davies_bouldin": float(davies_bouldin_score(Z, labels)),
    }

def bootstrap_stability_from_D(
    D: np.ndarray, k: int, B: int = BOOTSTRAPS, subsample: float = SUBSAMPLE, seed: int = SEED
) -> float:
    rng  = np.random.default_rng(seed + k)
    runs = []
    idx  = np.arange(D.shape[0])
    m    = max(2, int(subsample * len(idx)))

    for b in range(B):
        take  = rng.choice(idx, size=m, replace=False)
        D_sub = D[np.ix_(take, take)]
        labels_b, _ = pam_kmedoids(D_sub, k, seed + 137 * b)
        runs.append((take, labels_b.astype(int)))

    if len(runs) < 2:
        return np.nan

    aris, n_pairs = 0.0, 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            idx_i, lab_i = runs[i]
            idx_j, lab_j = runs[j]
            common, pi, pj = np.intersect1d(idx_i, idx_j, assume_unique=False, return_indices=True)
            if len(common) < 3:
                continue
            aris += adjusted_rand_score(lab_i[pi], lab_j[pj])
            n_pairs += 1
    return float(aris / n_pairs) if n_pairs else np.nan

def choose_k(Z: np.ndarray, seed: int = SEED, k_range=K_RANGE):
    """Return (grid, D) where grid is a list of dicts with stability + internal metrics at each K."""
    D = pairwise_distances(Z, metric="euclidean")
    summary = []
    for k in k_range:
        stab = bootstrap_stability_from_D(D, k, seed=seed)
        labels_k, _ = pam_kmedoids(D, k, seed + 999 + k)
        im = internal_metrics(Z, labels_k)
        summary.append({"K": int(k), "stability_ARI": float(stab), **im})
    return summary, D

def pick_k_by_rule(summary_rows: list, stability_tol: float = STABILITY_TOL) -> int:
    """
    Stability-first acceptance rule:
      1) Find max stability_ARI (S*).
      2) Keep any K with stability >= (1 - stability_tol) * S*.
      3) Among those, pick by highest silhouette, then highest CH, then lowest DB.
    """
    import pandas as pd
    df = pd.DataFrame(summary_rows).copy()
    # safety for missing columns
    for c in ["silhouette", "calinski_harabasz", "davies_bouldin"]:
        if c not in df.columns:
            df[c] = np.nan
    Sstar = df["stability_ARI"].max()
    near  = df[df["stability_ARI"] >= (1 - stability_tol) * Sstar].copy()
    if near.empty:
        return int(df.sort_values("stability_ARI", ascending=False).iloc[0]["K"])
    near = near.sort_values(
        by=["silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return int(near.iloc[0]["K"])


# ============ Main ============
def run():
    np.random.seed(SEED)
    C, P, Y = load_views()
    strata = make_strata(C)

    results    = []
    all_labels = []

    # P is already indexed by eid; its columns are feature names
    p_cols = list(P.columns)

    for s in STRATA_LABELS:
        ids_s = strata.index[strata == s]
        n     = len(ids_s)
        print(f"\n--- {s}: n={n} ---")
        if n < 50:
            print(f"[SKIP] {s}: too few samples for robust clustering.")
            continue

        Xs = P.loc[ids_s, p_cols].values

        # PCA: retain ≈80% variance
        pca = PCA(n_components=0.80, svd_solver="full")
        Z   = pca.fit_transform(Xs)
        print(f"[PCA] {s}: components={Z.shape[1]} (≈80% var)")

        # Grid search over K with stability + internal metrics
        grid, D = choose_k(Z, seed=SEED + 7, k_range=K_RANGE)

        # chosen K by rule, and K that maximizes stability (for transparency)
        chosenK = pick_k_by_rule(grid, stability_tol=STABILITY_TOL)
        bestK_stab = int(pd.DataFrame(grid).sort_values("stability_ARI", ascending=False).iloc[0]["K"])
        print(f"[SelectK] {s}: chosen K={chosenK} (best-by-stability K={bestK_stab})")

        # ---- Final fit (best-of-N) for the chosen K ----
        labels_final, medoids = pam_kmedoids_best_of_n(D, chosenK, n_init=N_INIT_FINAL, seed=SEED + 2024)
        im_final = internal_metrics(Z, labels_final)

        df_labels = pd.DataFrame({
            "eid": ids_s,
            "stratum": s,
            "cluster_id": labels_final.astype(int),
            "label": [f"{s}_{c}" for c in labels_final],
            "K": chosenK,
            "pca_components": Z.shape[1],
        }).set_index("eid")
        all_labels.append(df_labels)

        result = {
            "stratum": s,
            "bestK_rule": chosenK,
            "bestK_stability": bestK_stab,
            **im_final,
            "n": int(n),
            "pca_components": int(Z.shape[1]),
            "grid": grid,
        }
        with open(OUTD / f"metrics_{s}.json", "w") as f:
            json.dump(result, f, indent=2)
        results.append(result)

        # ---- Optional: emit High_MM K=5 and K=6 side-by-side ----
        if s == "High_MM" and EMIT_HIGHMM_BOTH:
            present_Ks = {int(r["K"]) for r in grid}
            for k_alt in HIGHMM_BOTH_KS:
                if k_alt not in present_Ks:
                    print(f"[INFO] Skipping High_MM K={k_alt} (not in K_RANGE/grid).")
                    continue
                labels_k, _ = pam_kmedoids_best_of_n(D, k_alt, n_init=N_INIT_FINAL, seed=SEED + 3000 + k_alt)
                im_k = internal_metrics(Z, labels_k)
                df_k = pd.DataFrame({
                    "eid": ids_s,
                    "stratum": s,
                    "cluster_id": labels_k.astype(int),
                    "label": [f"{s}_{c}" for c in labels_k],
                    "K": k_alt,
                    "pca_components": Z.shape[1],
                }).set_index("eid")
                df_k.to_csv(OUTD / f"mmsp_clusters_{s}_K{k_alt}.csv")
                with open(OUTD / f"metrics_{s}_K{k_alt}.json", "w") as f:
                    json.dump({"stratum": s, "K": k_alt, **im_k, "n": int(n)}, f, indent=2)
                print(f"[Saved] {OUTD / f'mmsp_clusters_{s}_K{k_alt}.csv'} (+ metrics)")

    if all_labels:
        labels = pd.concat(all_labels).sort_index()
        labels.to_csv(OUTD / "mmsp_clusters.csv")
        with open(OUTD / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved:", OUTD / "mmsp_clusters.csv")
    else:
        print("\n[WARN] No strata were clustered.")

if __name__ == "__main__":
    import pandas as pd  # used in run() for bestK_stab line
    run()
