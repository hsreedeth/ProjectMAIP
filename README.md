**ProjectMAIP** is an interpretable multimorbidity-stratified phenotyping (MMSP) pipeline for clinical cohorts. It builds three complementary “views” of each patient—Comorbidity (C), Physiology (P), and Social/Context (S)—then performs PCA on the P-view and k-medoids clustering within multimorbidity strata. Model selection is stability-first (bootstrap ARI), then separation (silhouette, Calinski–Harabasz, Davies–Bouldin). Outputs include phenotype profiles (P-view medians, C-view prevalences), Kaplan–Meier/log-rank and Cox results for external validity, and compact rulecards (tiny trees) for point-of-care interpretability—no black box required.

The repo is organised for reproducibility: `data/` (raw → processed), `src/` (e.g., `make_views.py`, `run_mmsp.py`, `external_validation.py`, `run_qc.py`), and `reports/` (figures/tables). Quick start: place your cohort in `data/00_raw/`, run preprocessing & QC, execute MMSP, then external validation; seeds and requirements are pinned for consistent results. Real data are not bundled; scripts assume a wide table with `eid`, outcomes (e.g., LOS/cost/survival), and standard physiology/comorbidity fields. Issues and PRs that improve robustness, fairness checks, or clinician-facing rulecards are very welcome.

---

Upcoming update: Redundancy management of processed data across pipelines. Specifically, between scr/scripts and notebooks. Every refernces in the pipeline will be consistent.





