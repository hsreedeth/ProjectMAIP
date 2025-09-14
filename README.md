This project finds stable ICU patient phenotypes in the public SUPPORT-II dataset using reproducible pipelines. 
The features are split into three views (Comorbidity, Physiology, Context),
cluster within multimorbidity strata, and is validated with survival, LOS, and cost.


**Quick start guide**
---


> Put in raw data at `data/00_raw/support2.csv`.

> Preprocess & build views (adds eid, scales P-view):
```python src/<preprocess_script>.py```

> QC & final cleaning:
```
python src/<qc_script>.py --visualize
python src/<clean_script>.py
```

> Run MMSP (PCA -> PAM per stratum then stability-first K selection):
```python src/run_mmsp_phase1_pam.py```

> External validation (KM/log-rank, Cox, LOS/cost):
```python src/<external_validation>.py```


**Outputs land in:**


> Views: `data/01_processed/ (C_view.csv, P_view_scaled.csv, S_view.csv, Y_validation.csv)`

Clusters & metrics: 

`data/02_clusters/ (mmsp_clusters.csv, metrics_*.json)`


`Figures/tables: reports/`
