# Benchmark data and evaluation

Data and weights are not included. The original loaders, splits, normalization, geometric preprocessing and force primitives are retained. Download datasets from their original providers and obey their licenses.

| Task | Path relative to `--data-root` | Split | Representation |
|---|---|---|---|
| Airfoil | `fno/airfoil/naca` | 1000 train / 200 test | 221*51 |
| Pipe | `fno/pipe` | 1000 / 200 | 129*129 |
| Darcy | `fno` | 1000 / 200 | 85*85 subsampling |
| Elasticity | `fno` | 1000 / 200 | 972 points |
| Navier-Stokes | `fno` | 1000 / 200 | 64*64; 10 observed -> 10 future frames |
| Plasticity | `fno/plas_N987_T20.mat` | 900 / 80 | 101*31; 20 times, 4 channels |
| AirfRANS | `AirfRANS/Dataset` | 720 train + 80 validation / 200 test | 32,000 sampled training points; full-field evaluation |
| ShapeNet-Car | `mlcfd_data/training_data` | 789 / 100, fold 0 | Variable point count |

Standard datasets follow the upstream [Transolver preparation instructions](https://github.com/thuml/Transolver/tree/75e0f67643806a81cd1d3f6adc88dd8c02416fe7/PDE-Solving-StandardBenchmark). The original links are [FNO data](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) and [Geo-FNO data](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8).

Keep the original filenames:

```text
fno/
  piececonst_r421_N1024_smooth1.mat
  piececonst_r421_N1024_smooth2.mat
  plas_N987_T20.mat
  NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat
  elasticity/Meshes/Random_UnitCell_XY_10.npy
  elasticity/Meshes/Random_UnitCell_sigma_10.npy
  airfoil/naca/NACA_Cylinder_X.npy
  airfoil/naca/NACA_Cylinder_Y.npy
  airfoil/naca/NACA_Cylinder_Q.npy
  pipe/Pipe_X.npy
  pipe/Pipe_Y.npy
  pipe/Pipe_Q.npy
```

`--data-path` overrides a Standard task directory; for Plasticity it is the complete MAT filename. The retained Pipe loader uses its original first-1200 selection. Do not reorder files or silently change train/test selection when comparing existing experiments.

AirfRANS is provided by [Extrality/AirfRANS](https://github.com/Extrality/AirfRANS), with the upstream [Dataset.zip](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip). Keep `manifest.json` and each case's internal VTU and aerofoil VTP files. Native evaluation repeatedly samples until points are covered, averages repeated predictions, reverses normalization and applies boundary postprocessing before aerodynamic metrics. The `full`, `scarce`, `reynolds` and `aoa` task variants are retained. The unified launcher accepts the actual `Dataset` path for LinearNO/PhyLoopNO train and evaluation. The original Transolver evaluation entry instead expects its parent directory.

Car data are provided by [Nobuyuki Umetani](http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip). `--data-dir` is the raw `training_data` directory and `--cache-dir` is the separate preprocessing cache. Retain the raw `quadpress_smpl.vtk` and `hexvelo_smpl.vtk` files for each case: cached graph arrays cannot replace the force calculation's geometry. The preserved dataset still constructs PyG objects and graph edges even where a model does not consume the edges. Radius graph construction requires the matching graph extension (newer PyG: pyg-lib>=0.6.0; older PyG: torch-cluster).

PhyLoopNO/LinearNO Car force evaluation uses the actual sample directory and surface fields via `cdlno/linearno/car_metrics.py`. The separately retained original Transolver evaluator contains upstream force-path conventions; review its native `utils/drag_coefficient.py` configuration when using that baseline. The extraction does not silently reinterpret historical upstream force results.

Field relative-L2 errors are reduced per sample before averaging; NS step/full and Plasticity time aggregation remain distinct. Physical and normalized units, training and evaluation masks, and coefficient correlations are recorded separately. The original Transolver AirfRANS repository explicitly corrected its published field-result description to MSE; do not treat those reference numbers as physical relative L2.
