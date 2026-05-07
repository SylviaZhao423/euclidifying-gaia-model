# Euclidifying Gaia: Star/Galaxy Classifier for Gaia DR3

An XGBoost classifier trained to identify extended (non-point-like) sources in the Gaia DR3 catalogue using Euclid Q1 morphological classifications as training labels.

## Models

Multiple model variants are provided, trading recall for precision:

| Model | Directory | Star:Galaxy ratio | Best for |
|-------|-----------|-------------------|----------|
| **Q1-only** (default) | `models/q1_only/` | 2.8:1 | Higher recall, general use |
| ERO 3:1 | `models/q1_ero_mix_3/` | 3:1 | |
| ERO 4:1 | `models/q1_ero_mix_4/` | 4:1 | |
| ERO 5:1 | `models/q1_ero_mix_5/` | 5:1 | Balanced |
| ERO 6:1 | `models/q1_ero_mix_6/` | 6:1 | |
| ERO 7:1 | `models/q1_ero_mix_7/` | 7:1 | |
| ERO 8:1 | `models/q1_ero_mix_8/` | 8:1 | |
| **ERO 8.63:1** | `models/q1_ero_mix_8.6/` | 8.63:1 (all ERO) | Highest precision, purer catalogues |

- **Algorithm**: XGBoost gradient-boosted decision trees
- **Training data**: 238,868 Gaia DR3 sources cross-matched with Euclid Q1 PHZ star/galaxy labels (G >= 17)
- **Ensemble**: 10 models trained with different random seeds; predictions are the ensemble mean
- **Output**: P(gal) probability (0 = star, 1 = galaxy)

## Performance (Q1 test set, 10-seed ensemble mean)

| Threshold | Galaxy Precision | Galaxy Recall | Galaxy F1 |
|-----------|-----------------|---------------|-----------|
| P >= 0.5  | 0.874 +/- 0.003 | 0.524 +/- 0.005 | 0.655 +/- 0.004 |
| P >= 0.9  | 0.960           | 0.378           | 0.543           |

ROC AUC: 0.855 +/- 0.002

## Features

The classifier uses 14 Gaia DR3 parameters. See `features.py` for the full list and SQL expressions for derived columns.

The top features by importance are:
1. **Corrected BP/RP excess factor** (C*) — 40% of discriminatory power
2. **G - G_RP** colour index
3. **Astrometric chi-squared** (chi2_AL)
4. **Position error**

## Quick start

```bash
pip install xgboost numpy pandas
```

```python
from predict import load_ensemble, predict_pgal
import numpy as np

models = load_ensemble()

# Your Gaia features: shape (n_sources, 14)
# See features.py for the required column order
X = np.load("your_gaia_features.npy")

p_gal = predict_pgal(models, X)
galaxy_candidates = p_gal >= 0.9
```

Or from command line with a CSV:
```bash
python predict.py your_sources.csv                           # Q1-only (default)
python predict.py your_sources.csv --model q1_ero_mix_5         # balanced
python predict.py your_sources.csv --model q1_ero_mix_8.6      # highest precision
```

The CSV must contain columns matching the 14 feature names in `features.py`.

## Recommended thresholds

| Application | Threshold | Purity at 1% galaxy fraction |
|-------------|-----------|------------------------------|
| Exploratory | P >= 0.5  | ~17%                         |
| Balanced    | P >= 0.7  | ~44%                         |
| High-purity | P >= 0.9  | ~75%                         |

Purity depends on the true galaxy fraction in your sample. Use Bayesian recalibration for precise estimates — see the paper for details.

## Important notes

- **Always use `predict_proba()`**, not `get_booster().predict()`. The latter skips calibration and gives different results.
- The model is trained on sources with G >= 17. Performance below this magnitude is not validated.
- NaN values in input features are handled internally by XGBoost.

## Citation

Zhao, S. (2026). *Euclidifying Gaia: Classifying Non-Point-Like Sources in Gaia DR3 Catalogue with Euclid-Supervised Machine Learning Model.* Cambridge Part III Astrophysics Research Project.

## Interactive visualisation

An interactive 3D UMAP visualisation with per-source Euclid cutouts is available at [euclidifying-gaia.com](https://euclidifying-gaia.com).

## License

MIT
