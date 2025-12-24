# CSIRO Image2Biomass Prediction

Predicting pasture biomass from drone imagery using vision transformers and ensemble methods. Built for the Kaggle CSIRO competition ($75k prize pool).

## Problem

Farmers need to know if there's enough grass for their livestock. Current methods are either too slow (manual sampling) or unreliable (sensors, remote sensing). This project predicts 5 biomass targets from images:

- **Dry_Green_g**: Green vegetation mass
- **Dry_Dead_g**: Dead vegetation mass  
- **Dry_Clover_g**: Clover-specific mass
- **GDM_g**: Green dry matter
- **Dry_Total_g**: Total biomass (weighted 0.5 in scoring)

**Evaluation**: Weighted R² across all predictions, with Dry_Total_g carrying half the weight.

## Approach

### Feature Extraction
- **DINOv2** vision transformer (4 variants: small/base/large/giant)
- Multi-stream processing: split images vertically, extract features, average per stream
- Tile subdivision option for finer spatial detail
- 768-dim embeddings → PCA to 75 components

### Auxiliary Classification
Two MLPs predict intermediate features via 5-fold CV:
- **State classifier**: 4 Australian states (92% bagging accuracy)
- **Species classifier**: 15 pasture species (78% bagging accuracy)

These features are concatenated with PCA features as inputs to the final regressor.

### Regression
**XGBoost** with Bayesian hyperparameter optimization:
- Train/test split by unique images (not rows) to prevent leakage
- 80 total features: 75 PCA + 5 one-hot encoded targets
- Best params: depth=10, lr=0.047, 384 trees

## Results

**Test R² = 0.70** (global weighted)

Per-target performance:
```
Dry_Green_g:  0.64
GDM_g:        0.64  
Dry_Total_g:  0.57
Dry_Clover_g: 0.41
Dry_Dead_g:   0.32
```

**Overfit analysis**: Severe overfitting detected (train R²=0.998, gap=0.30). Regularization needs tuning.

## Structure

```
├── EDA.ipynb                    # Data exploration, distributions, sample viz
├── model.ipynb                  # Full pipeline: extraction → classification → regression
├── features-extracted/          # Cached DINOv2 embeddings (not tracked)
└── submission.csv              # Competition format predictions
```

## Key Insights

1. **Data leakage risk**: Each image has 5 rows (one per target). Must split by `image_id`, not sample rows.

2. **Feature importance**: `target_name` one-hot encoding is critical (top 3 features). The model learns different scaling per target type.

3. **Spatial strategy**: Vertical streaming captures typical drone flight patterns better than random crops.

4. **Overfitting**: High train accuracy but poor generalization. Next steps: increase `reg_alpha/lambda`, reduce `max_depth`, add early stopping.

## Requirements

```
torch >= 2.0
transformers >= 4.30
xgboost >= 2.0
scikit-optimize
pandas, numpy, PIL
```

Pre-trained models downloaded via `kagglehub`:
```python
kagglehub.model_download("metaresearch/dinov2/pyTorch/small")
```

## Usage

Run cells sequentially in `model.ipynb`. Toggle cached features:
```python
dataset = os.path.exists(path_to_cached_features)  # True = load, False = extract
```

Config options:
```python
dino_version = 'small'  # small/base/large/giant
streams = 2             # vertical splits
tiles = 1               # subdivisions per stream
```

## Next Steps

- Fix overfitting: stronger regularization, smaller trees
- Test larger DINOv2 variants (giant not fully explored)
- Add NDVI and height features directly to XGBoost
- Ensemble multiple DINOv2 scales
- Domain-specific augmentation (brightness/color for seasonal variation)

---

Competition hosted by CSIRO & Meat & Livestock Australia. Dataset: 357 training images, 5 targets per image, Australian pastures across 4 states and 15 species.
