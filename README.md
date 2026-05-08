# Microscopy Image Classification Using Machine Learning and Deep Learning

This project compares three approaches for classifying microscopy images from the COMI BPAEC dataset into three cell-structure classes:

- actin
- mitochondria
- nucleus

The goal was to build a fair image-classification workflow, compare a classical model with deep-learning models, and evaluate the models on a held-out image group rather than a random split.

## Dataset

The dataset is stored under:

```text
COMI/dataset/BPAEC
```

The class folders are:

```text
actin/
mitochondria/
nucleus/
```

Each class contains Z-group folders. The project uses the following split:

| Split | Z groups | Images |
|---|---:|---:|
| Train | Z004-Z008 | 1,485 |
| Validation | Z009 | 297 |
| Test | Z010 | 297 |

Splitting by Z group helps reduce leakage because images from the same group can be visually related.

## Models

Three models were trained and compared:

1. **Random Forest + PCA**
   - Images resized to 64 x 64 grayscale
   - Flattened into pixel vectors
   - Standardized with `StandardScaler`
   - Reduced with PCA while preserving about 95% variance
   - Classified with a 500-tree Random Forest

2. **Baseline CNN**
   - Images resized to 224 x 224
   - Grayscale images duplicated to three channels
   - Custom convolutional network trained from scratch
   - Training-only augmentation

3. **ResNet18**
   - Images resized to 224 x 224
   - Grayscale images duplicated to three channels
   - ImageNet normalization
   - Pretrained ResNet18 with a replaced three-class output layer
   - Warmup training followed by fine-tuning

## Test Results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest + PCA | 95.96% | 95.94% | 95.95% | 95.93% |
| Baseline CNN | 76.77% | 81.53% | 76.60% | 76.52% |
| ResNet18 | 95.62% | 95.88% | 95.55% | 95.58% |

Random Forest + PCA had the highest score on this test split, with ResNet18 very close behind. The Baseline CNN had the largest error pattern, mostly from mitochondria images being predicted as nucleus.

## Repository Layout

```text
COMI/
  dataset/BPAEC/                  # microscopy images

comi_comparison_project/
  src/
    comi_pipeline.py              # data loading, training, evaluation, plotting
  notebooks/
    comi_comparison_training.ipynb
    comi_colab_training.ipynb
    comi_presentation_materials.ipynb
  results/
    model_comparison.csv
    comi_split_summary.csv
    figures/                      # confusion matrices, training curves, examples
    materials/                    # PCA plots, feature maps, Grad-CAM, comparison panels
    random_forest_pca/            # RF model, scaler, PCA, metrics
    baseline_cnn/                 # CNN checkpoint, history, metrics
    resnet18/                     # ResNet18 checkpoint, history, metrics
```

## Running the Project

The main project notebook is:

```text
comi_comparison_project/notebooks/comi_comparison_training.ipynb
```

Recommended workflow:

1. Open the notebook in Google Colab.
2. Set the runtime to GPU.
3. Update `PROJECT_ROOT` and `DATASET_ROOT` if needed.
4. Install the packages from:

```text
comi_comparison_project/requirements_colab.txt
```

5. Run the notebook from top to bottom.

To regenerate the presentation figures, run:

```text
comi_comparison_project/notebooks/comi_presentation_materials.ipynb
```

## Main Takeaways

- Z-group splitting gives a stricter test than random image splitting.
- PCA reduced the Random Forest input from 4,096 pixel values to 38 components while preserving about 95% variance.
- Transfer learning helped ResNet18 generalize much better than the CNN trained from scratch.
- Nucleus was the easiest class for the strongest models; mitochondria caused the most confusion.
