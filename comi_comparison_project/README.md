# COMI Comparison Project

This standalone project extends the original COMI workflow into a three-model comparison project using:

- `Random Forest + PCA`
- a baseline custom CNN
- a transfer-learning `ResNet18` model
- a separate presentation-material generation layer

The required model order in the main notebook is:

1. `Random Forest + PCA`
2. `Baseline CNN`
3. `ResNet18`

## What You Do Manually

You do **not** need to label the images manually.
You do **not** need to reorganize the COMI dataset manually.

You only need to do these Colab-side steps:

1. Upload this `comi_comparison_project` folder to Google Drive.
2. Make sure the COMI dataset is also available in Google Drive.
3. Open `notebooks/comi_comparison_training.ipynb` in Colab.
4. Update the `PROJECT_ROOT` and `DATASET_ROOT` paths in the notebook.
5. In Colab, set `Runtime > Change runtime type > GPU`.
6. Run the notebook cells from top to bottom.
7. After training completes, open `notebooks/comi_presentation_materials.ipynb` to generate extra presentation assets.

If you also see a copied legacy notebook from the original project, ignore it and use only the two notebooks listed above.

The Colab requirements file intentionally avoids reinstalling `torch` so Colab keeps its GPU-enabled PyTorch environment.

## Expected Dataset Root

Point the notebook to this folder:

`COMI/dataset/BPAEC`

The code expects these class folders:

- `actin`
- `mitochondria`
- `nucleus`

It also expects the `Z004` to `Z010` subfolders inside each class.

## Project Outputs

Running the main notebook will create these artifacts under the project `results/` directory:

- split manifest CSV
- class-count summary CSV
- Random Forest scaler, PCA, and trained model artifacts
- training history CSVs for CNN and ResNet18
- metric JSON files for all three models
- classification report CSVs for all three models
- prediction CSVs for all three models
- saved best model checkpoints for CNN and ResNet18
- class distribution plot
- training curves
- confusion matrices
- correct / incorrect prediction galleries
- confidence histograms
- final model comparison table and comparison chart

Running the presentation-materials notebook will create additional artifacts under `results/materials/`, including:

- Random Forest featurization panels
- PCA explained-variance and PCA-scatter plots
- pixel-grid-to-vector illustrations
- CNN and ResNet preprocessing panels
- selected feature-map visualizations
- embedding projection plots
- Grad-CAM figures
- cross-model explanation graphics

## Files

- `src/comi_pipeline.py`: reusable data, training, evaluation, RF/PCA, and visualization helpers
- `notebooks/comi_comparison_training.ipynb`: main training and comparison notebook
- `notebooks/comi_presentation_materials.ipynb`: separate presentation-material notebook
- `report_outline.md`: report and presentation outline
- `requirements_colab.txt`: packages to install in Colab
