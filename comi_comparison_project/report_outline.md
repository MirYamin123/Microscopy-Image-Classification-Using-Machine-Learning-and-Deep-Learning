# COMI Comparison Project Report Outline

## 1. Objective
- Build a deep learning model that classifies microscopy images into `actin`, `mitochondria`, and `nucleus`.
- Compare a classical ML baseline and two deep-learning models:
  - `Random Forest + PCA`
  - baseline CNN
  - transfer-learning `ResNet18`
- Evaluate performance on a held-out test split built by `Z` group to reduce leakage.

## 2. Why COMI
- Clean, pre-labeled microscopy classes
- Balanced class distribution
- Suitable for an end-to-end image-classification workflow
- Strong fit for a course project that must include training, validation, testing, and interpretation

## 3. Dataset Description
- Dataset used: COMI `BPAEC`
- Classes: `actin`, `mitochondria`, `nucleus`
- Image type: grayscale microscopy images
- Split strategy:
  - Train: `Z004`, `Z005`, `Z006`, `Z007`, `Z008`
  - Validation: `Z009`
  - Test: `Z010`
- Reason for split strategy: neighboring `Z` slices are visually related, so random splits can inflate performance

## 4. Preprocessing
- Resize images to `224x224`
- Convert grayscale to 3 channels
- Normalize pixel values
- Use data augmentation for training only

## 5. Models
- Classical baseline: Random Forest on flattened `64x64` grayscale images after standardization and PCA
- Baseline model: small CNN
- Main model: pretrained `ResNet18` with a 3-class output layer
- Loss: cross-entropy
- Optimizer: Adam
- Early stopping based on validation macro F1

## 6. Evaluation
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion matrix
- Training and validation curves

## 7. Error Analysis
- Show correct and incorrect examples
- Identify the most confused classes
- Use Grad-CAM to visualize what the `ResNet18` model attends to

## 8. Results Discussion
- Compare Random Forest, baseline CNN, and `ResNet18`
- Discuss whether transfer learning improved performance over both the classical ML baseline and the simple CNN
- Explain the likely biological reasons behind difficult cases

## 9. Limitations
- Single dataset domain
- Only three classes
- Performance may not generalize to other imaging conditions or cell types

## 10. Future Work
- Test deeper backbones such as EfficientNet
- Use cross-validation by `Z` group
- Explore self-supervised pretraining or domain-specific microscopy models
