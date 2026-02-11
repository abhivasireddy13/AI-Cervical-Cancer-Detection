# Explainable-AI-Cervical-Cancer-Detection-Framework
This repository contains the experimental implementation corresponding to the IEEE Access paper:

**An Explainable Multi-Level Framework for Cervical Cancer Detection using Traditional Computer Vision and Deep Learning**

The proposed framework integrates traditional computer vision techniques, machine learning models, deep learning architectures, and explainable AI (XAI) methods to enable transparent and interpretable cervical cancer detection from cytology images.

---

## 📄 Associated Research Paper

**Authors:**  
Medha Chawla; Pratik Chakraborty; Sriram Sunderrajan; Rani Oomman Panicker  

**Journal:** IEEE Access, 2026  
**DOI:** https://doi.org/10.1109/ACCESS.2026.3654146

This repository is provided to support reproducibility, transparency, and future research.

---



## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Objectives](#objectives)
- [Proposed Framework](#proposed-framework)
  - [Machine Learning Branch](#machine-learning-branch)
  - [Deep Learning Branch](#deep-learning-branch)
  - [Explainable AI Module](#explainable-ai-module)
- [Evaluation Protocol](#evaluation-protocol)
- [Results Summary](#results-summary)
- [Requirements](#requirements)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Introduction

Cervical cancer remains one of the leading causes of cancer-related mortality among women worldwide. Early identification of precancerous cellular changes through cervical cytology screening is essential for effective clinical intervention.

This work proposes an explainable multi-level diagnostic framework that combines handcrafted feature-based learning and deep representation learning, ensuring both strong classification performance and clinical interpretability.

---

## Datasets

The framework is evaluated using three publicly available benchmark cervical cytology datasets:

- SIPaKMeD Dataset  
- Mendeley Liquid-Based Cytology (LBC) Dataset  
- Herlev Dataset  

These datasets include multiple cervical cell categories corresponding to different lesion grades.

---

## Objectives

The primary objectives of this study are:

1. To develop a unified multi-level cervical cancer detection framework.
2. To integrate traditional computer vision, ML, and DL approaches.
3. To ensure interpretability using explainable AI techniques.
4. To evaluate robustness across datasets and lesion grades.

---

## Proposed Framework

The framework consists of two complementary analytical branches.

### Machine Learning Branch

Handcrafted feature extraction techniques:
- Histogram of Oriented Gradients (HOG)
- Local Binary Patterns (LBP)
- Scale-Invariant Feature Transform with Bag-of-Visual-Words (SIFT-BoVW)

These features are classified using:
- Logistic Regression
- Random Forest
- Linear Support Vector Machine (SVM)

---

### Deep Learning Branch

Deep learning-based feature learning is performed using:

- AlexNet
- ResNet50
- EfficientNet-B0
- Custom CNN–Transformer architecture

These models learn hierarchical representations directly from cytology images.

---

### Explainable AI Module

To ensure transparency and trustworthiness, the following XAI techniques are applied:

- Feature Importance for ML models
- Grad-CAM and Integrated Gradients for DL models


Explainability is analyzed across models, features, datasets, and lesion grades.

---

## Evaluation Protocol

- Five-fold cross-validation applied across all datasets
- Evaluation metrics include:
  - Accuracy
  - Macro F1-score
  - Sensitivity
  - Specificity
  - Area Under the ROC Curve (AUC)

---

## Results Summary

Key findings reported in the paper include:

- Random Forest with LBP features achieved the best performance in the ML branch.
- ResNet50 achieved the highest accuracy across all datasets in the DL branch.
- XAI analysis revealed meaningful cellular morphology patterns contributing to model decisions.

---

## Requirements

All dependencies are listed in:

```
requirements.txt
```

---

## Usage

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Run the provided training and evaluation scripts or notebooks to reproduce experimental results.

---

## Citation

If you use this work, please cite:

```bibtex
@ARTICLE{11352795,
  author={Chawla, Medha and Chakraborty, Pratik and Sunderrajan, Sriram and Panicker, Rani Oomman},
  journal={IEEE Access},
  title={An Explainable Multi-Level Framework for Cervical Cancer Detection using Traditional Computer Vision and Deep Learning},
  year={2026},
  pages={1-1},
  doi={10.1109/ACCESS.2026.3654146}
}
```

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


