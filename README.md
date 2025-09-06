# WiDS 2025 Datathon: Brain Imaging Analysis for ADHD and Sex Prediction

## Overview

This project is my submission for the **2025 Women in Data Science (WiDS) Datathon Challenge**. The challenge focused on analyzing functional brain imaging data to predict an individual's sex and ADHD diagnosis, with the goal of improving understanding of brain health in children and adolescents and the development of personalized medicine.

## Problem Statement

The WiDS 2025 Datathon challenged participants to:
- Predict an individual's **sex** (Male/Female) from functional brain imaging data
- Predict **ADHD diagnosis** (ADHD/No ADHD) from the same dataset
- Utilize functional connectome matrices and demographic/behavioral data
- Improve understanding of brain health patterns in children and adolescents

## Dataset

The dataset includes:
- **Functional Connectome Matrices**: Brain connectivity data (Pearson correlation matrices)
- **Quantitative Metadata**: Behavioral and demographic measurements
- **Categorical Metadata**: Categorical demographic information
- **Training Solutions**: Ground truth labels for ADHD and sex

### Key Features:
- **Training Data**: 1,213 participants with complete data
- **Test Data**: 304 participants for final predictions
- **Features**: ~19,000+ features including connectome data and metadata
- **Targets**: Binary classification for both ADHD and sex prediction

## Methodology

### 1. Data Preprocessing
- **Missing Value Imputation**: 
  - Categorical variables: Mode imputation
  - Numerical variables: Mean/median based on distribution skewness
- **Outlier Handling**: 
  - IQR-based outlier detection
  - Capping for highly skewed variables
  - Removal for normally distributed variables
- **Feature Engineering**:
  - One-hot encoding for categorical variables
  - Standard scaling for numerical features
  - Feature selection using SelectKBest with f_classif

### 2. Model Development
- **Algorithm**: XGBoost Classifier
- **Hyperparameter Tuning**: RandomizedSearchCV with stratified cross-validation
- **Class Balancing**: Computed class weights for imbalanced datasets
- **Feature Selection**: Top 500 most relevant features for each target

### 3. Model Performance

| Model | Accuracy | F1-Score | ROC AUC |
|-------|----------|----------|---------|
| ADHD Prediction | 82.3% | 81.3% | 82.9% |
| Sex Prediction | 75.3% | 74.7% | 72.2% |

### 4. Model Explainability
- **SHAP Analysis**: Feature importance and model interpretability
- **Feature Importance Plots**: Top contributing features for each prediction task
- **Confusion Matrices**: Detailed performance analysis

## Key Findings

### ADHD Prediction
- Strong performance with 82.3% accuracy
- Functional connectome features were highly predictive
- Specific brain connectivity patterns associated with ADHD diagnosis

### Sex Prediction
- Moderate performance with 75.3% accuracy
- Brain connectivity differences between sexes were detectable
- Some features showed significant sex-based patterns

## Technical Implementation

### Dependencies
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Model Explainability**: shap
- **Statistical Analysis**: scipy

### Key Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import shap
```

## File Structure

```
WIDS/
├── WiDS_2025_Brain_Imaging_Analysis.ipynb  # Main analysis notebook
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
└── predictions.csv                          # Final predictions (if generated)
```

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   - Open `WiDS_2025_Brain_Imaging_Analysis.ipynb` in Jupyter Notebook
   - Execute cells sequentially
   - Ensure data files are in the correct directory

3. **Data Requirements**:
   - Training quantitative metadata
   - Training categorical metadata
   - Test quantitative metadata
   - Test categorical metadata
   - Training solutions (labels)
   - Functional connectome matrices (training and test)

## Results

The final model achieved:
- **ADHD Prediction**: 82.3% accuracy with strong F1-score of 81.3%
- **Sex Prediction**: 75.3% accuracy with balanced F1-score of 74.7%

These results demonstrate the potential of functional brain imaging data for predicting both clinical conditions and demographic characteristics, contributing to the field of personalized medicine and brain health research.

## Future Improvements

1. **Advanced Feature Engineering**: 
   - Network analysis features from connectome matrices
   - Graph-based features and centrality measures

2. **Model Architecture**:
   - Deep learning approaches (CNNs for connectome data)
   - Ensemble methods combining multiple algorithms

3. **Data Augmentation**:
   - Synthetic data generation for imbalanced classes
   - Cross-validation with different data splits

## Acknowledgments

- **WiDS 2025 Datathon** for providing the dataset and challenge
- **Stanford University** for organizing the Women in Data Science initiative
- **Open source community** for the excellent machine learning libraries

## Contact

For questions about this analysis or collaboration opportunities, please feel free to reach out.

---

*This project was developed as part of the WiDS 2025 Datathon Challenge, focusing on advancing women in data science and contributing to brain health research.*
