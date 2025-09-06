# Project Structure

This document outlines the organization of the WiDS 2025 Brain Imaging Analysis project.

## Directory Structure

```
WIDS/
├── WiDS_2025_Brain_Imaging_Analysis.ipynb  # Main analysis notebook
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
├── .gitignore                              # Git ignore rules
├── PROJECT_STRUCTURE.md                    # This file
└── data/                                   # Data directory (if data is included)
    ├── train_quantitative_metadata.xlsx
    ├── train_categorical_metadata.xlsx
    ├── test_quantitative_metadata.xlsx
    ├── test_categorical_metadata.xlsx
    ├── training_solutions.xlsx
    ├── train_functional_connectome_matrices.csv
    └── test_functional_connectome_matrices.csv
```

## File Descriptions

### Core Files
- **WiDS_2025_Brain_Imaging_Analysis.ipynb**: Main Jupyter notebook containing the complete analysis pipeline
- **README.md**: Comprehensive project documentation including methodology, results, and usage instructions
- **requirements.txt**: Python package dependencies required to run the analysis
- **.gitignore**: Specifies files and directories to ignore in version control

### Data Files (Expected)
- **Training Data**: Quantitative and categorical metadata, functional connectome matrices, and ground truth labels
- **Test Data**: Quantitative and categorical metadata, and functional connectome matrices for prediction

## Notebook Sections

The main notebook is organized into the following sections:

1. **Data Loading and Exploration**
   - Import libraries and load datasets
   - Initial data exploration and visualization

2. **Data Preprocessing**
   - Missing value imputation
   - Outlier detection and handling
   - Feature engineering and encoding

3. **Model Development**
   - Feature selection
   - Hyperparameter tuning
   - Model training and validation

4. **Model Evaluation**
   - Performance metrics
   - Feature importance analysis
   - SHAP explainability

5. **Final Predictions**
   - Test set predictions
   - Submission file generation

## Usage Notes

- Ensure all data files are placed in the correct directory structure
- Install dependencies using `pip install -r requirements.txt`
- Run the notebook cells sequentially for best results
- The notebook is designed to be self-contained and reproducible
