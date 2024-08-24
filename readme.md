# GeoPredictors

GeoPredictors is a comprehensive suite designed for the classification of geological facies and prediction of permeability using advanced machine learning models. This project leverages the Corebreakout library for preprocessing core images and utilizes a modified ResNet34 for lithofacies classification, achieving significant accuracy improvements. For permeability prediction, models like XGBoost Regressor and Random Forest Regressor are employed, focusing on optimizing performance through rigorous model selection and hyper-parameter tuning.

## Installation

### Conda Environment

To ensure compatibility and ease of setup, we recommend using a Conda environment. You can create and activate the environment as follows:

```bash
conda env create -f environment.yml
conda activate arcadia
```

## Dependencies

Once the environment is activated, navigate to the project directory and install the necessary dependencies via the provided setup.py:

```bash
pip install .
```

This command will install all required packages and the GeoPredictors package itself.

## Usage

The project is structured to provide clear insights and easy navigation through the analysis process:

### Data Preprocessing:

Utilizes the Corebreakout library for extracting and segmenting core column images.
Lithofacies Classification: Employs a modified ResNet34 model for accurate classification of lithofacies from core images.
### Permeability Prediction:

Integrates machine learning models such as XGBoost for predicting permeability based on core sample data.
Notebooks
For detailed examples and explanations of the methodologies employed, please refer to the Jupyter notebooks included in the project. These notebooks offer a step-by-step guide to the preprocessing, model training, and evaluation phases, alongside insightful visualizations.

## Optimization

Code optimization and quality assurance have been addressed through linting with pylint, with the final code rating standing at 8.48/10.

## Visualization

The project includes visualization tools for displaying the predicted facies alongside core images, mimicking the wireline data presentation format for intuitive analysis.
