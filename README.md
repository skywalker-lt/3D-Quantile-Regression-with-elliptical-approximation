# The Plotly Implementation of 3D-Extended Quantile Regression Model Proposed in a Math AAHL Extended Essay

## Description

This repository provides a Plotly-based implementation of a **3D-extended quantile regression model**. For detailed explanation of the methodology, please refer to **Section 2 ~ 3** in the original essay. The code enables examinars to reproduce and validate the sample calculation presented in **Section 3** of the essay.

<img width="899" alt="Screenshot 2024-09-12 at 20 25 49" src="https://github.com/user-attachments/assets/08a1eea5-4891-4f6d-96b2-899793713178">


## Device Requirements

To run the code and reproduce the results, your device needs to satisfy the following requirements:

- **Mac**: ≥ MacOS 12
- **Windows**: ≥ Windows10
- **Linux**: ≥ Ubuntu 18.04

Additionally, you need to have **Anaconda 3** or a higher version of conda installed.

## Setup

To get started with the environment setup, follow the instructions:

1. Open a shell-based terminal (or command prompt). Clone this repository and run the following commands to create and activate a new Anaconda environment:

   ```bash
   git clone https://github.com/skywalker-lt/3D-Quantile-Regression-with-elliptical-approximation.git
   conda env create -n 3D_QuantileReg
   conda activate 3D_QuantileReg
   ```
2. Install the dependencies:

   ```
   pip install numpy pandas plotly statsmodels pyyaml
   ```
3. Prepare your data:

   In the file ```yourdataset.yaml```, replace **Your_Dataset.csv** with the path or filename of your dataset (**Your dataset must be in .csv format, otherwise the program will raise an error**).
   Also change the variable names in the ```config.yaml``` file to match the column headers in your dataset for the three variables (e.g., time, concentration, and wavelength).

## Evaluate

To perform regression on the sample data (same as the one used for sample calculation in the essay), you can simply run the following command in your terminal:

   ```
   python regression.py --config plotly_sample.yaml
   ```
   
If your are using your own datase, run

   ```
   python regression.py --config yourdataset.yaml
   ```
## References:

Thanks for the data sourced from Plotly datasets:
https://github.com/plotly/datasets

Additional references and other datasets used in the essay can be found in the bibliography of the original work. Thank you!
