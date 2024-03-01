# Age Classification Model

This project aims to build a machine learning model for classifying age groups based on heart rate and breath rate data.

## Overview

The project consists of several components:

- `data_processing.py`: Module for loading and preprocessing the dataset.
- `model.py`: Module for building, compiling, and training the age classification model.
- `main.py`: Script to orchestrate the data processing, model training, and evaluation.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/age-classification.git
    ```

2. Install the required Python packages:


## Usage

1. Place your dataset file (`AgeClassified.csv`) in the same directory as the Python files.

2. Run the `main.py` script:

    ```bash
    python main.py
    ```

## File Descriptions

### `data_processing.py`

This module contains functions to load and preprocess the dataset.

### `model.py`

This module contains functions to build, compile, and train the age classification model.

### `main.py`

The main script that orchestrates data processing, model training, and evaluation.

## Dataset

The dataset (`AgeClassified.csv`) contains heart rate and breath rate data along with age classifications.

## Model Evaluation

The model is evaluated using the mean squared error (MSE) metric.

## Requirements

- Python 
- numpy
- pandas
- scikit-learn
- tensorflow


## Research Paper Citation

The radar dataset used in this project is sourced from the research paper titled "Radar Recorded Child Vital Sign Public Dataset and Deep Learning-Based Age Group Classification Framework for Vehicular Application".

**By Authors:**
- Sungwon Yoo
- Shahzad Ahmed
- Sun Kang
- Duhyun Hwang
- Jungjun Lee
- Jungduck Son
- Sung Ho Cho

**Paper Link:** [Radar Recorded Child Vital Sign Public Dataset and Deep Learning-Based Age Group Classification Framework for Vehicular Application](https://www.mdpi.com/1424-8220/21/7/2412)

**Dataset Link:** https://figshare.com/s/936cf9f0dd25296495d3

