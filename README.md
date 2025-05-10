# lib-ml

## Overview

`lib-ml` is a Python library created as part of the REMLA Restaurant Sentiment Analysis application.

Its purpose is to encapsulate the common machine learning preprocessing logic required for the restaurant sentiment analysis task. This ensures consistency between the model training pipeline (`model-training`) and the prediction service (`model-service`), both of which depend on this library.

Currently, it provides two functions for preprocessing review data:
- `preprocess_dataset`: Method to preprocess the entire training dataset.
- `preprocess_input`: Method to preprocess the new input text using the same preprocessing steps as in the training dataset.

## Installation

This library is intended to be installed directly from its Git tag using `pip`.

To install a specific version (e.g., `v0.1.0`) as a dependency in another component (like `model-training` or `model-service`):

```bash
pip install git+https://github.com/remla25-team12/lib-ml.git@v<tag-name>

# Example:
# pip install git+https://github.com/remla25-team12/lib-ml.git@v0.1.0
# Replace <tag_name> with the desired release tag (e.g., v0.1.0). 

# Or
# lib-ml @ git+https://github.com/remla25-team12/lib-ml.git@v0.1.0 
# You can add this line to the requirements.txt file of your project.
```

## Usage 

Once installed, import and use the preprocessing functions as needed.

Example usage within model-training or model-service:

```bash
from lib_ml.preprocessing import preprocess_dataset

# Preprocess the dataset
X, y, cv = preprocess_dataset(dataset)
```
