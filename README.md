# Drift Detection and Embedding Tracking Framework

This project provides a framework for detecting data drift and tracking embedding distributions using various vector-based and distribution-based metrics. It supports multiple models, datasets, and drift strengths, enabling experiments and visualisations.

This provides a high-level workflow of using LLMs and DL experiments, creating embeddings, testing vectors and distribution-based drifts on synthetic and real-world streams.  
![High-level workflow](images/workflow0.png)

---

## Features

- **Drift Detection**: Supports both vector-based and distribution-based metrics for detecting data drift.
- **Embedding Tracking**: Tracks embeddings using KLL sketches, histograms, and PCA-based dimensionality reduction.
- **Metrics**: Includes metrics such as KL divergence, Jensen-Shannon divergence, Wasserstein distance, and others.
- **Visualisation**: Generates detailed plots for analysing drift detection results.
- **Extensibility**: Easily add new models, datasets, or metrics.

---

## File Structure

### Core Files
- `distribution_experiment.py`: Runs experiments focused on distribution-based metrics.
- `drift_detection.py`: Implements drift detection using both vector-based and distribution-based approaches.
- `embedding_tracker.py`: Tracks embeddings and computes distances using various methods.
- `metrics.py`: Defines vector-based and distribution-based metrics.
- `plot.py`: Generates visualisations for experiment results.
- `utils.py`: Utility functions for data loading, embedding extraction, and drift introduction.

### Configuration
- `config.py`: Contains default arguments for models, datasets, and experiment parameters.

### Experiments
- `vector_experiment.py`: Placeholder for vector-based experiments (under development).

### Tests
- `tests/`: Contains unit tests for KLL transformations and other components.

### Results
- Experiment results and plots are saved in the `data/` directory.

---

## Installation
Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running Experiments
Run the `distribution_experiment.py` script to evaluate distribution-based metrics:
```bash
python distribution_experiment.py
```

or 
```bash
python vector_experiment.py
```

Run the `drift_detection.py` script to evaluate both vector-based and distribution-based metrics:
```bash
python drift_detection.py
```

### Generating Plots
To generate visualisations for experiment results:
```bash
python plot.py
```

### Configuration
Modify `config.py` to customise models, datasets, and experiment parameters.

---

## Key Parameters

- **Models**: Specify models in `config.py` (e.g., `distilbert-base-uncased`, `google/mobilebert-uncased`).
- **Datasets**: Add datasets in `config.py` (e.g., `ag_news`).
- **Drift Strengths**: Control the level of drift introduced during experiments.
- **Metrics**: Choose from vector-based (e.g., Euclidean, cosine) or distribution-based (e.g., KL divergence, Wasserstein).

---

## Outputs

- **Results**: Saved as `results.json` in the `data/` directory.
- **Plots**: Includes similarity trends, memory usage, and overhead comparisons.


# Baseline Experiment 
The baseline experiment is located at `/baseline-experiment/`. It contains the following files:
- `baseline_experiment.py`: The main script for running the baseline experiment. This file needs to have the amazon dataset downloaded and unzipped together with the product mapper in the `data/` directory. 

Here's how you can update your README with the provided image descriptions and paths from your GitLab repo. I’ll insert a new section after the **Outputs** section, and format the image markdown for clarity:

---

## Visual Overview

This section provides visualizations illustrating concepts and workflows from the framework:

**Image 1**: Controlled Simulation of Embedding Drift. Text data is shuffled token-wise for LLMs, while tabular features undergo incremental shifts for DeepFM.  
![Controlled Simulation of Embedding Drift](images/workflow1.png)

**Image 2**: Embedding Drift Detection via Full and Compressed Representations. Comparison of histograms and KLL-based summaries for detecting embedding shifts.  
![Embedding Drift Detection](images/workflow2.png)

**Image 3**: Distance-Based Embedding Drift Detection. Geometric shifts tracked over time using distance metrics.  
![Distance-Based Embedding Drift Detection](images/workflow3.png)

