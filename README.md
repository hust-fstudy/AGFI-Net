# AGFI-Net: Adaptive Graph Feature Interaction Network for Event-Based Action Recognition

The code will be released after the paper is published.

## Overview

![Framework](./assets/Framework.svg)

## Installation

### Requirements

All the codes are tested in the following environment:

- Linux (Ubuntu 20.04)
- Python 3.12
- PyTorch 2.4.0
- CUDA 11.8

### Dataset Preparation

All datasets should be downloaded and placed within the `dataset` directory, adhering to the folder naming rules and structure specified for the `DvsGesture` dataset as provided in the project.

## Quick Start

Clone the repository to your local machine:

```
git clone https://github.com/hust-fstudy/AGFI-Net
cd AGFI-Net
```

Once the dataset is specified in the `dataset_dict` dictionary within the `main` function of the `run_recognition.py` file, we can train and test it using the following command:

```bash
python run_recognition.py
```

