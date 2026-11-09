# Handwritten Text Recognition using TrOCR

## Project Overview

This project focuses on Handwritten Text Recognition (HTR) using Microsoft's TrOCR (Transformer-based Optical Character Recognition) model.

The project uses the IAM Handwriting Database to fine-tune and evaluate a pretrained TrOCR model for recognizing handwritten text.

## Objectives

- Understand Transformer-based handwritten text recognition.
- Prepare and preprocess the IAM handwriting dataset.
- Use a pretrained TrOCR model.
- Fine-tune TrOCR on handwritten text images.
- Evaluate recognition performance using Character Error Rate (CER) and Word Error Rate (WER).
- Compare reproduced results with the original research paper.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- TrOCR
- Google Colab
- Pandas
- NumPy
- Matplotlib
- JiWER

## Dataset

**Dataset:** IAM Handwriting Database

The dataset contains handwritten text-line images and their corresponding transcriptions.

Dataset splits after filtering valid annotations:

| Split | Samples |
|---|---:|
| Training | 5,419 |
| Validation | 1,539 |
| Testing | 1,480 |
| **Total** | **8,438** |

## Model

**Pretrained model:** `microsoft/trocr-small-stage1`

The model is loaded using Hugging Face Transformers.

## Baseline Evaluation

The pretrained model was evaluated on the validation dataset before fine-tuning.

| Metric | Baseline Result |
|---|---:|
| CER | 13.20% |
| WER | 35.24% |

## Project Structure

```text
TrOCR-Handwritten-Text-Recognition/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── TrOCR_Assignment_02.ipynb
│
├── src/
│   ├── dataset_preparation.py
│   ├── model.py
│   ├── training.py
│   └── evaluation.py
│
├── results/
├── plots/
└── reports/
```

## Evaluation Metrics

- **Character Error Rate (CER):** Measures the difference between predicted and actual text at the character level.
- **Word Error Rate (WER):** Measures recognition errors at the word level.

## Project Status

- [x] Dataset preparation
- [x] Dataset splitting
- [x] Pretrained TrOCR model loading
- [x] Baseline evaluation
- [x] Initial source-code organization
- [ ] Model fine-tuning
- [ ] Final testing
- [ ] Training and validation graphs
- [ ] Comparison with original research paper
- [ ] Technical report

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

PyTorch may require a separate installation depending on your hardware and CUDA configuration.

## Author

BSCS Student

## Note

This repository is part of an academic assignment on Handwritten Text Recognition using Transformer-based models.
