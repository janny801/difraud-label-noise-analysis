# DiFrauD Label Noise Analysis

Dataset Cartography and label noise mitigation for fraud detection datasets.

## Academic Context
- **Course**: Security Analytics (University Course Project)
- **Timeline**: Oct. 2025 – Dec. 2025

## Overview & Key Highlights
- **Dataset Cartography**: Analyzed training dynamics (confidence $\mu$ and variability $\sigma$ across epochs) on 95k+ samples to map fraud data distributions into easy, ambiguous, and hard-to-learn partitions.
- **Label Noise Mitigation**: Filtered noisy and ambiguous sample representations to prevent overfitting to mislabeled data and stabilize training.
- **Model Architectures & Baselines**:
  - **Sentence-Level Embeddings + MLP**: SentenceBERT static representations tuned via Optuna with Weights & Biases (wandb) experiment tracking.
  - **Token-Level Embeddings + Bi-LSTM**: Pre-trained GloVe embeddings for sequential modeling.
  - **Transformer Fine-Tuning**: Pre-trained RoBERTa fine-tuned via Hugging Face Trainer.
  - **Classical Baselines**: Support Vector Machines (SVM).
- **Impact**: Improved downstream classification accuracy by **14%** across noisy benchmarks after filtering ambiguous representations.

## Contents
- `DiFrauD_label_noise_analysis.ipynb`: Complete pipeline covering data loading, feature extraction, model implementations, hyperparameter optimization, and evaluation comparisons.

## Stack
- Python, PyTorch, Hugging Face Transformers, Sentence-Transformers, scikit-learn, Optuna, Weights & Biases (W&B).
