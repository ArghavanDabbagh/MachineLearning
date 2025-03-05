# Deep Learning and Generative Models

This project explores deep learning techniques for image classification and generative models using PyTorch. The primary focus is on transfer learning, class imbalance handling, and generative models.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training Models](#training-models)
- [Evaluating Models](#evaluating-models)
- [Generative Models](#generative-models)
- [Results](#results)

## Project Structure

```
project/
├── data/
│   ├── chest_xray/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/

```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- scikit-learn

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download the dataset and place it in the `data/chest_xray` directory.
2. The dataset should have the following structure:
```
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test/
```

## Training Models

### 1. Transfer Learning

1. Load and explore the data.
2. Address class imbalance using weighted sampling.
3. Train a CNN model with more than 18 layers.
4. Train a pre-trained ResNet50 model using transfer learning.

### 2. Fine-tuning Models

1. Fine-tune the pre-trained ResNet50 model by unfreezing the last block and the fully connected layer.

## Evaluating Models

1. Evaluate the trained models on the test set.
2. Generate confusion matrices and classification reports.

## Generative Models

1. Load the CIFAR-10 dataset.
2. Define the VQ-VAE model, including the encoder, VQEmbeddingEMA, and decoder.
3. Train the VQ-VAE model.
4. Evaluate the model and visualize reconstructions.
5. Generate new images using the trained VQ-VAE model.

## Results

- The project demonstrates the effectiveness of transfer learning and fine-tuning in image classification tasks.
- The VQ-VAE model successfully generates new images based on the learned embeddings.

## Acknowledgments

- The dataset used in this project is from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- The VQ-VAE implementation is inspired by the original paper by Aaron van den Oord et al.

```
