# Machine Learning and Graph Analysis

This project explores various machine learning techniques, including supervised and unsupervised learning, as well as graph analysis using NetworkX. The primary focus is on text classification, dimensionality reduction, and community detection in graphs.

## Table of Contents

- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Graph Analysis](#graph-analysis)
- [Results](#results)


## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- NetworkX
- python-louvain
- powerlaw

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download the datasets and place them in the `data` directory.
2. The datasets should include:
   - `imdb_dataset.csv`: IMDB movie reviews dataset.
   - `fruit_data_with_colors.txt`: Fruit dataset with color features.
   - `hero-comic-network.csv`: Marvel hero-comic network dataset.

## Supervised Learning

### 1. Naive Bayes Classifier

1. Load and preprocess the IMDB dataset.
2. Vectorize the text data using CountVectorizer.
3. Train a Naive Bayes classifier on the training data.
4. Evaluate the model on the test data and generate a classification report.

### 2. SVM Classifier

1. Load and preprocess the IMDB dataset.
2. Vectorize the text data using CountVectorizer.
3. Train an SVM classifier on the training data.
4. Evaluate the model on the test data and generate a classification report.

### 3. KNN Classifier

1. Load and preprocess the IMDB dataset.
2. Vectorize the text data using CountVectorizer.
3. Train a KNN classifier on the training data.
4. Evaluate the model on the test data and generate a classification report.

## Unsupervised Learning

### 1. Dimensionality Reduction

1. Load the Breast Cancer and Fruit datasets.
2. Apply PCA to reduce the dimensionality of the datasets.
3. Visualize the PCA-transformed data.

### 2. Manifold Learning

1. Apply MDS and t-SNE to the Breast Cancer and Fruit datasets.
2. Visualize the transformed data.

## Graph Analysis

### 1. Zachary's Karate Club

1. Load and visualize the Zachary's Karate Club graph.
2. Apply hierarchical clustering and visualize the dendrogram.
3. Apply the Louvain method for community detection and visualize the communities.
4. Apply the Girvan-Newman algorithm for community detection and evaluate the results.

### 2. Marvel Universe Network

1. Load and visualize the Marvel hero-comic network.
2. Apply the Louvain method for community detection and visualize the communities.

## Results

- The project demonstrates the effectiveness of various machine learning techniques for text classification and dimensionality reduction.
- The graph analysis section showcases different community detection algorithms and their applications to real-world networks.

## Acknowledgments

- The IMDB dataset is from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
- The fruit dataset is from [University of California, Irvine](https://archive.ics.uci.edu/ml/datasets/Fruit+Data+with+Colors).
- The Marvel hero-comic network dataset is from [Kaggle](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network).

```
