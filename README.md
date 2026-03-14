# News-Classification-By-Clustering

Mini project for the **Natural Language Processing course (ICT3.013)** at the **University of Science and Technology of Hanoi (USTH)**.

This project explores **unsupervised topic discovery** on large-scale academic text data using **text preprocessing, TF-IDF encoding, dimensionality reduction, and K-Means clustering**. The goal is to automatically group documents with similar semantic content without using labeled training data.

---

# Project Overview

The rapid growth of digital text data makes manual classification impractical. Many real-world datasets lack labels, making **unsupervised learning techniques** important for discovering hidden structure in text corpora.

This project builds a complete NLP pipeline:

1. Text cleaning and preprocessing  
2. Feature extraction with TF-IDF  
3. Dimensionality reduction  
4. Document clustering with K-Means  
5. Visualization and evaluation of clusters  

The system automatically groups documents based on semantic similarity and evaluates clustering quality using several metrics.

---

# Dataset

We used the **`UniverseTBD/arxiv-abstracts-large`** dataset from HuggingFace.

Dataset characteristics:

- Over **2 million scientific papers**
- Fields include:
  - paper ID
  - title
  - authors
  - categories
  - abstract

For computational feasibility:

- **30,000 abstracts** were randomly sampled
- arXiv categories were grouped into **13 broader disciplines**
- Each document was assigned a **primary category** for evaluation purposes

Scientific abstracts were chosen because they concisely summarize research topics.

---

# Project Pipeline

## 1. Text Preprocessing

Raw text contains noise and linguistic variation. The preprocessing pipeline includes:

- Unicode normalization (NFKC)
- Lowercasing
- Removing:
  - URLs
  - emails
  - special characters
- Stopword removal using **NLTK**
- Lemmatization using **spaCy (`en_core_web_sm`)**
- Numeric normalization
- Domain abbreviation preservation (`ai`, `ml`, `dl`, `nlp`)

This step improves the semantic consistency of the corpus before vectorization.

---

## 2. Feature Encoding

Text documents were converted into numerical vectors using **TF-IDF (Term Frequency – Inverse Document Frequency)**.

Configuration:

TF-IDF highlights **rare but informative terms**, which is especially useful for distinguishing scientific topics.

---

## 3. Dimensionality Reduction

The TF-IDF matrix is extremely high-dimensional.

We applied:

**TruncatedSVD**

- Reduced features to **50 dimensions**
- Produced dense vectors suitable for clustering

For visualization:

**PCA** was used to project the vectors into **2D space**, enabling cluster visualization.

---

## 4. Clustering Method

The project uses **K-Means clustering** due to its scalability and efficiency in high-dimensional spaces.

Algorithm workflow:

1. Initialize `K` cluster centroids  
2. Assign documents to nearest centroid  
3. Update centroid positions  
4. Repeat until convergence  

Experiments were conducted with:

max_features = 200000
min_df = 5
max_df = 0.95

to determine the optimal number of clusters.

---

# Evaluation Metrics

Cluster quality was evaluated using: k = 10->60

- **Silhouette Score**
- **Calinski–Harabasz Index**
- **Davies–Bouldin Index**
- **Elbow Method**

These metrics assess:

- cluster separation
- intra-cluster cohesion
- inter-cluster distance

Visualization with PCA scatter plots also helped validate clustering structure.

---

# Results

The clustering model successfully grouped research abstracts into meaningful topic clusters.

Key observations:

- Distinct clusters formed for major academic fields
- PCA visualization showed clear separation between topics
- Some documents appeared between clusters, reflecting interdisciplinary research

Overall, **TF-IDF + K-Means** proved to be an effective and computationally efficient approach for unsupervised topic discovery.

---

# Tech Stack

Python libraries used:

- **scikit-learn**
- **spaCy**
- **NLTK**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **WordCloud**

---

# Project Structure
```
NLP-News-Clustering/
│
├── data/
│ └── sampled_arxiv_dataset.csv
│
├── notebooks/
│ └── EDA_and_experiments.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── vectorization.py
│ ├── clustering.py
│ └── visualization.py
│
├── models/
│ └── kmeans_model.pkl
│
├── results/
│ ├── cluster_visualization.png
│ └── evaluation_metrics.csv
│
└── README.md
```
---

# Installation

```bash
git clone https://github.com/your-username/nlp-news-clustering.git
cd nlp-news-clustering
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

# Running the Project
Example pipeline:
```
python src/preprocessing.py
python src/vectorization.py
python src/clustering.py
python src/visualization.py
```
# Authors
USTH – ICT3.013 Natural Language Processing

- Đoàn Minh Cường
- Nguyễn Trung Hiền
- Nguyễn Gia Huy
- Nguyễn Thuý Ngọc
- Trần Phú Thái
- Nguyễn Song Thắng
- Vũ Phạm Diệp Thảo
- Nguyễn Mạnh Khánh An

Lecturer: Dr. Phạm Quang Nhật Minh