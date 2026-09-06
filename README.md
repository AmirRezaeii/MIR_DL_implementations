# Modern Information Retrieval — Deep Learning Assignments

Practical assignments from the Deep Learning component of the **Modern Information Retrieval** course. Each notebook implements a core NLP technique from scratch or with PyTorch/HuggingFace.

---

## Notebooks

### `word_embedding.ipynb` — Word Embeddings
Trains and compares three word embedding models on a book metadata corpus (titles, descriptions, genres):
- **Skip-Gram Word2Vec** built from scratch with negative sampling
- **FastText** via Gensim (subword n-grams for rare word handling)
- **Native FastText** (Facebook's optimized implementation)

Models are evaluated on odd-one-out semantic tasks and visualized with PCA projections.

---

### `book_genre_classification_and_clustering.ipynb` — Classification & Clustering
A full text ML pipeline for predicting book genre from description text — all classifiers, metrics, and vectorizers implemented from scratch (no scikit-learn).

**Part 1 — Classification:** Compares TF-IDF and precomputed BGE sentence embeddings (1024-dim) as input representations across four classifiers: Naive Bayes, kNN, Logistic Regression, and MLP. Includes class imbalance handling via resampling strategies selected by development macro-F1, per-class error analysis, and TF-IDF weight interpretation.

**Part 2 — Clustering:** Unsupervised grouping using K-Means, Spherical K-Means (cosine distance), and DBSCAN — all from scratch. Clusters are evaluated with internal metrics (silhouette) and external metrics (NMI against true genres).

---

### `Bert.ipynb` — BERT Fine-Tuning
Compares three adaptation strategies for book genre classification using `bert-base-uncased`:

- **Full Fine-Tuning** — all layers updated
- **LoRA** — low-rank adapters injected into attention weights; only A and B matrices trained
- **Few-Shot** — frozen encoder, only classification head trained on limited data

Followed by a layer-wise **representation analysis**: `[CLS]` embeddings are extracted from every BERT layer before and after fine-tuning and projected with PCA to visualize how genre clusters form and sharpen across layers.

---

## Requirements
```
torch, transformers, peft, gensim, fasttext, datasets, numpy, matplotlib
```
All notebooks run on **Google Colab** (free T4 GPU). `word_embedding` and the classification notebook can run on CPU.
