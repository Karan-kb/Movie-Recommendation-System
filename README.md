# Algorithmic Movie Recommendation Engine using High-Dimensional Vector Embeddings

This repository contains the source code for an end-to-end Machine Learning-driven recommendation engine. The platform translates rich categorical metadata (genres, titles, descriptions, and structural tags) into high-dimensional numerical spaces to compute personalized similarity indexing for content recovery.

The system serves as a practical implementation of **Information Retrieval (IR)** and **Content-Based Filtering (CBF)** algorithms optimized for low-latency web inference.

---

## 🏗️ Algorithmic Core & Mathematical Framework

Recommendation pipelines must elegantly handle textual dimensionality and dataset sparsity. This project maps metadata components using a clear, mathematically sound natural language processing (NLP) matrix workflow:

### 1. Vector Space Transformation & Tokenization
Textual descriptions, key phrases, and genre labels are extracted, parsed, and tokenized. The tokens undergo vectorization to transform structural strings into a numeric matrix representation:
* **Feature Extraction Matrix:** Text blocks are vectorized using an optimized dictionary approach (e.g., token-count vectors or TF-IDF weights) to isolate the most high-variance attributes across thousands of dynamic content inputs.
* **Stop-Word Filtration:** Ingestion pipelines automatically isolate and eliminate non-informative syntax constraints to drastically reduce structural noise in the vector array.

### 2. Angular Proximity via Cosine Similarity
To evaluate content relevance, the application measures the angular distance between feature vectors within the high-dimensional space rather than relying on absolute magnitude calculations. Proximity is determined using the **Cosine Similarity** formula:

\[\text{Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}\]

* **Proximity Matching:** A score near `1.0` denotes high structural alignment, enabling the inference engine to rank and extract the top K nearest-neighbor candidates for immediate display.
* **Cold-Start Resiliency:** Because it maps spatial relationships between product metadata vectors rather than relying on collaborative user tracking matrices, the framework bypasses the classic cold-start constraint for new, unrated entries.

---

## 🛠️ Operational Architecture & Modules

* **Data Preprocessing & Ingestion Pipeline:** Automated data cleaning routines that normalize schema inconsistencies, handle missing feature vectors, and concatenate raw categorical values into unified token blocks.
* **Similarity Indexing Compute Core:** Highly optimized matrix multiplication engine that calculates distance profiles across thousands of entries in parallel vectors.
* **Streamlit Visualization Layer:** A decoupled, low-overhead micro-frontend designed to facilitate real-time query inputs and render top recommendation arrays without heavy page refreshes.

---

## ⚙️ Tech Stack & Requirements

* **Primary Engine Language:** Python
* **Mathematical & Data Matrix Toolkits:** NumPy, Pandas
* **Machine Learning Pipeline Libraries:** Scikit-Learn (`CountVectorizer` / `TfidfVectorizer`, Cosine Pairwise Computations)
* **Interface & Prototyping Vector:** Streamlit Web Engine

---

## 🚀 Execution & Deployment Pipeline

### Step 1: Clone Repository & System Setup
Clone the framework files locally and prepare your internal runtime configurations:
```bash
git clone https://github.com
cd Movie-Recommendation-System
```

### Step 2: Initialize Virtual Workspace & Dependencies
Instantiate a clean python virtual execution environment to eliminate package dependency drift:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run Interactive Vector Inference Application
Launch the graphical micro-frontend server to execute live text queries and visualize nearest-neighbor metadata matches:
```bash
streamlit run app.py
```
