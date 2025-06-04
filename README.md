# ITCare-Insight-Model  

### Trainig Machine Learning Model for IT Helpdesk to predict assign groups for issues.
This repository contains all code for preprocessing data, building the hybrid Q&A search index, and evaluating time-series and classification models used in ITCare Insight’s predictive analytics.  

---  

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Installation and Setup](#installation-and-setup)  
- [Data Preprocessing](#data-preprocessing)  
- [Advanced Model Training](#advanced-model-training)   
- [QnA Suggestion Implementation](#qna-suggestion-implementation)

---  

## Prerequisites
- Python 3.8+
- Access to Dataset/incident_report_q&a_cleaned.csv
- pip

---  

## Installation and Setup  

1. Clone the Repository

```bash
git clone https://github.com/OshadiPalliyaguruge/ITCare-Insight-Model.git 
cd ITCare-Insight-Model
```

2. Install Dependencies  
```bash
pip install -r requirements.txt
```

---  

## Data Preprocessing

Before any modeling or indexing, raw data must be profiled and cleaned. We follow two phases:

**Phase 1: Understand the Dataset**  
- Check dataset shape (number of rows and columns)  
- Inspect data types and value distributions for each column  
- Identify missing values and duplicate records  

**Phase 2: Data Cleaning & Transformation**  
- Remove duplicate rows to ensure unique records  
- Handle missing values (e.g., imputation or removal)  
- Filter out or rename irrelevant columns  
- Normalize or encode fields as needed for downstream tasks  

Once these steps complete, save the cleaned CSV (e.g. `incident_report_preprocessed_final_98000_cleaned.csv`) in `Dataset/` and proceed with preprocessing, indexing, or model training.  

#### *Files in `Dataset_Preprocessing` folder will do these preprocessings.*

---  

## Advanced Model Training

This section describes how to train the hybrid neural network (classification) model that combines BERT embeddings with one-hot encoded features to predict the Assigned Group.


### 1. Launch the Notebook
From the project root, start Jupyter:  
```bash
jupyter notebook advanced_model.ipynb
```

### 2. Configure Your Environment
**GPU Check (optional):**  
  In the first cell, run:
  ```bash
  !nvidia-smi
  torch.cuda.empty_cache()
  ```
**Dataset Path:**  
Ensure the CSV path in the “Load dataset” cell matches to correct path for your CSV file.


### 3. Run Training Cells
Execute cells in order:
1. Imports & setup
2. One-hot encode categorical features & generate BERT embeddings
3. Combine features and split into train/val/test
4. Define `NeuralNetworkWithPrecomputedBERT` model class
5. Training loop with AdamW, LR scheduler, and early stopping
6. Evaluation on test set

The notebook will automatically create an `output_<timestamp>/` folder and save:
- `best_trained_model_<timestamp>.pth` (PyTorch state dict)  
- `advanced_trained_encoder.pkl` (OneHotEncoder)  
- `advanced_new_y_mapping.pkl` (label mapping)  
- `advanced_metadata.pkl` (input/output sizes)
- 

#### Model Training Completion 
  ![image](https://github.com/user-attachments/assets/9545bfef-ede6-4c48-8466-6d1fb012151f)
  

### 4. Load & Use Your Model  

---  


## QnA Suggestion Implementation

### 1. Preprocessing & Indexing
Run `preprocess_and_index.py` to:  
- Tokenize, remove stopwords, and lemmatize summaries and resolutions
- Build an inverted index for quick token lookup
- Compute TF-IDF vectors and SBERT embeddings

Save all artifacts into `QnA_Suggestions/preprocessed_data_<timestamp>.pkl`

```bash
python preprocess_and_index.py
```

### 2. Hybrid Q&A Search  

Use `search.py` to load the latest preprocessed pickle and perform hybrid search:  
- Exact match
- Token-based inverted index scoring
- TF-IDF cosine similarity
- SBERT semantic similarity
- Weighted combination of scores

Example usage:  
```bash
python search.py
```

```python
from search import search
results = search("How do I reset my password?", top_n=3, min_similarity=0.3)
``` 


#### *With these components, you can reproduce data preprocessing, power the hybrid Q&A suggestion engine, and evaluate time-series and classification performance for the ITCare Insight project.*
