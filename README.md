# A Modular RAG-Based System for Automatic Short Answer Scoring with Feedback

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Zero-Shot Pipeline](#zero-shot-pipeline)
- [RAG Pipeline](#rag-pipeline)
  - [ColBERT Environment Setup](#step-1-set-up-colbert-environment)
  - [Indexing with ColBERT](#step-2-indexing-with-colbert)
  - [Running the ColBERT Server](#step-3-start-the-colbert-server)
  - [Running the RAG Pipeline](#step-4-run-rag-pipeline)
  - [Majority Class Server](#step-5-run-the-majority-class-server)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository contains the code for the paper **"A Modular RAG-Based System for Automatic Short Answer Scoring with Feedback"**, which explores the use of Retrieval-Augmented Generation (RAG) for scoring short answers and providing detailed feedback to students. The system is designed to optimize scoring through modular pipelines, including zero-shot and few-shot setups, and offers an efficient method for integrating feedback.

---

## Installation

To get started, clone this repository and set up the necessary environments for running the notebooks:

1. Clone the repository:
   ```bash
   git clone https://github.com/mennafateen/modular-asas-f-rag
   cd modular-asas-f-rag
   ```

2. Install dependencies for the zero-shot and RAG pipelines by setting up the respective environments.

### Environment Setup for Zero-Shot and RAG Pipelines

1. **Create and activate a virtual environment** (recommended:  `conda`):
   ```bash
   conda create --name asas-env python=3.10
   conda activate asas-env
   ```
   
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Zero-Shot (ASAS-F-Z) Pipeline

1. **Run the zero-shot pipeline notebook** for automatic short answer scoring:
   ```bash
   jupyter notebook asas-f-z.ipynb
   ```

---

## Optimized Few-Shot (ASAS-F-Opt) Pipeline

1. **Run the optimized pipeline notebook** for automatic short answer scoring:
   ```bash
   jupyter notebook asas-f-opt.ipynb
   ```

---

## RAG Pipeline

### 1: Setting Up ColBERT Environment

For the RAG pipeline, an additional environment for ColBERT is required:

1. **Create and activate a new virtual environment**:
   ```bash
   conda create --name colbert-env python=3.10
   conda activate colbert-env
   ```

2. **Install the ColBERT server requirements**:
   ```bash
   pip install -r requirements_server.txt
   ```

### Step 2: Indexing with ColBERT

1. **Run the ColBERT indexing notebook** to create the required index:
   ```bash
   jupyter notebook colbert_indexing.ipynb
   ```

###  3: Starting the ColBERT Server

1. **Run the ColBERT server notebook** to start the server for retrieval operations:
   ```bash
   jupyter notebook colbert_server.ipynb
   ```

### 4: Running the RAG Pipeline

1. **Switch to the original environment** (from `requirements.txt`) and **run the RAG pipeline**:
   ```bash
   jupyter notebook asas-f-rag.ipynb
   ```

###  5: Running the Majority Class Scorer

1. **Use the ColBERT environment** to **run the majority class server**:
   ```bash
   jupyter notebook majority_class_server.ipynb
   ```

---
## Evaluation

The evaluation of the system is conducted on the ASAS dataset, which is available in the `data` directory. The dataset contains short answers for multiple-choice questions, along with the correct answers and the corresponding scores. The evaluation metrics include accuracy, F1-score, RMSE, BLEU, ROUGE and BERTScore scores.

You can evaluate the generated outputs which are located in the `generated_results` directory by running the evaluation notebook:

```bash
jupyter notebook evaluation.ipynb
```

---


## Citation

If you find this work useful in your research, please cite our paper:

```

```

---

