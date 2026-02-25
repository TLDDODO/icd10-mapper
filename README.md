# ICD-9 Chapter Mapping Pipeline

An NLP pipeline that maps clinical text descriptions to ICD-9 disease chapters using embedding-based vector retrieval with human-in-the-loop validation.

Built as a portfolio project demonstrating skills relevant to medical claims text analysis and ICD coding automation.

---

## What This Does

**Input:** A clinical text description (e.g. `"acute myocardial infarction no ST elevation"`)  
**Output:** Top-3 matching ICD-9 chapters with confidence scores

```
#1  Circulatory System                   score=0.891
#2  Symptoms, Signs, Ill-defined         score=0.512
#3  Respiratory System                   score=0.441
```

---

## Pipeline

```
Clinical text
     ↓
spaCy preprocessing (tokenize, normalize, preserve clinical negations)
     ↓
Dual embedding models:
  ├─ all-MiniLM-L6-v2       (sentence-transformers, general)
  └─ Bio_ClinicalBERT        (transformers, clinical domain)
     ↓
Cosine similarity retrieval over 18-chapter ICD index
     ↓
Top-3 predictions + confidence scores
     ↓
Evaluation: top-1/top-3 accuracy, macro F1, confusion matrix
     ↓
Human-in-the-loop CLI validation
```

---

## Data

- **D_ICD_DIAGNOSES.csv** — 14,567 ICD-9 codes with full text descriptions (MIMIC-III Demo)
- **DIAGNOSES_ICD.csv** — 1,761 real patient diagnosis records

Chapter labels are derived automatically from ICD-9 numeric ranges (no manual annotation required).

---

## Key Design Decisions

**Why preserve clinical negations?**  
Words like `no`, `denies`, `without` change clinical meaning entirely. Standard stopword removal would strip these, corrupting the semantics.

**Why 18-chapter index instead of 14k ICD index?**  
Querying a model's own training descriptions creates data leakage — top-1 would trivially be itself. The chapter-level index keeps retrieval honest and evaluatable.

**Why compare two models?**  
MiniLM is fast and general. Bio_ClinicalBERT is trained on clinical text (PubMed + MIMIC). Comparing them shows which performs better on medical terminology — this is the kind of ablation study expected in production NLP work.

---

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas scikit-learn sentence-transformers transformers torch spacy numpy matplotlib seaborn

# Download spaCy model
python -m spacy download en_core_web_sm
```

Place data files in `data/raw/`:
```
data/raw/
├── D_ICD_DIAGNOSES.csv
└── DIAGNOSES_ICD.csv
```

---

## Usage

```bash
# Run full pipeline (both models)
python main.py

# Run MiniLM only (faster, ~2 min)
python main.py --model minilm

# Run with human review session
python main.py --model minilm --review
```

---

## Results

| Model | Top-1 Acc | Top-3 Acc | Macro F1 |
|-------|-----------|-----------|----------|
| MiniLM | 44.2% | 69.4% | 0.4119 |
| Bio_ClinicalBERT | 43.5% | 65.0% | 0.3926 |

*Results populated after running `python main.py`*

---

## Project Structure

```
icd10-mapper/
├── data/raw/                  ← MIMIC Demo CSV files
├── src/
│   ├── preprocess.py          ← spaCy text normalization
│   ├── embedder.py            ← MiniLM + ClinicalBERT embeddings
│   ├── retriever.py           ← cosine similarity vector search
│   ├── evaluator.py           ← accuracy, F1, confusion matrix
│   └── human_review.py        ← CLI validation tool
├── results/                   ← evaluation outputs, review logs
├── main.py
└── README.md
```
