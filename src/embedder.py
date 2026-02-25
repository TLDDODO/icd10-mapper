"""
embedder.py
-----------
Generates sentence embeddings using two models for comparison:
  1. all-MiniLM-L6-v2       (sentence-transformers, general purpose)
  2. Bio_ClinicalBERT        (transformers, clinical domain)

Both produce normalized vectors ready for cosine similarity.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# ── Model 1: MiniLM (sentence-transformers) ───────────────────────────────────

class MiniLMEmbedder:
    """
    General-purpose sentence embedder.
    Fast, good baseline. Used as comparison against ClinicalBERT.
    """

    def __init__(self):
        print("[MiniLM] Loading model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[MiniLM] Ready.")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized vectors.
        Returns numpy array of shape (n, 384).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True   # L2 normalize → cosine sim = dot product
        )
        return embeddings


# ── Model 2: Bio_ClinicalBERT (transformers) ──────────────────────────────────

class ClinicalBERTEmbedder:
    """
    Clinical domain sentence embedder using Bio_ClinicalBERT.
    Uses mean pooling over last hidden states with attention mask.
    Vectors are L2 normalized for cosine similarity.
    """

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(self):
        print("[ClinicalBERT] Loading model (first run will download ~400MB)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[ClinicalBERT] Ready on {self.device}.")

    def _mean_pool(self, token_embeddings: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling: average token embeddings, weighted by attention mask.
        This is the correct way to get sentence vectors from BERT.
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts in batches. Returns L2-normalized numpy array.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            if i % 200 == 0:
                print(f"[ClinicalBERT] Encoding {i}/{len(texts)}...")

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)

            embeddings = self._mean_pool(
                output.last_hidden_state,
                encoded["attention_mask"]
            )

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


# ── Chapter Index Builder ─────────────────────────────────────────────────────

# 18 ICD-9 chapter descriptions used as the retrieval index.
# These are the "documents" in our vector search.
CHAPTER_DESCRIPTIONS = {
    "Infectious and Parasitic Diseases":
        "infectious diseases bacterial viral parasitic infection fever sepsis tuberculosis HIV",
    "Neoplasms":
        "cancer tumor malignant benign neoplasm carcinoma lymphoma leukemia",
    "Endocrine, Nutritional, Metabolic, Immunity":
        "diabetes thyroid obesity metabolic disorder hormone endocrine nutrition",
    "Diseases of the Blood":
        "anemia bleeding coagulation blood disorder platelet hemoglobin",
    "Mental Disorders":
        "depression anxiety psychosis schizophrenia mental disorder psychiatric",
    "Nervous System and Sense Organs":
        "stroke seizure epilepsy neuropathy brain spinal cord eye ear vision hearing",
    "Circulatory System":
        "heart failure myocardial infarction hypertension arrhythmia coronary cardiac",
    "Respiratory System":
        "pneumonia asthma COPD bronchitis respiratory failure lung breathing",
    "Digestive System":
        "gastritis ulcer liver cirrhosis bowel gastrointestinal intestinal",
    "Genitourinary System":
        "kidney renal urinary tract bladder prostate reproductive",
    "Pregnancy, Childbirth, Puerperium":
        "pregnancy labor delivery obstetric maternal prenatal postpartum",
    "Skin and Subcutaneous Tissue":
        "dermatitis cellulitis wound skin rash ulcer abscess",
    "Musculoskeletal and Connective Tissue":
        "arthritis fracture bone joint muscle orthopedic spine",
    "Congenital Anomalies":
        "congenital anomaly birth defect genetic malformation",
    "Perinatal Conditions":
        "newborn neonatal premature birth infant perinatal",
    "Symptoms, Signs, Ill-defined Conditions":
        "pain fever nausea fatigue symptom unspecified ill-defined",
    "Injury and Poisoning":
        "fracture trauma injury accident poisoning overdose burn",
    "Supplementary V Codes":
        "history screening checkup vaccination follow-up status",
}


def build_chapter_index(embedder) -> tuple[list[str], np.ndarray]:
    """
    Encode all chapter descriptions into vectors.
    Returns (chapter_names, chapter_embeddings).
    """
    chapter_names = list(CHAPTER_DESCRIPTIONS.keys())
    chapter_texts = list(CHAPTER_DESCRIPTIONS.values())
    print(f"[Index] Building chapter index ({len(chapter_names)} chapters)...")
    chapter_embeddings = embedder.encode(chapter_texts, batch_size=18)
    print("[Index] Done.")
    return chapter_names, chapter_embeddings


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_texts = [
        "acute myocardial infarction no ST elevation",
        "type 2 diabetes mellitus without complications",
        "community acquired pneumonia bilateral"
    ]

    print("=== MiniLM Test ===")
    minilm = MiniLMEmbedder()
    vecs = minilm.encode(sample_texts)
    print(f"Shape: {vecs.shape}")
    print(f"Sample norm: {np.linalg.norm(vecs[0]):.4f} (should be ~1.0)\n")

    print("=== Chapter Index Test ===")
    names, embs = build_chapter_index(minilm)
    print(f"Chapter index shape: {embs.shape}")
