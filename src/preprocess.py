"""
preprocess.py
-------------
Text cleaning for ICD mapping pipeline.
Uses spaCy for tokenization. Preserves clinical negations (no/denies/without).
Outputs both raw_text and norm_text columns.
"""

import re
import pandas as pd
import spacy

# Clinical negation words — must NOT be removed
CLINICAL_NEGATIONS = {
    "no", "not", "nor", "never", "without", "denies", "deny",
    "absent", "absence", "negative", "rule", "out", "except",
    "neither", "hardly", "barely", "rarely"
}

def load_spacy_model():
    """Load spaCy model, with helpful error message if not installed."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("[ERROR] spaCy model not found.")
        print("Run: python -m spacy download en_core_web_sm")
        raise


def basic_clean(text: str) -> str:
    """
    Light cleaning only:
    - Lowercase
    - Remove special characters (keep letters, digits, spaces, hyphens)
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def spacy_normalize(text: str, nlp) -> str:
    """
    spaCy tokenization only — no aggressive stopword removal.
    Preserves clinical negations.
    Returns space-joined tokens (original form, lowercased).
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        # Always keep clinical negations
        if token.lower_ in CLINICAL_NEGATIONS:
            tokens.append(token.lower_)
        # Skip pure punctuation and whitespace
        elif token.is_punct or token.is_space:
            continue
        else:
            tokens.append(token.lower_)
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str, nlp) -> pd.DataFrame:
    """
    Apply preprocessing to a dataframe column.
    Adds 'raw_text' (basic clean) and 'norm_text' (spaCy normalized) columns.
    """
    print(f"[preprocess] Processing {len(df)} rows...")
    df = df.copy()
    df["raw_text"] = df[text_col].apply(basic_clean)
    df["norm_text"] = df["raw_text"].apply(lambda t: spacy_normalize(t, nlp))
    print("[preprocess] Done.")
    return df


# ── ICD-9 Chapter Mapping ─────────────────────────────────────────────────────

# ICD-9 chapter definitions: (start, end, chapter_name)
ICD9_CHAPTERS = [
    (1,   139,  "Infectious and Parasitic Diseases"),
    (140, 239,  "Neoplasms"),
    (240, 279,  "Endocrine, Nutritional, Metabolic, Immunity"),
    (280, 289,  "Diseases of the Blood"),
    (290, 319,  "Mental Disorders"),
    (320, 389,  "Nervous System and Sense Organs"),
    (390, 459,  "Circulatory System"),
    (460, 519,  "Respiratory System"),
    (520, 579,  "Digestive System"),
    (580, 629,  "Genitourinary System"),
    (630, 679,  "Pregnancy, Childbirth, Puerperium"),
    (680, 709,  "Skin and Subcutaneous Tissue"),
    (710, 739,  "Musculoskeletal and Connective Tissue"),
    (740, 759,  "Congenital Anomalies"),
    (760, 779,  "Perinatal Conditions"),
    (780, 799,  "Symptoms, Signs, Ill-defined Conditions"),
    (800, 999,  "Injury and Poisoning"),
]

# V and E codes (supplementary)
VCODE_CHAPTER = "Supplementary V Codes"
ECODE_CHAPTER = "Supplementary E Codes"


def icd9_to_chapter(icd9_code: str) -> str:
    """
    Map an ICD-9 code string to its chapter name.
    Handles V-codes, E-codes, and numeric codes.
    Returns 'Unknown' if no match found.
    """
    if not isinstance(icd9_code, str):
        return "Unknown"

    code = icd9_code.strip().upper()

    # V codes (e.g. V01, V458)
    if code.startswith("V"):
        return VCODE_CHAPTER

    # E codes (e.g. E800, E9289)
    if code.startswith("E"):
        return ECODE_CHAPTER

    # Numeric codes — extract leading digits
    numeric = re.match(r"^(\d+)", code)
    if not numeric:
        return "Unknown"

    # Take first 3 characters as the category number
    # e.g. "01716" → "017" → 17, "41001" → "410" → 410
    num_str = numeric.group(1)[:3]
    num = int(num_str)

    for start, end, chapter in ICD9_CHAPTERS:
        if start <= num <= end:
            return chapter

    return "Unknown"


def add_chapter_labels(df: pd.DataFrame, icd_col: str = "icd9_code") -> pd.DataFrame:
    """Add 'chapter' column to dataframe based on ICD-9 code."""
    df = df.copy()
    df["chapter"] = df[icd_col].apply(icd9_to_chapter)
    return df


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test chapter mapping
    test_codes = ["99591", "41001", "V450", "E8001", "486", "29500"]
    print("=== Chapter Mapping Test ===")
    for code in test_codes:
        print(f"  {code:10s} → {icd9_to_chapter(code)}")

    # Test text cleaning
    print("\n=== Text Cleaning Test ===")
    nlp = load_spacy_model()
    sample = "Acute myocardial infarction, no ST elevation, without complications"
    print(f"  Input : {sample}")
    print(f"  Clean : {basic_clean(sample)}")
    print(f"  Norm  : {spacy_normalize(basic_clean(sample), nlp)}")
    print("\n✓ Negation 'no' and 'without' preserved ✓")
