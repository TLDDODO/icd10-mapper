"""
retriever.py
------------
Vector similarity retrieval over ICD-9 chapter index.
Input: text query (string)
Output: top-k chapters with similarity scores
"""

import numpy as np


def cosine_similarity_matrix(query_vecs: np.ndarray,
                              index_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and index vectors.
    Both inputs must be L2-normalized (norm=1).
    When normalized, cosine similarity = dot product.
    Returns matrix of shape (n_queries, n_index).
    """
    return query_vecs @ index_vecs.T


def retrieve_top_k(
    query_text: str,
    embedder,
    chapter_names: list[str],
    chapter_embeddings: np.ndarray,
    k: int = 3
) -> list[dict]:
    """
    Given a text query, return top-k matching ICD chapters.

    Args:
        query_text: input clinical text
        embedder: MiniLMEmbedder or ClinicalBERTEmbedder
        chapter_names: list of chapter name strings
        chapter_embeddings: (n_chapters, dim) normalized vectors
        k: number of results to return

    Returns:
        List of dicts: [{"rank": 1, "chapter": "...", "score": 0.87}, ...]
    """
    # Encode query
    query_vec = embedder.encode([query_text])   # shape (1, dim)

    # Cosine similarity with all chapters
    sims = cosine_similarity_matrix(query_vec, chapter_embeddings)[0]  # shape (n_chapters,)

    # Get top-k indices
    top_indices = np.argsort(sims)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank": rank,
            "chapter": chapter_names[idx],
            "score": float(sims[idx])
        })

    return results


def batch_retrieve(
    texts: list[str],
    embedder,
    chapter_names: list[str],
    chapter_embeddings: np.ndarray,
    k: int = 3,
    batch_size: int = 64
) -> list[list[dict]]:
    """
    Batch version of retrieve_top_k for evaluation.
    Returns list of top-k results for each input text.
    """
    print(f"[Retriever] Encoding {len(texts)} queries...")
    query_embeddings = embedder.encode(texts, batch_size=batch_size)

    print("[Retriever] Computing similarities...")
    sim_matrix = cosine_similarity_matrix(query_embeddings, chapter_embeddings)
    # shape: (n_texts, n_chapters)

    all_results = []
    for i, sims in enumerate(sim_matrix):
        top_indices = np.argsort(sims)[::-1][:k]
        results = [
            {
                "rank": rank + 1,
                "chapter": chapter_names[idx],
                "score": float(sims[idx])
            }
            for rank, idx in enumerate(top_indices)
        ]
        all_results.append(results)

    print("[Retriever] Done.")
    return all_results


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from embedder import MiniLMEmbedder, build_chapter_index

    embedder = MiniLMEmbedder()
    chapter_names, chapter_embeddings = build_chapter_index(embedder)

    test_queries = [
        "acute myocardial infarction with ST elevation",
        "type 2 diabetes mellitus poorly controlled",
        "community acquired pneumonia with fever",
        "fractured femur after fall",
    ]

    print("\n=== Retrieval Test ===")
    for query in test_queries:
        results = retrieve_top_k(query, embedder, chapter_names, chapter_embeddings, k=3)
        print(f"\nQuery: {query}")
        for r in results:
            print(f"  #{r['rank']} {r['chapter']:<45} score={r['score']:.3f}")
