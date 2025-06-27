import re
import string
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    """ "
    Cleans the input text by converting it to lowercase, removing digits, punctuation
    Args:
        text(str) : The input text to be cleaned.
    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text


def generate_companies_embeddings(
    df: pd.DataFrame, embeddings_model
) -> [pd.DataFrame, np.ndarray]:
    """
    Generates embedding based on the "Name" and "Description" columns of the DataFrame.
    Args:
        df (pd.DataFrame) : The input DataFrame containing company data.
        embeddings_model: The model used to generate embeddings
    Returns:
        pd.DataFrame: DataFrame with cleaned text and embeddings.
        np.ndarray: Array of embeddings for the companies.
    """

    df["concatenated_name_description"] = (
        df["Description"].fillna(" ").astype(str)
    )

    # Clean the contancated name description
    df["concatenated_cleaned_text"] = df["concatenated_name_description"].apply(
        clean_text
    )

    # Generate embeddings
    embeddings = embeddings_model.encode(
        df["concatenated_cleaned_text"],
        show_progress_bar=False,
        device="cude" if embeddings_model.device.type == "cuda" else "cpu",
    )
    df["embeddings"] = (
        embeddings.tolist()
    )  # Convert to list for DataFrame compatibility

    return df


def mmr(doc_embeddings, query_embedding, top_k=10, lambda_param=0.5):
    """
    Computes the Maximal Marginal Relevance (MMR) for document embeddings based on a query embedding.
    Args:
        doc_embeddings (np.ndarray): Array of document embeddings.
        query_embedding (np.ndarray): Embedding of the query.
        top_k (int): Number of top documents to select.
        lambda_param (float): Trade-off parameter between relevance and diversity.
    Returns:
        list: Indices of the selected top_k documents.
    """
    # Initialize
    selected = []
    remaining = list(range(len(doc_embeddings)))

    # Compute initial similarities
    similarity_to_query = cosine_similarity([query_embedding], doc_embeddings).flatten()

    # Select the most relevant doc first
    selected.append(np.argmax(similarity_to_query))
    remaining.remove(selected[0])

    # Iteratively select the rest
    for _ in range(top_k - 1):
        mmr_score = []
        for idx in remaining:
            relevance = similarity_to_query[idx]
            diversity = max(
                cosine_similarity(
                    [doc_embeddings[idx]], doc_embeddings[selected]
                ).flatten()
            )
            score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_score.append(score)

        selected_idx = remaining[np.argmax(mmr_score)]
        selected.append(selected_idx)
        remaining.remove(selected_idx)

    return selected
