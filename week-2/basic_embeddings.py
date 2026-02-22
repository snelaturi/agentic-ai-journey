from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()
client = OpenAI()


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """Get embedding vector for text"""
    text = text.replace("\n", " ")  # Clean text
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors"""
    # Convert to numpy arrays
    a = np.array(vec1)
    b = np.array(vec2)

    # Cosine similarity formula
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_product / (norm_a * norm_b)


def euclidean_distance(vec1: list, vec2: list) -> float:
    """Calculate Euclidean distance between two vectors"""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.linalg.norm(a - b)


def dot_product_similarity(vec1: list, vec2: list) -> float:
    """Calculate dot product similarity"""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b)


def main():
    print("=" * 70)
    print(" " * 20 + "EMBEDDINGS DEMO")
    print("=" * 70)

    # Test sentences
    sentences = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "I love eating pizza",
        "Python is a programming language",
        "Dogs are loyal pets"
    ]

    print("\nGenerating embeddings...\n")

    # Get embeddings for all sentences
    embeddings = {}
    for sentence in sentences:
        emb = get_embedding(sentence)
        embeddings[sentence] = emb
        print(f"✓ Embedded: '{sentence}'")
        print(f"  Vector length: {len(emb)} dimensions")
        print(f"  First 5 values: {emb[:5]}\n")

    # Compare similarities
    print("=" * 70)
    print("SIMILARITY COMPARISONS")
    print("=" * 70)

    base_sentence = sentences[0]  # "The cat sat on the mat"
    base_embedding = embeddings[base_sentence]

    print(f"\nComparing to: '{base_sentence}'\n")

    similarities = []
    for sentence in sentences:
        if sentence == base_sentence:
            continue

        emb = embeddings[sentence]

        cos_sim = cosine_similarity(base_embedding, emb)
        euc_dist = euclidean_distance(base_embedding, emb)
        dot_prod = dot_product_similarity(base_embedding, emb)

        similarities.append((sentence, cos_sim, euc_dist, dot_prod))

        print(f"Sentence: '{sentence}'")
        print(f"  Cosine Similarity:    {cos_sim:.4f}")
        print(f"  Euclidean Distance:   {euc_dist:.4f}")
        print(f"  Dot Product:          {dot_prod:.4f}")
        print()

    # Sort by cosine similarity (most similar first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    print("=" * 70)
    print("RANKING BY SIMILARITY (Cosine)")
    print("=" * 70)

    for i, (sentence, cos_sim, _, _) in enumerate(similarities, 1):
        print(f"{i}. {sentence}")
        print(f"   Similarity: {cos_sim:.4f}")
        print()


if __name__ == "__main__":
    main()