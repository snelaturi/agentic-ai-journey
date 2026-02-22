from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os

load_dotenv()

# Initialize ChromaDB with OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

client = chromadb.Client(Settings(
    persist_directory="./chroma_openai_db",
    anonymized_telemetry=False
))

print("="*70)
print(" " * 15 + "ChromaDB with OpenAI Embeddings")
print("="*70)

# Create collection with OpenAI embeddings
collection = client.create_collection(
    name="knowledge_base",
    embedding_function=openai_ef,
    metadata={"description": "Knowledge Base with OpenAI Embeddings"}
)

# Add technical documentation
documents = [
    # Programming
    "Python is a high-level, interpreted programming language known for its simplicity and readability.",
    "JavaScript is primarily used for web development and runs in browsers.",
    "Java is an object-oriented language widely used for enterprise applications.",

    # AI/ML
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Natural language processing (NLP) helps computers understand human language.",

    # Databases
    "SQL is a language for managing relational databases.",
    "MongoDB is a NoSQL database that stores data in JSON-like documents.",
    "Vector databases store embeddings for semantic search.",

    # Cloud
    "AWS provides cloud computing services including EC2 and S3.",
    "Docker containers package applications with their dependencies.",
    "Kubernetes orchestrates containerized applications at scale."
]

ids = [f"doc_{i}" for i in range(len(documents))]

metadatas = [
    {"category": "programming", "topic": "python"},
    {"category": "programming", "topic": "javascript"},
    {"category": "programming", "topic": "java"},
    {"category": "ai", "topic": "machine-learning"},
    {"category": "ai", "topic": "deep-learning"},
    {"category": "ai", "topic": "nlp"},
    {"category": "database", "topic": "sql"},
    {"category": "database", "topic": "nosql"},
    {"category": "database", "topic": "vector-db"},
    {"category": "cloud", "topic": "aws"},
    {"category": "cloud", "topic": "docker"},
    {"category": "cloud", "topic": "kubernetes"}
]

print("\nAdding documents to collection...")
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)
print(f"✓ Added {len(documents)} documents with OpenAI embeddings")


print("SEMANTIC SEARCH TESTS")

#query = "How do I work with data?"
query = "What languages are good for beginners?"
print(f"\nQuery: '{query}'")
print("Filter: category = 'ai'")
print("-"*70)

results = collection.query(
    query_texts=[query],
    n_results=3
)

for i, (doc, metadata) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0]
), 1):
    print(f"\n{i}. {doc}")
    print(f"   Topic: {metadata['topic']}")

# Clean up
print("\n" + "="*70)
client.delete_collection(name="knowledge_base")
print("✓ Cleaned up collection")










