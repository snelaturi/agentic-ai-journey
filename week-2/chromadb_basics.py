import chromadb
from chromadb.config import Settings

# Initializing ChromaDB client
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

# 1. CREATE COLLECTION
collection = client.create_collection(
    name="test_collection",
    metadata={"description":"This is a test collection"}
)

print(f"✓ Created collection: {collection.name}")


# 2. ADD DOCUMENTS
print("\n2. Adding documents...")
documents = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Machine learning is a subset of AI",
    "Dogs are loyal pets",
    "Cats are independent animals"
]

ids = [f"doc_{i}" for i in range(len(documents))]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=[{"category": "tech"} if i < 3 else {"category": "animals"}
               for i in range(len(documents))]
)
print(f"✓ Added {len(documents)} documents")
count = collection.count()
print(f" Collection has {count} documents")

# 4. QUERY (Semantic Search)
print("\n4. Querying collection...")
query_text = "programming languages"
results = collection.query(
    query_texts=[query_text],
    n_results=3
)

print(f"\nQuery: '{query_text}'")
print("\nTop 3 results:")
for i, (doc, distance, metadata) in enumerate(zip(
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0]
), 1):
    print(f"\n{i}. {doc}")
    print(f"   Distance: {distance:.4f}")
    print(f"   Metadata: {metadata}")

# 5. GET SPECIFIC DOCUMENTS
print("\n5. Getting specific documents...")
specific_docs = collection.get(
    ids=["doc_0", "doc_3"]
)
print(f"✓ Retrieved {len(specific_docs['ids'])} documents:")
for doc_id, doc in zip(specific_docs['ids'], specific_docs['documents']):
    print(f"  {doc_id}: {doc}")

# 6. UPDATE DOCUMENT
print("\n6. Updating document...")
collection.update(
    ids=["doc_0"],
    documents=["Python is an amazing programming language"],
    metadatas=[{"category": "tech", "updated": True}]
)
print("✓ Updated doc_0")

# Verify update
updated = collection.get(ids=["doc_0"])
print(f"  New content: {updated['documents'][0]}")

# 7. DELETE DOCUMENT
print("\n7. Deleting document...")
collection.delete(ids=["doc_4"])
print("✓ Deleted doc_4")
print(f"  New count: {collection.count()}")

# 8. FILTER BY METADATA
print("\n8. Filtering by metadata...")
tech_docs = collection.get(
    where={"category": "tech"}
)
print(f"✓ Found {len(tech_docs['ids'])} tech documents:")
for doc_id, doc in zip(tech_docs['ids'], tech_docs['documents']):
    print(f"  {doc_id}: {doc}")

# 9. DELETE COLLECTION
print("\n9. Cleaning up...")
client.delete_collection(name="test_collection")
print("✓ Deleted collection")


