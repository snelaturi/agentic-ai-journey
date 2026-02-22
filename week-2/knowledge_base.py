from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any

load_dotenv()


class KnowledgeBase:
    """Vector-based knowledge base for semantic search"""

    def __init__(self, collection_name: str = "knowledge_base"):
        # Setup OpenAI embedding function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./knowledge_base_db"
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.openai_ef
            )
            print(f"✓ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.openai_ef,
                metadata={"description": "Technical knowledge base"}
            )
            print(f"✓ Created new collection: {collection_name}")

    def add_documents(
            self,
            documents: List[str],
            metadatas: List[Dict[str, Any]] = None,
            ids: List[str] = None
    ):
        """Add documents to knowledge base"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{}] * len(documents)

        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

        print(f"✓ Added {len(documents)} documents")

    def search(
            self,
            query: str,
            n_results: int = 5,
            filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search knowledge base"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            })

        return formatted_results

    def get_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Get document by ID"""
        result = self.collection.get(ids=[doc_id])

        if result['ids']:
            return {
                'id': result['ids'][0],
                'document': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
        return None

    def update_document(
            self,
            doc_id: str,
            document: str = None,
            metadata: Dict[str, Any] = None
    ):
        """Update existing document"""
        update_args = {'ids': [doc_id]}

        if document:
            update_args['documents'] = [document]
        if metadata:
            update_args['metadatas'] = [metadata]

        self.collection.update(**update_args)
        print(f"✓ Updated {doc_id}")

    def delete_document(self, doc_id: str):
        """Delete document by ID"""
        self.collection.delete(ids=[doc_id])
        print(f"✓ Deleted {doc_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name
        }


def demo():
    """Demo the knowledge base"""
    print("=" * 70)
    print(" " * 20 + "KNOWLEDGE BASE DEMO")
    print("=" * 70)

    # Initialize
    kb = KnowledgeBase("tech_docs")

    # Add Spring Boot documentation
    spring_boot_docs = [
        "Spring Boot is an opinionated framework for building production-ready applications quickly.",
        "Spring Boot auto-configuration automatically configures your Spring application based on dependencies.",
        "Spring Boot Actuator provides production-ready features like health checks and metrics.",
        "Spring Data JPA simplifies database access with repository abstractions.",
        "Spring Security provides authentication and authorization features.",
        "@RestController annotation marks a class as a REST API controller.",
        "Spring Boot uses embedded Tomcat by default for running web applications.",
        "@Autowired annotation enables dependency injection in Spring.",
        "application.properties or application.yml configure Spring Boot applications.",
        "Spring Boot DevTools provides automatic restart during development."
    ]

    metadatas = [
        {"framework": "spring-boot", "topic": "overview"},
        {"framework": "spring-boot", "topic": "autoconfiguration"},
        {"framework": "spring-boot", "topic": "actuator"},
        {"framework": "spring-boot", "topic": "data-jpa"},
        {"framework": "spring-boot", "topic": "security"},
        {"framework": "spring-boot", "topic": "annotations"},
        {"framework": "spring-boot", "topic": "web"},
        {"framework": "spring-boot", "topic": "annotations"},
        {"framework": "spring-boot", "topic": "configuration"},
        {"framework": "spring-boot", "topic": "devtools"}
    ]

    # Add documents
    print("\nAdding Spring Boot documentation...")
    kb.add_documents(spring_boot_docs, metadatas)

    # Get stats
    stats = kb.get_stats()
    print(f"\nKnowledge Base Stats:")
    print(f"  Total documents: {stats['total_documents']}")

    # Test searches
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH EXAMPLES")
    print("=" * 70)

    queries = [
        "How do I set up a REST API?",
        "What's the best way to connect to a database?",
        "How can I monitor my application in production?",
        "How do I configure my application?"
    ]

    for query in queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 70)

        results = kb.search(query, n_results=2)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['document']}")
            print(f"   Topic: {result['metadata']['topic']}")
            print(f"   Relevance: {1 - result['distance']:.4f}")


if __name__ == "__main__":
    demo()