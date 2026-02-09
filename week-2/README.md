# Week 2: RAG & Vector Databases

**Duration:** 12 hours  
**Goal:** Master Retrieval-Augmented Generation

## 📋 Overview

This week focuses on building Retrieval-Augmented Generation (RAG) systems that combine the power of LLMs with external knowledge bases. You'll learn embeddings, vector databases, and advanced retrieval techniques to create intelligent Q&A systems.

## 🎯 Learning Objectives

- Understand embeddings and vector representations
- Master vector similarity search
- Set up and use ChromaDB
- Build end-to-end RAG pipelines
- Implement document chunking strategies
- Learn hybrid search techniques
- Apply re-ranking for better retrieval
- Deploy production-ready RAG systems

## 📅 Daily Breakdown

### Day 8-9: Embeddings & Vector Search (4 hrs)

**Topics:**
- What are embeddings?
- OpenAI embedding models
- Vector similarity (cosine, euclidean, dot product)
- Introduction to vector databases
- ChromaDB setup and operations
- CRUD operations with vectors
- Collection management
- Metadata filtering

**Hands-on:**
- Generate embeddings with OpenAI
- Calculate similarity between texts
- Set up ChromaDB locally
- Store and query document embeddings
- Experiment with different distance metrics
- Build simple semantic search

**Key Concepts:**
- Text → Vector transformation
- Dimensionality of embeddings
- Similarity vs distance metrics
- Vector database vs traditional database

**Exercises:**
- `exercises/embeddings_basics.py` - Generate and compare embeddings
- `exercises/chromadb_intro.py` - CRUD operations with ChromaDB
- `exercises/semantic_search.py` - Build basic semantic search

---

### Day 10-12: Technical Documentation Q&A System (4 hrs)

**Topics:**
- RAG architecture overview
- Document loading and preprocessing
- Text chunking strategies
- Chunk size optimization
- Retrieval pipeline design
- Prompt engineering for RAG
- Context window management
- Source attribution

**Hands-on:**
- Load technical documentation (Python docs, API docs, etc.)
- Implement chunking strategies
- Create embedding pipeline
- Build retrieval system
- Integrate with LLM for answer generation
- Add source citations
- Handle edge cases

**RAG Pipeline:**
1. **Document Ingestion** → Load docs
2. **Chunking** → Split into optimal sizes
3. **Embedding** → Convert to vectors
4. **Storage** → Save in ChromaDB
5. **Query** → User question
6. **Retrieval** → Find relevant chunks
7. **Generation** → LLM creates answer
8. **Response** → Return with sources

**Deliverable:** Technical Documentation Q&A System

**Project Structure:**
```
projects/tech-docs-qa/
├── src/
│   ├── ingest.py          # Document loading & chunking
│   ├── embeddings.py      # Embedding generation
│   ├── retriever.py       # Vector search
│   ├── generator.py       # LLM answer generation
│   └── main.py            # Complete pipeline
├── data/
│   └── docs/              # Documentation files
├── tests/
├── README.md
└── requirements.txt
```

---

### Day 13-14: Advanced RAG Techniques (4 hrs)

**Topics:**
- Limitations of basic RAG
- Hybrid search (vector + keyword)
- BM25 algorithm
- Re-ranking strategies
- Cross-encoder models
- Query expansion
- Multi-query retrieval
- Parent-child chunking
- Metadata filtering
- Performance optimization

**Hands-on:**
- Implement BM25 keyword search
- Combine vector + keyword search
- Add re-ranking with cross-encoders
- Implement query expansion
- Test with complex queries
- Benchmark retrieval quality
- Optimize for speed and accuracy

**Advanced Techniques:**
- **Hybrid Search:** Combine semantic + keyword search
- **Re-ranking:** Use cross-encoder to re-score results
- **Query Expansion:** Generate multiple query variations
- **Hypothetical Document Embeddings (HyDE):** Generate hypothetical answers first
- **Maximal Marginal Relevance (MMR):** Reduce redundancy in results

**Deliverable:** Blog post comparing basic vs advanced RAG

---

## 🚀 Deliverables

### 1. Technical Documentation Q&A System
**Location:** `projects/tech-docs-qa/`

**Features:**
- ✅ Load and process technical documentation
- ✅ Intelligent chunking with overlap
- ✅ Semantic search with ChromaDB
- ✅ LLM-powered answer generation
- ✅ Source attribution and citations
- ✅ Web UI or CLI interface
- ✅ Support for multiple document types (PDF, Markdown, HTML)

**Tech Stack:**
- ChromaDB for vector storage
- OpenAI embeddings (text-embedding-3-small)
- OpenAI GPT-4 for generation
- LangChain for orchestration
- Optional: Streamlit/Gradio for UI

**Example Queries:**
- "How do I use async/await in Python?"
- "What's the difference between == and is?"
- "Explain Python decorators with examples"

---

### 2. Advanced RAG System with Hybrid Search
**Location:** `projects/tech-docs-qa/` (enhanced version)

**Advanced Features:**
- ✅ Hybrid search (vector + BM25)
- ✅ Re-ranking with cross-encoders
- ✅ Query expansion
- ✅ Metadata filtering
- ✅ Performance benchmarking

**Technologies:**
- Rank-BM25 for keyword search
- sentence-transformers for re-ranking
- Custom query expansion logic

---

### 3. Blog Post: "Basic vs Advanced RAG"
**Location:** `blog/rag-comparison.md`

**Contents:**
- Introduction to RAG
- Basic RAG architecture walkthrough
- Limitations of basic RAG
- Advanced techniques explained
- Performance comparison (metrics)
- Code examples
- When to use each approach
- Conclusion and best practices

**Metrics to Compare:**
- Retrieval accuracy
- Response quality
- Latency
- Cost per query

---

## 📚 Resources

### Official Documentation
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Sentence Transformers](https://www.sbert.net/)

### Papers & Articles
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.00083)
- "Building RAG Systems: Best Practices"
- "Chunking Strategies for RAG"
- "Evaluating RAG Systems"

### Tutorials
- [Building Production-Ready RAG Applications](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/)
- [Advanced RAG Techniques](https://www.pinecone.io/learn/advanced-rag/)
- [ChromaDB Tutorial Series](https://docs.trychroma.com/getting-started)

### Videos
- "RAG from Scratch" - LangChain
- "Vector Databases Explained"
- "Embeddings Deep Dive"
- "Building Production RAG Systems"

### Books/Courses
- "Generative AI with LangChain" - Ben Auffarth
- DeepLearning.AI - Building Applications with Vector Databases
- "Vector Databases: A Practical Guide"

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
# Activate virtual environment
source ../../venv/bin/activate

# Install required packages
pip install openai python-dotenv
pip install langchain langchain-openai langchain-community
pip install chromadb
pip install pypdf python-docx markdown
pip install rank-bm25
pip install sentence-transformers
pip install streamlit  # Optional: for UI
pip install gradio     # Optional: alternative UI

# Update requirements
pip freeze > ../../requirements.txt
```

### 2. Set Up ChromaDB
```bash
# ChromaDB will create a local database automatically
# Default location: ./chroma_db/
```

### 3. Update .env File
Add to your existing `.env`:
```bash
# Existing keys
OPENAI_API_KEY=sk-your-key-here

# ChromaDB (if using cloud version)
CHROMA_API_KEY=your-chroma-key  # Optional

# Embedding model
EMBEDDING_MODEL=text-embedding-3-small
```

### 4. Prepare Documentation
```bash
# Create data directory
mkdir -p projects/tech-docs-qa/data/docs

# Add your documentation files
# Examples:
# - Python documentation (markdown)
# - API documentation (PDF)
# - Technical guides (HTML)
```

---

## 📝 Project Structure
```
week-2/
├── README.md
├── notes.md
├── resources.md
│
├── exercises/
│   ├── embeddings_basics.py
│   ├── chromadb_intro.py
│   ├── semantic_search.py
│   ├── chunking_strategies.py
│   ├── hybrid_search.py
│   └── reranking_demo.py
│
└── blog/
    └── rag-comparison.md

projects/tech-docs-qa/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   └── main.py
├── data/
│   └── docs/
├── tests/
│   └── test_retrieval.py
├── ui/
│   └── streamlit_app.py
├── chroma_db/          # Created automatically
├── README.md
├── requirements.txt
└── .env
```

---

## 💡 Code Templates

### Embeddings Basics
```python
# exercises/embeddings_basics.py
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """Generate embedding for text"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    """Calculate cosine similarity"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example usage
text1 = "The cat sat on the mat"
text2 = "A feline rested on the rug"
text3 = "Python is a programming language"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

print(f"Similarity (text1, text2): {cosine_similarity(emb1, emb2):.4f}")
print(f"Similarity (text1, text3): {cosine_similarity(emb1, emb3):.4f}")
```

### ChromaDB Setup
```python
# exercises/chromadb_intro.py
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client
client = chromadb.Client(Settings(
    persist_directory="./chroma_db"
))

# Create or get collection
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"description": "My first ChromaDB collection"}
)

# Add documents
collection.add(
    documents=[
        "Python is a high-level programming language",
        "JavaScript is used for web development",
        "Machine learning models need training data"
    ],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[
        {"source": "python_docs", "category": "programming"},
        {"source": "js_docs", "category": "web"},
        {"source": "ml_guide", "category": "ai"}
    ]
)

# Query
results = collection.query(
    query_texts=["What is Python?"],
    n_results=2
)

print(results)
```

### Basic RAG Pipeline
```python
# projects/tech-docs-qa/src/rag_pipeline.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

class RAGPipeline:
    def __init__(self, docs_path, persist_dir="./chroma_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.vectorstore = None
        self.qa_chain = None
        
    def ingest_documents(self, docs_path):
        """Load and process documents"""
        loader = DirectoryLoader(docs_path, glob="**/*.md")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
    def setup_qa_chain(self):
        """Create QA chain"""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        
    def query(self, question):
        """Query the RAG system"""
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }
```

---

## ✅ Completion Checklist

### Day 8-9: Embeddings & Vector Search
- [ ] Understand embedding concepts
- [ ] Generate embeddings with OpenAI API
- [ ] Calculate similarity between vectors
- [ ] Install and set up ChromaDB
- [ ] Perform CRUD operations
- [ ] Build semantic search demo
- [ ] Experiment with different metrics
- [ ] Complete exercises
- [ ] Commit code to GitHub

### Day 10-12: RAG System
- [ ] Design RAG architecture
- [ ] Implement document loader
- [ ] Create chunking strategy
- [ ] Build embedding pipeline
- [ ] Set up vector store
- [ ] Implement retriever
- [ ] Integrate LLM for generation
- [ ] Add source citations
- [ ] Test with sample queries
- [ ] Create UI (optional)
- [ ] Write project README
- [ ] Commit project to GitHub

### Day 13-14: Advanced RAG
- [ ] Understand hybrid search
- [ ] Implement BM25 search
- [ ] Combine vector + keyword search
- [ ] Add re-ranking with cross-encoders
- [ ] Implement query expansion
- [ ] Test advanced techniques
- [ ] Benchmark performance
- [ ] Write comparison blog post
- [ ] Update project with advanced features
- [ ] Commit everything to GitHub

---

## 🎓 Key Concepts Mastered

- [ ] Text embeddings and vector representations
- [ ] Vector similarity metrics
- [ ] ChromaDB operations
- [ ] Document chunking strategies
- [ ] RAG pipeline architecture
- [ ] Retrieval techniques
- [ ] Prompt engineering for RAG
- [ ] Hybrid search (semantic + keyword)
- [ ] Re-ranking methods
- [ ] Query optimization
- [ ] Performance evaluation

---

## 📊 Time Tracking

| Day | Hours Spent | Topics Covered |
|-----|-------------|----------------|
| 8   |             |                |
| 9   |             |                |
| 10  |             |                |
| 11  |             |                |
| 12  |             |                |
| 13  |             |                |
| 14  |             |                |
| **Total** | **/ 12 hrs** |          |

---

## 🧪 Testing & Evaluation

### Retrieval Quality Metrics
```python
# Test your RAG system
test_queries = [
    "How do I handle exceptions in Python?",
    "What is the difference between lists and tuples?",
    "Explain Python generators"
]

for query in test_queries:
    result = rag.query(query)
    print(f"Q: {query}")
    print(f"A: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
    print("---")
```

### Performance Benchmarks
- **Retrieval Latency:** < 500ms
- **End-to-end Query Time:** < 3s
- **Retrieval Accuracy:** > 80%
- **Answer Relevance:** Measured by human eval

---

## 🚧 Challenges & Solutions

### Challenge 1: Optimal Chunk Size
**Problem:**
Chunks too small → Loss of context  
Chunks too large → Irrelevant information

**Solution:**
- Test different sizes (500, 1000, 1500 tokens)
- Use RecursiveCharacterTextSplitter
- Add overlap (200 tokens)
- Evaluate retrieval quality

---

### Challenge 2: Retrieval Accuracy
**Problem:**
Basic vector search misses relevant documents

**Solution:**
- Implement hybrid search
- Add re-ranking
- Use query expansion
- Filter by metadata

---

### Challenge 3: Answer Quality
**Problem:**
LLM generates incorrect answers despite good retrieval

**Solution:**
- Improve prompt engineering
- Add explicit instructions
- Use chain-of-thought prompting
- Include examples in prompt

---

## 📈 Performance Comparison

| Metric | Basic RAG | Advanced RAG | Improvement |
|--------|-----------|--------------|-------------|
| Retrieval Accuracy | % | % | % |
| Query Latency | ms | ms | ms |
| Answer Relevance | /10 | /10 | |
| Cost per Query | $ | $ | $ |

---

## 🔗 Project Links

- **GitHub Repository:** [Your repo link]
- **Tech Docs Q&A Demo:** [If deployed]
- **Blog Post:** [Link to blog]
- **Documentation:** [Project docs]

---

## 📞 Next Steps

After completing Week 2:
1. ✅ Push RAG system to GitHub
2. ✅ Publish blog post
3. ✅ Update main README.md
4. 📖 Review RAG best practices
5. 🎯 Prepare for Week 3: ReAct & Tool-Use Agents
6. 🚀 Optional: Deploy RAG system to production

---

## 💡 Bonus Ideas

- Add support for PDFs, DOCX files
- Implement conversation memory in RAG
- Build a web UI with Streamlit
- Add caching for faster responses
- Implement multi-language support
- Create a Chrome extension for docs search
- Deploy to cloud (Hugging Face Spaces, Streamlit Cloud)
EOF
