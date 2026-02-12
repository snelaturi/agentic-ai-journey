# Week 5: Multi-Agent Systems & Production

**Duration:** 14 hours  
**Goal:** Build production-grade multi-agent application

## 📋 Overview

This week is the culmination of your agentic AI journey. You'll build a production-ready multi-agent system with proper architecture, FastAPI backend, async processing, monitoring, and deployment. The flagship project is a Financial Compliance Platform that demonstrates enterprise-grade agent orchestration.

## 🎯 Learning Objectives

- Master multi-agent communication and coordination
- Build state machines with LangGraph
- Design production-grade agent architectures
- Implement FastAPI backends for agents
- Deploy multi-agent systems to production
- Add monitoring, logging, and observability
- Implement event-driven architectures
- Build scalable, async agent workflows
- Create comprehensive API documentation
- Deploy with Docker and CI/CD

## 📅 Daily Breakdown

### Day 29-30: Multi-Agent Fundamentals (4 hrs)

#### Study Phase (1.5 hrs)

**Multi-Agent Communication Patterns:**

1. **Centralized Communication**
````
   Agent 1 ←→ Orchestrator ←→ Agent 2
````
   - Single coordinator manages all agents
   - Easier to debug and control
   - Single point of failure

2. **Peer-to-Peer Communication**
````
   Agent 1 ←→ Agent 2 ←→ Agent 3
````
   - Direct agent-to-agent communication
   - More flexible and resilient
   - Harder to coordinate

3. **Broadcast Communication**
````
   Agent 1 → All Agents
   All Agents → Agent 1
````
   - One agent broadcasts to all
   - Useful for consensus and voting
   - Can be noisy

**Agent Coordination Strategies:**

- **Sequential:** Agents execute one after another
- **Parallel:** Agents execute simultaneously
- **Hierarchical:** Agents in layers, top-down execution
- **Market-based:** Agents bid for tasks
- **Voting/Consensus:** Agents vote on decisions

**Required Reading:**
- 📄 [AutoGen Paper](https://arxiv.org/abs/2308.08155) - "AutoGen: Enabling Next-Gen LLM Applications"
- 📖 [AutoGen Documentation](https://microsoft.github.io/autogen/)
- 📄 [Multi-Agent Systems for LLMs](https://arxiv.org/abs/2402.05120)

**AutoGen Key Concepts:**
````python
# AutoGen basic pattern
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user",
    code_execution_config={"work_dir": "coding"}
)

# Initiate conversation
user_proxy.initiate_chat(
    assistant,
    message="Build a simple calculator"
)
````

---

#### LangGraph Tutorial (1.5 hrs)

**What is LangGraph?**
- Library for building stateful, multi-agent workflows
- Graph-based architecture for agent coordination
- Built on top of LangChain
- Enables complex agent interactions

**Core Concepts:**

1. **State**
````python
   from typing import TypedDict, Annotated
   
   class AgentState(TypedDict):
       messages: Annotated[list, "conversation history"]
       next_agent: str
       task_result: dict
````

2. **Nodes** (Agents/Functions)
````python
   def researcher_agent(state: AgentState):
       # Do research
       return {"messages": [...], "next_agent": "writer"}
````

3. **Edges** (Flow Control)
````python
   graph.add_edge("researcher", "writer")
   graph.add_conditional_edges(
       "writer",
       should_continue,
       {
           "continue": "reviewer",
           "end": END
       }
   )
````

4. **Graph Execution**
````python
   app = graph.compile()
   result = app.invoke(initial_state)
````

**LangGraph Multi-Agent Pattern:**
````
START
  ↓
[Supervisor] → Decides which agent to call
  ↓
[Agent 1] → Executes task
  ↓
[Supervisor] → Reviews result, decides next
  ↓
[Agent 2] → Executes next task
  ↓
[Supervisor] → Final review
  ↓
END
````

**Required Tutorial:**
- 🎥 [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/)
- 📖 [Multi-Agent Workflows Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- 💻 [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

---

#### Build: Multi-Agent Debate System (1 hr)

**System Architecture:**
````
User Question
    ↓
[Debate Moderator]
    ↓
[Pro Agent] ←→ [Con Agent]
    ↓        ↓
  Arguments  Arguments
    ↓        ↓
[Judge Agent]
    ↓
Final Decision + Reasoning
````

**Agents:**

1. **Pro Agent**
   - Argues in favor of the proposition
   - Provides evidence and reasoning
   - Responds to counterarguments

2. **Con Agent**
   - Argues against the proposition
   - Provides contradicting evidence
   - Challenges pro arguments

3. **Judge Agent**
   - Evaluates both arguments
   - Considers evidence quality
   - Makes final decision
   - Provides reasoning

**Example Debate:**
````
Topic: "AI will create more jobs than it destroys by 2030"

Pro Agent: "AI will create new job categories like AI trainers, 
prompt engineers, and automation specialists. Historical tech 
revolutions always created more jobs than they destroyed..."

Con Agent: "Unlike previous revolutions, AI can replace both 
manual AND cognitive labor. Studies show 47% of jobs are at 
risk of automation within 20 years..."

Judge: "While both sides present valid points, the evidence 
suggests... [makes decision with reasoning]"
````

**Implementation:**
````python
# projects/multi-agent-debate/src/debate_system.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class DebateState(TypedDict):
    topic: str
    pro_arguments: List[str]
    con_arguments: List[str]
    round: int
    decision: str
    reasoning: str

def pro_agent(state: DebateState) -> DebateState:
    """Generate pro argument"""
    # Generate argument in favor
    argument = llm.invoke(f"Argue FOR: {state['topic']}")
    state["pro_arguments"].append(argument)
    return state

def con_agent(state: DebateState) -> DebateState:
    """Generate con argument"""
    # Generate argument against
    argument = llm.invoke(f"Argue AGAINST: {state['topic']}")
    state["con_arguments"].append(argument)
    return state

def judge_agent(state: DebateState) -> DebateState:
    """Make final decision"""
    decision = llm.invoke(f"""
    Evaluate these arguments:
    PRO: {state['pro_arguments']}
    CON: {state['con_arguments']}
    Make a decision and explain reasoning.
    """)
    state["decision"] = decision
    return state

# Build graph
graph = StateGraph(DebateState)
graph.add_node("pro", pro_agent)
graph.add_node("con", con_agent)
graph.add_node("judge", judge_agent)

# Add edges for 3 rounds of debate
for i in range(3):
    if i == 0:
        graph.set_entry_point("pro")
    graph.add_edge("pro", "con")
    if i < 2:
        graph.add_edge("con", "pro")
    else:
        graph.add_edge("con", "judge")

graph.add_edge("judge", END)

app = graph.compile()
````

**Deliverable:** `projects/multi-agent-debate/`

**Project Structure:**
````
projects/multi-agent-debate/
├── src/
│   ├── __init__.py
│   ├── debate_system.py      # Main debate logic
│   ├── agents/
│   │   ├── pro_agent.py
│   │   ├── con_agent.py
│   │   └── judge_agent.py
│   └── utils/
│       └── prompts.py
├── examples/
│   └── debate_examples.py
├── tests/
│   └── test_debate.py
├── ui/
│   └── streamlit_app.py      # Interactive debate UI
├── README.md
└── requirements.txt
````

---

### Day 31-35: FLAGSHIP PROJECT - Financial Compliance Platform (10 hrs)

#### 🏗️ System Architecture
````
┌─────────────────────────────────────────────────────┐
│                  API Gateway (FastAPI)               │
│  ┌────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  /upload   │  │ /analyze │  │  /reports    │   │
│  └────────────┘  └──────────┘  └──────────────┘   │
└───────────────────────┬─────────────────────────────┘
                        ↓
              ┌─────────────────────┐
              │ Orchestrator Agent  │
              │  (LangGraph State   │
              │      Machine)       │
              └─────────────────────┘
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
┌───────────────┐ ┌──────────────┐ ┌─────────────┐
│  Document     │ │  Compliance  │ │    Risk     │
│  Ingestion    │ │   Analysis   │ │ Assessment  │
│    Agent      │ │    Agent     │ │   Agent     │
└───────────────┘ └──────────────┘ └─────────────┘
        ↓               ↓               ↓
        └───────────────┼───────────────┘
                        ↓
              ┌─────────────────────┐
              │ Report Generation   │
              │      Agent          │
              └─────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌───────────────┐            ┌──────────────────┐
│  PostgreSQL   │            │   ChromaDB       │
│  (Metadata)   │            │  (Vector Store)  │
└───────────────┘            └──────────────────┘
````

**Data Flow:**
1. User uploads document via API
2. Orchestrator receives task
3. Document Ingestion Agent extracts content
4. Compliance Agent checks rules
5. Risk Agent calculates scores
6. Report Agent generates summary
7. Results returned via WebSocket

---

#### Day 31: Project Setup & Architecture (2 hrs)

**FastAPI Project Structure:**
````
projects/financial-compliance-platform/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry
│   ├── config.py                # Configuration
│   ├── dependencies.py          # FastAPI dependencies
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py     # Document upload
│   │   │   ├── analysis.py      # Analysis endpoints
│   │   │   └── reports.py       # Report endpoints
│   │   └── websocket.py         # Real-time updates
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Main orchestrator
│   │   ├── ingestion_agent.py   # Document processing
│   │   ├── compliance_agent.py  # Compliance checks
│   │   ├── risk_agent.py        # Risk assessment
│   │   └── report_agent.py      # Report generation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py          # SQLAlchemy models
│   │   └── schemas.py           # Pydantic schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_service.py
│   │   ├── vector_store.py      # ChromaDB operations
│   │   └── llm_service.py       # OpenAI interactions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── pdf_processor.py
│       ├── cost_tracker.py
│       └── logger.py
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_agents.py
│   │   └── test_services.py
│   └── integration/
│       └── test_api.py
│
├── migrations/              # Alembic migrations
│   └── versions/
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml        # GitHub Actions
│
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   └── DEPLOYMENT.md
│
├── .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
├── README.md
└── alembic.ini
````

**Database Schema (PostgreSQL):**
````sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    upload_date TIMESTAMP DEFAULT NOW(),
    status VARCHAR(50),
    user_id UUID,
    metadata JSONB
);

-- Analysis results
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    compliance_score FLOAT,
    risk_score FLOAT,
    findings JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Compliance rules
CREATE TABLE compliance_rules (
    id UUID PRIMARY KEY,
    rule_name VARCHAR(255),
    rule_description TEXT,
    rule_type VARCHAR(100),
    severity VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE
);

-- API usage tracking
CREATE TABLE api_usage (
    id UUID PRIMARY KEY,
    endpoint VARCHAR(255),
    user_id UUID,
    tokens_used INTEGER,
    cost DECIMAL(10, 4),
    timestamp TIMESTAMP DEFAULT NOW()
);
````

**Docker Compose Setup:**
````yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/compliance_db
      - CHROMA_HOST=chromadb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - chromadb
      - redis
    volumes:
      - ../uploads:/app/uploads

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=compliance_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

volumes:
  postgres_data:
  chroma_data:
````

**CI/CD Skeleton (GitHub Actions):**
````yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest --cov=app tests/
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run linters
        run: |
          pip install black flake8 mypy
          black --check app/
          flake8 app/
          mypy app/

  deploy:
    needs: [test, lint]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Deployment commands
          echo "Deploying to production..."
````

**Configuration Setup:**
````python
# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Financial Compliance Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str
    
    # Vector Store
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Authentication
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".pdf", ".docx", ".txt"]
    
    # Monitoring
    SENTRY_DSN: str = ""
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
````

**FastAPI Main Application:**
````python
# app/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import get_settings
from app.api.routes import documents, analysis, reports
from app.models.database import engine, Base
from app.utils.logger import setup_logging

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application...")
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    logger.info("Shutting down application...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Routes
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "vector_store": "connected"
    }
````

---

#### Day 32: Document Ingestion Agent (2 hrs)

**Agent Responsibilities:**
- Extract text from PDFs
- Identify tables and structure
- Extract metadata (dates, entities, amounts)
- Chunk documents intelligently
- Generate embeddings
- Store in vector database

**Implementation:**
````python
# app/agents/ingestion_agent.py
from typing import Dict, List, Any
import pypdf
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

class DocumentIngestionAgent:
    """Agent for processing and ingesting documents"""
    
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def process_document(
        self,
        file_path: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Process document through complete ingestion pipeline
        """
        try:
            # Step 1: Extract text
            logger.info(f"Extracting text from {file_path}")
            text = self._extract_text(file_path)
            
            # Step 2: Extract tables
            logger.info("Extracting tables")
            tables = self._extract_tables(file_path)
            
            # Step 3: Extract metadata
            logger.info("Extracting metadata")
            metadata = await self._extract_metadata(text)
            
            # Step 4: Chunk text
            logger.info("Chunking document")
            chunks = self._chunk_document(text)
            
            # Step 5: Generate embeddings
            logger.info("Generating embeddings")
            embeddings = self.embeddings.embed_documents(chunks)
            
            # Step 6: Store in vector database
            logger.info("Storing in vector database")
            self._store_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            
            return {
                "status": "success",
                "document_id": document_id,
                "chunks_created": len(chunks),
                "tables_found": len(tables),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def _extract_tables(self, file_path: str) -> List[Dict]:
        """Extract tables from PDF"""
        tables = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    tables.append({
                        "page": page.page_number,
                        "data": table
                    })
        return tables
    
    async def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract structured metadata using LLM"""
        prompt = f"""Extract key metadata from this document:

{text[:2000]}  # First 2000 chars

Extract and return as JSON:
- document_type (e.g., "financial_report", "contract", "invoice")
- date
- company_name
- key_entities (people, organizations)
- financial_amounts
- important_dates
"""
        
        response = await self.llm.ainvoke(prompt)
        # Parse JSON from response
        import json
        try:
            metadata = json.loads(response.content)
        except:
            metadata = {"raw": response.content}
        
        return metadata
    
    def _chunk_document(self, text: str) -> List[str]:
        """Split document into chunks"""
        return self.text_splitter.split_text(text)
    
    def _store_chunks(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict
    ):
        """Store chunks in vector database"""
        self.vector_store.add_documents(
            ids=[f"{document_id}_chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {**metadata, "chunk_index": i, "document_id": document_id}
                for i in range(len(chunks))
            ]
        )
````

**PDF Processing Utilities:**
````python
# app/utils/pdf_processor.py
import pdfplumber
from typing import List, Dict

class PDFProcessor:
    """Advanced PDF processing utilities"""
    
    @staticmethod
    def extract_with_layout(pdf_path: str) -> List[Dict]:
        """Extract text while preserving layout"""
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages.append({
                    "page_number": page.page_number,
                    "text": page.extract_text(),
                    "layout": page.extract_words(),
                    "tables": page.extract_tables()
                })
        return pages
    
    @staticmethod
    def detect_document_structure(pages: List[Dict]) -> Dict:
        """Detect document structure (headers, sections, etc.)"""
        # Implement structure detection logic
        pass
    
    @staticmethod
    def extract_figures_and_charts(pdf_path: str) -> List:
        """Extract images and charts"""
        # Implement image extraction
        pass
````

---

#### Day 33: Compliance & Risk Agents (2 hrs)

**Compliance Analysis Agent:**
````python
# app/agents/compliance_agent.py
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ComplianceAnalysisAgent:
    """Agent for checking regulatory compliance"""
    
    def __init__(self, llm, rules_db):
        self.llm = llm
        self.rules_db = rules_db
    
    async def analyze_compliance(
        self,
        document_id: str,
        document_text: str,
        relevant_rules: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze document for compliance violations
        """
        findings = []
        
        for rule in relevant_rules:
            logger.info(f"Checking rule: {rule['rule_name']}")
            
            # Check rule against document
            result = await self._check_rule(
                document_text=document_text,
                rule=rule
            )
            
            if result["violated"]:
                findings.append({
                    "rule_id": rule["id"],
                    "rule_name": rule["rule_name"],
                    "severity": rule["severity"],
                    "description": result["description"],
                    "evidence": result["evidence"],
                    "recommendation": result["recommendation"]
                })
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(findings)
        
        return {
            "document_id": document_id,
            "compliance_score": compliance_score,
            "total_rules_checked": len(relevant_rules),
            "violations_found": len(findings),
            "findings": findings,
            "overall_status": self._determine_status(compliance_score)
        }
    
    async def _check_rule(
        self,
        document_text: str,
        rule: Dict
    ) -> Dict[str, Any]:
        """Check a single compliance rule"""
        
        prompt = f"""Analyze this document for compliance with the following rule:

Rule: {rule['rule_name']}
Description: {rule['rule_description']}
Type: {rule['rule_type']}

Document excerpt:
{document_text[:3000]}

Determine:
1. Is this rule violated? (yes/no)
2. What is the evidence?
3. What is the recommendation to fix it?

Return as JSON:
{{
    "violated": true/false,
    "description": "explanation",
    "evidence": "specific text from document",
    "recommendation": "how to fix"
}}
"""
        
        response = await self.llm.ainvoke(prompt)
        
        # Parse response
        import json
        try:
            result = json.loads(response.content)
        except:
            result = {
                "violated": False,
                "description": "Error parsing result",
                "evidence": "",
                "recommendation": ""
            }
        
        return result
    
    def _calculate_compliance_score(self, findings: List[Dict]) -> float:
        """Calculate overall compliance score (0-100)"""
        if not findings:
            return 100.0
        
        # Weight by severity
        severity_weights = {
            "critical": 10,
            "high": 5,
            "medium": 2,
            "low": 1
        }
        
        total_deductions = sum(
            severity_weights.get(f["severity"], 1)
            for f in findings
        )
        
        # Score out of 100
        score = max(0, 100 - (total_deductions * 5))
        return score
    
    def _determine_status(self, score: float) -> str:
        """Determine overall compliance status"""
        if score >= 90:
            return "compliant"
        elif score >= 70:
            return "needs_attention"
        else:
            return "non_compliant"
````

**Risk Assessment Agent:**
````python
# app/agents/risk_agent.py
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class RiskAssessmentAgent:
    """Agent for assessing financial and regulatory risk"""
    
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
    
    async def assess_risk(
        self,
        document_id: str,
        document_text: str,
        compliance_findings: List[Dict]
    ) -> Dict[str, Any]:
        """
        Comprehensive risk assessment
        """
        # Assess different risk categories
        financial_risk = await self._assess_financial_risk(document_text)
        regulatory_risk = self._assess_regulatory_risk(compliance_findings)
        operational_risk = await self._assess_operational_risk(document_text)
        reputational_risk = await self._assess_reputational_risk(document_text)
        
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(
            financial_risk,
            regulatory_risk,
            operational_risk,
            reputational_risk
        )
        
        return {
            "document_id": document_id,
            "overall_risk_score": overall_risk["score"],
            "risk_level": overall_risk["level"],
            "risk_breakdown": {
                "financial": financial_risk,
                "regulatory": regulatory_risk,
                "operational": operational_risk,
                "reputational": reputational_risk
            },
            "key_risk_factors": overall_risk["key_factors"],
            "mitigation_recommendations": overall_risk["recommendations"]
        }
    
    async def _assess_financial_risk(self, text: str) -> Dict:
        """Assess financial risk indicators"""
        prompt = f"""Analyze financial risk in this document:

{text[:2000]}

Identify:
1. Financial irregularities
2. Unusual transactions
3. Cash flow concerns
4. Debt levels
5. Liquidity issues

Return risk score (0-100) and key factors.
"""
        response = await self.llm.ainvoke(prompt)
        # Parse and return
        return self._parse_risk_response(response.content)
    
    def _assess_regulatory_risk(self, findings: List[Dict]) -> Dict:
        """Assess regulatory risk based on compliance findings"""
        if not findings:
            return {"score": 0, "level": "low", "factors": []}
        
        # Calculate risk from violations
        severity_scores = {
            "critical": 90,
            "high": 70,
            "medium": 40,
            "low": 20
        }
        
        max_severity_score = max(
            severity_scores.get(f["severity"], 0)
            for f in findings
        )
        
        return {
            "score": max_severity_score,
            "level": self._score_to_level(max_severity_score),
            "factors": [f["rule_name"] for f in findings]
        }
    
    async def _assess_operational_risk(self, text: str) -> Dict:
        """Assess operational risk"""
        # Implementation similar to financial risk
        pass
    
    async def _assess_reputational_risk(self, text: str) -> Dict:
        """Assess reputational risk"""
        # Implementation
        pass
    
    def _calculate_overall_risk(
        self,
        financial: Dict,
        regulatory: Dict,
        operational: Dict,
        reputational: Dict
    ) -> Dict:
        """Calculate weighted overall risk score"""
        
        # Weighted average
        weights = {
            "financial": 0.3,
            "regulatory": 0.4,
            "operational": 0.2,
            "reputational": 0.1
        }
        
        overall_score = (
            financial["score"] * weights["financial"] +
            regulatory["score"] * weights["regulatory"] +
            operational["score"] * weights["operational"] +
            reputational["score"] * weights["reputational"]
        )
        
        return {
            "score": overall_score,
            "level": self._score_to_level(overall_score),
            "key_factors": self._identify_key_factors(
                financial, regulatory, operational, reputational
            ),
            "recommendations": self._generate_recommendations(overall_score)
        }
    
    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to risk level"""
        if score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"
````

**Multi-Agent Coordination with LangGraph:**
````python
# app/agents/orchestrator.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import logging

logger = logging.getLogger(__name__)

class AnalysisState(TypedDict):
    document_id: str
    document_path: str
    ingestion_result: dict
    compliance_result: dict
    risk_result: dict
    report: str
    status: str
    errors: list

class FinancialComplianceOrchestrator:
    """Orchestrator for multi-agent compliance analysis"""
    
    def __init__(
        self,
        ingestion_agent,
        compliance_agent,
        risk_agent,
        report_agent
    ):
        self.ingestion_agent = ingestion_agent
        self.compliance_agent = compliance_agent
        self.risk_agent = risk_agent
        self.report_agent = report_agent
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("ingest", self._ingest_node)
        workflow.add_node("compliance", self._compliance_node)
        workflow.add_node("risk", self._risk_node)
        workflow.add_node("report", self._report_node)
        
        # Define flow
        workflow.set_entry_point("ingest")
        workflow.add_edge("ingest", "compliance")
        workflow.add_edge("compliance", "risk")
        workflow.add_edge("risk", "report")
        workflow.add_edge("report", END)
        
        return workflow.compile()
    
    async def _ingest_node(self, state: AnalysisState) -> AnalysisState:
        """Document ingestion node"""
        logger.info(f"Ingesting document: {state['document_id']}")
        
        result = await self.ingestion_agent.process_document(
            file_path=state["document_path"],
            document_id=state["document_id"]
        )
        
        state["ingestion_result"] = result
        state["status"] = "ingested"
        return state
    
    async def _compliance_node(self, state: AnalysisState) -> AnalysisState:
        """Compliance analysis node"""
        logger.info(f"Analyzing compliance: {state['document_id']}")
        
        # Get document text from ingestion result
        # Fetch relevant rules
        # Run compliance analysis
        
        result = await self.compliance_agent.analyze_compliance(
            document_id=state["document_id"],
            document_text="...",  # From vector store
            relevant_rules=[]  # From rules database
        )
        
        state["compliance_result"] = result
        state["status"] = "compliance_checked"
        return state
    
    async def _risk_node(self, state: AnalysisState) -> AnalysisState:
        """Risk assessment node"""
        logger.info(f"Assessing risk: {state['document_id']}")
        
        result = await self.risk_agent.assess_risk(
            document_id=state["document_id"],
            document_text="...",
            compliance_findings=state["compliance_result"]["findings"]
        )
        
        state["risk_result"] = result
        state["status"] = "risk_assessed"
        return state
    
    async def _report_node(self, state: AnalysisState) -> AnalysisState:
        """Report generation node"""
        logger.info(f"Generating report: {state['document_id']}")
        
        report = await self.report_agent.generate_report(
            ingestion=state["ingestion_result"],
            compliance=state["compliance_result"],
            risk=state["risk_result"]
        )
        
        state["report"] = report
        state["status"] = "completed"
        return state
    
    async def analyze_document(self, document_id: str, document_path: str):
        """Run complete analysis workflow"""
        initial_state = {
            "document_id": document_id,
            "document_path": document_path,
            "status": "pending",
            "errors": []
        }
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        return final_state
````

---

#### Day 34: Report Generator & WebSocket (2 hrs)

**Report Generation Agent:**
````python
# app/agents/report_agent.py
from typing import Dict, Any
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)

class ReportGenerationAgent:
    """Agent for generating executive reports"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def generate_report(
        self,
        ingestion: Dict,
        compliance: Dict,
        risk: Dict
    ) -> Dict[str, Any]:
        """
        Generate comprehensive executive report
        """
        # Generate executive summary
        executive_summary = await self._generate_executive_summary(
            compliance, risk
        )
        
        # Generate detailed findings
        detailed_findings = self._format_detailed_findings(
            compliance, risk
        )
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            compliance, risk
        )
        
        # Generate HTML report
        html_report = self._render_html_report(
            executive_summary=executive_summary,
            compliance=compliance,
            risk=risk,
            detailed_findings=detailed_findings,
            recommendations=recommendations
        )
        
        # Generate PDF (optional)
        # pdf_report = self._generate_pdf(html_report)
        
        return {
            "executive_summary": executive_summary,
            "compliance_score": compliance["compliance_score"],
            "risk_score": risk["overall_risk_score"],
            "detailed_findings": detailed_findings,
            "recommendations": recommendations,
            "html_report": html_report
        }
    
    async def _generate_executive_summary(
        self,
        compliance: Dict,
        risk: Dict
    ) -> str:
        """Generate executive summary using LLM"""
        
        prompt = f"""Generate an executive summary for this financial compliance analysis:

Compliance Score: {compliance['compliance_score']}/100
Violations Found: {len(compliance['findings'])}
Risk Score: {risk['overall_risk_score']}/100
Risk Level: {risk['risk_level']}

Key Findings:
{self._format_key_findings(compliance, risk)}

Write a concise 2-3 paragraph executive summary suitable for C-level executives.
"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    def _render_html_report(self, **kwargs) -> str:
        """Render HTML report from template"""
        
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Compliance Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #2c3e50; color: white; padding: 20px; }
        .summary { background: #ecf0f1; padding: 20px; margin: 20px 0; }
        .score { font-size: 48px; font-weight: bold; }
        .findings { margin: 20px 0; }
        .violation { border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; }
        .recommendation { border-left: 4px solid #27ae60; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Financial Compliance Analysis Report</h1>
        <p>Generated: {{ timestamp }}</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>{{ executive_summary }}</p>
        
        <div style="display: flex; justify-content: space-around; margin-top: 20px;">
            <div>
                <div class="score" style="color: {{ 'green' if compliance.compliance_score >= 90 else 'orange' if compliance.compliance_score >= 70 else 'red' }}">
                    {{ compliance.compliance_score }}
                </div>
                <p>Compliance Score</p>
            </div>
            <div>
                <div class="score" style="color: {{ 'green' if risk.overall_risk_score < 40 else 'orange' if risk.overall_risk_score < 70 else 'red' }}">
                    {{ risk.overall_risk_score }}
                </div>
                <p>Risk Score</p>
            </div>
        </div>
    </div>
    
    <div class="findings">
        <h2>Detailed Findings</h2>
        {% for finding in compliance.findings %}
        <div class="violation">
            <h3>{{ finding.rule_name }}</h3>
            <p><strong>Severity:</strong> {{ finding.severity }}</p>
            <p>{{ finding.description }}</p>
            <p><strong>Evidence:</strong> {{ finding.evidence }}</p>
        </div>
        {% endfor %}
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        {% for rec in recommendations %}
        <div class="recommendation">
            <h3>{{ rec.title }}</h3>
            <p>{{ rec.description }}</p>
        </div>
        {% endfor %}
    </div>
</body>
</html>
        """)
        
        from datetime import datetime
        kwargs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return template.render(**kwargs)
````

**WebSocket for Real-Time Updates:**
````python
# app/api/websocket.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(
                json.dumps(message)
            )
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_text(json.dumps(message))

manager = ConnectionManager()

# In main.py, add WebSocket endpoint:
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages
            await manager.send_message(
                client_id,
                {"type": "ack", "data": data}
            )
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Usage in orchestrator:
async def _notify_progress(self, client_id: str, status: str, data: dict):
    """Send real-time progress updates"""
    await manager.send_message(client_id, {
        "type": "progress",
        "status": status,
        "data": data
    })
````

---

#### Day 35: Production Features (2 hrs)

**Error Handling and Retries:**
````python
# app/utils/retry.py
from functools import wraps
import asyncio
import logging

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, base_delay=1):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached for {func.__name__}: {str(e)}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}. "
                        f"Retrying in {delay}s... Error: {str(e)}"
                    )
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

# Usage:
@retry_with_backoff(max_retries=3, base_delay=2)
async def call_llm_api(...):
    # API call that might fail
    pass
````

**Logging and Monitoring:**
````python
# app/utils/logger.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(level: str = "INFO"):
    """Set up structured logging"""
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # JSON formatter for structured logs
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# app/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
api_requests = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

api_latency = Histogram(
    'api_latency_seconds',
    'API latency',
    ['endpoint']
)

active_analyses = Gauge(
    'active_analyses',
    'Number of active document analyses'
)

# Usage in routes:
@router.post("/analyze")
async def analyze_document(...):
    start_time = time.time()
    active_analyses.inc()
    
    try:
        result = await orchestrator.analyze_document(...)
        api_requests.labels(endpoint="/analyze", method="POST", status="success").inc()
        return result
    except Exception as e:
        api_requests.labels(endpoint="/analyze", method="POST", status="error").inc()
        raise
    finally:
        active_analyses.dec()
        api_latency.labels(endpoint="/analyze").observe(time.time() - start_time)
````

**Token Usage Tracking:**
````python
# app/utils/cost_tracker.py
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class CostTracker:
    """Track API usage and costs"""
    
    # OpenAI pricing (as of 2024)
    PRICING = {
        "gpt-4": {
            "input": 0.03 / 1000,   # per token
            "output": 0.06 / 1000
        },
        "gpt-3.5-turbo": {
            "input": 0.001 / 1000,
            "output": 0.002 / 1000
        },
        "text-embedding-3-small": {
            "input": 0.00002 / 1000
        }
    }
    
    def __init__(self, db_session):
        self.db = db_session
        self.current_usage = {
            "tokens": 0,
            "cost": 0.0
        }
    
    def track_completion(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: str = None
    ):
        """Track a completion API call"""
        pricing = self.PRICING.get(model, self.PRICING["gpt-4"])
        
        cost = (
            input_tokens * pricing["input"] +
            output_tokens * pricing["output"]
        )
        
        self.current_usage["tokens"] += input_tokens + output_tokens
        self.current_usage["cost"] += cost
        
        # Store in database
        self._store_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            user_id=user_id
        )
        
        logger.info(
            f"API call tracked: {model}, "
            f"tokens={input_tokens + output_tokens}, "
            f"cost=${cost:.4f}"
        )
    
    def track_embedding(
        self,
        model: str,
        tokens: int,
        user_id: str = None
    ):
        """Track an embedding API call"""
        pricing = self.PRICING.get(model, self.PRICING["text-embedding-3-small"])
        cost = tokens * pricing["input"]
        
        self.current_usage["tokens"] += tokens
        self.current_usage["cost"] += cost
        
        self._store_usage(
            model=model,
            input_tokens=tokens,
            output_tokens=0,
            cost=cost,
            user_id=user_id
        )
    
    def get_usage_summary(self, user_id: str = None) -> Dict:
        """Get usage summary from database"""
        # Query database for usage stats
        pass
````

**Rate Limiting:**
````python
# app/api/middleware.py
from fastapi import Request, HTTPException
from redis import Redis
import time

class RateLimiter:
    """Rate limiting middleware"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def check_rate_limit(
        self,
        request: Request,
        limit: int = 60,  # requests per minute
        window: int = 60  # seconds
    ):
        """Check if request should be rate limited"""
        # Get client identifier (IP or user ID)
        client_id = request.client.host
        if hasattr(request.state, "user"):
            client_id = request.state.user.id
        
        key = f"rate_limit:{client_id}"
        
        # Get current count
        current = self.redis.get(key)
        
        if current is None:
            # First request in window
            self.redis.setex(key, window, 1)
            return True
        
        current = int(current)
        
        if current >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {limit} requests per {window} seconds."
            )
        
        # Increment counter
        self.redis.incr(key)
        return True

# Usage in routes:
@router.post("/analyze")
async def analyze_document(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    await rate_limiter.check_rate_limit(request)
    # ... rest of endpoint
````

**API Authentication (JWT):**
````python
# app/auth/jwt.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return user_id
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

# Usage in routes:
@router.post("/analyze")
async def analyze_document(
    user_id: str = Depends(verify_token)
):
    # User is authenticated
    pass
````

**Unit and Integration Tests:**
````python
# tests/unit/test_compliance_agent.py
import pytest
from app.agents.compliance_agent import ComplianceAnalysisAgent

@pytest.fixture
def compliance_agent():
    return ComplianceAnalysisAgent(llm=mock_llm, rules_db=mock_db)

@pytest.mark.asyncio
async def test_analyze_compliance_no_violations(compliance_agent):
    """Test compliance analysis with no violations"""
    result = await compliance_agent.analyze_compliance(
        document_id="test-1",
        document_text="Sample compliant document",
        relevant_rules=[mock_rule]
    )
    
    assert result["compliance_score"] == 100
    assert len(result["findings"]) == 0
    assert result["overall_status"] == "compliant"

# tests/integration/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_document():
    """Test document upload endpoint"""
    with open("test_document.pdf", "rb") as f:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    assert response.status_code == 200
    assert "document_id" in response.json()

def test_analyze_document():
    """Test analysis endpoint"""
    response = client.post(
        "/api/v1/analysis/analyze",
        json={"document_id": "test-document-id"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "compliance_score" in data
    assert "risk_score" in data
````

---

### Weekend: Deploy & Document

**Deployment to Render/Railway:**
````yaml
# render.yaml
services:
  - type: web
    name: compliance-platform-api
    env: docker
    dockerfilePath: ./docker/Dockerfile
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: compliance-db
          property: connectionString
      - key: OPENAI_API_KEY
        sync: false
      - key: SECRET_KEY
        generateValue: true
    
  - type: worker
    name: compliance-platform-worker
    env: docker
    dockerfilePath: ./docker/Dockerfile.worker
    
databases:
  - name: compliance-db
    databaseName: compliance
    user: compliance_user
````

**Demo Video Script:**
````markdown
# Demo Video Outline (5-7 minutes)

## Introduction (30s)
- "Welcome to the Financial Compliance Platform"
- "Multi-agent system for automated compliance analysis"

## Architecture Overview (1 min)
- Show system diagram
- Explain agent responsibilities
- Highlight key technologies

## Live Demo (3 mins)
1. Upload document via API
2. Show WebSocket real-time updates
3. View analysis results
4. Explore generated report
5. Show metrics dashboard

## Technical Highlights (1 min)
- LangGraph orchestration
- FastAPI performance
- Production features

## Results & Metrics (1 min)
- Processing time
- Accuracy metrics
- Cost per analysis

## Conclusion (30s)
- GitHub repository
- Documentation
- Contact information
````

**Comprehensive README:**
````markdown
# Financial Compliance Platform

Production-grade multi-agent system for automated financial document compliance analysis.

## 🚀 Features

- **Multi-Agent Architecture**: Orchestrated agents for ingestion, compliance, risk, and reporting
- **Real-Time Updates**: WebSocket support for live progress tracking
- **Production Ready**: Error handling, rate limiting, authentication, monitoring
- **Scalable**: Async processing, Docker deployment, horizontal scaling
- **Cost Tracking**: Built-in token usage and cost monitoring
- **Comprehensive API**: RESTful API with Swagger documentation

## 🏗️ Architecture

[Include architecture diagram]

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- OpenAI API key

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/yourusername/financial-compliance-platform
cd financial-compliance-platform

# Set up environment
cp .env.example .env
# Edit .env with your credentials

# Using Docker
docker-compose up -d

# Or local development
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

## 📖 API Documentation

Access Swagger UI at: http://localhost:8000/docs

### Key Endpoints

**Upload Document**
```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data

Returns: {"document_id": "uuid"}
```

**Analyze Document**
```bash
POST /api/v1/analysis/analyze
{
  "document_id": "uuid"
}

Returns: {
  "compliance_score": 85.5,
  "risk_score": 45.2,
  "findings": [...],
  "report_url": "..."
}
```

## 🧪 Testing
```bash
# Run tests
pytest

# With coverage
pytest --cov=app tests/

# Integration tests only
pytest tests/integration/
```

## 📊 Monitoring

- Metrics: http://localhost:8000/metrics
- Health: http://localhost:8000/health
- Logs: `docker-compose logs -f api`

## 🚀 Deployment

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment instructions.

## 📈 Performance

- Average analysis time: 45s
- Throughput: 100 documents/hour
- Accuracy: 92% compliance detection
- Cost: $0.15 per analysis

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 📞 Contact

- GitHub: [@yourusername]
- Email: your.email@example.com
````

---

## 🚀 Deliverables

### 1. Multi-Agent Debate System
**Location:** `projects/multi-agent-debate/`

**Features:**
- ✅ Pro/Con agents with structured arguments
- ✅ Judge agent with reasoning
- ✅ LangGraph state machine
- ✅ Multiple debate rounds
- ✅ Interactive UI

---

### 2. Financial Compliance Platform
**Location:** `projects/financial-compliance-platform/`

**Features:**
- ✅ FastAPI backend with async processing
- ✅ Multi-agent orchestration with LangGraph
- ✅ Document ingestion (PDF processing)
- ✅ Compliance analysis agent
- ✅ Risk assessment agent
- ✅ Report generation agent
- ✅ PostgreSQL + ChromaDB storage
- ✅ WebSocket real-time updates
- ✅ JWT authentication
- ✅ Rate limiting
- ✅ Comprehensive logging and monitoring
- ✅ Token usage tracking
- ✅ Docker deployment
- ✅ CI/CD pipeline
- ✅ Unit and integration tests
- ✅ API documentation

---

## 📚 Resources

### Multi-Agent Systems
- 📄 [AutoGen Paper](https://arxiv.org/abs/2308.08155)
- 📄 [Multi-Agent LLM Systems](https://arxiv.org/abs/2402.05120)
- 📖 [AutoGen Documentation](https://microsoft.github.io/autogen/)
- 📖 [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

### Production Best Practices
- 📖 [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- 📖 [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- 📖 [Kubernetes for ML](https://kubernetes.io/docs/tutorials/)
- 📄 "Building Production-Ready ML Systems"

### Monitoring & Observability
- 📖 [Prometheus Documentation](https://prometheus.io/docs/)
- 📖 [Grafana Dashboards](https://grafana.com/docs/)
- 📖 [OpenTelemetry](https://opentelemetry.io/docs/)

---

## ✅ Completion Checklist

### Day 29-30: Multi-Agent Fundamentals
- [ ] Study multi-agent communication patterns
- [ ] Complete AutoGen tutorial
- [ ] Complete LangGraph tutorial
- [ ] Understand state machines
- [ ] Build debate system
  - [ ] Implement Pro agent
  - [ ] Implement Con agent
  - [ ] Implement Judge agent
  - [ ] Create LangGraph workflow
  - [ ] Add UI
- [ ] Test with multiple topics
- [ ] Document architecture
- [ ] Commit to GitHub

### Day 31: Project Setup
- [ ] Design system architecture
- [ ] Set up FastAPI project structure
- [ ] Configure PostgreSQL
- [ ] Set up ChromaDB
- [ ] Create Docker Compose
- [ ] Set up CI/CD skeleton
- [ ] Create database schema
- [ ] Write initial documentation

### Day 32: Document Ingestion
- [ ] Implement PDF text extraction
- [ ] Implement table extraction
- [ ] Build metadata extractor
- [ ] Create chunking logic
- [ ] Integrate embedding generation
- [ ] Connect to vector store
- [ ] Write tests
- [ ] Document agent

### Day 33: Compliance & Risk
- [ ] Design compliance rule engine
- [ ] Implement Compliance Agent
- [ ] Build risk assessment logic
- [ ] Implement Risk Agent
- [ ] Create LangGraph orchestration
- [ ] Test multi-agent coordination
- [ ] Add error handling
- [ ] Write tests

### Day 34: Report & WebSocket
- [ ] Implement Report Generation Agent
- [ ] Create HTML templates
- [ ] Build PDF generation (optional)
- [ ] Set up WebSocket endpoint
- [ ] Implement connection manager
- [ ] Add real-time progress updates
- [ ] Test streaming
- [ ] Write tests

### Day 35: Production Features
- [ ] Implement retry logic
- [ ] Add structured logging
- [ ] Set up Prometheus metrics
- [ ] Implement cost tracking
- [ ] Add rate limiting
- [ ] Implement JWT authentication
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Add health checks
- [ ] Configure monitoring

### Weekend: Deploy & Document
- [ ] Deploy to cloud platform
- [ ] Configure DNS (if needed)
- [ ] Set up monitoring dashboards
- [ ] Record demo video
- [ ] Write comprehensive README
- [ ] Create API documentation
- [ ] Write deployment guide
- [ ] Create architecture diagrams
- [ ] Add usage examples
- [ ] Final testing
- [ ] Push to GitHub
- [ ] Share project

---

## 🎓 Key Concepts Mastered

- [ ] Multi-agent communication patterns
- [ ] AutoGen framework
- [ ] LangGraph state machines
- [ ] Agent orchestration
- [ ] FastAPI production patterns
- [ ] Async processing
- [ ] Event-driven architecture
- [ ] WebSocket real-time updates
- [ ] Docker containerization
- [ ] Database design for agents
- [ ] Vector store integration
- [ ] API authentication & authorization
- [ ] Rate limiting strategies
- [ ] Structured logging
- [ ] Prometheus metrics
- [ ] Cost tracking
- [ ] CI/CD pipelines
- [ ] Integration testing
- [ ] Production deployment

---

## 📊 Time Tracking

| Day | Hours Spent | Topics Covered |
|-----|-------------|----------------|
| 29  |             |                |
| 30  |             |                |
| 31  |             |                |
| 32  |             |                |
| 33  |             |                |
| 34  |             |                |
| 35  |             |                |
| Weekend |         |                |
| **Total** | **/ 14 hrs** |          |

---

## 📈 Platform Metrics

| Metric | Value |
|--------|-------|
| API Endpoints | 12+ |
| Agents | 5 |
| Average Analysis Time | 45s |
| Throughput | 100 docs/hour |
| Test Coverage | % |
| Uptime | % |
| Cost per Analysis | $ |

---

## 🚧 Challenges & Solutions

### Challenge 1: Agent Coordination
**Problem:**
Managing state across multiple agents

**Solution:**
- Used LangGraph for explicit state management
- Clear state transitions between agents
- Checkpoints for error recovery

---

### Challenge 2: Real-Time Updates
**Problem:**
Keeping users informed during long-running processes

**Solution:**
- Implemented WebSocket connections
- Progress updates at each agent step
- Error notifications

---

### Challenge 3: Cost Management
**Problem:**
LLM API costs can escalate quickly

**Solution:**
- Comprehensive token tracking
- Caching strategies
- Rate limiting
- Cost alerts

---

## 🎯 Week 5 Reflection

**What went well:**
- 

**What was difficult:**
- 

**What I would do differently:**
- 

**Key takeaways:**
- 

**Most valuable skill learned:**
- 

**Ready for Week 6?** [ ] Yes [ ] Need more practice

---

## 🔗 Project Links

- **GitHub Repository:** [Your repo link]
- **Live Demo:** [Deployment URL]
- **API Documentation:** [Swagger URL]
- **Demo Video:** [Loom/YouTube link]
- **Architecture Docs:** [Link]
- **Monitoring Dashboard:** [Grafana link]

---

## 📞 Next Steps

After completing Week 5:
1. ✅ Deploy platform to production
2. ✅ Create demo video
3. ✅ Write comprehensive documentation
4. ✅ Update main README.md
5. 🎯 Prepare for Week 6: Optimization & Scaling
6. 📝 Write technical blog post
7. 🚀 Add platform to portfolio
8. 💼 Consider commercial applications

---

## 💡 Bonus Ideas

- Add more agent types (e.g., Translation Agent, Summarization Agent)
- Implement agent marketplace
- Build agent analytics dashboard
- Add A/B testing for agents
- Create agent performance comparisons
- Implement agent versioning
- Build agent debugging tools
- Add multi-tenancy support
- Create white-label version
- Build agent SDK for custom agents

---

**Week 5 Checkpoint:** ✅ Production multi-agent system deployed
