# Week 3: Custom Tools & SQL Agents

**Duration:** 12 hours  
**Goal:** Build agents that interact with databases and APIs

## 📋 Overview

This week focuses on building custom tools that extend agent capabilities beyond text generation. You'll create tools for database access, API integration, file operations, email automation, and web scraping, then combine them into powerful multi-tool agents.

## 🎯 Learning Objectives

- Design and implement custom agent tools
- Build database query agents with natural language
- Integrate multiple APIs into agent workflows
- Master tool calling and orchestration
- Implement error handling and validation
- Create reusable tool libraries
- Build production-ready SQL agents
- Orchestrate multi-API research workflows

## 📅 Daily Breakdown

### Day 15-16: Build 5 Custom Tools (4 hrs)

**Topics:**
- Tool design principles
- Function schemas and signatures
- Input validation and error handling
- Tool documentation best practices
- Testing custom tools
- Tool composition patterns

**Tools to Build:**

#### 1. Database Tool (SQLite/PostgreSQL)
**Capabilities:**
- Execute SELECT queries
- Insert/Update/Delete operations
- Schema inspection
- Transaction management
- Connection pooling

**Use Cases:**
- Query customer data
- Fetch product information
- Analyze sales records

---

#### 2. API Integration Tool
**Capabilities:**
- REST API calls (GET, POST, PUT, DELETE)
- Authentication (API keys, OAuth)
- Rate limiting
- Response parsing
- Error handling

**Example APIs:**
- Weather API (OpenWeatherMap)
- News API
- Stock market data (Alpha Vantage)
- GitHub API

---

#### 3. File Operations Tool
**Capabilities:**
- Read/write files (TXT, JSON, CSV)
- Parse structured data
- File system navigation
- Directory operations
- File search and filtering

**Use Cases:**
- Process log files
- Generate reports
- Read configuration files

---

#### 4. Email Automation Tool
**Capabilities:**
- Send emails (SMTP)
- Read emails (IMAP)
- Email templates
- Attachment handling
- Email filtering

**Use Cases:**
- Send notifications
- Process incoming requests
- Automated reporting

---

#### 5. Web Scraper Tool
**Capabilities:**
- HTML parsing (BeautifulSoup)
- CSS/XPath selectors
- Dynamic content (Selenium - optional)
- Pagination handling
- Data extraction

**Use Cases:**
- Extract product prices
- Gather news articles
- Collect research data

---

**Deliverable:** Custom Tools Library (`tools/`)

**Project Structure:**
```
projects/custom-tools-library/
├── tools/
│   ├── __init__.py
│   ├── database_tool.py
│   ├── api_tool.py
│   ├── file_tool.py
│   ├── email_tool.py
│   └── scraper_tool.py
├── tests/
│   ├── test_database_tool.py
│   ├── test_api_tool.py
│   └── ...
├── examples/
│   └── tool_usage_examples.py
├── README.md
└── requirements.txt
```

---

### Day 17-19: Natural Language to SQL Agent (4 hrs)

**Topics:**
- Text-to-SQL fundamentals
- Database schema understanding
- SQL query generation
- Query validation and safety
- Result formatting
- Multi-table queries
- Aggregations and joins
- Query optimization

**Architecture:**
```
User Question
    ↓
[Schema Retrieval] ← Database Schema
    ↓
[Query Generation] ← LLM with examples
    ↓
[Query Validation] ← SQL parser
    ↓
[Execute Query] ← Database
    ↓
[Format Results] ← Result formatter
    ↓
Natural Language Answer
```

**Key Components:**

1. **Schema Inspector**
   - Extract table structures
   - Identify relationships
   - Get sample data
   - Generate schema descriptions

2. **Query Generator**
   - Prompt engineering for SQL
   - Few-shot examples
   - Chain-of-thought reasoning
   - Error handling

3. **Safety Layer**
   - Read-only queries
   - Query validation
   - SQL injection prevention
   - Timeout limits

4. **Result Interpreter**
   - Format query results
   - Generate natural language summaries
   - Create visualizations (optional)

**Sample Database:** E-commerce (customers, orders, products, reviews)

**Example Queries:**
- "Show me the top 5 customers by total spending"
- "What's the average order value by month?"
- "Which products have ratings above 4.5?"
- "Find customers who haven't ordered in 6 months"

**Deliverable:** Natural Language to SQL Agent

**Project Structure:**
```
projects/nl-to-sql-agent/
├── src/
│   ├── __init__.py
│   ├── schema_inspector.py
│   ├── query_generator.py
│   ├── query_validator.py
│   ├── result_formatter.py
│   └── sql_agent.py
├── data/
│   ├── ecommerce.db        # SQLite database
│   └── sample_data.sql     # Sample data
├── prompts/
│   ├── sql_generation.txt
│   └── few_shot_examples.json
├── tests/
├── ui/
│   └── streamlit_app.py
├── README.md
└── requirements.txt
```

---

### Day 20-21: Multi-API Research Agent (4 hrs)

**Topics:**
- Multi-tool orchestration
- Agent planning and reasoning
- Tool selection strategies
- Parallel tool execution
- Result aggregation
- Complex workflow design

**Research Agent Capabilities:**

1. **Web Search** (via API)
   - Search across multiple sources
   - Extract key information
   - Filter relevant results

2. **News Aggregation**
   - Fetch latest news
   - Filter by topic/date
   - Summarize articles

3. **Stock Data**
   - Get real-time quotes
   - Historical data
   - Company information

4. **Weather Data**
   - Current conditions
   - Forecasts
   - Historical weather

5. **Data Storage**
   - Save research results
   - Create reports
   - Export to various formats

**Use Case Example:**
*"Research the impact of recent weather patterns on agricultural stock prices in California"*

**Agent Workflow:**
1. Search for California weather news
2. Fetch weather data for agricultural regions
3. Get stock prices for agricultural companies
4. Find related news articles
5. Analyze correlations
6. Generate comprehensive report

**Deliverable:** Multi-API Research Agent

**Project Structure:**
```
projects/multi-api-research-agent/
├── src/
│   ├── __init__.py
│   ├── tools/
│   │   ├── web_search_tool.py
│   │   ├── news_tool.py
│   │   ├── stock_tool.py
│   │   ├── weather_tool.py
│   │   └── storage_tool.py
│   ├── agent/
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── research_agent.py
│   └── utils/
│       ├── report_generator.py
│       └── data_aggregator.py
├── config/
│   └── api_config.yaml
├── outputs/
│   └── reports/
├── tests/
├── examples/
│   └── research_examples.py
├── README.md
└── requirements.txt
```

---

## 🚀 Deliverables

### 1. Custom Tools Library
**Location:** `projects/custom-tools-library/`

**Features:**
- ✅ 5 production-ready tools
- ✅ Comprehensive error handling
- ✅ Input validation
- ✅ Type hints and documentation
- ✅ Unit tests for each tool
- ✅ Usage examples
- ✅ Integration with LangChain/OpenAI

**Quality Standards:**
- Type annotations on all functions
- Docstrings with examples
- Error messages that guide users
- Logging for debugging
- Configuration via environment variables

---

### 2. Natural Language to SQL Agent
**Location:** `projects/nl-to-sql-agent/`

**Features:**
- ✅ Convert natural language to SQL
- ✅ Support complex queries (joins, aggregations)
- ✅ Safety checks and validation
- ✅ Natural language result explanations
- ✅ Schema-aware query generation
- ✅ Web UI for interaction
- ✅ Query history and caching

**Supported Query Types:**
- Simple SELECT queries
- Multi-table JOINs
- Aggregations (SUM, AVG, COUNT, etc.)
- GROUP BY and HAVING
- Subqueries
- Date/time operations

**Tech Stack:**
- OpenAI GPT-4 for query generation
- SQLAlchemy for database interaction
- SQLGlot for query parsing/validation
- Streamlit for UI
- LangChain for orchestration

---

### 3. Multi-API Research Agent
**Location:** `projects/multi-api-research-agent/`

**Features:**
- ✅ Orchestrate multiple API calls
- ✅ Intelligent tool selection
- ✅ Parallel execution
- ✅ Result aggregation and synthesis
- ✅ Comprehensive report generation
- ✅ Caching for efficiency
- ✅ Cost tracking

**Research Capabilities:**
- Market research
- Competitive analysis
- Trend analysis
- News monitoring
- Data correlation studies

**Tech Stack:**
- LangChain Agents
- Multiple API integrations
- Async/await for parallel calls
- Jinja2 for report templates
- Rich for CLI output

---

## 📚 Resources

### Official Documentation
- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

### SQL & Databases
- [SQL Tutorial - W3Schools](https://www.w3schools.com/sql/)
- [SQLite Python Tutorial](https://docs.python.org/3/library/sqlite3.html)
- [PostgreSQL Python Tutorial](https://www.psycopg.org/docs/)
- "SQL for Data Analysis" - Cathy Tanimura

### API Integration
- [Requests Documentation](https://requests.readthedocs.io/)
- [REST API Best Practices](https://stackoverflow.blog/2020/03/02/best-practices-for-rest-api-design/)
- [API Authentication Methods](https://roadmap.sh/guides/api-authentication)

### Web Scraping
- [Web Scraping with Python](https://realpython.com/beautiful-soup-web-scraper-python/)
- [Scrapy Documentation](https://docs.scrapy.org/)
- [Selenium Python Docs](https://selenium-python.readthedocs.io/)

### Papers & Articles
- "Text-to-SQL: A Survey" - Recent NLP techniques
- "LangChain Agents Deep Dive"
- "Building Production-Ready AI Agents"
- "Multi-Agent Orchestration Patterns"

### Videos
- "Building Custom LangChain Tools" - YouTube
- "Text-to-SQL with GPT-4" - Tutorial
- "API Integration Best Practices"
- "Web Scraping Masterclass"

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
# Activate virtual environment
source ../../venv/bin/activate

# Core dependencies
pip install openai python-dotenv
pip install langchain langchain-openai langchain-community

# Database tools
pip install sqlalchemy psycopg2-binary

# API & Web tools
pip install requests beautifulsoup4 lxml
pip install selenium  # Optional: for dynamic content

# Email tools
pip install secure-smtplib

# Data processing
pip install pandas numpy

# CLI & UI
pip install rich streamlit

# Testing
pip install pytest pytest-asyncio

# Update requirements
pip freeze > ../../requirements.txt
```

### 2. Set Up Database
```bash
# Create SQLite database for testing
cd projects/nl-to-sql-agent/data

# Run setup script (you'll create this)
python setup_database.py
```

### 3. Update .env File
```bash
# Add to existing .env in project root
cat >> ../../.env << 'ENV'

# Week 3 API Keys
OPENWEATHER_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here

# Database (if using PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost/dbname

# Email (for email tool)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Other APIs
SERPAPI_KEY=your_serpapi_key
ENV
```

### 4. Get Required API Keys

**Free APIs:**
- **OpenWeather:** https://openweathermap.org/api
- **News API:** https://newsapi.org/
- **Alpha Vantage:** https://www.alphavantage.co/support/#api-key
- **GitHub:** https://github.com/settings/tokens

**Paid/Freemium:**
- **SerpAPI:** https://serpapi.com/ (Google search)

---

## 💡 Code Templates

### Custom Tool Template
```python
# tools/base_tool.py
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class BaseTool(BaseModel):
    """Base class for all custom tools"""
    
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _validate_input(self, **kwargs) -> Dict[str, Any]:
        """Validate tool input"""
        raise NotImplementedError
    
    def _execute(self, **kwargs) -> Any:
        """Execute tool logic"""
        raise NotImplementedError
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the tool with error handling"""
        try:
            validated_input = self._validate_input(**kwargs)
            result = self._execute(**validated_input)
            return {
                "success": True,
                "result": result,
                "error": None
            }
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {str(e)}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
```

### Database Tool Example
```python
# tools/database_tool.py
from typing import List, Dict, Any
import sqlite3
from pathlib import Path
from .base_tool import BaseTool

class DatabaseTool(BaseTool):
    """Tool for executing SQL queries"""
    
    name: str = "database_query"
    description: str = "Execute SQL queries on the database"
    db_path: str
    
    def _validate_input(self, query: str, **kwargs) -> Dict[str, Any]:
        """Validate SQL query"""
        # Basic SQL injection prevention
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise ValueError(f"Query contains forbidden keyword: {keyword}")
        
        return {"query": query}
    
    def _execute(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        results = [dict(row) for row in rows]
        
        conn.close()
        return results
    
    def get_schema(self) -> Dict[str, List[str]]:
        """Get database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            schema[table] = columns
        
        conn.close()
        return schema
```

### API Tool Example
```python
# tools/api_tool.py
import requests
from typing import Dict, Any, Optional
from .base_tool import BaseTool

class APITool(BaseTool):
    """Generic API integration tool"""
    
    name: str = "api_call"
    description: str = "Make HTTP API calls"
    base_url: str
    api_key: Optional[str] = None
    
    def _validate_input(
        self,
        endpoint: str,
        method: str = "GET",
        **kwargs
    ) -> Dict[str, Any]:
        """Validate API call parameters"""
        if method not in ["GET", "POST", "PUT", "DELETE"]:
            raise ValueError(f"Invalid HTTP method: {method}")
        
        return {
            "endpoint": endpoint,
            "method": method,
            "params": kwargs.get("params", {}),
            "data": kwargs.get("data", {})
        }
    
    def _execute(
        self,
        endpoint: str,
        method: str,
        params: Dict,
        data: Dict
    ) -> Any:
        """Execute API call"""
        url = f"{self.base_url}/{endpoint}"
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=data if data else None,
            headers=headers,
            timeout=30
        )
        
        response.raise_for_status()
        return response.json()
```

### SQL Agent Example
```python
# projects/nl-to-sql-agent/src/sql_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

class NLToSQLAgent:
    """Natural Language to SQL Agent"""
    
    def __init__(self, db_path: str):
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Create toolkit with database tools
        self.toolkit = SQLDatabaseToolkit(
            db=self.db,
            llm=self.llm
        )
        
        # Create agent
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type="openai-tools"
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query database using natural language"""
        try:
            result = self.agent.invoke({"input": question})
            return {
                "success": True,
                "answer": result["output"],
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "error": str(e)
            }
    
    def get_schema_info(self) -> str:
        """Get formatted schema information"""
        return self.db.get_table_info()
```

---

## ✅ Completion Checklist

### Day 15-16: Custom Tools
- [ ] Design tool architecture
- [ ] Implement Database Tool
  - [ ] Connection management
  - [ ] Query execution
  - [ ] Schema inspection
  - [ ] Error handling
- [ ] Implement API Tool
  - [ ] REST operations
  - [ ] Authentication
  - [ ] Rate limiting
  - [ ] Response parsing
- [ ] Implement File Tool
  - [ ] Read/write operations
  - [ ] Multiple formats (TXT, JSON, CSV)
  - [ ] Directory operations
- [ ] Implement Email Tool
  - [ ] SMTP setup
  - [ ] Send emails
  - [ ] Templates
  - [ ] Attachments
- [ ] Implement Web Scraper Tool
  - [ ] HTML parsing
  - [ ] Data extraction
  - [ ] Error handling
- [ ] Write unit tests for all tools
- [ ] Create usage documentation
- [ ] Commit tools library to GitHub

### Day 17-19: SQL Agent
- [ ] Set up sample e-commerce database
- [ ] Implement schema inspector
- [ ] Build query generator with prompts
- [ ] Add few-shot examples
- [ ] Implement query validator
- [ ] Add safety checks
- [ ] Build result formatter
- [ ] Create complete SQL agent
- [ ] Test with various queries
- [ ] Build Streamlit UI
- [ ] Write comprehensive README
- [ ] Commit project to GitHub

### Day 20-21: Research Agent
- [ ] Integrate 5 API tools
- [ ] Design agent planning logic
- [ ] Implement tool orchestration
- [ ] Add parallel execution
- [ ] Build result aggregator
- [ ] Create report generator
- [ ] Test complex research workflows
- [ ] Add caching mechanism
- [ ] Implement cost tracking
- [ ] Create usage examples
- [ ] Write documentation
- [ ] Commit project to GitHub

---

## 🎓 Key Concepts Mastered

- [ ] Custom tool design patterns
- [ ] Function calling with LLMs
- [ ] Database interaction from agents
- [ ] SQL query generation
- [ ] Query validation and safety
- [ ] API integration patterns
- [ ] Web scraping techniques
- [ ] Email automation
- [ ] Multi-tool orchestration
- [ ] Agent planning and reasoning
- [ ] Error handling strategies
- [ ] Production-ready tool development

---

## 📊 Time Tracking

| Day | Hours Spent | Topics Covered |
|-----|-------------|----------------|
| 15  |             |                |
| 16  |             |                |
| 17  |             |                |
| 18  |             |                |
| 19  |             |                |
| 20  |             |                |
| 21  |             |                |
| **Total** | **/ 12 hrs** |          |

---

## 🧪 Testing Examples

### Test Database Tool
```python
# tests/test_database_tool.py
import pytest
from tools.database_tool import DatabaseTool

def test_simple_query():
    tool = DatabaseTool(db_path="test.db")
    result = tool.run(query="SELECT * FROM users LIMIT 5")
    
    assert result["success"] == True
    assert len(result["result"]) <= 5

def test_dangerous_query():
    tool = DatabaseTool(db_path="test.db")
    result = tool.run(query="DROP TABLE users")
    
    assert result["success"] == False
    assert "forbidden" in result["error"].lower()
```

### Test SQL Agent
```python
# Test queries for SQL agent
test_cases = [
    {
        "question": "Show me the top 5 customers by total spending",
        "expected_tables": ["customers", "orders"]
    },
    {
        "question": "What's the average order value?",
        "expected_aggregation": "AVG"
    },
    {
        "question": "Which products are out of stock?",
        "expected_filter": "stock = 0"
    }
]

for test in test_cases:
    result = sql_agent.query(test["question"])
    print(f"Q: {test['question']}")
    print(f"A: {result['answer']}")
```

---

## 🚧 Challenges & Solutions

### Challenge 1: SQL Injection Prevention
**Problem:**
User input could contain malicious SQL

**Solution:**
- Whitelist allowed operations
- Use parameterized queries
- Validate queries before execution
- Run in read-only mode
- Add timeout limits

---

### Challenge 2: API Rate Limiting
**Problem:**
Multiple API calls exceed rate limits

**Solution:**
- Implement exponential backoff
- Add request caching
- Queue requests
- Use async/await for efficiency
- Monitor usage

---

### Challenge 3: Tool Reliability
**Problem:**
External services fail or return errors

**Solution:**
- Comprehensive error handling
- Retry logic with backoff
- Fallback mechanisms
- Detailed logging
- User-friendly error messages

---

## 📈 Tool Performance Metrics

| Tool | Success Rate | Avg Response Time | Error Rate |
|------|--------------|-------------------|------------|
| Database Tool | % | ms | % |
| API Tool | % | ms | % |
| File Tool | % | ms | % |
| Email Tool | % | ms | % |
| Scraper Tool | % | ms | % |

---

## 🔗 Project Links

- **GitHub Repository:** [Your repo link]
- **Tools Library:** [Link to tools]
- **SQL Agent Demo:** [If deployed]
- **Research Agent Demo:** [If deployed]
- **Documentation:** [Tool docs]

---

## 📞 Next Steps

After completing Week 3:
1. ✅ Push all tools and agents to GitHub
2. ✅ Test tools with edge cases
3. ✅ Update main README.md
4. 📖 Review agent orchestration patterns
5. 🎯 Prepare for Week 4: Multi-Agent Systems
6. 🚀 Optional: Package tools as a library
7. 📝 Optional: Write blog post on tool design

---

## 💡 Bonus Ideas

- Create a tool registry/marketplace
- Build a visual tool builder UI
- Add monitoring and analytics
- Create Docker containers for tools
- Build a tool testing framework
- Add authentication/authorization
- Create tool usage analytics
- Build a tool documentation generator
- Implement tool versioning
- Create tool composition patterns
EOF