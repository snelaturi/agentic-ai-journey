
# Week 1: LLM Fundamentals

**Duration:** 12 hours  
**Goal:** Master OpenAI API + Basic Agents

## 📋 Overview

This week focuses on building a strong foundation with Large Language Models (LLMs) and creating practical AI agents using the OpenAI API and LangChain framework.

## 🎯 Learning Objectives

- Understand OpenAI API fundamentals
- Master prompt engineering techniques
- Implement conversational memory in chatbots
- Build agents with function calling capabilities
- Learn LangChain basics for agent orchestration
- Deploy agents with real-world integrations

## 📅 Daily Breakdown

### Day 1-2: OpenAI API & Conversational Agents (4 hrs)
**Topics:**
- OpenAI API setup and authentication
- Chat completions API
- System vs user vs assistant messages
- Temperature and token management
- Conversation history and memory patterns
- Streaming responses

**Hands-on:**
- Set up OpenAI account and API key
- Build basic chatbot
- Implement conversation memory
- Add conversation summarization

**Deliverable:** Chatbot with Memory Agent

---

### Day 3-4: Function Calling & Tool Use (4 hrs)
**Topics:**
- Function calling fundamentals
- Defining function schemas
- Handling function calls and responses
- Error handling and validation
- Multi-function agents

**Hands-on:**
- Create weather lookup function
- Build calculator functions
- Integrate external APIs
- Handle edge cases

**Deliverable:** Weather/Calculator Agent

---

### Day 5-7: LangChain & Production Agent (4 hrs)
**Topics:**
- LangChain framework introduction
- Chains, agents, and tools
- Memory management in LangChain
- API integrations (News API)
- Sentiment analysis basics
- Agent deployment considerations

**Hands-on:**
- Set up LangChain
- Build news fetching tool
- Implement sentiment analysis
- Create financial news agent

**Deliverable:** Financial News Sentiment Agent

---

## 🚀 Deliverables

### 1. Chatbot with Memory Agent
**Location:** `exercises/chatbot_with_memory.py`

**Features:**
- Maintains conversation context
- Remembers user preferences
- Summarizes long conversations
- Graceful error handling

**Technologies:** OpenAI API, python-dotenv

---

### 2. Weather/Calculator Agent
**Location:** `exercises/weather_calculator_agent.py`

**Features:**
- Get current weather for any city
- Perform mathematical calculations
- Handle multi-step requests
- Function calling with validation

**Technologies:** OpenAI Function Calling, Weather API

---

### 3. Financial News Sentiment Agent
**Location:** `projects/financial-news-agent/`

**Features:**
- Fetch latest financial news
- Analyze sentiment (positive/negative/neutral)
- Generate investment insights
- Filter by stock symbols or topics

**Technologies:** LangChain, OpenAI, News API, Sentiment Analysis

---

## 📚 Resources

### Official Documentation
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

### Tutorials
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)

### Articles
- "A Complete Guide to Function Calling in OpenAI"
- "Building Conversational AI with Memory"
- "LangChain for Beginners"

### Videos
- [OpenAI API Crash Course](https://www.youtube.com/)
- [Function Calling Deep Dive](https://www.youtube.com/)
- [LangChain in 30 Minutes](https://www.youtube.com/)

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
# Activate virtual environment
source ../../venv/bin/activate

# Install required packages
pip install openai python-dotenv langchain langchain-openai requests
pip install newsapi-python textblob

# Update requirements
pip freeze > ../../requirements.txt
```

### 2. Set Up API Keys
Create `.env` file in project root:
```bash
OPENAI_API_KEY=sk-your-key-here
WEATHER_API_KEY=your-weather-api-key
NEWS_API_KEY=your-news-api-key
```

### 3. Get API Keys
- **OpenAI:** https://platform.openai.com/api-keys
- **Weather API:** https://openweathermap.org/api
- **News API:** https://newsapi.org/

---

## ✅ Completion Checklist

### Day 1-2: OpenAI API Basics
- [ ] Set up OpenAI API account
- [ ] Understand API authentication
- [ ] Make first API call
- [ ] Implement basic chatbot
- [ ] Add conversation memory
- [ ] Test with different prompts
- [ ] Handle errors gracefully
- [ ] Commit chatbot to GitHub

### Day 3-4: Function Calling
- [ ] Understand function calling concepts
- [ ] Define function schemas
- [ ] Implement weather function
- [ ] Implement calculator functions
- [ ] Test multi-function calls
- [ ] Add input validation
- [ ] Handle API errors
- [ ] Commit agent to GitHub

### Day 5-7: LangChain Agent
- [ ] Install LangChain
- [ ] Understand chains and agents
- [ ] Set up News API
- [ ] Create news fetching tool
- [ ] Implement sentiment analysis
- [ ] Build complete agent
- [ ] Test with real queries
- [ ] Create project documentation
- [ ] Commit to GitHub

---

## 🎓 Key Concepts Mastered

- [ ] OpenAI Chat Completions API
- [ ] Prompt engineering best practices
- [ ] Conversation memory patterns
- [ ] Function calling and tool use
- [ ] LangChain framework basics
- [ ] API integration techniques
- [ ] Error handling strategies
- [ ] Agent deployment considerations

---

## 🔗 Project Links

- **GitHub Repository:** [Link to your repo]
- **Chatbot Demo:** [If deployed]
- **Weather Agent Demo:** [If deployed]
- **Financial News Agent:** [If deployed]

---

## 📊 Time Tracking

| Day | Hours Spent | Topics Covered |
|-----|-------------|----------------|
| 1   |             |                |
| 2   |             |                |
| 3   |             |                |
| 4   |             |                |
| 5   |             |                |
| 6   |             |                |
| 7   |             |                |
| **Total** | **/ 12 hrs** |          |


---

## 📞 Next Steps

After completing Week 1:
1. Push all three agents to GitHub
2. Update main README.md progress tracker
3. Review and refactor code
4. Prepare for Week 2: Advanced Agent Patterns
5. Optional: Deploy one agent to a web interface

---