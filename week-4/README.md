# Week 4: Agent Patterns & Architectures

**Duration:** 14 hours  
**Goal:** Master agent reasoning patterns

## 📋 Overview

This week dives deep into advanced agent reasoning patterns that enable AI systems to think, plan, and improve iteratively. You'll implement ReAct (Reasoning + Acting), planning agents that decompose complex tasks, and self-reflective agents that critique and improve their own outputs.

## 🎯 Learning Objectives

- Understand and implement the ReAct pattern
- Master task decomposition and planning
- Build agents that can self-critique and improve
- Implement agent memory systems
- Learn iterative refinement patterns
- Study AutoGPT and BabyAGI architectures
- Apply advanced prompting strategies
- Build production-ready reasoning agents

## 📅 Daily Breakdown

### Day 22-23: ReAct Pattern (4 hrs)

**The ReAct Framework:**
ReAct (Reasoning + Acting) interleaves reasoning traces and task-specific actions, enabling agents to dynamically reason, plan, and act while maintaining interpretability.

#### Study Phase (1.5 hrs)

**Core Concepts:**

1. **Thought-Action-Observation Loop**
````
   Thought: [Agent's reasoning about next step]
   Action: [Specific action to take]
   Observation: [Result from action]
   → Repeat until task complete
````

2. **Why ReAct Works:**
   - Explicit reasoning improves decision quality
   - Actions ground reasoning in reality
   - Observations provide feedback
   - Transparent thought process

3. **ReAct vs Standard Prompting:**
   - **Standard:** Direct question → Answer
   - **ReAct:** Question → Think → Act → Observe → Think → Answer

**Required Reading:**
- 📄 [ReAct Paper](https://arxiv.org/abs/2210.03629) - "ReAct: Synergizing Reasoning and Acting in Language Models"
- 📖 [LangChain ReAct Agent](https://python.langchain.com/docs/modules/agents/agent_types/react)

**Key Paper Insights:**
- ReAct outperforms chain-of-thought on complex tasks
- Combination of reasoning and acting is crucial
- Few-shot examples significantly improve performance
- Self-correction emerges naturally

---

#### Implementation Phase (2.5 hrs)

**Build ReAct Agent from Scratch**

**Components to Implement:**

1. **Reasoning Loop**
````python
   while not task_complete:
       thought = generate_thought(state)
       action = select_action(thought)
       observation = execute_action(action)
       state = update_state(observation)
````

2. **Action Selection**
   - Parse LLM output for actions
   - Map to available tools
   - Handle invalid actions
   - Support action parameters

3. **Observation Processing**
   - Execute tool calls
   - Format results
   - Handle errors
   - Update agent context

4. **Thought Generation**
   - Analyze current state
   - Consider available actions
   - Plan next step
   - Self-critique if needed

**Available Actions for Demo:**
- `search(query)` - Web search
- `calculate(expression)` - Math operations
- `get_current_date()` - Current date/time
- `finish(answer)` - Complete task

**Example ReAct Trace:**
````
Question: What is the capital of the country where the 2024 Olympics were held?

Thought: I need to find out where the 2024 Olympics were held first.
Action: search("2024 Olympics host country")
Observation: The 2024 Summer Olympics were held in Paris, France.

Thought: Now I know the Olympics were in France. The capital of France is Paris, which is also the host city.
Action: finish("Paris")
````

**Comparison Study:**
Build two versions:
1. **With ReAct:** Full thought-action-observation loop
2. **Without ReAct:** Direct prompting

**Metrics to Compare:**
- Success rate on complex questions
- Number of tool calls
- Response accuracy
- Interpretability

**Test Cases:**
````python
test_cases = [
    "What is 15% of the GDP of the country with the largest population?",
    "How many days until the next leap year?",
    "What was the temperature in New York when the last iPhone was released?",
    "Compare the population density of the top 3 most populous cities"
]
````

**Deliverable:** `projects/react-agent-implementation/`

**Project Structure:**
````
projects/react-agent-implementation/
├── src/
│   ├── __init__.py
│   ├── react_agent.py          # Main ReAct agent
│   ├── standard_agent.py       # Baseline for comparison
│   ├── thought_generator.py    # Reasoning logic
│   ├── action_parser.py        # Parse actions from LLM
│   ├── tools.py                # Available actions
│   └── executor.py             # Execute actions
├── prompts/
│   ├── react_prompt.txt        # ReAct system prompt
│   ├── few_shot_examples.json  # Example traces
│   └── standard_prompt.txt     # Baseline prompt
├── experiments/
│   ├── run_comparison.py       # Compare both agents
│   └── analysis.ipynb          # Results analysis
├── tests/
│   └── test_react_agent.py
├── blog/
│   └── react-deep-dive.md      # Blog post
├── README.md
└── requirements.txt
````

**Blog Post Requirements:**
- Explain ReAct pattern in simple terms
- Show example traces with explanations
- Compare performance with/without ReAct
- Discuss when to use ReAct
- Include code snippets
- Share lessons learned

---

### Day 24-25: Planning Agents (4 hrs)

**Planning Agent Architecture:**
````
High-Level Goal
    ↓
[Task Decomposition]
    ↓
Subtasks: [Task1, Task2, Task3, ...]
    ↓
[Sequential Execution]
    ↓
    For each subtask:
        - Execute
        - Validate result
        - Self-correct if needed
        - Update plan if required
    ↓
Final Result
````

#### Study Phase (1 hr)

**Core Concepts:**

1. **Plan-and-Execute Pattern**
   - Planning phase: Decompose goal into steps
   - Execution phase: Execute steps sequentially
   - Monitoring: Track progress and adapt

2. **Task Decomposition Strategies**
   - Hierarchical breakdown
   - Dependency analysis
   - Resource estimation
   - Risk assessment

3. **Self-Correction Mechanisms**
   - Validate outputs
   - Detect errors
   - Adjust plan dynamically
   - Learn from failures

**Required Reading:**
- 📄 "Plan-and-Execute Agents" - LangChain docs
- 📖 "Task Decomposition for LLMs" - Research papers
- 🎥 "Building Planning Agents" - Tutorial videos

---

#### Build Phase (3 hrs)

**Project Planning Agent**

**Goal:** Take a high-level software development goal and break it into executable subtasks.

**Example Input:**
````
"Build a REST API for user management"
````

**Expected Output (Plan):**
````
1. Design database schema
   - Users table (id, email, password_hash, created_at)
   - Sessions table for authentication
   
2. Set up project structure
   - Create Flask/FastAPI app
   - Configure database connection
   - Set up environment variables
   
3. Implement user registration
   - Create registration endpoint
   - Add email validation
   - Hash passwords
   - Store in database
   
4. Implement authentication
   - Create login endpoint
   - Generate JWT tokens
   - Add token validation middleware
   
5. Implement CRUD operations
   - GET /users (list users)
   - GET /users/{id} (get user)
   - PUT /users/{id} (update user)
   - DELETE /users/{id} (delete user)
   
6. Add error handling
   - Validation errors
   - Authentication errors
   - Database errors
   
7. Write tests
   - Unit tests for each endpoint
   - Integration tests
   - Authentication tests
   
8. Create documentation
   - API documentation (OpenAPI/Swagger)
   - Setup instructions
   - Usage examples
````

**Agent Capabilities:**

1. **Task Decomposition**
   - Break goal into logical steps
   - Identify dependencies
   - Order tasks appropriately
   - Estimate complexity

2. **Execution**
   - Execute each subtask
   - Generate code/documentation
   - Run validation checks
   - Report progress

3. **Self-Correction**
   - Detect errors in outputs
   - Retry with modifications
   - Adjust plan if needed
   - Learn from mistakes

4. **Validation**
   - Check if subtask completed correctly
   - Verify dependencies met
   - Ensure quality standards
   - Gate progress on validation

**Implementation Details:**
````python
class PlanningAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.planner = create_planner()
        self.executor = create_executor()
        self.validator = create_validator()
    
    def process_goal(self, goal: str) -> PlanResult:
        # Phase 1: Planning
        plan = self.planner.decompose(goal)
        
        # Phase 2: Execution
        results = []
        for task in plan.tasks:
            result = self.execute_task(task)
            
            # Validate and self-correct
            if not self.validator.validate(result):
                result = self.self_correct(task, result)
            
            results.append(result)
            
            # Update plan if needed
            if result.requires_plan_update:
                plan = self.planner.replan(goal, results)
        
        return PlanResult(plan=plan, results=results)
    
    def execute_task(self, task: Task) -> TaskResult:
        # Execute with error handling
        try:
            output = self.executor.execute(task)
            return TaskResult(success=True, output=output)
        except Exception as e:
            return TaskResult(success=False, error=str(e))
    
    def self_correct(self, task: Task, result: TaskResult) -> TaskResult:
        # Analyze failure
        analysis = self.llm.analyze_error(task, result)
        
        # Retry with corrections
        corrected_task = self.llm.modify_task(task, analysis)
        return self.execute_task(corrected_task)
````

**Test Scenarios:**
````python
test_goals = [
    "Build a REST API for user management",
    "Create a data pipeline to analyze customer behavior",
    "Implement a recommendation system for e-commerce",
    "Build a chatbot for customer support"
]
````

**Deliverable:** `projects/planning-agent/`

**Project Structure:**
````
projects/planning-agent/
├── src/
│   ├── __init__.py
│   ├── planning_agent.py       # Main agent
│   ├── planner.py              # Task decomposition
│   ├── executor.py             # Task execution
│   ├── validator.py            # Output validation
│   └── self_corrector.py       # Error correction
├── prompts/
│   ├── decomposition.txt       # Planning prompts
│   ├── execution.txt           # Execution prompts
│   └── validation.txt          # Validation prompts
├── examples/
│   ├── rest_api_plan.py        # Example runs
│   ├── data_pipeline_plan.py
│   └── chatbot_plan.py
├── tests/
│   └── test_planning_agent.py
├── ui/
│   └── streamlit_app.py        # Interactive UI
├── README.md
└── requirements.txt
````

---

### Day 26-28: Advanced Agent Patterns (6 hrs)

#### Study Phase (1 hr)

**Advanced Patterns:**

1. **Reflexion Pattern**
   - Agent reflects on its outputs
   - Generates self-critique
   - Iteratively improves
   - Learns from mistakes

2. **Self-Critique Loop**
````
   Generate Output
       ↓
   Self-Critique (find issues)
       ↓
   Improve Output
       ↓
   Repeat until quality threshold met
````

3. **Iterative Improvement**
   - Multiple refinement passes
   - Progressive enhancement
   - Quality metrics
   - Stopping criteria

**Required Reading:**
- 📄 [Reflexion Paper](https://arxiv.org/abs/2303.11366) - "Reflexion: Language Agents with Verbal Reinforcement Learning"
- 📖 "Self-Critique for LLMs"
- 🎥 "Building Self-Improving Agents"

---

#### Build: Code Review Agent (3 hrs)

**Agent Capabilities:**

1. **Code Analysis**
   - Syntax and style issues
   - Performance problems
   - Security vulnerabilities
   - Best practice violations
   - Code complexity

2. **Improvement Suggestions**
   - Specific, actionable feedback
   - Prioritized by importance
   - With code examples
   - Explanation of benefits

3. **Test Case Generation**
   - Unit tests
   - Edge cases
   - Integration tests
   - Test data

4. **Iterative Refactoring**
   - Apply suggestions
   - Re-analyze code
   - Repeat until quality threshold
   - Track improvements

**Example Workflow:**
````python
# Input code
code = """
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price'] * item['quantity']
    return total
"""

# Agent output (Iteration 1)
"""
Issues Found:
1. No input validation - what if items is None or not a list?
2. No error handling - what if 'price' or 'quantity' keys are missing?
3. Could use more Pythonic approach with sum() and list comprehension
4. No type hints
5. Missing docstring

Suggested Refactoring:
"""

refactored_code = """
from typing import List, Dict, Union

def calculate_total(items: List[Dict[str, Union[int, float]]]) -> float:
    \"\"\"
    Calculate total cost from list of items.
    
    Args:
        items: List of dicts with 'price' and 'quantity' keys
        
    Returns:
        Total cost as float
        
    Raises:
        ValueError: If items is invalid or missing required keys
    \"\"\"
    if not items:
        return 0.0
    
    try:
        return sum(
            item['price'] * item['quantity'] 
            for item in items
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid item format: {e}")
"""

# Test cases generated
"""
def test_calculate_total():
    # Normal case
    assert calculate_total([
        {'price': 10, 'quantity': 2},
        {'price': 5, 'quantity': 3}
    ]) == 35
    
    # Empty list
    assert calculate_total([]) == 0
    
    # None input
    with pytest.raises(ValueError):
        calculate_total(None)
    
    # Missing keys
    with pytest.raises(ValueError):
        calculate_total([{'price': 10}])
"""
````

**Iterative Refinement Process:**
````python
class CodeReviewAgent:
    def __init__(self):
        self.max_iterations = 3
        self.quality_threshold = 8.0  # out of 10
    
    def review_and_improve(self, code: str) -> RefactoredCode:
        current_code = code
        iteration = 0
        
        while iteration < self.max_iterations:
            # Analyze current code
            issues = self.analyze(current_code)
            quality_score = self.calculate_quality(issues)
            
            if quality_score >= self.quality_threshold:
                break
            
            # Generate improvements
            suggestions = self.generate_suggestions(issues)
            
            # Apply refactoring
            current_code = self.refactor(current_code, suggestions)
            
            # Generate tests
            tests = self.generate_tests(current_code)
            
            iteration += 1
        
        return RefactoredCode(
            code=current_code,
            tests=tests,
            iterations=iteration,
            final_quality_score=quality_score
        )
````

**Supported Languages:**
- Python
- Java
- JavaScript/TypeScript (bonus)

**Deliverable:** `projects/code-review-agent/`

**Project Structure:**
````
projects/code-review-agent/
├── src/
│   ├── __init__.py
│   ├── code_review_agent.py    # Main agent
│   ├── analyzers/
│   │   ├── python_analyzer.py
│   │   └── java_analyzer.py
│   ├── refactorer.py           # Code refactoring
│   ├── test_generator.py       # Generate tests
│   └── quality_scorer.py       # Calculate quality
├── prompts/
│   ├── analysis.txt
│   ├── suggestions.txt
│   ├── refactoring.txt
│   └── test_generation.txt
├── examples/
│   ├── python_examples/
│   │   ├── example1_before.py
│   │   ├── example1_after.py
│   │   └── example1_tests.py
│   └── java_examples/
├── tests/
│   └── test_code_review_agent.py
├── ui/
│   └── streamlit_app.py
├── README.md
└── requirements.txt
````

---

#### Agent Memory and Learning (2 hrs)

**Implement Memory System:**

1. **Short-Term Memory**
   - Current conversation context
   - Recent actions and observations
   - Working memory for current task

2. **Long-Term Memory**
   - Vector database of past experiences
   - Successful patterns
   - Failed approaches to avoid
   - User preferences

3. **Learning Mechanism**
   - Store successful strategies
   - Retrieve relevant past experiences
   - Apply learned patterns to new problems
   - Improve over time

**Implementation:**
````python
from langchain.memory import ConversationBufferMemory
from chromadb import Client

class AgentMemory:
    def __init__(self):
        # Short-term memory
        self.conversation_memory = ConversationBufferMemory(
            return_messages=True
        )
        
        # Long-term memory
        self.vector_db = Client()
        self.experience_collection = self.vector_db.create_collection(
            name="agent_experiences"
        )
    
    def store_experience(
        self,
        task: str,
        approach: str,
        result: str,
        success: bool
    ):
        """Store experience in long-term memory"""
        self.experience_collection.add(
            documents=[approach],
            metadatas=[{
                "task": task,
                "result": result,
                "success": success
            }],
            ids=[f"exp_{uuid.uuid4()}"]
        )
    
    def retrieve_similar_experiences(
        self,
        current_task: str,
        n_results: int = 3
    ) -> List[Experience]:
        """Retrieve similar past experiences"""
        results = self.experience_collection.query(
            query_texts=[current_task],
            n_results=n_results
        )
        return self._format_experiences(results)
    
    def learn_from_feedback(self, feedback: str):
        """Update strategies based on feedback"""
        # Extract lessons from feedback
        lessons = self.extract_lessons(feedback)
        
        # Store in memory
        for lesson in lessons:
            self.store_experience(
                task=lesson.context,
                approach=lesson.what_worked,
                result=lesson.outcome,
                success=True
            )
````

---

### Weekend Exploration

#### Study Advanced Architectures

**1. AutoGPT Architecture**
- Autonomous agent that sets and pursues goals
- Web browsing and information gathering
- Memory management
- File operations
- Code execution

**Key Components:**
````
AutoGPT
├── Goal setting
├── Task prioritization
├── Long-term memory
├── Internet access
├── File management
└── Self-improvement
````

**2. BabyAGI Architecture**
- Task-driven autonomous agent
- Creates, prioritizes, and executes tasks
- Uses OpenAI and vector databases
- Iterative task management

**Key Components:**
````
BabyAGI
├── Task creation agent
├── Task prioritization agent
├── Execution agent
└── Context retrieval (vector DB)
````

**Resources:**
- 🔗 [AutoGPT GitHub](https://github.com/Significant-Gravitas/AutoGPT)
- 🔗 [BabyAGI GitHub](https://github.com/yoheinakajima/babyagi)
- 📄 "Autonomous Agents Survey" - Papers

---

#### Experiment with Prompting Strategies

**Advanced Techniques:**

1. **Chain-of-Thought (CoT)**
````
   Let's think step by step:
   1. First, we need to...
   2. Then, we should...
   3. Finally, we can...
````

2. **Tree of Thoughts (ToT)**
````
   Explore multiple reasoning paths:
   Path 1: ...
   Path 2: ...
   Path 3: ...
   Evaluate and choose best path
````

3. **Self-Consistency**
````
   Generate multiple solutions
   Vote on most common answer
   Use ensemble approach
````

4. **ReAct + Chain-of-Thought**
````
   Thought: [Step-by-step reasoning]
   Action: [Based on reasoning]
   Observation: [Result]
````

**Experiments to Run:**
````python
prompting_experiments = [
    {
        "name": "Standard Prompting",
        "technique": "Direct question",
        "test_cases": math_problems
    },
    {
        "name": "Chain-of-Thought",
        "technique": "Step-by-step reasoning",
        "test_cases": math_problems
    },
    {
        "name": "ReAct",
        "technique": "Thought-Action-Observation",
        "test_cases": complex_tasks
    },
    {
        "name": "Self-Consistency",
        "technique": "Multiple generations + voting",
        "test_cases": ambiguous_questions
    }
]
````

---

#### Document Patterns Learned

**Create Pattern Library:**
````markdown
# Agent Design Patterns

## 1. ReAct Pattern
**When to use:** Complex tasks requiring multiple steps
**Structure:** Thought → Action → Observation loop
**Benefits:** Transparency, self-correction
**Drawbacks:** More API calls, slower

## 2. Plan-and-Execute
**When to use:** Large projects with clear subtasks
**Structure:** Planning phase + Execution phase
**Benefits:** Organized, scalable
**Drawbacks:** Rigid, hard to adapt mid-execution

## 3. Reflexion Pattern
**When to use:** Quality-critical outputs
**Structure:** Generate → Critique → Improve loop
**Benefits:** High-quality outputs, self-improvement
**Drawbacks:** Time-consuming, expensive

## 4. Autonomous Agent
**When to use:** Long-running, goal-oriented tasks
**Structure:** Goal → Tasks → Execute → Adapt
**Benefits:** Autonomous, adaptive
**Drawbacks:** Can go off-track, hard to control
````

---

## 🚀 Deliverables

### 1. ReAct Agent Implementation
**Location:** `projects/react-agent-implementation/`

**Features:**
- ✅ Full ReAct loop implementation
- ✅ Comparison with standard prompting
- ✅ Multiple tool integrations
- ✅ Comprehensive examples
- ✅ Performance benchmarks
- ✅ Blog post explaining ReAct

**Tech Stack:** OpenAI GPT-4, Custom tools, LangChain (optional)

---

### 2. Planning Agent
**Location:** `projects/planning-agent/`

**Features:**
- ✅ Task decomposition from high-level goals
- ✅ Sequential execution with validation
- ✅ Self-correction on errors
- ✅ Dynamic plan adjustment
- ✅ Progress tracking
- ✅ Interactive UI

**Use Cases:**
- Software development planning
- Data pipeline design
- Project management
- Workflow automation

---

### 3. Code Review Agent
**Location:** `projects/code-review-agent/`

**Features:**
- ✅ Multi-language support (Python, Java)
- ✅ Code analysis and critique
- ✅ Refactoring suggestions
- ✅ Test case generation
- ✅ Iterative improvement
- ✅ Quality scoring
- ✅ Memory of past reviews

**Quality Checks:**
- Style and syntax
- Performance
- Security
- Best practices
- Test coverage
- Documentation

---

### 4. Pattern Documentation
**Location:** `week-4/patterns/`

**Contents:**
- ReAct pattern guide
- Planning pattern guide
- Reflexion pattern guide
- Comparison matrix
- When to use each pattern
- Implementation tips
- Common pitfalls

---

## 📚 Resources

### Papers (Must Read)
- 📄 [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- 📄 [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- 📄 [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- 📄 [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

### Documentation
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LangChain ReAct](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

### GitHub Repositories
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [BabyAGI](https://github.com/yoheinakajima/babyagi)
- [LangChain Examples](https://github.com/langchain-ai/langchain/tree/master/templates)

### Tutorials & Courses
- "Building ReAct Agents" - DeepLearning.AI
- "Advanced Agent Patterns" - LangChain tutorials
- "Prompt Engineering Guide" - OpenAI

### Videos
- "ReAct Explained" - YouTube
- "Building Autonomous Agents" - Conference talks
- "Agent Design Patterns" - Technical workshops

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
````bash
# Activate virtual environment
source ../../venv/bin/activate

# Install core dependencies
pip install openai python-dotenv
pip install langchain langchain-openai langchain-community

# Code analysis tools
pip install pylint black flake8 mypy
pip install radon  # Code complexity

# For Java analysis (optional)
pip install javalang

# UI and visualization
pip install streamlit plotly

# Testing
pip install pytest pytest-cov

# Update requirements
pip freeze > ../../requirements.txt
````

### 2. Environment Variables
````bash
# Already in .env, but verify:
OPENAI_API_KEY=sk-your-key-here

# Add if needed:
ANTHROPIC_API_KEY=your-anthropic-key
````

---

## 💡 Code Templates

### ReAct Agent Template
````python
# projects/react-agent-implementation/src/react_agent.py
from typing import List, Dict, Any
from openai import OpenAI
import json

class ReActAgent:
    """ReAct agent implementation"""
    
    def __init__(self, tools: List[callable]):
        self.client = OpenAI()
        self.tools = {tool.__name__: tool for tool in tools}
        self.max_iterations = 10
        
    def run(self, question: str) -> str:
        """Run ReAct loop"""
        context = f"Question: {question}\n\n"
        
        for i in range(self.max_iterations):
            # Generate thought and action
            response = self._generate_step(context)
            
            # Parse response
            thought = self._extract_thought(response)
            action = self._extract_action(response)
            
            context += f"Thought: {thought}\n"
            context += f"Action: {action}\n"
            
            # Check if finished
            if action.startswith("finish"):
                answer = self._extract_answer(action)
                return answer
            
            # Execute action
            observation = self._execute_action(action)
            context += f"Observation: {observation}\n\n"
        
        return "Max iterations reached without answer"
    
    def _generate_step(self, context: str) -> str:
        """Generate next thought and action"""
        prompt = self._build_prompt(context)
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def _execute_action(self, action: str) -> str:
        """Execute tool call"""
        try:
            # Parse action: tool_name(args)
            tool_name, args = self._parse_action(action)
            
            if tool_name in self.tools:
                result = self.tools[tool_name](*args)
                return str(result)
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing action: {str(e)}"
    
    @property
    def system_prompt(self) -> str:
        return """You are a helpful agent that uses the ReAct pattern.

Available actions:
- search(query) - Search the web
- calculate(expression) - Perform calculations
- get_current_date() - Get current date
- finish(answer) - Return final answer

Format your response as:
Thought: [your reasoning about what to do next]
Action: [action to take]

Always think step-by-step and use actions to gather information."""
````

### Planning Agent Template
````python
# projects/planning-agent/src/planning_agent.py
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Task:
    id: int
    description: str
    dependencies: List[int]
    status: str = "pending"
    result: Any = None

class PlanningAgent:
    """Agent that decomposes and executes plans"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
    def process_goal(self, goal: str) -> Dict[str, Any]:
        """Process high-level goal"""
        # Phase 1: Decompose into tasks
        tasks = self.decompose(goal)
        
        # Phase 2: Execute tasks
        results = []
        for task in tasks:
            result = self.execute_task(task)
            results.append(result)
            
            # Validate and correct if needed
            if not self.validate(result):
                result = self.correct(task, result)
        
        return {
            "goal": goal,
            "plan": tasks,
            "results": results,
            "success": all(r.success for r in results)
        }
    
    def decompose(self, goal: str) -> List[Task]:
        """Decompose goal into subtasks"""
        prompt = f"""Break down this goal into specific, actionable subtasks:

Goal: {goal}

Provide a step-by-step plan with clear dependencies.
Format as JSON list of tasks."""
        
        response = self.llm.invoke(prompt)
        tasks = self._parse_tasks(response.content)
        return tasks
    
    def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task"""
        prompt = f"""Execute this task:

Task: {task.description}

Provide detailed implementation or output."""
        
        response = self.llm.invoke(prompt)
        return TaskResult(
            task_id=task.id,
            output=response.content,
            success=True
        )
````

### Code Review Agent Template
````python
# projects/code-review-agent/src/code_review_agent.py
class CodeReviewAgent:
    """Agent that reviews and improves code"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.max_iterations = 3
        
    def review(self, code: str, language: str = "python") -> ReviewResult:
        """Review code with iterative improvement"""
        current_code = code
        iterations = []
        
        for i in range(self.max_iterations):
            # Analyze code
            issues = self.analyze(current_code, language)
            
            # Calculate quality score
            score = self.calculate_quality(issues)
            
            iterations.append({
                "iteration": i + 1,
                "issues": issues,
                "score": score,
                "code": current_code
            })
            
            # Stop if quality is good enough
            if score >= 8.0:
                break
            
            # Improve code
            current_code = self.refactor(current_code, issues)
        
        # Generate tests
        tests = self.generate_tests(current_code)
        
        return ReviewResult(
            original_code=code,
            final_code=current_code,
            tests=tests,
            iterations=iterations,
            final_score=score
        )
    
    def analyze(self, code: str, language: str) -> List[Issue]:
        """Analyze code for issues"""
        prompt = f"""Analyze this {language} code and identify issues:
```{language}
{code}
```

Provide:
1. Style and syntax issues
2. Performance problems
3. Security vulnerabilities
4. Best practice violations
5. Missing error handling
6. Missing documentation

Format as structured list."""
        
        response = self.llm.invoke(prompt)
        return self._parse_issues(response.content)
    
    def refactor(self, code: str, issues: List[Issue]) -> str:
        """Refactor code based on issues"""
        prompt = f"""Refactor this code to fix these issues:

Original code:
```python
{code}
```

Issues to fix:
{self._format_issues(issues)}

Provide improved code with all issues addressed."""
        
        response = self.llm.invoke(prompt)
        return self._extract_code(response.content)
````

---

## ✅ Completion Checklist

### Day 22-23: ReAct Pattern
- [ ] Read ReAct paper
- [ ] Understand thought-action-observation loop
- [ ] Design ReAct agent architecture
- [ ] Implement reasoning loop
- [ ] Implement action parsing
- [ ] Implement observation processing
- [ ] Create 4+ tools for agent
- [ ] Build standard agent for comparison
- [ ] Run experiments on test cases
- [ ] Analyze performance differences
- [ ] Write blog post on ReAct
- [ ] Create examples and demos
- [ ] Commit to GitHub

### Day 24-25: Planning Agent
- [ ] Study plan-and-execute pattern
- [ ] Understand task decomposition
- [ ] Design planning agent architecture
- [ ] Implement task decomposer
- [ ] Implement task executor
- [ ] Add validation logic
- [ ] Implement self-correction
- [ ] Test with software development goals
- [ ] Build interactive UI
- [ ] Create comprehensive examples
- [ ] Write documentation
- [ ] Commit to GitHub

### Day 26-28: Code Review Agent
- [ ] Study Reflexion paper
- [ ] Understand self-critique patterns
- [ ] Design code review agent
- [ ] Implement code analyzer
- [ ] Implement refactoring logic
- [ ] Add test generation
- [ ] Implement iterative improvement
- [ ] Add quality scoring
- [ ] Integrate memory system
- [ ] Test with Python and Java code
- [ ] Create UI for code review
- [ ] Document patterns learned
- [ ] Commit to GitHub

### Weekend Exploration
- [ ] Study AutoGPT architecture
- [ ] Study BabyAGI architecture
- [ ] Experiment with CoT prompting
- [ ] Experiment with ToT prompting
- [ ] Try self-consistency approach
- [ ] Document all patterns
- [ ] Create pattern comparison matrix
- [ ] Write lessons learned

---

## 🎓 Key Concepts Mastered

- [ ] ReAct pattern (Reasoning + Acting)
- [ ] Thought-Action-Observation loops
- [ ] Plan-and-Execute pattern
- [ ] Task decomposition strategies
- [ ] Self-correction mechanisms
- [ ] Reflexion and self-critique
- [ ] Iterative improvement
- [ ] Agent memory systems
- [ ] Long-term learning
- [ ] Autonomous agent architectures
- [ ] Advanced prompting strategies
- [ ] Agent orchestration patterns

---

## 📊 Time Tracking

| Day | Hours Spent | Topics Covered |
|-----|-------------|----------------|
| 22  |             |                |
| 23  |             |                |
| 24  |             |                |
| 25  |             |                |
| 26  |             |                |
| 27  |             |                |
| 28  |             |                |
| Weekend |         |                |
| **Total** | **/ 14 hrs** |          |

---

## 🧪 Experiments & Results

### ReAct vs Standard Prompting

| Metric | Standard | ReAct | Improvement |
|--------|----------|-------|-------------|
| Success Rate | % | % | % |
| Avg. Steps | | | |
| Tool Calls | | | |
| Accuracy | % | % | % |
| Cost per Query | $ | $ | $ |

### Planning Agent Performance

| Goal Type | Success Rate | Avg. Subtasks | Self-Corrections |
|-----------|--------------|---------------|------------------|
| API Development | % | | |
| Data Pipeline | % | | |
| Chatbot | % | | |

### Code Review Quality

| Iteration | Quality Score | Issues Found | Issues Fixed |
|-----------|---------------|--------------|--------------|
| 1 | /10 | | |
| 2 | /10 | | |
| 3 | /10 | | |

---

## 🚧 Challenges & Solutions

### Challenge 1: ReAct Loops
**Problem:**
Agent gets stuck in infinite loops

**Solution:**
- Add max iteration limit
- Implement loop detection
- Add progress tracking
- Use self-critique to break loops

---

### Challenge 2: Task Decomposition Quality
**Problem:**
Plans are too vague or too detailed

**Solution:**
- Add few-shot examples
- Use structured output format
- Validate task dependencies
- Iteratively refine plan

---

### Challenge 3: Code Refactoring Quality
**Problem:**
Refactored code introduces new bugs

**Solution:**
- Add validation step
- Generate tests first
- Run tests after refactoring
- Use static analysis tools

---

## 🎯 Week 4 Reflection

**What went well:**
- 

**What was difficult:**
- 

**What I would do differently:**
- 

**Key takeaways:**
- 

**Most impactful pattern learned:**
- 

**Ready for Week 5?** [ ] Yes [ ] Need more practice

---

## 📈 Pattern Comparison Matrix

| Pattern | Best For | Pros | Cons | Cost | Complexity |
|---------|----------|------|------|------|------------|
| ReAct | Complex multi-step tasks | Transparent, self-correcting | More API calls | Medium | Medium |
| Planning | Large projects | Organized, scalable | Rigid | Low | Low |
| Reflexion | Quality-critical | High quality | Time-consuming | High | High |
| Standard | Simple queries | Fast, cheap | Limited reasoning | Low | Low |

---

## 🔗 Project Links

- **GitHub Repository:** [Your repo link]
- **ReAct Agent:** [Demo link]
- **Planning Agent:** [Demo link]
- **Code Review Agent:** [Demo link]
- **Blog Post:** [ReAct deep dive]
- **Pattern Documentation:** [Link]

---

## 📞 Next Steps

After completing Week 4:
1. ✅ Push all agents to GitHub
2. ✅ Publish ReAct blog post
3. ✅ Update main README.md
4. 📖 Review advanced agent patterns
5. 🎯 Prepare for Week 5: Multi-Agent Systems
6. 🚀 Optional: Deploy agents as APIs
7. 📝 Optional: Write comparison article

---

## 💡 Bonus Ideas

- Build agent pattern library
- Create pattern selection tool
- Implement hybrid patterns
- Add visualization of agent reasoning
- Build agent debugging tools
- Create performance benchmarks
- Add agent monitoring dashboard
- Implement agent version control
- Create agent testing framework
- Build agent orchestration platform

---


**Week 4 Checkpoint:** ✅ Understand ReAct, planning, self-reflection patterns

