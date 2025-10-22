# GraphFlow - Human-in-the-Loop System 

## 📁 Project Diagrams

- [GraphFlow Architecture Diagram](https://drive.google.com/file/d/1NGgb0e0XFwjVnxFjnU5Mw6Iu4PyFj8vY/view?usp=sharing)
- [Linear GraphFlow Diagram](https://drive.google.com/file/d/17yuaP9ECqrm3WKNdhC0ztHkazf1oFJFi/view?usp=sharing)


A sophisticated workflow agent built with LangGraph, LangChain, and LangSmith that handles user queries through structured nodes with dynamic human-in-the-loop (HITL) interactions.

## Overview

GraphFlow is an intelligent query processing system that automatically routes user queries to appropriate handlers (web search, email drafting) while maintaining contextual awareness and providing dynamic human-in-the-loop interactions when needed.

## Core Architecture

### System Components

1. **Router Node**: Determines query type and routes to appropriate handler
2. **Web Search Node**: Handles search queries using Serper integration
3. **Email Drafter Node**: Creates email drafts using Gemini LLM
4. **Human-in-the-Loop Node**: Manages dynamic user interactions
5. **Memory Node**: Handles state persistence and token management

### Technology Stack

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and tool management
- **Gemini LLM**: Google's advanced language model
- **Serper**: Real-time web search API
- **SQLite**: Persistent memory and state storage
- **Streamlit**: Web interface
- **Tiktoken**: Token counting and management

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LangGraph
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv myvenv
   source myvenv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGSMITH_PROJECT=graphflow-project
   ```

## Usage

### Command Line Interface

Run the CLI application:
```bash
python cli.py
```

Available Commands:
- `help` - Show help information
- `quit` - Exit the application
- `clear` - Clear conversation history
- `status` - Show system status
- `history` - Show recent conversation history

### Web Interface

Run the Streamlit web app:
```bash
streamlit run web_app.py
```

The web interface will be available at `http://localhost:8501`

### Programmatic Usage

```python
from graphflow import GraphFlow

graphflow = GraphFlow()
response = graphflow.process_query("Search for latest AI news")
print(response['content'])
```

## System Architecture

### Workflow Flow



1. **User Input Processing**: Receives and parses user queries
2. **Query Classification**: Routes to appropriate processing node
3. **Tool Execution**: Web Search or Email Draft generation
4. **Response Generation**: AI-generated response creation
5. **HITL Check**: Dynamic human interaction when needed
6. **Memory Storage**: Persists to SQLite database
7. **Token Management**: Automatic summarization when limits exceeded

### Node Interaction Logic

The system implements conditional branching with iterative processing:

- **Real-time Search Path**: Handles external information needs, updates messages and tool outputs
- **Email Drafting Path**: Manages email composition tasks, updates messages and tool outputs
- **Context-aware Processing**: Handles references to previous chat history
- **Iterative HITL Processing**: Continues until human interaction flag is cleared
- **Direct Response Path**: For greetings and simple queries after context fetching

### State Management

```python
class GraphFlowState(TypedDict):
    messages: List[BaseMessage]
    current_node: str
    user_query: str
    tool_output: str
    hitl_flag: bool
    hitl_response: str
```

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Google Gemini API key (Required)
- `SERPER_API_KEY`: Serper search API key (Required)
- `LANGSMITH_API_KEY`: LangSmith API key for tracing (Optional)
- `LANGSMITH_PROJECT`: LangSmith project name (Optional)

### Token Management

The system automatically manages token limits (default: 1500 tokens) by:
- Summarizing older messages when context exceeds limits
- Maintaining recent conversation context
- Storing full history in SQLite database

## Database Schema

### Conversations Table
- `id`: Primary key
- `timestamp`: Interaction timestamp
- `node_type`: Type of node that processed the query
- `user_query`: Original user input
- `ai_response`: AI-generated response
- `tool_output`: Tool execution results
- `hitl_interaction`: HITL interaction details
- `token_count`: Token count for the interaction
- `summary`: Summarized content for token management

### Memory State Table
- `id`: Primary key
- `timestamp`: State timestamp
- `state_data`: Serialized state information
- `node_state`: Current node state

## Project Structure

```
LangGraph/
├── graphflow.py          # Main GraphFlow implementation
├── cli.py               # Command-line interface
├── web_app.py           # Streamlit web interface
├── config.py            # Configuration management
├── linear_graphflow.py  # Simplified linear implementation
├── requirements.txt     # Python dependencies
├── conversations.db     # SQLite database (auto-created)
└── README.md           # Documentation
```

## Development

### Adding New Nodes

1. Create a new node function:
   ```python
   def new_node(state: GraphFlowState) -> GraphFlowState:
       # Node implementation
       return state
   ```

2. Add the node to the workflow:
   ```python
   workflow.add_node("new_node", new_node)
   ```

3. Update routing logic in `router_node` and `should_continue`

### Customizing HITL Interactions

Modify the `hitl_node` function to add custom interaction patterns:
```python
def hitl_node(state: GraphFlowState) -> GraphFlowState:
    # Add custom HITL logic based on current_node
    pass
```

## Troubleshooting

### Common Issues

- **API Key Errors**: Ensure all required API keys are set in `.env`
- **Database Errors**: Check SQLite file permissions
- **Token Limit Errors**: Adjust `LIMIT_TOKENS` in `config.py`
- **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging:
```bash
export DEBUG=1
python cli.py
```

---

# Linear GraphFlow - Sequential LangChain Implementation



This is a simplified, linear implementation of the GraphFlow system using LangChain with sequential processing and in-memory context management.

## Overview

The Linear GraphFlow processes user queries through a simple sequential flow:

```
User Input → Memory & Context → Web Search (Optional) → Email Drafting (Optional) → Output
```

Unlike the complex LangGraph implementation, this version uses:

- LangChain chains for sequential processing
- In-memory context management (no SQLite)
- Simple conditional logic for optional steps
- Easy-to-understand linear flow

## Architecture

### Linear Flow Diagram


### Architecture Comparison

The linear implementation provides a streamlined alternative to the complex graph-based workflow, focusing on simplicity and ease of maintenance.

## Files

- `linear_graphflow.py` - Main implementation with LangChain
- `test_linear_demo.py` - Demo script without API dependencies
- `visualize_linear_flow.py` - Creates flow diagrams
- `Linear_GraphFlow_Visualization.ipynb` - Jupyter notebook with visualizations

## Quick Start

### Run the Demo (No API Keys Required)

```bash
python test_linear_demo.py
```

This demonstrates the linear flow with mock LLM responses.

### Run with Real APIs

```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export SERPER_API_KEY="your_serper_api_key"
python linear_graphflow.py
```

## Key Features

### Sequential Processing

- Step 1: Memory & Context Retrieval
- Step 2: Web Search (if query needs external info)
- Step 3: Email Drafting (if query is email-related)
- Step 4: Output Generation

### In-Memory Context Management

```python
class InMemoryContextManager:
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}
```

### Conditional Steps

- Web Search: Triggered by keywords like "search", "find", "what is", "how to"
- Email Drafting: Triggered by keywords like "email", "draft", "write", "send"

## Comparison with LangGraph

| Feature               | Linear GraphFlow | LangGraph       |
| --------------------- | ---------------- | --------------- |
| Complexity            | Simple           | Complex         |
| Memory Usage          | Low (in-memory)  | Higher (SQLite) |
| Processing Speed      | Fast             | Moderate        |
| Maintainability       | High             | Moderate        |
| Debugging             | Easy             | Complex         |
| Development Speed     | Fast             | Slower          |

## Example Usage

```python
from linear_graphflow import LinearGraphFlow

graphflow = LinearGraphFlow()
result = graphflow.process_query("What is machine learning?")
print(result['response'])

result = graphflow.process_query("Draft an email to my team")
print(result['response'])
```

## Processing Flow Examples

### Search Query

```
Input: "What is artificial intelligence?"
→ Memory: Retrieve context
→ Web Search: Search for AI information
→ Output: Comprehensive answer based on search results
```

### Email Query

```
Input: "Draft an email to schedule a meeting"
→ Memory: Retrieve context
→ Email Drafting: Generate email content
→ Output: Complete email draft
```

### Simple Query

```
Input: "Hello, how are you?"
→ Memory: Retrieve context
→ Output: Conversational response
```

## Technical Details

### Dependencies

- `langchain` - Core LangChain functionality
- `langchain-google-genai` - Google Gemini integration
- `langchain-community` - Community tools (Serper)
- `tiktoken` - Token counting
- `matplotlib` - Visualization
- `pandas` - Data analysis

### Configuration

```python
# Token limit for context management
LIMIT_TOKENS = 1500

# LLM configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)
```

## Notes

- The linear implementation is much simpler than the LangGraph version
- No complex state management or conditional routing
- Easy to understand and modify
- Perfect for learning LangChain concepts
- Great for prototyping before moving to more complex systems

Linear GraphFlow provides a clean, efficient alternative to complex graph-based workflows when you need straightforward sequential processing with optional steps.
