# GraphFlow - Human-in-the-Loop System

A sophisticated workflow agent built with LangGraph, LangChain, and LangSmith that handles user queries through structured nodes with dynamic human-in-the-loop (HITL) interactions.

## 🚀 Features

- **Intelligent Routing**: Automatically routes queries to appropriate nodes (Web Search, Email Draft)
- **Dynamic HITL Interactions**: Context-aware human-in-the-loop prompts based on current state
- **Memory Management**: SQLite-based persistent memory with token limit handling
- **Web Search Integration**: Real-time search using Tavily API
- **Email Drafting**: AI-powered email composition using Gemini LLM
- **Token Management**: Automatic message summarization to maintain context within limits
- **Multiple Interfaces**: CLI and Web interfaces available

## 🏗️ Architecture

### Core Components

1. **Router Node**: Determines query type and routes to appropriate handler
2. **Web Search Node**: Handles search queries using Tavily integration
3. **Email Drafter Node**: Creates email drafts using Gemini LLM
4. **Human-in-the-Loop Node**: Manages dynamic user interactions
5. **Memory Node**: Handles state persistence and token management

### Technology Stack

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and tool management
- **Gemini LLM**: Google's advanced language model
- **Tavily**: Real-time web search API
- **SQLite**: Persistent memory storage
- **Streamlit**: Web interface
- **Tiktoken**: Token counting and management

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LangGraph
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv myvenv
   source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
   LANGSMITH_PROJECT=graphflow-project  # Optional
   ```

## 🚀 Usage

### Command Line Interface

Run the CLI application:
```bash
python cli.py
```

**Available Commands**:
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

# Initialize GraphFlow
graphflow = GraphFlow()

# Process a query
response = graphflow.process_query("Search for latest AI news")
print(response['content'])
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `LANGSMITH_API_KEY` | No | LangSmith API key for tracing |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

### Token Management

The system automatically manages token limits (default: 1500 tokens) by:
- Summarizing older messages when context exceeds limits
- Maintaining recent conversation context
- Storing full history in SQLite database

## 📊 Database Schema

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

## 🔄 Workflow Flow

1. **User Input** → Router Node
2. **Query Classification** → Route to appropriate node
3. **Tool Execution** → Web Search or Email Draft
4. **Response Generation** → AI-generated response
5. **HITL Check** → Dynamic human interaction if needed
6. **Memory Storage** → Persist to SQLite
7. **Token Management** → Summarize if needed

## 🛠️ Development

### Project Structure

```
LangGraph/
├── graphflow.py          # Main GraphFlow implementation
├── cli.py               # Command-line interface
├── web_app.py           # Streamlit web interface
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── conversations.db     # SQLite database (auto-created)
└── README.md           # This file
```

### Adding New Nodes

1. Create a new node function following the pattern:
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

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set in `.env`
2. **Database Errors**: Check SQLite file permissions
3. **Token Limit Errors**: Adjust `LIMIT_TOKENS` in `config.py`
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export DEBUG=1
python cli.py
```

## 📈 Performance

- **Memory Usage**: Optimized with message summarization
- **Response Time**: Typically 2-5 seconds per query
- **Database**: SQLite for lightweight persistence
- **Token Efficiency**: Automatic context management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- Google for Gemini LLM
- Tavily for search capabilities
- Streamlit for web interface capabilities

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue in the repository

---

**GraphFlow** - Making AI interactions more human and contextual! 🤖✨