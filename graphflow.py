"""
GraphFlow - Human-in-the-Loop System with LangGraph, LangChain, and LangSmith
A workflow agent that handles user queries via structured nodes with dynamic HITL interactions.
"""

import os
import json
import sqlite3
from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
from datetime import datetime
import tiktoken

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Token limit for context management
LIMIT_TOKENS = 1500

class GraphFlowState(TypedDict):
    """State management for GraphFlow nodes"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_node: str
    hitl_active: bool
    user_query: str
    search_results: Optional[Dict[str, Any]]
    email_draft: Optional[str]
    email_recipient: Optional[str]
    memory_data: Dict[str, Any]
    token_count: int

class MemoryManager:
    """SQLite-based memory management for GraphFlow"""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                node_type TEXT,
                user_query TEXT,
                ai_response TEXT,
                tool_output TEXT,
                hitl_interaction TEXT,
                token_count INTEGER,
                summary TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                state_data TEXT,
                node_state TEXT
            )
        ''')

        # Lightweight schema migration: ensure required columns exist
        def ensure_columns(table: str, required: List[str]):
            cursor.execute(f"PRAGMA table_info({table})")
            existing_cols = {row[1] for row in cursor.fetchall()}
            for col in required:
                if col not in existing_cols:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")

        ensure_columns(
            "conversations",
            [
                "node_type",
                "user_query",
                "ai_response",
                "tool_output",
                "hitl_interaction",
                "token_count",
                "summary",
            ],
        )

        ensure_columns(
            "memory_state",
            [
                "state_data",
                "node_state",
            ],
        )
        
        conn.commit()
        conn.close()
    
    def store_interaction(self, node_type: str, user_query: str, ai_response: str, 
                         tool_output: str = None, hitl_interaction: str = None, 
                         token_count: int = 0, summary: str = None):
        """Store interaction in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (node_type, user_query, ai_response, tool_output, hitl_interaction, token_count, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (node_type, user_query, ai_response, tool_output, hitl_interaction, token_count, summary))
        
        conn.commit()
        conn.close()
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversations 
            ORDER BY id DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        cols = [col[0] for col in cursor.description]
        conn.close()
        
        return [dict(zip(cols, row)) for row in results]
    
    def store_state(self, state_data: Dict[str, Any], node_state: str):
        """Store current state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert messages to serializable format
        serializable_data = state_data.copy()
        if 'messages' in serializable_data:
            serializable_data['messages'] = [
                {
                    'type': type(msg).__name__,
                    'content': str(msg.content) if hasattr(msg, 'content') else str(msg)
                } for msg in serializable_data['messages']
            ]
        
        cursor.execute('''
            INSERT INTO memory_state (state_data, node_state)
            VALUES (?, ?)
        ''', (json.dumps(serializable_data), node_state))
        
        conn.commit()
        conn.close()

class TokenManager:
    """Handle token counting and message summarization"""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def summarize_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Summarize older messages to stay within token limit"""
        total_tokens = sum(self.count_tokens(str(msg.content)) for msg in messages)
        
        if total_tokens <= LIMIT_TOKENS:
            return messages
        
        # Keep recent messages and summarize older ones
        recent_messages = messages[-5:]  # Keep last 5 messages
        older_messages = messages[:-5]
        
        if older_messages:
            # Create summary of older messages
            summary_content = f"Previous conversation summary: {len(older_messages)} messages about various topics including search queries and email drafting."
            summary_message = SystemMessage(content=summary_content)
            
            return [summary_message] + recent_messages
        
        return recent_messages

# Initialize managers
memory_manager = MemoryManager()
token_manager = TokenManager()

# Initialize Tavily search tool
tavily_tool = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"), max_results=5)
print(tavily_tool.invoke({"query": "test query"}))

@tool
def web_search_tool(query: str) -> str:
    """Search the web for real-time information"""
    try:
        results = tavily_tool.invoke({"query": query})
        return json.dumps(results)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def email_draft_tool(query: str, recipient: str = None) -> str:
    """Draft an email based on user query"""
    try:
        prompt = f"Draft an email based on this request: {query}"
        if recipient:
            prompt += f" for recipient: {recipient}"
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Email drafting error: {str(e)}"

def router_node(state: GraphFlowState) -> GraphFlowState:
    """Router Node - Determine query type and route accordingly"""
    user_query = state["user_query"].lower()
    
    # Simple routing logic based on keywords
    if any(keyword in user_query for keyword in ["search", "find", "look up", "what is", "how to"]):
        next_node = "web_search"
    elif any(keyword in user_query for keyword in ["email", "draft", "send", "message", "write"]):
        next_node = "email_draft"
    else:
        # Ask for clarification
        next_node = "hitl_clarification"
    
    # Store routing decision internally
    memory_manager.store_interaction(
        node_type="router",
        user_query=state["user_query"],
        ai_response=f"Internal routing to {next_node}",
        hitl_interaction="routing_decision"
    )
    
    return {
        "current_node": next_node,
        "hitl_active": next_node == "hitl_clarification"
    }

def post_processing_node(state: GraphFlowState) -> GraphFlowState:
    """Post-processing node to handle HITL routing after main nodes complete"""
    updates = {}
    
    # Check if we have results that need HITL interaction
    if "search_results" in state and state["search_results"]:
        updates.update({
            "current_node": "hitl_web_search",
            "hitl_active": True
        })
    elif "email_draft" in state and state["email_draft"]:
        updates.update({
            "current_node": "hitl_email_draft",
            "hitl_active": True
        })
    elif state.get("current_node") == "hitl_clarification":
        updates.update({
            "hitl_active": True
        })
    
    # Debug: Log state updates
    print(f"Post-processing updates: {updates}")
    
    return updates

def web_search_node(state: GraphFlowState) -> GraphFlowState:
    """Web Search Node - Handle search queries with Tavily"""
    try:
        # Perform web search
        search_results = web_search_tool.invoke(state["user_query"])
        
        # Debug: Log raw search results
        print(f"Raw search results: {search_results}")
        
        if not search_results:
            error_response = "No search results found. Please try a different query."
            return {
                "messages": [AIMessage(content=error_response)],
                "search_results": None
            }
        
        # Attempt to parse JSON
        try:
            search_data = json.loads(search_results)
        except json.JSONDecodeError as json_err:
            error_response = f"Failed to parse search results: {str(json_err)}. Please try rephrasing your query."
            return {
                "messages": [AIMessage(content=error_response)],
                "search_results": None
            }
        
        # Generate response using Gemini
        search_context = f"Search results for '{state['user_query']}':\n{json.dumps(search_data, indent=2)}"
        response = llm.invoke([
            HumanMessage(content=f"Based on these search results, provide a helpful response: {search_context}")
        ])
        
        # Store interaction
        memory_manager.store_interaction(
            node_type="web_search",
            user_query=state["user_query"],
            ai_response=response.content,
            tool_output=search_results,
            token_count=token_manager.count_tokens(response.content)
        )
        
        return {
            "messages": [AIMessage(content=response.content)],
            "search_results": search_data
        }
        
    except Exception as e:
        error_response = f"I encountered an error while searching: {str(e)}. Please try rephrasing your query."
        return {
            "messages": [AIMessage(content=error_response)],
            "search_results": None
        }
def email_draft_node(state: GraphFlowState) -> GraphFlowState:
    """Email Drafter Node - Draft emails using Gemini"""
    try:
        # Draft email
        email_content = email_draft_tool.invoke(state["user_query"])
        
        # Store interaction
        memory_manager.store_interaction(
            node_type="email_draft",
            user_query=state["user_query"],
            ai_response=email_content,
            token_count=token_manager.count_tokens(email_content)
        )
        
        return {
            "messages": [AIMessage(content=f"Here's your email draft:\n\n{email_content}")],
            "email_draft": email_content
        }
        
    except Exception as e:
        error_response = f"I encountered an error while drafting the email: {str(e)}. Please try again."
        return {
            "messages": [AIMessage(content=error_response)]
        }

def hitl_node(state: GraphFlowState) -> GraphFlowState:
    """Human-in-the-Loop Node - Handle dynamic user interactions"""
    current_node = state["current_node"]
    
    if current_node == "hitl_clarification":
        clarification = "I'm not sure what you'd like me to help with. Would you like me to:\n1. Search for information on a topic\n2. Help draft an email\n\nPlease let me know which option you prefer or provide more details about your request."
        
        return {
            "messages": [AIMessage(content=clarification)],
            "hitl_active": True
        }
    
    elif current_node == "hitl_web_search":
        question = "Would you like me to search for more specific information or expand on any of these results?"
        return {
            "messages": [AIMessage(content=question)],
            "hitl_active": True
        }
    
    elif current_node == "hitl_email_draft":
        question = "Would you like me to adjust the tone, add more details, or make any other changes to this email draft?"
        return {
            "messages": [AIMessage(content=question)],
            "hitl_active": True
        }
    
    return state

def memory_node(state: GraphFlowState) -> GraphFlowState:
    """Memory Node - Manage state and context"""
    # Store current state
    memory_manager.store_state(state, state["current_node"])
    
    # Check token limits and summarize if needed
    if state["messages"]:
        summarized_messages = token_manager.summarize_messages(list(state["messages"]))
        state["messages"] = summarized_messages
        state["token_count"] = sum(token_manager.count_tokens(str(msg.content)) for msg in summarized_messages)
    
    return state


def should_continue(state: GraphFlowState) -> str:
    print(f"Routing from router, current_node: {state['current_node']}")
    if state["current_node"] == "web_search":
        return "web_search"
    elif state["current_node"] == "email_draft":
        return "email_draft"
    elif state["current_node"] == "hitl_clarification":
        return "hitl"
    else:
        return "post_processing"

def should_continue_after_processing(state: GraphFlowState) -> str:
    print(f"Routing from post_processing, hitl_active: {state.get('hitl_active', False)}")
    if state.get("hitl_active", False):
        return "hitl"
    else:
        return END
        
# Create the GraphFlow workflow
def create_graphflow_workflow():
    """Create and configure the GraphFlow workflow"""
    workflow = StateGraph(GraphFlowState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("email_draft", email_draft_node)
    workflow.add_node("post_processing", post_processing_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("memory", memory_node)
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        should_continue,
        {
            "web_search": "web_search",
            "email_draft": "email_draft",
            "hitl": "hitl",
            "post_processing": "post_processing"
        }
    )
    
    # Add edges from main nodes to post-processing
    workflow.add_edge("web_search", "post_processing")
    workflow.add_edge("email_draft", "post_processing")
    
    # Add conditional edges from post-processing
    workflow.add_conditional_edges(
        "post_processing",
        should_continue_after_processing,
        {
            "hitl": "hitl",
            END: END
        }
    )
    
    # Add edges to memory
    workflow.add_edge("hitl", "memory")
    workflow.add_edge("memory", END)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    return workflow.compile()

class GraphFlow:
    """Main GraphFlow application class"""
    
    def __init__(self):
        self.workflow = create_graphflow_workflow()
        self.memory_manager = memory_manager
        self.token_manager = token_manager
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query through the GraphFlow workflow"""
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "current_node": "router",
                "hitl_active": False,
                "user_query": user_query,
                "search_results": None,
                "email_draft": None,
                "email_recipient": None,
                "memory_data": {},
                "token_count": 0
            }
            
            # Process through workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract response
            if result["messages"]:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return {
                        "content": last_message.content,
                        "node_type": result["current_node"],
                        "hitl_active": result["hitl_active"]
                    }
            
            return {
                "content": "I'm processing your request. Please wait a moment.",
                "node_type": result["current_node"],
                "hitl_active": result["hitl_active"]
            }
            
        except Exception as e:
            return {
                "content": f"I encountered an error: {str(e)}. Please try again.",
                "node_type": "error",
                "hitl_active": False
            }

def main():
    """Main application entry point"""
    print("GraphFlow - Human-in-the-Loop System")
    print("=====================================")
    print("Type 'quit' to exit\n")
    
    # Initialize GraphFlow
    graphflow = GraphFlow()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process query
            response = graphflow.process_query(user_input)
            
            # Display response
            print(f"\nGraphFlow: {response['content']}\n")
            
            # Handle HITL interactions
            if response.get('hitl_active', False):
                follow_up = input("Follow-up (or 'continue' to proceed): ").strip()
                if follow_up.lower() not in ['continue', 'next', '']:
                    # Process follow-up as new query
                    follow_up_response = graphflow.process_query(follow_up)
                    print(f"\nGraphFlow: {follow_up_response['content']}\n")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
