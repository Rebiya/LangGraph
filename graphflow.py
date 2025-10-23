"""
GraphFlow - Human-in-the-Loop System with LangGraph, LangChain, and LangSmith
A workflow agent that handles user queries via structured nodes with dynamic HITL interactions.
"""

import os
import json
import sqlite3
import re
import threading
import time
from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Token limit for context management
LIMIT_TOKENS = 1500
MAX_ITERATIONS = 5  # Maximum HITL iterations before forced termination

# Structured output models for LLM responses
class RouterResponse(BaseModel):
    intent: str = Field(description="The classified intent: greeting, search, email, or memory")
    response: str = Field(description="The response message to the user")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)

class HITLResponse(BaseModel):
    decision: str = Field(description="The decision: CONTINUE or END")
    response: str = Field(description="The friendly response to the user")
    reason: str = Field(description="Brief reason for the decision")

# Output parsers
router_parser = PydanticOutputParser(pydantic_object=RouterResponse)
hitl_parser = PydanticOutputParser(pydantic_object=HITLResponse)

class GraphFlowState(TypedDict):
    """State management for GraphFlow nodes"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_node: str
    previous_node: str
    hitl_active: bool
    hitl_flag: bool
    user_query: str
    search_results: Optional[Dict[str, Any]]
    email_draft: Optional[str]
    email_recipient: Optional[str]
    memory_data: Dict[str, Any]
    token_count: int
    query_type: str
    response: str
    context: str
    iteration_count: int
    max_iterations_reached: bool
    api_keys_available: Dict[str, bool]

class MemoryManager:
    """SQLite-based memory management for GraphFlow with concurrency handling"""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
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
        """Store interaction in database with concurrency handling"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO conversations 
                    (node_type, user_query, ai_response, tool_output, hitl_interaction, token_count, summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (node_type, user_query, ai_response, tool_output, hitl_interaction, token_count, summary))
                
                conn.commit()
                conn.close()
            except sqlite3.OperationalError as e:
                print(f"[MemoryManager Error]: Database timeout: {str(e)}")
                time.sleep(0.1)  # Brief delay before retry
                # Retry once
                try:
                    conn = sqlite3.connect(self.db_path, timeout=60.0)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO conversations 
                        (node_type, user_query, ai_response, tool_output, hitl_interaction, token_count, summary)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (node_type, user_query, ai_response, tool_output, hitl_interaction, token_count, summary))
                    conn.commit()
                    conn.close()
                except Exception as retry_e:
                    print(f"[MemoryManager Error]: Retry failed: {str(retry_e)}")
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history with concurrency handling"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
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
            except sqlite3.OperationalError as e:
                print(f"[MemoryManager Error]: Database timeout: {str(e)}")
                return []
    
    def get_semantic_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get semantically relevant conversation history using simple text matching"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()
                
                # Simple semantic matching using LIKE queries
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE user_query LIKE ? OR ai_response LIKE ? OR summary LIKE ?
                    ORDER BY id DESC 
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
                
                results = cursor.fetchall()
                cols = [col[0] for col in cursor.description]
                conn.close()
                
                return [dict(zip(cols, row)) for row in results]
            except sqlite3.OperationalError as e:
                print(f"[MemoryManager Error]: Database timeout: {str(e)}")
                return []
    
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
    """Handle token counting and message summarization with LLM-based summarization"""
    
    def __init__(self):
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using Gemini-compatible estimator"""
        return count_tokens_gemini(text)
    
    def summarize_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Summarize older messages to stay within token limit using LLM-based summarization"""
        total_tokens = sum(self.count_tokens(str(msg.content)) for msg in messages)
        
        if total_tokens <= LIMIT_TOKENS:
            return messages
        
        # Keep recent messages and summarize older ones
        recent_messages = messages[-3:]  # Keep last 3 messages
        older_messages = messages[:-3]
        
        if older_messages and len(older_messages) > 2:
            try:
                # Use LLM to create intelligent summary
                older_content = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in older_messages])
                summary_prompt = f"""Summarize the following conversation history concisely while preserving key information:

{older_content}

Provide a brief summary that captures the main topics, decisions, and outcomes."""
                
                summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
                summary_content = f"Previous conversation summary: {summary_response.content}"
                summary_message = SystemMessage(content=summary_content)
                
                return [summary_message] + recent_messages
            except Exception as e:
                print(f"[TokenManager Error]: LLM summarization failed: {str(e)}")
                # Fallback to simple summary
                summary_content = f"Previous conversation summary: {len(older_messages)} messages about various topics."
                summary_message = SystemMessage(content=summary_content)
                return [summary_message] + recent_messages
        
        return recent_messages

# Initialize managers
memory_manager = MemoryManager()
token_manager = TokenManager()

# API key validation
def check_api_keys() -> Dict[str, bool]:
    """Check availability of required API keys"""
    return {
        "google_api": bool(os.getenv("GOOGLE_API_KEY")),
        "serper_api": bool(os.getenv("SERPER_API_KEY"))
    }

# Enhanced safe execution with iteration limits
def safe_node_execution(node_func, state, node_name: str):
    """Wrap node execution with try/except, iteration limits, and safe state updates"""
    try:
        # Check iteration limits for HITL nodes
        if node_name == "hitl" and state.get("iteration_count", 0) >= MAX_ITERATIONS:
            print(f"[{node_name} Warning]: Maximum iterations ({MAX_ITERATIONS}) reached, forcing termination")
            state.update({
                "current_node": "end",
                "hitl_active": False,
                "hitl_flag": False,
                "max_iterations_reached": True
            })
            return state
        
        result = node_func(state)
        # Merge returned partial state with current state
        state.update(result)
        # Ensure essential keys exist
        for key in ["messages", "current_node", "hitl_active", "hitl_flag"]:
            if key not in state:
                state[key] = [] if key == "messages" else False
        return state
    except Exception as e:
        err_msg = f"[{node_name} Error]: {str(e)}"
        print(err_msg)
        # Append error message to conversation messages
        if "messages" not in state or state["messages"] is None:
            state["messages"] = []
        state["messages"].append(AIMessage(content=err_msg))
        # End workflow gracefully
        state["current_node"] = "end"
        state["hitl_active"] = False
        state["hitl_flag"] = False
        return state


# 2️⃣ Gemini-compatible token estimator
def count_tokens_gemini(text: str) -> int:
    """Estimate tokens by simple word split for Gemini LLMs"""
    return len(text.split())

# 3️⃣ Append messages instead of overwriting
def append_message(state: GraphFlowState, message: BaseMessage):
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(message)
    return state

# 4️⃣ Safe router parsing
def parse_router_response(response_text: str):
    """Robustly parse INTENT & RESPONSE from router LLM output"""
    intent, router_resp = "search", ""
    try:
        lines = response_text.splitlines()
        for line in lines:
            if line.startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip().lower()
            elif line.startswith("RESPONSE:"):
                router_resp = line.split(":", 1)[1].strip()
        # Fallback: regex if formatting unexpected
        if not intent:
            match = re.search(r"INTENT:\s*(\w+)", response_text)
            if match:
                intent = match.group(1).lower()
        if not router_resp:
            match = re.search(r"RESPONSE:\s*(.*)", response_text)
            if match:
                router_resp = match.group(1).strip()
    except Exception as e:
        print(f"[Router Parse Error]: {str(e)}")
    return intent, router_resp

# 5️⃣ Safe HITL parsing
def parse_hitl_response(response_text: str):
    """Extract DECISION, RESPONSE, REASON from HITL output robustly"""
    decision, human_resp, reason = "END", "Thank you!", "Defaulted decision"
    try:
        match = re.search(r"DECISION:(.*)\nRESPONSE:(.*)\nREASON:(.*)", response_text, re.S)
        if match:
            decision, human_resp, reason = [s.strip() for s in match.groups()]
        else:
            for line in response_text.splitlines():
                if line.startswith("DECISION:"):
                    decision = line.split(":", 1)[1].strip().upper()
                elif line.startswith("RESPONSE:"):
                    human_resp = line.split(":", 1)[1].strip()
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
    except Exception as e:
        print(f"[HITL Parse Error]: {str(e)}")
    return decision, human_resp, reason

def starter_node(state: GraphFlowState) -> GraphFlowState:
    """Starter Node - Main entry point that initializes the flow"""
    print("🚀 Starting GraphFlow workflow...")
    
    # Initialize state with default values
    updates = {
        "current_node": "router",
        "previous_node": "starter",
        "hitl_active": False,
        "hitl_flag": False,
        "query_type": "unknown",
        "response": "",
        "context": "",
        "iteration_count": 0
    }
    
    # Store initial interaction
    memory_manager.store_interaction(
        node_type="starter",
        user_query=state["user_query"],
        ai_response="Workflow initialized",
        hitl_interaction="flow_start"
    )
    
    print(f"Transition: starter -> router")
    return updates

def end_node(state: GraphFlowState) -> GraphFlowState:
    """End Node - Terminates flow, no LLM, just finalizes session"""
    print("🏁 Ending GraphFlow workflow...")
    
    # Save the last message and full state to SQLite
    memory_manager.store_state(state, "end")
    
    # Store final interaction
    memory_manager.store_interaction(
        node_type="end",
        user_query=state["user_query"],
        ai_response=state.get("response", "Workflow completed"),
        hitl_interaction="flow_end",
        summary=f"Completed {state.get('iteration_count', 0)} iterations"
    )
    
    # Reset transient flags
    state.update({"hitl_flag": False, "hitl_active": False})
    
    # Append final message if not already present
    final_message = "Thank you for using GraphFlow! Have a great day!"
    state = append_message(state, AIMessage(content=final_message))
    
    print(f"Transition: end -> END")
    return {
        "current_node": END,
        "previous_node": "end",
        "hitl_active": False,
        "hitl_flag": False
    }

# Initialize Tavily search tool

# Initialize Serper API wrapper with your API key from environment
serper_tool = GoogleSerperAPIWrapper(k=2, api_key=os.getenv("SERPER_API_KEY"))

@tool
def web_search_tool(query: str) -> dict:
    """Perform a robust web search using Serper.dev and return structured results."""
    try:
        # Get raw results from Serper
        raw_results = serper_tool.results(query)  # Returns dict with 'organic'

        # Debug: log raw Serper response for visibility
        print("===== Serper Raw Response =====")
        print(f"Type: {type(raw_results)}")
        print(f"Content: {raw_results}")
        print("================================")

        # Extract organic results safely
        organic_results = raw_results.get("organic", [])
        if not organic_results:
            return {"results": []}

        # Limit to top 2 results
        formatted_results = [
            {
                "title": r.get("title", "No title"),
                "content": r.get("snippet", "No content"),
                "url": r.get("link", "")
            }
            for r in organic_results[:2]
        ]

        return {"results": formatted_results}

    except Exception as e:
        error_msg = f"Search service unavailable: {str(e)}"
        print(f"===== Serper Error =====")
        print(error_msg)
        print("========================")
        return {"error": error_msg}

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
    """Router Node - LLM-driven query analyzer with structured outputs and API key guards"""
    user_query = state["user_query"]
    
    # Check API keys availability
    api_keys = check_api_keys()
    state["api_keys_available"] = api_keys
    
    # Summarize & manage tokens
    state = append_message(state, HumanMessage(content=user_query))
    summarized_messages = token_manager.summarize_messages(state["messages"])
    state["messages"] = summarized_messages
    
    # LLM call with structured output
    try:
        system_msg = SystemMessage(content=f"""You are a polite and warm assistant. Classify the user's intent and respond appropriately.

        Classify the user's intent as one of: greeting, search, email, or memory.
        
        If it's a greeting (hi, hello, hey, good morning, etc.), respond with a polite, human-like greeting.
        If it's a search task (looking for information, facts, current events), indicate it needs web search.
        If it's an email task (drafting, writing, sending emails), indicate it needs email drafting.
        If it's context-based (asking about previous conversations, memory), indicate it needs memory retrieval.
        
        {router_parser.get_format_instructions()}""")
        
        response = llm.invoke([system_msg, HumanMessage(content=user_query)])
        parsed_response = router_parser.parse(response.content)
        intent = parsed_response.intent
        router_resp = parsed_response.response
        confidence = parsed_response.confidence
        
        # If confidence is low, default to search
        if confidence < 0.5:
            intent = "search"
            router_resp = "I'll help you search for information."
            
    except Exception as e:
        print(f"[Router LLM Error]: {str(e)}")
        intent, router_resp = "search", f"Error: {str(e)}"

    # Append response message
    state = append_message(state, AIMessage(content=router_resp))
    
    # Decide next node safely with API key checks
    if intent == "search" and not api_keys.get("serper_api", False):
        intent = "end"
        router_resp = "I'm sorry, but the search functionality is not available. Please check your API configuration."
    elif intent == "email" and not api_keys.get("google_api", False):
        intent = "end"
        router_resp = "I'm sorry, but the email functionality is not available. Please check your API configuration."
    
    next_node = {
        "greeting": "end",
        "search": "websearch",
        "email": "email_drafter",
        "memory": "memory"
    }.get(intent, "websearch")
    
    # Update state
    state.update({
        "current_node": next_node,
        "previous_node": "router",
        "query_type": intent,
        "response": router_resp,
        "hitl_active": False,
        "hitl_flag": False
    })
    
    # Log to SQLite
    memory_manager.store_interaction("router", user_query, router_resp)
    print(f"Transition: router -> {next_node}")
    return state



# --- Web Search Node for GraphFlow ---
def websearch_node(state: GraphFlowState) -> GraphFlowState:
    """Web Search Node - Integrates Serper + LLM with API key guards and safe error handling."""
    user_query = state.get("user_query", "")
    
    # Check API key availability
    api_keys = state.get("api_keys_available", {})
    if not api_keys.get("serper_api", False):
        error_msg = "Search functionality is not available. Please check your SERPER_API_KEY configuration."
        state = append_message(state, AIMessage(content=error_msg))
        return {
            "current_node": "end",
            "previous_node": "websearch",
            "hitl_active": False,
            "hitl_flag": False
        }
    
    # Token management before LLM call
    summarized_messages = token_manager.summarize_messages(state["messages"])
    state["messages"] = summarized_messages
    
    try:
        # Perform web search
        search_output = web_search_tool.invoke(user_query)

        # Log the processed search output
        print(f"===== Processed Search Output for '{user_query}' =====")
        print(json.dumps(search_output, indent=2))
        print("================================================")

        # Check for errors
        if "error" in search_output:
            response_text = f"Sorry, I couldn't fetch search results: {search_output['error']}. Please try rephrasing your query."
            print(f"Error response: {response_text}")
            state = append_message(state, AIMessage(content=response_text))
            return {
                "search_results": None,
                "current_node": "end",
                "previous_node": "websearch",
                "hitl_active": False,
                "hitl_flag": False
            }

        # Extract results safely
        search_results = search_output.get("results", [])
        if not search_results:
            response_text = "No search results found. Please try rephrasing your query."
            print(f"No results: {response_text}")
            state = append_message(state, AIMessage(content=response_text))
            return {
                "search_results": None,
                "current_node": "end",
                "previous_node": "websearch",
                "hitl_active": False,
                "hitl_flag": False
            }

        # Format search results for LLM
        formatted_results = "\n".join([
            f"• {r['title']}: {r['content'][:200]}..." for r in search_results
        ])
        search_context = f"Search results for '{user_query}':\n{formatted_results}"
        print(f"===== Search Context for LLM =====")
        print(search_context)
        print("=================================")

        # Call your LLM (Gemini or other)
        response = llm.invoke([
            HumanMessage(content=f"Based on these search results, provide a concise and informative answer:\n{search_context}")
        ])

        print(f"===== LLM Response =====")
        print(f"{response.content[:100]}...")
        print("=======================")

        # Enable iteration for web search - set HITL active for user feedback
        print(f"Transition: websearch -> hitl")
        state = append_message(state, AIMessage(content=response.content))
        return {
            "search_results": search_results,
            "hitl_active": True,
            "hitl_flag": True,
            "current_node": "hitl",
            "previous_node": "websearch",
            "response": response.content,
            "iteration_count": state.get("iteration_count", 0) + 1
        }

    except Exception as e:
        error_response = f"I encountered an issue with the search: {str(e)}. Please try rephrasing your query."
        print(f"===== Web Search Node Error =====")
        print(error_response)
        print("================================")
        state = append_message(state, AIMessage(content=error_response))
        return {
            "search_results": None,
            "current_node": "end",
            "previous_node": "websearch",
            "hitl_active": False,
            "hitl_flag": False
        }


def email_drafter_node(state: GraphFlowState) -> GraphFlowState:
    """Email Drafter Node - Draft emails using Gemini"""
    # Token management before LLM call
    summarized_messages = token_manager.summarize_messages(state["messages"])
    state["messages"] = summarized_messages
    
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
        
        # Enable iteration for email draft - set HITL active for user feedback
        print(f"Transition: email_drafter -> hitl")
        response_text = f"Here's your email draft:\n\n{email_content}"
        state = append_message(state, AIMessage(content=response_text))
        return {
            "email_draft": email_content,
            "hitl_active": True,
            "hitl_flag": True,
            "current_node": "hitl",
            "previous_node": "email_drafter",
            "response": response_text,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
        
    except Exception as e:
        error_response = f"I encountered an error while drafting the email: {str(e)}. Please try again."
        state = append_message(state, AIMessage(content=error_response))
        return {
            "current_node": "end",
            "previous_node": "email_drafter",
            "hitl_active": False,
            "hitl_flag": False
        }

def hitl_node(state: GraphFlowState) -> GraphFlowState:
    """Human-in-the-Loop Node - LLM-powered friendly human interaction with structured outputs and iteration limits"""
    # Store the previous node name before calling LLM
    previous_node = state.get("previous_node", "router")
    user_query = state.get("user_query", "")
    response = state.get("response", "")
    query_type = state.get("query_type", "unknown")
    iteration_count = state.get("iteration_count", 0)
    
    # Check iteration limits
    if iteration_count >= MAX_ITERATIONS:
        print(f"[HITL Warning]: Maximum iterations ({MAX_ITERATIONS}) reached, forcing termination")
        termination_msg = f"I've reached the maximum number of iterations ({MAX_ITERATIONS}). Let me end this conversation here."
        state = append_message(state, AIMessage(content=termination_msg))
        return {
            "current_node": "end",
            "previous_node": "hitl",
            "hitl_active": False,
            "hitl_flag": False,
            "max_iterations_reached": True,
            "response": termination_msg
        }
    
    # Token management before LLM call
    summarized_messages = token_manager.summarize_messages(state["messages"])
    state["messages"] = summarized_messages
    
    # System message for HITL LLM with structured output
    system_message = SystemMessage(content=f"""You are a helpful assistant managing human-agent interactions. Be friendly, conversational, and clarify user intent.

    Your job is to:
    1. Engage in natural, friendly conversation with the user
    2. Understand their feedback and needs
    3. Decide whether to continue the current task (search/email) or finish the conversation
    4. Be encouraging and helpful in your responses

    Based on the user's response, decide:
    - CONTINUE: If they want to iterate (more search, email changes, etc.)
    - END: If they're satisfied or want to finish

    {hitl_parser.get_format_instructions()}""")
    
    # Context about what was just completed
    context_info = ""
    if previous_node == "websearch":
        context_info = f"We just completed a web search for: '{user_query}'. The search results were: {response[:200]}..."
    elif previous_node == "email_drafter":
        context_info = f"We just drafted an email for: '{user_query}'. The draft was: {response[:200]}..."
    
    try:
        llm_response = llm.invoke([
            system_message,
            HumanMessage(content=f"""Context: {context_info}
            
            User's current query/feedback: '{user_query}'
            
            Please engage with the user and decide whether to continue or end the conversation.""")
        ])
        
        response_text = llm_response.content.strip()
        print(f"🤖 HITL LLM Response: {response_text}")
        
        # Use structured parsing
        try:
            parsed_response = hitl_parser.parse(response_text)
            decision = parsed_response.decision
            human_resp = parsed_response.response
            reason = parsed_response.reason
        except Exception as parse_error:
            print(f"[HITL Parse Error]: {str(parse_error)}")
            # Fallback to safe parsing
            decision, human_resp, reason = parse_hitl_response(response_text)
        
        # Store HITL interaction
        memory_manager.store_interaction(
            node_type="hitl",
            user_query=user_query,
            ai_response=human_resp,
            hitl_interaction=f"decision_{decision.lower()}",
            summary=reason
        )
        
        # Append response message
        state = append_message(state, AIMessage(content=human_resp))
        
        if decision == "CONTINUE":
            # Continue with the previous node
            print(f"Transition: hitl -> {previous_node}")
            return {
                "current_node": previous_node,
                "previous_node": "hitl",
                "hitl_active": True,
                "hitl_flag": True,
                "response": human_resp,
                "iteration_count": iteration_count + 1,
                "user_query": user_query  # Update user query for next iteration
            }
        else:
            # End the conversation
            print(f"Transition: hitl -> end")
            return {
                "current_node": "end",
                "previous_node": "hitl",
                "hitl_active": False,
                "hitl_flag": False,
                "response": human_resp
            }
            
    except Exception as e:
        print(f"❌ HITL LLM Error: {str(e)}")
        # Fallback response
        fallback_response = "I understand. Is there anything else I can help you with?"
        print(f"Transition: hitl -> end (error fallback)")
        state = append_message(state, AIMessage(content=fallback_response))
        return {
            "current_node": "end",
            "previous_node": "hitl",
            "hitl_active": False,
            "hitl_flag": False,
            "response": fallback_response
        }

def memory_node(state: GraphFlowState) -> GraphFlowState:
    """Memory Node - State persistence and semantic context retrieval"""
    user_query = state["user_query"]
    
    # Store current state
    memory_manager.store_state(state, state["current_node"])
    
    # Retrieve semantic context from memory
    semantic_messages = memory_manager.get_semantic_memory(user_query, limit=5)
    
    # If no semantic matches, fall back to recent messages
    if not semantic_messages:
        semantic_messages = memory_manager.get_recent_messages(limit=3)
    
    # Build context from retrieved interactions
    context_parts = []
    for msg in semantic_messages:
        if msg.get('node_type') in ['websearch', 'email_drafter', 'hitl']:
            context_parts.append(f"{msg['node_type']}: {msg['ai_response'][:100]}...")
    
    context = "\n".join(context_parts) if context_parts else "No relevant context available."
    
    # Check token limits and summarize if needed
    if state["messages"]:
        summarized_messages = token_manager.summarize_messages(list(state["messages"]))
        state["messages"] = summarized_messages
        state["token_count"] = sum(token_manager.count_tokens(str(msg.content)) for msg in summarized_messages)
    
    # Store memory interaction
    memory_manager.store_interaction(
        node_type="memory",
        user_query=user_query,
        ai_response=f"Retrieved semantic context: {context[:200]}...",
        hitl_interaction="semantic_retrieval"
    )
    
    # Append response message
    response_text = f"Based on our previous conversations: {context[:200]}..."
    state = append_message(state, AIMessage(content=response_text))
    
    print(f"Transition: memory -> end")
    return {
        "current_node": "end",
        "previous_node": "memory",
        "context": context,
        "response": response_text,
        "hitl_active": False,
        "hitl_flag": False
    }


def should_continue(state: GraphFlowState) -> str:
    """Determine next node based on current state"""
    current_node = state["current_node"]
    
    # Direct routing from router
    if current_node == "websearch":
        return "websearch"
    elif current_node == "email_drafter":
        return "email_drafter"
    elif current_node == "memory":
        return "memory"
    elif current_node == "hitl":
        return "hitl"
    elif current_node == "end":
        return "end"
    else:
        return END

def hitl_routing(state: GraphFlowState) -> str:
    """Proper conditional function for HITL routing"""
    if state.get("hitl_flag"):
        return state.get("previous_node", "router")
    else:
        return "end"

def websearch_routing(state: GraphFlowState) -> str:
    """Routing from websearch node"""
    if state.get("hitl_active", False):
        return "hitl"
    else:
        return "end"

def email_drafter_routing(state: GraphFlowState) -> str:
    """Routing from email_drafter node"""
    if state.get("hitl_active", False):
        return "hitl"
    else:
        return "end"
        
# Create the GraphFlow workflow
def create_graphflow_workflow():
    """Create and configure the GraphFlow workflow with safe execution"""
    workflow = StateGraph(GraphFlowState)
    
    # Add nodes wrapped in safe execution
    workflow.add_node("starter", lambda state: safe_node_execution(starter_node, state, "starter"))
    workflow.add_node("router", lambda state: safe_node_execution(router_node, state, "router"))
    workflow.add_node("websearch", lambda state: safe_node_execution(websearch_node, state, "websearch"))
    workflow.add_node("email_drafter", lambda state: safe_node_execution(email_drafter_node, state, "email_drafter"))
    workflow.add_node("hitl", lambda state: safe_node_execution(hitl_node, state, "hitl"))
    workflow.add_node("memory", lambda state: safe_node_execution(memory_node, state, "memory"))
    workflow.add_node("end", lambda state: safe_node_execution(end_node, state, "end"))
    
    # Add edges from starter to router
    workflow.add_edge("starter", "router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        should_continue,
        {
            "websearch": "websearch",
            "email_drafter": "email_drafter",
            "memory": "memory",
            "hitl": "hitl",
            "end": "end",
            END: END
        }
    )
    
    # Add conditional edges for iteration (WebSearch and Email only)
    workflow.add_conditional_edges(
        "websearch",
        websearch_routing
    )
    
    workflow.add_conditional_edges(
        "email_drafter", 
        email_drafter_routing
    )
    
    # Add edges from HITL back to nodes for iteration
    workflow.add_conditional_edges(
        "hitl",
        hitl_routing
    )
    
    # Add edges to end
    workflow.add_edge("memory", "end")
    workflow.add_edge("end", END)
    
    # Set entry point
    workflow.set_entry_point("starter")
    
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
            # Initialize state with all required fields
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "current_node": "starter",
                "previous_node": "",
                "hitl_active": False,
                "hitl_flag": False,
                "user_query": user_query,
                "search_results": None,
                "email_draft": None,
                "email_recipient": None,
                "memory_data": {},
                "token_count": 0,
                "query_type": "unknown",
                "response": "",
                "context": "",
                "iteration_count": 0,
                "max_iterations_reached": False,
                "api_keys_available": check_api_keys()
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
            # FIXED: Ensure all exception objects are converted to strings
            error_message = f"I encountered an error: {str(e)}. Please try again."
            return {
                "content": error_message,
                "node_type": "error",
                "hitl_active": False
            }

def main():
    """Main application entry point with graceful error handling"""
    print("GraphFlow - Human-in-the-Loop System (Enhanced)")
    print("===============================================")
    print("Type 'quit' to exit\n")
    
    # Initialize GraphFlow
    graphflow = GraphFlow()
    
    # Check API keys on startup
    api_keys = check_api_keys()
    print(f"API Keys Status: {api_keys}")
    if not api_keys.get("google_api", False):
        print("Warning: GOOGLE_API_KEY not found. Some features may not work.")
    if not api_keys.get("serper_api", False):
        print("Warning: SERPER_API_KEY not found. Search functionality will be limited.")
    print()
    
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
            print("\n\nGoodbye! Thanks for using GraphFlow!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()
