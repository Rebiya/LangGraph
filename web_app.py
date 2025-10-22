"""
GraphFlow Web Interface
A Streamlit-based web interface for the GraphFlow system
"""

import streamlit as st
import json
from datetime import datetime
from graphflow import GraphFlow
from config import validate_config

# Page configuration
st.set_page_config(
    page_title="GraphFlow - HITL System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'graphflow' not in st.session_state:
        try:
            st.session_state.graphflow = GraphFlow()
            st.session_state.conversation_history = []
            st.session_state.current_hitl = False
        except Exception as e:
            st.error(f"Failed to initialize GraphFlow: {str(e)}")
            st.stop()

def display_conversation_history():
    """Display conversation history in sidebar"""
    st.sidebar.header("📝 Conversation History")
    
    if st.session_state.conversation_history:
        for i, msg in enumerate(reversed(st.session_state.conversation_history[-10:])):
            with st.sidebar.expander(f"Message {len(st.session_state.conversation_history) - i}"):
                st.write(f"**You:** {msg['user']}")
                st.write(f"**GraphFlow:** {msg['assistant']}")
                if 'node_type' in msg:
                    st.caption(f"Node: {msg['node_type']}")
    else:
        st.sidebar.write("No conversation yet")

def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.header("🔧 System Status")
    
    try:
        # Check memory manager
        memory_status = "✅ Connected" if st.session_state.graphflow.memory_manager else "❌ Error"
        st.sidebar.write(f"Memory: {memory_status}")
        
        # Check token manager
        token_status = "✅ Active" if st.session_state.graphflow.token_manager else "❌ Error"
        st.sidebar.write(f"Token Manager: {token_status}")
        
        # Check workflow
        workflow_status = "✅ Ready" if st.session_state.graphflow.workflow else "❌ Error"
        st.sidebar.write(f"Workflow: {workflow_status}")
        
        # Show recent database entries
        try:
            recent = st.session_state.graphflow.memory_manager.get_recent_messages(3)
            st.sidebar.write(f"Recent DB entries: {len(recent)}")
        except:
            st.sidebar.write("DB entries: ❌ Error")
    
    except Exception as e:
        st.sidebar.error(f"Status check failed: {str(e)}")

def clear_conversation():
    """Clear conversation history"""
    st.session_state.conversation_history = []
    st.session_state.current_hitl = False
    st.rerun()

def main():
    """Main web application"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🤖 GraphFlow - Human-in-the-Loop System")
    st.markdown("Powered by LangGraph, LangChain, and Gemini LLM")
    
    # Sidebar
    with st.sidebar:
        display_system_status()
        st.divider()
        display_conversation_history()
        st.divider()
        
        if st.button("🗑️ Clear Conversation", type="secondary"):
            clear_conversation()
        
        st.markdown("### 💡 Example Queries")
        st.markdown("""
        - "Search for latest AI news"
        - "Draft an email to my boss"
        - "Find Python programming tutorials"
        - "Write a professional email"
        """)
    
    # Main chat interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "💬 Ask GraphFlow anything:",
            placeholder="Type your message here...",
            key="user_input"
        )
    
    with col2:
        if st.button("🚀 Send", type="primary", use_container_width=True):
            if user_input.strip():
                process_user_input(user_input.strip())
    
    # Display conversation
    if st.session_state.conversation_history:
        st.markdown("### 💬 Conversation")
        
        for msg in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.write(msg['user'])
            
            with st.chat_message("assistant"):
                st.write(msg['assistant'])
                
                # Show additional info if available
                if 'node_type' in msg and msg['node_type'] != 'unknown':
                    st.caption(f"Processed by: {msg['node_type']}")
                
                if 'hitl_active' in msg and msg['hitl_active']:
                    st.info("🔄 Human-in-the-Loop interaction active")
    
    # HITL interaction area
    if st.session_state.current_hitl:
        st.markdown("### 🔄 Human-in-the-Loop Interaction")
        st.info("GraphFlow is waiting for your input to continue the conversation.")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            hitl_input = st.text_input(
                "Follow-up response:",
                placeholder="Provide additional input or clarification...",
                key="hitl_input"
            )
        
        with col2:
            if st.button("✅ Submit", type="primary", use_container_width=True):
                if hitl_input.strip():
                    process_user_input(hitl_input.strip())
                else:
                    st.warning("Please enter a response")

def process_user_input(user_input: str):
    """Process user input through GraphFlow"""
    try:
        # Process query
        with st.spinner("🤔 Processing your request..."):
            response = st.session_state.graphflow.process_query(user_input)
        
        # Store in conversation history
        st.session_state.conversation_history.append({
            'user': user_input,
            'assistant': response['content'],
            'node_type': response.get('node_type', 'unknown'),
            'hitl_active': response.get('hitl_active', False),
            'timestamp': datetime.now().isoformat()
        })
        
        # Update HITL state
        st.session_state.current_hitl = response.get('hitl_active', False)
        
        # Rerun to update the interface
        st.rerun()
    
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        st.session_state.conversation_history.append({
            'user': user_input,
            'assistant': f"Sorry, I encountered an error: {str(e)}",
            'node_type': 'error',
            'hitl_active': False,
            'timestamp': datetime.now().isoformat()
        })
        st.rerun()

if __name__ == "__main__":
    try:
        # Validate configuration
        validate_config()
        main()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.markdown("""
        Please ensure you have set up your API keys in the .env file:
        - GOOGLE_API_KEY
        - TAVILY_API_KEY
        
        Example .env file:
        ```
        GOOGLE_API_KEY=your_google_api_key_here
        TAVILY_API_KEY=your_tavily_api_key_here
        ```
        """)
    except Exception as e:
        st.error(f"Fatal Error: {str(e)}")
