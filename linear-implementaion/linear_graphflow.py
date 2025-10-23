"""
Linear GraphFlow - Sequential LangChain Implementation
A simplified linear workflow that processes user queries through sequential steps:
User Input → Memory → Web Search → Email Drafting → Output
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import tiktoken

from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Token limit for context management
LIMIT_TOKENS = 1500

class InMemoryContextManager:
    """In-memory context management for Linear GraphFlow"""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def add_interaction(self, step: str, user_input: str, ai_response: str, 
                       metadata: Dict[str, Any] = None):
        """Add interaction to conversation history"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "user_input": user_input,
            "ai_response": ai_response,
            "metadata": metadata or {}
        }
        self.conversation_history.append(interaction)
        self.current_context.update(interaction)
    
    def get_recent_context(self, limit: int = 5) -> str:
        """Get recent conversation context as formatted string"""
        recent = self.conversation_history[-limit:] if self.conversation_history else []
        
        context_parts = []
        for interaction in recent:
            context_parts.append(
                f"[{interaction['step']}] User: {interaction['user_input']}\n"
                f"AI: {interaction['ai_response']}\n"
            )
        
        return "\n".join(context_parts)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def should_summarize(self) -> bool:
        """Check if conversation history needs summarization"""
        total_tokens = sum(
            self.count_tokens(interaction['user_input'] + interaction['ai_response'])
            for interaction in self.conversation_history
        )
        return total_tokens > LIMIT_TOKENS
    
    def summarize_history(self) -> str:
        """Summarize conversation history to stay within token limits"""
        if len(self.conversation_history) <= 3:
            return self.get_recent_context()
        
        # Keep recent interactions and summarize older ones
        recent = self.conversation_history[-2:]
        older = self.conversation_history[:-2]
        
        summary = f"Previous conversation summary: {len(older)} interactions about various topics including search queries and email drafting."
        
        recent_context = "\n".join([
            f"[{interaction['step']}] User: {interaction['user_input']}\nAI: {interaction['ai_response']}\n"
            for interaction in recent
        ])
        
        return f"{summary}\n\n{recent_context}"

class LinearGraphFlow:
    """Linear GraphFlow implementation using LangChain sequential processing"""
    
    def __init__(self):
        self.context_manager = InMemoryContextManager()
        self.serper_tool = GoogleSerperAPIWrapper(k=2, api_key=os.getenv("SERPER_API_KEY"))
        
        # Initialize prompts
        self.memory_prompt = PromptTemplate(
            input_variables=["user_query", "context"],
            template="""Based on the conversation context below, provide relevant information for the user's query.

Context:
{context}

User Query: {user_query}

Provide relevant context or indicate if no relevant context is available:"""
        )
        
        self.web_search_prompt = PromptTemplate(
            input_variables=["user_query", "search_results"],
            template="""Based on the search results below, provide a comprehensive answer to the user's query.

Search Results:
{search_results}

User Query: {user_query}

Provide a detailed and informative response:"""
        )
        
        self.email_draft_prompt = PromptTemplate(
            input_variables=["user_query", "context"],
            template="""Draft a professional email based on the user's request and any relevant context.

Context:
{context}

User Request: {user_query}

Draft the email:"""
        )
    
    def memory_step(self, user_query: str) -> Dict[str, Any]:
        """Step 1: Memory retrieval and context building"""
        print("🔄 Step 1: Memory & Context Retrieval")
        
        # Get recent context
        context = self.context_manager.get_recent_context()
        
        if not context:
            return {
                "step": "memory",
                "context": "No previous context available.",
                "needs_web_search": True,
                "needs_email_draft": False
            }
        
        # Use LLM to determine relevance and next steps
        try:
            memory_response = (self.memory_prompt | llm).invoke({
                "user_query": user_query,
                "context": context
            }).content
            
            # Determine next steps based on query type
            needs_web_search = any(keyword in user_query.lower() for keyword in 
                                 ["search", "find", "look up", "what is", "how to", "information"])
            needs_email_draft = any(keyword in user_query.lower() for keyword in 
                                  ["email", "draft", "send", "message", "write"])
            
            self.context_manager.add_interaction(
                step="memory",
                user_input=user_query,
                ai_response=memory_response,
                metadata={"needs_web_search": needs_web_search, "needs_email_draft": needs_email_draft}
            )
            
            return {
                "step": "memory",
                "context": memory_response,
                "needs_web_search": needs_web_search,
                "needs_email_draft": needs_email_draft
            }
            
        except Exception as e:
            return {
                "step": "memory",
                "context": f"Memory retrieval error: {str(e)}",
                "needs_web_search": True,
                "needs_email_draft": False
            }
    
    def web_search_step(self, user_query: str, context: str) -> Dict[str, Any]:
        """Step 2: Web search (if needed)"""
        print("🔍 Step 2: Web Search")
        
        try:
            # Perform web search
            search_results = self.serper_tool.results(user_query)
            
            # Extract and format results
            organic_results = search_results.get("organic", [])
            formatted_results = []
            
            for result in organic_results[:2]:  # Limit to top 2 results
                formatted_results.append({
                    "title": result.get("title", "No title"),
                    "content": result.get("snippet", "No content"),
                    "url": result.get("link", "")
                })
            
            if not formatted_results:
                return {
                    "step": "web_search",
                    "search_results": [],
                    "response": "No search results found. Please try rephrasing your query.",
                    "success": False
                }
            
            # Use LLM to process search results
            search_response = (self.web_search_prompt | llm).invoke({
                "user_query": user_query,
                "search_results": "\n".join([
                    f"• {r['title']}: {r['content']}" for r in formatted_results
                ])
            }).content
            
            self.context_manager.add_interaction(
                step="web_search",
                user_input=user_query,
                ai_response=search_response,
                metadata={"search_results": formatted_results}
            )
            
            return {
                "step": "web_search",
                "search_results": formatted_results,
                "response": search_response,
                "success": True
            }
            
        except Exception as e:
            error_response = f"Search failed: {str(e)}. Please try rephrasing your query."
            return {
                "step": "web_search",
                "search_results": [],
                "response": error_response,
                "success": False
            }
    
    def email_draft_step(self, user_query: str, context: str) -> Dict[str, Any]:
        """Step 3: Email drafting (if needed)"""
        print("📧 Step 3: Email Drafting")
        
        try:
            # Use LLM to draft email
            email_response = (self.email_draft_prompt | llm).invoke({
                "user_query": user_query,
                "context": context
            }).content
            
            self.context_manager.add_interaction(
                step="email_draft",
                user_input=user_query,
                ai_response=email_response,
                metadata={"type": "email_draft"}
            )
            
            return {
                "step": "email_draft",
                "email_content": email_response,
                "success": True
            }
            
        except Exception as e:
            error_response = f"Email drafting failed: {str(e)}. Please try again."
            return {
                "step": "email_draft",
                "email_content": error_response,
                "success": False
            }
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through linear workflow"""
        print(f"\n🚀 Processing Query: '{user_query}'\n")
        
        # Step 1: Memory & Context
        memory_result = self.memory_step(user_query)
        print(f"Memory Context: {memory_result['context'][:100]}...")
        
        # Step 2: Web Search (if needed)
        final_response = memory_result['context']
        search_results = []
        
        if memory_result['needs_web_search']:
            search_result = self.web_search_step(user_query, memory_result['context'])
            final_response = search_result['response']
            search_results = search_result['search_results']
            print(f"Search Results: {len(search_results)} results found")
        
        # Step 3: Email Drafting (if needed)
        email_content = None
        if memory_result['needs_email_draft']:
            email_result = self.email_draft_step(user_query, memory_result['context'])
            final_response = email_result['email_content']
            email_content = email_result['email_content']
            print(f"Email Draft: {len(email_content)} characters")
        
        # Check if we need to summarize history
        if self.context_manager.should_summarize():
            print("📝 Summarizing conversation history...")
            self.context_manager.summarize_history()
        
        return {
            "user_query": user_query,
            "response": final_response,
            "steps_completed": [
                memory_result['step'],
                "web_search" if memory_result['needs_web_search'] else None,
                "email_draft" if memory_result['needs_email_draft'] else None
            ],
            "search_results": search_results,
            "email_content": email_content,
            "context_length": len(self.context_manager.conversation_history)
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get full conversation history"""
        return self.context_manager.conversation_history
    
    def clear_context(self):
        """Clear conversation context"""
        self.context_manager.conversation_history = []
        self.context_manager.current_context = {}
        print("🧹 Context cleared")

def main():
    """Main application entry point"""
    print("Linear GraphFlow - Sequential LangChain Implementation")
    print("=====================================================")
    print("Type 'quit' to exit, 'clear' to clear context, 'history' to view history\n")
    
    # Initialize Linear GraphFlow
    graphflow = LinearGraphFlow()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                graphflow.clear_context()
                continue
            
            if user_input.lower() == 'history':
                history = graphflow.get_conversation_history()
                print(f"\n📚 Conversation History ({len(history)} interactions):")
                for i, interaction in enumerate(history, 1):
                    print(f"{i}. [{interaction['step']}] {interaction['user_input'][:50]}...")
                print()
                continue
            
            if not user_input:
                continue
            
            # Process query through linear workflow
            result = graphflow.process_query(user_input)
            
            # Display response
            print(f"\n🤖 Linear GraphFlow: {result['response']}\n")
            
            # Show additional info if available
            if result['search_results']:
                print(f"🔍 Found {len(result['search_results'])} search results")
            
            if result['email_content']:
                print(f"📧 Email draft created ({len(result['email_content'])} characters)")
            
            print(f"📊 Context: {result['context_length']} interactions stored\n")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
