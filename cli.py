#!/usr/bin/env python3
"""
GraphFlow CLI Interface
A simple command-line interface for the GraphFlow system
"""

import sys
import json
from graphflow import GraphFlow
from config import validate_config

def print_banner():
    """Print application banner"""
    print("=" * 50)
    print("🤖 GraphFlow - Human-in-the-Loop System")
    print("=" * 50)
    print("Powered by LangGraph, LangChain, and Gemini LLM")
    print("Type 'help' for commands or 'quit' to exit")
    print("=" * 50)

def print_help():
    """Print help information"""
    print("\n📖 Available Commands:")
    print("  help     - Show this help message")
    print("  quit     - Exit the application")
    print("  clear    - Clear conversation history")
    print("  status   - Show system status")
    print("  history  - Show recent conversation history")
    print("\n💡 Example Queries:")
    print("  'Search for latest AI news'")
    print("  'Draft an email to my boss about the project'")
    print("  'Find information about Python programming'")
    print("  'Write a professional email to a client'")

def print_status(graphflow):
    """Print system status"""
    print("\n🔧 System Status:")
    print(f"  Memory Database: {'✅ Connected' if graphflow.memory_manager else '❌ Error'}")
    print(f"  Token Manager: {'✅ Active' if graphflow.token_manager else '❌ Error'}")
    print(f"  Workflow: {'✅ Ready' if graphflow.workflow else '❌ Error'}")

def show_history(graphflow, limit=5):
    """Show recent conversation history"""
    try:
        history = graphflow.memory_manager.get_recent_messages(limit)
        if not history:
            print("\n📝 No conversation history found.")
            return
        
        print(f"\n📝 Recent Conversations (last {len(history)}):")
        print("-" * 40)
        
        for i, record in enumerate(history, 1):
            print(f"{i}. [{record['node_type']}] {record['user_query'][:50]}...")
            print(f"   Response: {record['ai_response'][:100]}...")
            print(f"   Time: {record['timestamp']}")
            print()
    
    except Exception as e:
        print(f"❌ Error retrieving history: {str(e)}")

def clear_history(graphflow):
    """Clear conversation history"""
    try:
        import sqlite3
        conn = sqlite3.connect(graphflow.memory_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations")
        cursor.execute("DELETE FROM memory_state")
        conn.commit()
        conn.close()
        print("✅ Conversation history cleared.")
    except Exception as e:
        print(f"❌ Error clearing history: {str(e)}")

def main():
    """Main CLI application"""
    try:
        # Validate configuration
        validate_config()
        
        # Initialize GraphFlow
        graphflow = GraphFlow()
        
        # Print banner
        print_banner()
        
        # Main interaction loop
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Goodbye! Thanks for using GraphFlow.")
                    break
                
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                
                elif user_input.lower() == 'status':
                    print_status(graphflow)
                    continue
                
                elif user_input.lower() == 'history':
                    show_history(graphflow)
                    continue
                
                elif user_input.lower() == 'clear':
                    clear_history(graphflow)
                    continue
                
                elif not user_input:
                    continue
                
                # Process user query
                print("\n🤔 Processing...")
                response = graphflow.process_query(user_input)
                
                # Display response
                print(f"\n🤖 GraphFlow: {response['content']}")
                
                # Handle HITL interactions
                if response.get('hitl_active', False):
                    print(f"\n💭 [HITL Active - {response.get('node_type', 'unknown')}]")
                    
                    while True:
                        follow_up = input("\n🔄 Follow-up (or 'done' to continue): ").strip()
                        
                        if follow_up.lower() in ['done', 'continue', 'next', '']:
                            break
                        
                        if follow_up.lower() in ['quit', 'exit', 'bye']:
                            print("\n👋 Goodbye! Thanks for using GraphFlow.")
                            sys.exit(0)
                        
                        # Process follow-up
                        follow_up_response = graphflow.process_query(follow_up)
                        print(f"\n🤖 GraphFlow: {follow_up_response['content']}")
                        
                        if not follow_up_response.get('hitl_active', False):
                            break
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using GraphFlow.")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'help' for assistance.")
    
    except ValueError as e:
        print(f"\n❌ Configuration Error: {str(e)}")
        print("\nPlease ensure you have set up your API keys in the .env file:")
        print("  - GOOGLE_API_KEY")
        print("  - TAVILY_API_KEY")
        print("\nExample .env file:")
        print("GOOGLE_API_KEY=your_google_api_key_here")
        print("TAVILY_API_KEY=your_tavily_api_key_here")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
