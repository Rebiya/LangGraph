#!/usr/bin/env python3
"""
GraphFlow Demo Script
Demonstrates the key features of the GraphFlow system
"""

import os
import time
from graphflow import GraphFlow
from config import validate_config

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)

def print_response(response, query):
    """Print formatted response"""
    print(f"\n👤 User: {query}")
    print(f"🤖 GraphFlow: {response['content']}")
    if response.get('hitl_active'):
        print("🔄 [HITL Active] - Human-in-the-Loop interaction enabled")
    print(f"📊 Node: {response.get('node_type', 'unknown')}")

def demo_web_search():
    """Demonstrate web search functionality"""
    print_section("Web Search Demo")
    
    queries = [
        "Search for latest AI news",
        "Find information about Python programming",
        "Look up recent developments in machine learning"
    ]
    
    for query in queries:
        print(f"\n🔍 Testing: {query}")
        time.sleep(1)  # Simulate processing time

def demo_email_drafting():
    """Demonstrate email drafting functionality"""
    print_section("Email Drafting Demo")
    
    queries = [
        "Draft an email to my boss about project status",
        "Write a professional email to a client",
        "Create an email for meeting follow-up"
    ]
    
    for query in queries:
        print(f"\n📧 Testing: {query}")
        time.sleep(1)  # Simulate processing time

def demo_hitl_interactions():
    """Demonstrate HITL interactions"""
    print_section("Human-in-the-Loop Demo")
    
    print("🔄 HITL interactions provide dynamic, contextual prompts")
    print("   based on the current state and tool outputs.")
    print("\nExample HITL prompts:")
    print("   • 'Do you want me to expand on any of these results?'")
    print("   • 'Would you like a deeper comparison of these options?'")
    print("   • 'Should I adjust the tone or add more details?'")

def demo_memory_management():
    """Demonstrate memory management"""
    print_section("Memory Management Demo")
    
    print("🧠 Memory features:")
    print("   • SQLite-based persistent storage")
    print("   • Token limit handling (1500 tokens)")
    print("   • Automatic message summarization")
    print("   • Conversation history tracking")
    print("   • State persistence across sessions")

def demo_token_management():
    """Demonstrate token management"""
    print_section("Token Management Demo")
    
    print("🔢 Token management features:")
    print("   • Automatic token counting using Tiktoken")
    print("   • Context summarization when limits exceeded")
    print("   • Recent message preservation")
    print("   • Efficient memory usage")

def main():
    """Main demo function"""
    print("🚀 GraphFlow Demo")
    print("Welcome to the GraphFlow Human-in-the-Loop System!")
    
    # Check configuration
    try:
        validate_config()
        print("\n✅ Configuration validated successfully")
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("Please set up your API keys in the .env file")
        return
    
    # Initialize GraphFlow
    try:
        print("\n🔧 Initializing GraphFlow...")
        graphflow = GraphFlow()
        print("✅ GraphFlow initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize GraphFlow: {e}")
        return
    
    # Run demos
    demo_web_search()
    demo_email_drafting()
    demo_hitl_interactions()
    demo_memory_management()
    demo_token_management()
    
    # Interactive demo
    print_section("Interactive Demo")
    print("Try these example queries:")
    print("1. 'Search for latest AI news'")
    print("2. 'Draft an email to my team'")
    print("3. 'Find Python tutorials'")
    print("4. 'Write a professional email'")
    
    print("\n🎉 Demo completed!")
    print("\nTo run the full system:")
    print("  • CLI: python cli.py")
    print("  • Web: streamlit run web_app.py")

if __name__ == "__main__":
    main()
