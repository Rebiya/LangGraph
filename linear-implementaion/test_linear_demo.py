#!/usr/bin/env python3
"""
Simple demo of Linear GraphFlow without API dependencies
This demonstrates the linear flow structure and logic
"""

class MockLLM:
    """Mock LLM for demonstration purposes"""
    
    def invoke(self, input_data):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        if isinstance(input_data, dict):
            if "user_query" in input_data and "context" in input_data:
                if "email" in input_data["user_query"].lower():
                    return MockResponse("Subject: Meeting Update\n\nHi Team,\n\nI wanted to update you on our upcoming meeting. Please let me know your availability.\n\nBest regards")
                else:
                    return MockResponse("Based on the context, here's a helpful response to your query.")
            elif "search_results" in input_data:
                return MockResponse("Based on the search results, here's what I found: The information you're looking for is available in the search results provided.")
        return MockResponse("I understand your request and will help you with that.")

class MockSerperTool:
    """Mock search tool for demonstration"""
    
    def results(self, query):
        return {
            "organic": [
                {
                    "title": f"Search result for: {query}",
                    "snippet": f"This is a sample search result about {query}. It contains relevant information that would help answer your question.",
                    "link": "https://example.com/result1"
                },
                {
                    "title": f"Another result for: {query}",
                    "snippet": f"Additional information about {query} that provides more context and details.",
                    "link": "https://example.com/result2"
                }
            ]
        }

class LinearGraphFlowDemo:
    """Demo version of Linear GraphFlow without API dependencies"""
    
    def __init__(self):
        self.llm = MockLLM()
        self.serper_tool = MockSerperTool()
        self.conversation_history = []
    
    def memory_step(self, user_query: str):
        """Step 1: Memory retrieval and context building"""
        print("🔄 Step 1: Memory & Context Retrieval")
        
        context = "No previous context available." if not self.conversation_history else "Previous conversation context available."
        
        # Determine next steps based on query type
        needs_web_search = any(keyword in user_query.lower() for keyword in 
                             ["search", "find", "look up", "what is", "how to", "information"])
        needs_email_draft = any(keyword in user_query.lower() for keyword in 
                              ["email", "draft", "send", "message", "write"])
        
        response = self.llm.invoke({
            "user_query": user_query,
            "context": context
        }).content
        
        self.conversation_history.append({
            "step": "memory",
            "user_input": user_query,
            "ai_response": response
        })
        
        return {
            "step": "memory",
            "context": response,
            "needs_web_search": needs_web_search,
            "needs_email_draft": needs_email_draft
        }
    
    def web_search_step(self, user_query: str, context: str):
        """Step 2: Web search (if needed)"""
        print("🔍 Step 2: Web Search")
        
        search_results = self.serper_tool.results(user_query)
        organic_results = search_results.get("organic", [])
        
        formatted_results = []
        for result in organic_results[:2]:
            formatted_results.append({
                "title": result.get("title", "No title"),
                "content": result.get("snippet", "No content"),
                "url": result.get("link", "")
            })
        
        search_response = self.llm.invoke({
            "user_query": user_query,
            "search_results": "\n".join([
                f"• {r['title']}: {r['content']}" for r in formatted_results
            ])
        }).content
        
        self.conversation_history.append({
            "step": "web_search",
            "user_input": user_query,
            "ai_response": search_response
        })
        
        return {
            "step": "web_search",
            "search_results": formatted_results,
            "response": search_response,
            "success": True
        }
    
    def email_draft_step(self, user_query: str, context: str):
        """Step 3: Email drafting (if needed)"""
        print("📧 Step 3: Email Drafting")
        
        email_response = self.llm.invoke({
            "user_query": user_query,
            "context": context
        }).content
        
        self.conversation_history.append({
            "step": "email_draft",
            "user_input": user_query,
            "ai_response": email_response
        })
        
        return {
            "step": "email_draft",
            "email_content": email_response,
            "success": True
        }
    
    def process_query(self, user_query: str):
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
            "context_length": len(self.conversation_history)
        }

def main():
    """Demo the Linear GraphFlow"""
    print("Linear GraphFlow Demo - Sequential LangChain Implementation")
    print("=" * 60)
    print("This demo shows the linear flow without API dependencies\n")
    
    # Initialize Linear GraphFlow
    graphflow = LinearGraphFlowDemo()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Draft an email to my team about the project update",
        "How does artificial intelligence work?",
        "Write an email to schedule a meeting"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {query}")
        print(f"{'='*60}")
        
        result = graphflow.process_query(query)
        
        print(f"\n📝 Response: {result['response']}")
        print(f"🔧 Steps completed: {[s for s in result['steps_completed'] if s]}")
        print(f"📊 Context length: {result['context_length']} interactions")
        
        if result['search_results']:
            print(f"🔍 Search results: {len(result['search_results'])} found")
        
        if result['email_content']:
            print(f"📧 Email content: {len(result['email_content'])} characters")
    
    print(f"\n{'='*60}")
    print("Demo completed! The linear flow processes queries through:")
    print("1. Memory & Context Retrieval")
    print("2. Web Search (if needed)")
    print("3. Email Drafting (if needed)")
    print("4. Output Generation")
    print("=" * 60)

if __name__ == "__main__":
    main()
