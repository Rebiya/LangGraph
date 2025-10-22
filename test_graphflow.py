"""
Test suite for GraphFlow
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import sqlite3

# Mock environment variables for testing
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAlJG9xI1csnN5JfzsNwRkY0hRLMQUcRzg'
os.environ['TAVILY_API_KEY'] = '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'

from graphflow import GraphFlow, MemoryManager, TokenManager, GraphFlowState
from config import validate_config

class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager functionality"""
    
    def setUp(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.memory_manager = MemoryManager(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database table creation"""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('conversations', tables)
        self.assertIn('memory_state', tables)
        
        conn.close()
    
    def test_store_interaction(self):
        """Test storing interactions"""
        self.memory_manager.store_interaction(
            node_type="test",
            user_query="test query",
            ai_response="test response",
            tool_output="test output"
        )
        
        # Verify data was stored
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE node_type = 'test'")
        result = cursor.fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[2], "test")  # node_type
        self.assertEqual(result[3], "test query")  # user_query
        
        conn.close()
    
    def test_get_recent_messages(self):
        """Test retrieving recent messages"""
        # Store some test data
        for i in range(5):
            self.memory_manager.store_interaction(
                node_type=f"test_{i}",
                user_query=f"query_{i}",
                ai_response=f"response_{i}"
            )
        
        # Get recent messages
        recent = self.memory_manager.get_recent_messages(3)
        
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0]['node_type'], 'test_4')  # Most recent first

class TestTokenManager(unittest.TestCase):
    """Test TokenManager functionality"""
    
    def setUp(self):
        self.token_manager = TokenManager()
    
    def test_count_tokens(self):
        """Test token counting"""
        text = "Hello, world!"
        token_count = self.token_manager.count_tokens(text)
        
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)
    
    def test_summarize_messages(self):
        """Test message summarization"""
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Create test messages
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm doing well, thanks!")
        ]
        
        # Test summarization
        summarized = self.token_manager.summarize_messages(messages)
        
        self.assertIsInstance(summarized, list)
        self.assertLessEqual(len(summarized), len(messages))

class TestConfig(unittest.TestCase):
    """Test configuration validation"""
    
    def test_validate_config_with_keys(self):
        """Test config validation with valid keys"""
        with patch.dict(os.environ, {
            'GOOGLE_API_KEY': 'AIzaSyAlJG9xI1csnN5JfzsNwRkY0hRLMQUcRzg',
            'TAVILY_API_KEY': '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'
        }):
            # Should not raise an exception
            validate_config()
    
    def test_validate_config_missing_keys(self):
        """Test config validation with missing keys"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                validate_config()

class TestGraphFlowIntegration(unittest.TestCase):
    """Test GraphFlow integration"""
    
    @patch('graphflow.llm')
    @patch('graphflow.web_search_tool')
    def test_web_search_flow(self, mock_search_tool, mock_llm):
        """Test web search workflow"""
        # Mock responses
        mock_search_tool.invoke.return_value = '{"results": [{"title": "Test", "content": "Test content"}]}'
        mock_llm.invoke.return_value = MagicMock(content="Search results: Test content")
        
        # Create GraphFlow instance
        graphflow = GraphFlow()
        
        # Test web search query
        response = graphflow.process_query("Search for AI news")
        
        self.assertIn('content', response)
        self.assertIsInstance(response['content'], str)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
