#!/usr/bin/env python3
"""
GraphFlow Setup Script
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        env_content = """# GraphFlow Environment Configuration
# Fill in your API keys

# Google Gemini API Key (Required)
GOOGLE_API_KEY=your_google_api_key_here

# Tavily Search API Key (Required)
TAVILY_API_KEY=your_tavily_api_key_here

# LangSmith API Key (Optional)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=graphflow-project
"""
        env_file.write_text(env_content)
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your API keys")
    else:
        print("✅ .env file already exists")

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")

def run_tests():
    """Run test suite"""
    print("🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "test_graphflow.py"])
        print("✅ All tests passed")
    except subprocess.CalledProcessError:
        print("⚠️  Some tests failed, but setup can continue")
    except FileNotFoundError:
        print("⚠️  Test file not found, skipping tests")

def main():
    """Main setup function"""
    print("🚀 GraphFlow Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create .env file
    create_env_file()
    
    # Create directories
    create_directories()
    
    # Run tests
    run_tests()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Run: python cli.py (for CLI interface)")
    print("3. Run: streamlit run web_app.py (for web interface)")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
