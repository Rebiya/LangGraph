#!/usr/bin/env python3
"""
GraphFlow Requirements Checker
Verifies that all required packages and API keys are available
"""

import sys
import importlib
import os
from pathlib import Path

# Required packages
REQUIRED_PACKAGES = [
    'langgraph',
    'langchain',
    'langchain_core',
    'langchain_community',
    'langchain_google_genai',
    'pydantic',
    'dotenv',
    'typing_extensions',
    'tavily',
    'langchain_tavily',
    'sqlalchemy',
    'google.generativeai',
    'requests',
    'httpx',
    'tiktoken',
    'streamlit'
]

# Required API keys
REQUIRED_API_KEYS = [
    'GOOGLE_API_KEY',
    'TAVILY_API_KEY'
]

# Optional API keys
OPTIONAL_API_KEYS = [
    'LANGSMITH_API_KEY'
]

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def check_packages():
    """Check if required packages are installed"""
    print("\n📦 Checking required packages...")
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 Checking API keys...")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("❌ python-dotenv not installed")
        return False
    
    missing_keys = []
    for key in REQUIRED_API_KEYS:
        if not os.getenv(key):
            print(f"❌ {key} - Not set")
            missing_keys.append(key)
        else:
            print(f"✅ {key}")
    
    # Check optional keys
    for key in OPTIONAL_API_KEYS:
        if os.getenv(key):
            print(f"✅ {key} (optional)")
        else:
            print(f"⚠️  {key} - Not set (optional)")
    
    if missing_keys:
        print(f"\n⚠️  Missing required API keys: {', '.join(missing_keys)}")
        print("   Please set them in your .env file")
        return False
    
    print("✅ All required API keys are configured")
    return True

def check_database():
    """Check database connectivity"""
    print("\n🗄️  Checking database...")
    try:
        import sqlite3
        db_path = "conversations.db"
        
        # Test database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_file_structure():
    """Check if required files exist"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        'graphflow.py',
        'cli.py',
        'web_app.py',
        'config.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files are present")
    return True

def main():
    """Main requirements check"""
    print("🔍 GraphFlow Requirements Checker")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_packages(),
        check_api_keys(),
        check_database(),
        check_file_structure()
    ]
    
    print("\n" + "=" * 50)
    print("📊 Summary")
    print("=" * 50)
    
    if all(checks):
        print("🎉 All requirements are met!")
        print("   You can now run GraphFlow:")
        print("   • CLI: python cli.py")
        print("   • Web: streamlit run web_app.py")
        return True
    else:
        print("❌ Some requirements are not met")
        print("   Please address the issues above before running GraphFlow")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
