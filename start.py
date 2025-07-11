#!/usr/bin/env python3
"""
AI Service Startup Script

Easy startup script with environment checking and helpful error messages.
"""
import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found!")
        print("📝 Creating .env from example...")
        
        # Copy from env.example
        example_file = Path("env.example")
        if example_file.exists():
            import shutil
            shutil.copy("env.example", ".env")
            print("✅ Created .env file from example")
            print("⚠️  Please edit .env and add your GEMINI API keys!")
            return False
        else:
            print("❌ env.example not found!")
            return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check required variables
    gemini_keys = os.getenv("GEMINI_KEYS")
    if not gemini_keys or gemini_keys.startswith("your_gemini"):
        print("❌ GEMINI_KEYS not configured!")
        print("🔑 Get your API keys from: https://ai.google.dev/")
        print("📝 Edit .env file and replace 'your_gemini_api_key_...' with real keys")
        return False
    
    print("✅ Environment configuration looks good!")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic_settings",
        "google.generativeai",
        "sentence_transformers",
        "chromadb",
        "fitz",  # PyMuPDF
        "docx",  # python-docx
        "pptx",  # python-pptx
        "PIL",   # Pillow
        "loguru"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "fitz":
                import fitz
            elif package == "docx":
                import docx
            elif package == "pptx":
                import pptx
            elif package == "PIL":
                import PIL
            elif package == "google.generativeai":
                import google.generativeai
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📥 Installing missing packages...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    print("✅ All dependencies are available!")
    return True

def start_service():
    """Start the AI service"""
    print("🚀 Starting AI Service...")
    
    # Import and start the service
    try:
        import uvicorn
        from main import app
        
        # Load settings
        from app.core.config import settings
        
        print(f"🌟 Starting {settings.app_name} v{settings.app_version}")
        print(f"🌐 Server will be available at: http://{settings.host}:{settings.port}")
        print(f"📖 API documentation: http://{settings.host}:{settings.port}/docs")
        print(f"📋 Alternative docs: http://{settings.host}:{settings.port}/redoc")
        print("\n" + "="*60)
        print("🤖 Rin-chan AI Tutor is ready to help students learn!")
        print("="*60 + "\n")
        
        # Start the server
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            access_log=True,
            log_level=settings.log_level.lower()
        )
        
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your .env configuration")
        print("2. Verify API keys are valid")
        print("3. Run: python test_service.py")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🤖 AI Service Startup")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed!")
        print("Please fix the configuration and try again.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed!")
        print("Please install required packages and try again.")
        sys.exit(1)
    
    # Start service
    print("\n🎯 All checks passed! Starting service...")
    start_service()

if __name__ == "__main__":
    main() 