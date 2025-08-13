"""
🚀 AI INTERNSHIP PROJECTS - LOCAL TESTING SCRIPT
===============================================

Quick script to test the application locally before deployment.
Helps verify everything works correctly.

Author: AI Intern
Version: 1.0.0 - Local Testing Edition
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'langchain',
        'groq',
        'plotly',
        'pandas',
        'opencv-python',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_files():
    """Check if all required files exist."""
    print("\n📁 Checking required files...")
    
    required_files = [
        'app.py',
        'master_app_enterprise.py',
        'usage_manager.py',
        'env_manager.py',
        'app_integrator.py',
        'requirements.txt',
        '.env'
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files are present!")
    return True

def check_env_file():
    """Check if .env file has the required variables."""
    print("\n🔐 Checking environment configuration...")
    
    if not Path('.env').exists():
        print("  ❌ .env file not found")
        return False
    
    with open('.env', 'r') as f:
        env_content = f.read()
    
    required_vars = ['GROQ_API_KEY']
    
    for var in required_vars:
        if var in env_content and 'your_' not in env_content:
            print(f"  ✅ {var} configured")
        else:
            print(f"  ⚠️  {var} needs configuration")
    
    print("✅ Environment file checked!")
    return True

def run_streamlit():
    """Run the Streamlit application."""
    print("\n🚀 Starting Streamlit application...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit: {e}")
        return False
    
    return True

def main():
    """Main function to run all checks and start the application."""
    print("🚀 AI INTERNSHIP PROJECTS - LOCAL TESTING")
    print("=" * 50)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run checks
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return
    
    if not check_files():
        print("\n❌ File check failed. Please ensure all files are present.")
        return
    
    check_env_file()
    
    print("\n" + "=" * 50)
    print("✅ All checks passed! Ready to start the application.")
    print("=" * 50)
    
    # Ask user if they want to start the app
    response = input("\n🚀 Start the application? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_streamlit()
    else:
        print("👋 Exiting. Run this script again when you're ready to test!")

if __name__ == "__main__":
    main()