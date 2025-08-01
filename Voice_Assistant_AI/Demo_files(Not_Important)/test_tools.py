#!/usr/bin/env python3
"""
Test script to verify the custom tools work correctly
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_tool_imports():
    """Test if custom tools can be imported without errors"""
    print("🧪 Testing tool imports...")
    
    try:
        # Test simple version tools
        from voice_app_simple import (
            SimpleSearchTool, 
            CalculatorTool, 
            EmailTool, 
            ReminderTool, 
            WeatherTool
        )
        
        print("✅ Simple version tools imported successfully")
        
        # Test tool instantiation
        search_tool = SimpleSearchTool()
        calc_tool = CalculatorTool()
        email_tool = EmailTool()
        reminder_tool = ReminderTool()
        weather_tool = WeatherTool()
        
        print("✅ All tools instantiated successfully")
        
        # Test tool properties
        assert search_tool.name == "web_search"
        assert calc_tool.name == "calculator"
        assert email_tool.name == "email_sender"
        assert reminder_tool.name == "reminder_scheduler"
        assert weather_tool.name == "weather_info"
        
        print("✅ Tool properties verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool import test failed: {e}")
        return False

def test_calculator_tool():
    """Test the calculator tool functionality"""
    print("\n🧮 Testing calculator tool...")
    
    try:
        from voice_app_simple import CalculatorTool
        
        calc_tool = CalculatorTool()
        
        # Test basic calculations
        test_cases = [
            ("2 + 2", "4"),
            ("10 * 5", "50"),
            ("100 / 4", "25.0"),
            ("2 ** 3", "8"),
        ]
        
        for expression, expected in test_cases:
            result = calc_tool.run(expression)
            print(f"  {expression} = {result}")
            assert expected in result
        
        print("✅ Calculator tool working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Calculator tool test failed: {e}")
        return False

def test_search_tool():
    """Test the search tool functionality"""
    print("\n🔍 Testing search tool...")
    
    try:
        from voice_app_simple import SimpleSearchTool
        
        search_tool = SimpleSearchTool()
        
        # Test search functionality
        result = search_tool.run("artificial intelligence")
        print(f"  Search result: {result[:100]}...")
        
        assert "Search results for" in result
        assert "artificial intelligence" in result
        
        print("✅ Search tool working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Search tool test failed: {e}")
        return False

def test_langchain_integration():
    """Test LangChain integration"""
    print("\n🔗 Testing LangChain integration...")
    
    try:
        from langchain.tools import Tool
        from voice_app_simple import CalculatorTool
        
        calc_tool = CalculatorTool()
        
        # Create LangChain Tool wrapper
        langchain_tool = Tool(
            name="Calculator",
            func=calc_tool.run,
            description="Perform mathematical calculations"
        )
        
        # Test the wrapped tool
        result = langchain_tool.run("5 + 3")
        print(f"  LangChain tool result: {result}")
        
        assert "8" in result
        
        print("✅ LangChain integration working correctly")
        return True
        
    except Exception as e:
        print(f"❌ LangChain integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 Voice Assistant Tools Test")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("voice_app_simple.py").exists():
        print("❌ Please run this test from the Voice_Assistant_AI directory")
        sys.exit(1)
    
    tests = [
        test_tool_imports,
        test_calculator_tool,
        test_search_tool,
        test_langchain_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your voice assistant tools are working correctly.")
        print("\n🚀 Ready to run: streamlit run voice_app_simple.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)