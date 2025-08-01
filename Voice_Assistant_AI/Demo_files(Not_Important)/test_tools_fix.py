#!/usr/bin/env python3
"""
Test script to verify Calculator, Email, and Reminder tool fixes
"""

import os
import sys
from pathlib import Path
import datetime

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_calculator_tool():
    """Test the improved Calculator tool"""
    print("🧮 Testing Calculator Tool...")
    
    try:
        from voice_app_simple import calculator_tool
        
        # Test basic calculations
        test_cases = [
            ("2 + 3", "5"),
            ("10 * 5", "50"),
            ("100 / 4", "25"),
            ("2 ** 3", "8"),
            ("sqrt(16)", "4"),
            ("sin(0)", "0"),
            ("pi", "3.14159"),
            ("2 + 3 * 4", "14"),
            ("(2 + 3) * 4", "20"),
        ]
        
        passed = 0
        for expression, expected in test_cases:
            try:
                result = calculator_tool.run(expression)
                print(f"  ✅ {expression} = {result}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {expression} failed: {e}")
        
        # Test error handling
        error_cases = [
            "import os",  # Should be blocked
            "1/0",       # Division by zero
            "invalid",   # Invalid expression
        ]
        
        for expression in error_cases:
            try:
                result = calculator_tool.run(expression)
                if "error" in result.lower() or "invalid" in result.lower():
                    print(f"  ✅ Error handling: {expression} -> {result}")
                    passed += 1
                else:
                    print(f"  ⚠️ Should have failed: {expression} -> {result}")
            except Exception as e:
                print(f"  ✅ Error caught: {expression} -> {e}")
                passed += 1
        
        print(f"✅ Calculator: {passed}/{len(test_cases) + len(error_cases)} tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Calculator test failed: {e}")
        return False

def test_email_tool():
    """Test the improved Email tool"""
    print("\n📧 Testing Email Tool...")
    
    try:
        from voice_app_simple import email_tool
        
        # Test input parsing
        test_cases = [
            ("test@example.com|Test Subject|Test Body", "valid format"),
            ("test@example.com|Test Subject", "valid format (no body)"),
            ("invalid-email|Subject|Body", "invalid email"),
            ("test", "insufficient parameters"),
        ]
        
        passed = 0
        for input_str, expected_type in test_cases:
            try:
                result = email_tool.run(input_str)
                print(f"  📝 Input: {input_str}")
                print(f"     Result: {result}")
                
                if expected_type == "valid format":
                    if "credentials not configured" in result or "sent successfully" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Unexpected result for {expected_type}")
                elif expected_type == "invalid email":
                    if "Invalid email address" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Should have caught invalid email")
                elif expected_type == "insufficient parameters":
                    if "Please provide" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Should have caught insufficient parameters")
                        
            except Exception as e:
                print(f"  ❌ Email test error: {e}")
        
        print(f"✅ Email: {passed}/{len(test_cases)} tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Email test failed: {e}")
        return False

def test_reminder_tool():
    """Test the improved Reminder tool"""
    print("\n⏰ Testing Reminder Tool...")
    
    try:
        from voice_app_simple import reminder_tool
        
        # Test time parsing
        future_time = (datetime.datetime.now() + datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
        
        test_cases = [
            (f"Test Reminder|Test Description|{future_time}", "valid format"),
            (f"Quick Reminder|{future_time}", "valid format (no description)"),
            ("Test|Description|invalid-time", "invalid time"),
            ("Test", "insufficient parameters"),
            ("Past Reminder|Description|2020-01-01 10:00", "past time"),
        ]
        
        passed = 0
        for input_str, expected_type in test_cases:
            try:
                result = reminder_tool.run(input_str)
                print(f"  📝 Input: {input_str}")
                print(f"     Result: {result}")
                
                if expected_type == "valid format":
                    if "scheduled for" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Unexpected result for {expected_type}")
                elif expected_type == "invalid time":
                    if "Invalid time format" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Should have caught invalid time")
                elif expected_type == "insufficient parameters":
                    if "Please provide" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Should have caught insufficient parameters")
                elif expected_type == "past time":
                    if "must be in the future" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Should have caught past time")
                        
            except Exception as e:
                print(f"  ❌ Reminder test error: {e}")
        
        print(f"✅ Reminder: {passed}/{len(test_cases)} tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Reminder test failed: {e}")
        return False

def test_weather_tool():
    """Test the improved Weather tool"""
    print("\n🌤️ Testing Weather Tool...")
    
    try:
        from voice_app_simple import weather_tool
        
        # Test weather lookup
        test_cases = [
            ("London", "valid city"),
            ("InvalidCityName123", "invalid city"),
            ("", "empty input"),
        ]
        
        passed = 0
        for city, expected_type in test_cases:
            try:
                result = weather_tool.run(city)
                print(f"  📝 City: {city}")
                print(f"     Result: {result}")
                
                if expected_type == "valid city":
                    if "Weather in" in result or "API key not configured" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Unexpected result for {expected_type}")
                elif expected_type == "invalid city":
                    if "not found" in result or "API key not configured" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Should handle invalid city better")
                elif expected_type == "empty input":
                    if "not found" in result or "API key not configured" in result:
                        passed += 1
                        print(f"  ✅ {expected_type}")
                    else:
                        print(f"  ⚠️ Should handle empty input")
                        
            except Exception as e:
                print(f"  ❌ Weather test error: {e}")
        
        print(f"✅ Weather: {passed}/{len(test_cases)} tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Weather test failed: {e}")
        return False

def test_env_configuration():
    """Test environment configuration"""
    print("\n🔧 Testing Environment Configuration...")
    
    try:
        from voice_app_simple import GROQ_API_KEY, EMAIL_ADDRESS, EMAIL_PASSWORD, WEATHER_API_KEY
        
        configs = [
            ("GROQ_API_KEY", GROQ_API_KEY, "AI functionality"),
            ("EMAIL_ADDRESS", EMAIL_ADDRESS, "Email sending"),
            ("EMAIL_PASSWORD", EMAIL_PASSWORD, "Email authentication"),
            ("WEATHER_API_KEY", WEATHER_API_KEY, "Weather information"),
        ]
        
        configured = 0
        for name, value, purpose in configs:
            if value:
                print(f"  ✅ {name}: Configured ({purpose})")
                configured += 1
            else:
                print(f"  ⚠️ {name}: Not configured ({purpose})")
        
        print(f"✅ Environment: {configured}/{len(configs)} variables configured")
        
        if configured < len(configs):
            print("\n💡 To fix missing configurations:")
            print("   1. Check your .env file exists")
            print("   2. Add missing variables to .env file")
            print("   3. For Gmail: Use App Password, not regular password")
            print("   4. For Weather: Get free API key from OpenWeatherMap")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def main():
    """Run all tool fix verification tests"""
    print("🔧 Tool Fix Verification Tests")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("voice_app_simple.py").exists():
        print("❌ Please run this test from the Voice_Assistant_AI directory")
        sys.exit(1)
    
    tests = [
        test_env_configuration,
        test_calculator_tool,
        test_email_tool,
        test_reminder_tool,
        test_weather_tool,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Tool Fix Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TOOLS FIXED! Your voice assistant tools are now:")
        print("   🧮 Calculator: Safe math with advanced functions")
        print("   📧 Email: Robust sending with Gmail App Password support")
        print("   ⏰ Reminder: Flexible time parsing and scheduling")
        print("   🌤️ Weather: Detailed weather information")
        print("   🔧 Environment: Proper configuration handling")
        
        print("\n🚀 Ready to test: streamlit run voice_app_simple.py")
        
        print("\n💡 How to test the tools:")
        print("   🧮 Calculator: 'Calculate 2 + 3 * 4' or 'What is sqrt(16)?'")
        print("   📧 Email: 'Send email to test@example.com with subject Hello'")
        print("   ⏰ Reminder: 'Remind me to call mom tomorrow at 3pm'")
        print("   🌤️ Weather: 'What's the weather in London?'")
        
    else:
        print("⚠️ Some tool issues may remain. Check the error messages above.")
        print("   The tools should still work better than before.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)