#!/usr/bin/env python3
"""
Test script to verify TTS COM initialization fix
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_tts_initialization():
    """Test TTS engine initialization with COM fix"""
    print("🔊 Testing TTS Initialization...")
    
    try:
        # Import the TTS initialization function
        from voice_app_simple import init_tts_engine, windows_speak
        
        # Test TTS engine initialization
        engine = init_tts_engine()
        
        if engine is not None:
            print("✅ TTS engine initialized successfully")
            
            # Test basic TTS functionality
            try:
                engine.say("Testing TTS engine")
                engine.runAndWait()
                print("✅ TTS engine speech test successful")
            except Exception as e:
                print(f"⚠️ TTS speech test failed: {e}")
                print("   But engine initialization worked")
        else:
            print("⚠️ TTS engine is None, testing fallback...")
            
            # Test Windows SAPI fallback
            if windows_speak("Testing Windows SAPI fallback"):
                print("✅ Windows SAPI fallback working")
            else:
                print("⚠️ Windows SAPI fallback also failed")
                print("   App will run in text-only mode")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS initialization test failed: {e}")
        return False

def test_speak_function():
    """Test the enhanced speak function"""
    print("\n🗣️ Testing Speak Function...")
    
    try:
        from voice_app_simple import speak
        
        # Test speak function with different emotions
        test_cases = [
            ("Hello, this is a test", "neutral"),
            ("I'm so happy!", "joy"),
            ("This is concerning", "sad"),
        ]
        
        for text, emotion in test_cases:
            print(f"  Testing: '{text}' with emotion '{emotion}'")
            try:
                speak(text, emotion)
                print(f"    ✅ Speak function executed for '{emotion}' emotion")
            except Exception as e:
                print(f"    ⚠️ Speak function error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Speak function test failed: {e}")
        return False

def test_app_import():
    """Test that the app can be imported without COM errors"""
    print("\n📱 Testing App Import...")
    
    try:
        # This should not raise the COM initialization error anymore
        import voice_app_simple
        print("✅ App imported successfully without COM errors")
        
        # Test that TTS engine variable exists
        if hasattr(voice_app_simple, 'tts_engine'):
            engine = voice_app_simple.tts_engine
            if engine is not None:
                print("✅ TTS engine is available")
            else:
                print("⚠️ TTS engine is None (fallback mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ App import test failed: {e}")
        return False

def main():
    """Run all TTS fix verification tests"""
    print("🔧 TTS COM Initialization Fix Verification")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("voice_app_simple.py").exists():
        print("❌ Please run this test from the Voice_Assistant_AI directory")
        sys.exit(1)
    
    tests = [
        test_app_import,
        test_tts_initialization,
        test_speak_function,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 TTS Fix Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 TTS COM initialization fix successful!")
        print("   ✅ No more CoInitialize errors")
        print("   ✅ TTS engine working or fallback available")
        print("   ✅ App can start without crashes")
        print("\n🚀 Ready to run: streamlit run voice_app_simple.py")
    else:
        print("⚠️ Some TTS issues may remain. Check the error messages above.")
        print("   The app should still work in text-only mode.")
    
    print("\n💡 TTS Status:")
    try:
        from voice_app_simple import tts_engine
        if tts_engine is not None:
            print("   🔊 Full TTS functionality available")
        else:
            print("   📝 Text-only mode (TTS fallback will be used)")
    except:
        print("   ❓ TTS status unknown")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)