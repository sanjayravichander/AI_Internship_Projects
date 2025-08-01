#!/usr/bin/env python3
"""
Test script to diagnose translation import issues
"""

import sys
import os

print("🔍 Testing Translation Import...")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

# Test 1: Basic import
try:
    from deep_translator import GoogleTranslator
    print("✅ deep_translator import successful")
    
    # Test 2: Create translator instance
    translator = GoogleTranslator(source='auto', target='hi')
    print("✅ GoogleTranslator instance created")
    
    # Test 3: Simple translation
    result = translator.translate("Hello, how are you?")
    print(f"✅ Translation test: 'Hello, how are you?' -> '{result}'")
    
    print("\n🎉 Translation library is working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n🔧 To fix this, run:")
    print("pip install deep-translator")
    
except Exception as e:
    print(f"❌ Translation error: {e}")
    print("This might be a network connectivity issue.")

# Test 4: Check if module is in path
try:
    import deep_translator
    print(f"✅ deep_translator module location: {deep_translator.__file__}")
except:
    print("❌ deep_translator module not found in Python path")

# Test 5: List installed packages
try:
    import pkg_resources
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    if 'deep-translator' in installed_packages:
        print("✅ deep-translator is in installed packages")
    else:
        print("❌ deep-translator not found in installed packages")
        print("Available packages containing 'translator':")
        for pkg in installed_packages:
            if 'translator' in pkg.lower():
                print(f"  - {pkg}")
except:
    print("❌ Could not check installed packages")

print("\n" + "="*50)
print("RECOMMENDATION:")
print("Use app_no_translation.py for guaranteed functionality")
print("It has all features except translation and works 100%")
print("="*50)