#!/usr/bin/env python3
"""
Test script to verify the fixes in groq_cartoonify.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def test_imports():
    """Test if all imports work correctly"""
    try:
        # Test basic imports
        import streamlit as st
        import base64
        from io import BytesIO
        from PIL import Image, ImageEnhance, ImageFilter
        import numpy as np
        import cv2
        import time
        import tempfile
        print("✅ Basic imports successful")
        
        # Test the main module
        import groq_cartoonify
        print("✅ groq_cartoonify module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_function_definitions():
    """Test if all functions are properly defined"""
    try:
        import groq_cartoonify
        
        # Check if key functions exist
        functions_to_check = [
            'process_video',
            'capture_image_from_camera', 
            'process_image_input',
            'apply_cartoon_filter',
            'enhance_cartoon_colors',
            'main'
        ]
        
        for func_name in functions_to_check:
            if hasattr(groq_cartoonify, func_name):
                print(f"✅ Function {func_name} exists")
            else:
                print(f"❌ Function {func_name} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error checking functions: {e}")
        return False

def test_video_function_structure():
    """Test if the video processing function has proper error handling"""
    try:
        import groq_cartoonify
        import inspect
        
        # Get the source code of process_video function
        source = inspect.getsource(groq_cartoonify.process_video)
        
        # Check for proper variable initialization
        if "temp_video_path = None" in source and "output_path = None" in source:
            print("✅ Video function has proper variable initialization")
        else:
            print("❌ Video function missing proper variable initialization")
            return False
            
        # Check for finally block
        if "finally:" in source:
            print("✅ Video function has proper cleanup in finally block")
        else:
            print("❌ Video function missing finally block")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error checking video function: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing groq_cartoonify.py fixes...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Function Definition Test", test_function_definitions),
        ("Video Function Structure Test", test_video_function_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes should work correctly.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)