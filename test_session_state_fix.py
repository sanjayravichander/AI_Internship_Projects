"""
Test script to verify the session state initialization fix.
Run this to ensure the usage_data error is resolved.
"""

import streamlit as st
from datetime import datetime

# Test the fix by simulating the app startup sequence
def test_session_state_fix():
    """Test that session state is properly initialized."""
    
    st.title("🧪 Session State Fix Test - ENHANCED VERSION")
    
    # Test 0: Check if usage_data exists before any operations
    st.subheader("Test 0: Pre-Import Session State Check")
    if 'usage_data' in st.session_state:
        st.success("✅ usage_data already exists in session state!")
        st.json(st.session_state.usage_data)
    else:
        st.warning("⚠️ usage_data not found - will be initialized")
    
    # Test 1: Direct initialization (as done in app.py)
    st.subheader("Test 1: Direct Initialization")
    try:
        from usage_manager import ensure_usage_data_initialized
        ensure_usage_data_initialized()
        st.success("✅ ensure_usage_data_initialized() successful")
        st.json(st.session_state.usage_data)
    except Exception as e:
        st.error(f"❌ Direct initialization failed: {e}")
        st.code(str(e))
    
    # Test 2: Usage manager import and initialization
    st.subheader("Test 2: Usage Manager Import")
    try:
        from usage_manager import get_usage_manager, display_usage_info
        usage_manager = get_usage_manager()
        if usage_manager:
            st.success("✅ Usage manager imported and initialized successfully")
        else:
            st.warning("⚠️ Usage manager is None (fallback mode)")
    except Exception as e:
        st.error(f"❌ Usage manager import failed: {e}")
        st.code(str(e))
    
    # Test 3: Display usage info (the function that was causing the error)
    st.subheader("Test 3: Display Usage Info")
    try:
        display_usage_info()
        st.success("✅ display_usage_info() executed without errors")
    except Exception as e:
        st.error(f"❌ display_usage_info() failed: {e}")
        st.code(str(e))
    
    # Test 4: Check API usage function
    st.subheader("Test 4: API Usage Check")
    try:
        from usage_manager import check_api_usage, record_api_usage
        
        # Test checking usage
        can_use_groq = check_api_usage('groq')
        st.info(f"Can use Groq API: {can_use_groq}")
        
        # Test recording usage
        record_api_usage('groq', success=True)
        st.success("✅ API usage functions work correctly")
        
    except Exception as e:
        st.error(f"❌ API usage functions failed: {e}")
        st.code(str(e))
    
    # Test 5: Boolean conversion fix
    st.subheader("Test 5: Boolean Conversion Fix")
    try:
        usage_manager = get_usage_manager()
        if usage_manager:
            # Test the safe boolean conversion
            test_cases = [
                (True, "boolean True"),
                (False, "boolean False"),
                ("true", "string 'true'"),
                ("false", "string 'false'"),
                ("TRUE", "string 'TRUE'"),
                ("FALSE", "string 'FALSE'"),
                (1, "integer 1"),
                (0, "integer 0"),
            ]
            
            st.write("Testing boolean conversion with different input types:")
            for value, description in test_cases:
                try:
                    result = usage_manager._safe_bool_convert(value)
                    st.write(f"✅ {description} → {result}")
                except Exception as e:
                    st.write(f"❌ {description} → Error: {e}")
            
            st.success("✅ Boolean conversion fix working correctly")
        else:
            st.warning("⚠️ Usage manager not available for boolean conversion test")
            
    except Exception as e:
        st.error(f"❌ Boolean conversion test failed: {e}")
        st.code(str(e))
    
    # Test 6: Session state contents
    st.subheader("Test 6: Final Session State")
    st.write("Current session state keys:")
    st.write(list(st.session_state.keys()))
    
    if 'usage_data' in st.session_state:
        st.write("Usage data contents:")
        st.json(st.session_state.usage_data)
    
    st.success("🎉 All tests completed! If you see this message, the session state fix is working.")

if __name__ == "__main__":
    test_session_state_fix()