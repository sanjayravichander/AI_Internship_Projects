"""
Safe utility functions for handling various data types in Streamlit Cloud
"""

def safe_bool_convert(value, default=False):
    """
    Safely convert any value to boolean, handling all edge cases.
    
    Args:
        value: Any value that needs to be converted to boolean
        default: Default value if conversion fails
        
    Returns:
        bool: Converted boolean value
    """
    if value is None:
        return default
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    if isinstance(value, str):
        return value.lower().strip() in ('true', '1', 'yes', 'on', 'enabled')
    
    # For any other type, try to convert to string first
    try:
        str_value = str(value).lower().strip()
        return str_value in ('true', '1', 'yes', 'on', 'enabled')
    except:
        return default

def safe_int_convert(value, default=0):
    """
    Safely convert any value to integer.
    
    Args:
        value: Any value that needs to be converted to integer
        default: Default value if conversion fails
        
    Returns:
        int: Converted integer value
    """
    if value is None:
        return default
    
    if isinstance(value, int):
        return value
    
    if isinstance(value, float):
        return int(value)
    
    if isinstance(value, bool):
        return int(value)
    
    if isinstance(value, str):
        try:
            # Handle string numbers
            return int(float(value.strip()))
        except:
            return default
    
    # For any other type, try to convert
    try:
        return int(value)
    except:
        return default

def safe_str_convert(value, default=""):
    """
    Safely convert any value to string.
    
    Args:
        value: Any value that needs to be converted to string
        default: Default value if conversion fails
        
    Returns:
        str: Converted string value
    """
    if value is None:
        return default
    
    if isinstance(value, str):
        return value
    
    try:
        return str(value)
    except:
        return default