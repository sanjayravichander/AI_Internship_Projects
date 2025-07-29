#!/usr/bin/env python3
"""
Rate limit handler for Groq API
"""

import time
import re
from typing import Dict, Any, Optional

class RateLimitHandler:
    """Handle Groq API rate limits gracefully"""
    
    def __init__(self):
        self.last_rate_limit_time = 0
        self.rate_limit_duration = 0
    
    def parse_rate_limit_error(self, error_message: str) -> Optional[Dict[str, Any]]:
        """Parse rate limit error message to extract useful information"""
        try:
            # Extract wait time from error message
            wait_time_match = re.search(r'Please try again in (\d+)m(\d+(?:\.\d+)?)s', error_message)
            if wait_time_match:
                minutes = int(wait_time_match.group(1))
                seconds = float(wait_time_match.group(2))
                wait_seconds = minutes * 60 + seconds
                
                # Extract usage information
                used_match = re.search(r'Used (\d+)', error_message)
                limit_match = re.search(r'Limit (\d+)', error_message)
                
                return {
                    'wait_seconds': wait_seconds,
                    'used_tokens': int(used_match.group(1)) if used_match else None,
                    'token_limit': int(limit_match.group(1)) if limit_match else None,
                    'is_rate_limit': True
                }
        except Exception:
            pass
        
        return None
    
    def is_rate_limit_error(self, error_message: str) -> bool:
        """Check if error is a rate limit error"""
        return 'rate_limit_exceeded' in error_message or 'Rate limit reached' in error_message
    
    def get_friendly_message(self, error_info: Dict[str, Any]) -> str:
        """Generate a user-friendly rate limit message"""
        if not error_info.get('is_rate_limit'):
            return "An error occurred while processing your request."
        
        wait_seconds = error_info.get('wait_seconds', 60)
        used_tokens = error_info.get('used_tokens')
        token_limit = error_info.get('token_limit')
        
        message = f"🚨 **API Rate Limit Reached**\n\n"
        
        if used_tokens and token_limit:
            percentage = (used_tokens / token_limit) * 100
            message += f"📊 **Token Usage**: {used_tokens:,} / {token_limit:,} ({percentage:.1f}%)\n\n"
        
        if wait_seconds < 120:  # Less than 2 minutes
            message += f"⏰ **Wait Time**: ~{int(wait_seconds)} seconds\n\n"
            message += "💡 **What to do**: Please wait a moment and try again.\n\n"
        else:
            minutes = int(wait_seconds / 60)
            message += f"⏰ **Wait Time**: ~{minutes} minutes\n\n"
            message += "💡 **What to do**: You've reached your daily token limit. Try again later or upgrade your plan.\n\n"
        
        message += "🔧 **Alternative**: The system will automatically switch to a more efficient model."
        
        return message
    
    def should_retry(self, error_info: Dict[str, Any]) -> bool:
        """Determine if we should retry the request"""
        if not error_info.get('is_rate_limit'):
            return False
        
        wait_seconds = error_info.get('wait_seconds', 0)
        return wait_seconds < 300  # Only retry if wait time is less than 5 minutes

def handle_groq_error(error_message: str) -> Dict[str, Any]:
    """Handle Groq API errors and return appropriate response"""
    handler = RateLimitHandler()
    
    if handler.is_rate_limit_error(error_message):
        error_info = handler.parse_rate_limit_error(error_message)
        if error_info:
            return {
                'is_rate_limit': True,
                'friendly_message': handler.get_friendly_message(error_info),
                'should_retry': handler.should_retry(error_info),
                'wait_seconds': error_info.get('wait_seconds', 60),
                'raw_error': error_message
            }
    
    return {
        'is_rate_limit': False,
        'friendly_message': f"An error occurred: {error_message}",
        'should_retry': False,
        'raw_error': error_message
    }