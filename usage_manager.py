"""
🚀 AI INTERNSHIP PROJECTS - USAGE MANAGER
==========================================

Comprehensive usage tracking and rate limiting system for public deployment.
Manages API quotas, session limits, and graceful degradation.

Author: AI Intern
Version: 1.0.0 - Public Deployment Edition
"""

import streamlit as st
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import hashlib
import os

class UsageManager:
    """
    Manages API usage, rate limiting, and quotas for public deployment.
    """
    
    def __init__(self):
        self.session_id = self._get_session_id()
        self._initialize_session_state()
        self._load_config()
    
    def _get_session_id(self) -> str:
        """Generate a unique session ID for tracking."""
        if 'session_id' not in st.session_state:
            # Create a unique session ID based on timestamp and random data
            session_data = f"{time.time()}_{hash(str(st.session_state))}"
            st.session_state.session_id = hashlib.md5(session_data.encode()).hexdigest()[:16]
        return st.session_state.session_id
    
    def _initialize_session_state(self):
        """Initialize session state for usage tracking."""
        if 'usage_data' not in st.session_state:
            st.session_state.usage_data = {
                'groq_requests': 0,
                'openai_requests': 0,
                'email_sends': 0,
                'weather_requests': 0,
                'session_start': datetime.now().isoformat(),
                'last_request_times': {},
                'total_requests': 0
            }
    
    def _safe_bool_convert(self, value, default='true'):
        """Safely convert a value to boolean, handling both string and bool types."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == 'true'
        return str(default).lower() == 'true'
    
    def _safe_int_convert(self, value, default):
        """Safely convert a value to integer with fallback."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return int(default)
    
    def _load_config(self):
        """Load configuration from Streamlit secrets or environment."""
        try:
            # Try to load from Streamlit secrets first (for cloud deployment)
            if hasattr(st, 'secrets') and st.secrets:
                self.config = {
                    'daily_limits': {
                        'groq': self._safe_int_convert(st.secrets.get('DAILY_GROQ_REQUESTS', 1000), 1000),
                        'openai': self._safe_int_convert(st.secrets.get('DAILY_OPENAI_REQUESTS', 100), 100),
                        'email': self._safe_int_convert(st.secrets.get('DAILY_EMAIL_SENDS', 50), 50),
                        'weather': self._safe_int_convert(st.secrets.get('DAILY_WEATHER_REQUESTS', 500), 500)
                    },
                    'session_limits': {
                        'groq': self._safe_int_convert(st.secrets.get('SESSION_GROQ_REQUESTS', 50), 50),
                        'openai': self._safe_int_convert(st.secrets.get('SESSION_OPENAI_REQUESTS', 10), 10),
                        'email': self._safe_int_convert(st.secrets.get('SESSION_EMAIL_SENDS', 5), 5),
                        'weather': self._safe_int_convert(st.secrets.get('SESSION_WEATHER_REQUESTS', 20), 20)
                    },
                    'rate_limits': {
                        'groq': self._safe_int_convert(st.secrets.get('GROQ_RATE_LIMIT', 30), 30),
                        'openai': self._safe_int_convert(st.secrets.get('OPENAI_RATE_LIMIT', 10), 10),
                        'email': self._safe_int_convert(st.secrets.get('EMAIL_RATE_LIMIT', 2), 2),
                        'weather': self._safe_int_convert(st.secrets.get('WEATHER_RATE_LIMIT', 10), 10)
                    },
                    'enable_rate_limiting': self._safe_bool_convert(st.secrets.get('ENABLE_RATE_LIMITING', 'true')),
                    'graceful_degradation': self._safe_bool_convert(st.secrets.get('GRACEFUL_DEGRADATION', 'true'))
                }
            else:
                # Fallback to environment variables or defaults
                self.config = {
                    'daily_limits': {
                        'groq': self._safe_int_convert(os.getenv('DAILY_GROQ_REQUESTS', 1000), 1000),
                        'openai': self._safe_int_convert(os.getenv('DAILY_OPENAI_REQUESTS', 100), 100),
                        'email': self._safe_int_convert(os.getenv('DAILY_EMAIL_SENDS', 50), 50),
                        'weather': self._safe_int_convert(os.getenv('DAILY_WEATHER_REQUESTS', 500), 500)
                    },
                    'session_limits': {
                        'groq': self._safe_int_convert(os.getenv('SESSION_GROQ_REQUESTS', 50), 50),
                        'openai': self._safe_int_convert(os.getenv('SESSION_OPENAI_REQUESTS', 10), 10),
                        'email': self._safe_int_convert(os.getenv('SESSION_EMAIL_SENDS', 5), 5),
                        'weather': self._safe_int_convert(os.getenv('SESSION_WEATHER_REQUESTS', 20), 20)
                    },
                    'rate_limits': {
                        'groq': self._safe_int_convert(os.getenv('GROQ_RATE_LIMIT', 30), 30),
                        'openai': self._safe_int_convert(os.getenv('OPENAI_RATE_LIMIT', 10), 10),
                        'email': self._safe_int_convert(os.getenv('EMAIL_RATE_LIMIT', 2), 2),
                        'weather': self._safe_int_convert(os.getenv('WEATHER_RATE_LIMIT', 10), 10)
                    },
                    'enable_rate_limiting': self._safe_bool_convert(os.getenv('ENABLE_RATE_LIMITING', 'true')),
                    'graceful_degradation': self._safe_bool_convert(os.getenv('GRACEFUL_DEGRADATION', 'true'))
                }
        except Exception as e:
            # Use conservative defaults if configuration fails
            st.warning(f"⚠️ Using default usage limits due to configuration error: {e}")
            self.config = {
                'daily_limits': {'groq': 500, 'openai': 50, 'email': 25, 'weather': 250},
                'session_limits': {'groq': 25, 'openai': 5, 'email': 3, 'weather': 10},
                'rate_limits': {'groq': 15, 'openai': 5, 'email': 1, 'weather': 5},
                'enable_rate_limiting': True,
                'graceful_degradation': True
            }
    
    def check_usage_limit(self, service: str) -> Dict[str, Any]:
        """
        Check if the service usage is within limits.
        
        Args:
            service: Service name ('groq', 'openai', 'email', 'weather')
            
        Returns:
            Dict with 'allowed', 'reason', 'remaining' keys
        """
        # Ensure usage_data is initialized
        self._initialize_session_state()
        
        current_usage = st.session_state.usage_data.get(f'{service}_requests', 0)
        session_limit = self.config['session_limits'].get(service, 10)
        
        # Check session limit
        if current_usage >= session_limit:
            return {
                'allowed': False,
                'reason': f'Session limit reached ({current_usage}/{session_limit})',
                'remaining': 0,
                'limit_type': 'session'
            }
        
        # Check rate limiting if enabled
        if self.config['enable_rate_limiting']:
            if not self._check_rate_limit(service):
                return {
                    'allowed': False,
                    'reason': f'Rate limit exceeded. Please wait a moment.',
                    'remaining': session_limit - current_usage,
                    'limit_type': 'rate'
                }
        
        return {
            'allowed': True,
            'reason': 'Within limits',
            'remaining': session_limit - current_usage,
            'limit_type': None
        }
    
    def _check_rate_limit(self, service: str) -> bool:
        """Check if the service is within rate limits."""
        # Ensure usage_data is initialized
        self._initialize_session_state()
        
        now = time.time()
        rate_limit = self.config['rate_limits'].get(service, 10)
        
        # Get last request times for this service
        service_times = st.session_state.usage_data['last_request_times'].get(service, [])
        
        # Remove requests older than 1 minute
        service_times = [t for t in service_times if now - t < 60]
        
        # Check if we're within the rate limit
        if len(service_times) >= rate_limit:
            return False
        
        # Update the request times
        service_times.append(now)
        st.session_state.usage_data['last_request_times'][service] = service_times
        
        return True
    
    def record_usage(self, service: str, success: bool = True):
        """Record usage of a service."""
        # Ensure usage_data is initialized
        self._initialize_session_state()
        
        if success:
            current_count = st.session_state.usage_data.get(f'{service}_requests', 0)
            st.session_state.usage_data[f'{service}_requests'] = current_count + 1
            st.session_state.usage_data['total_requests'] = st.session_state.usage_data.get('total_requests', 0) + 1
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        # Ensure usage_data is initialized
        self._initialize_session_state()
        
        usage_data = st.session_state.usage_data
        stats = {
            'session_id': self.session_id,
            'session_start': usage_data.get('session_start'),
            'total_requests': usage_data.get('total_requests', 0),
            'services': {}
        }
        
        for service in ['groq', 'openai', 'email', 'weather']:
            current_usage = usage_data.get(f'{service}_requests', 0)
            session_limit = self.config['session_limits'].get(service, 10)
            
            stats['services'][service] = {
                'used': current_usage,
                'limit': session_limit,
                'remaining': max(0, session_limit - current_usage),
                'percentage': min(100, (current_usage / session_limit) * 100) if session_limit > 0 else 0
            }
        
        return stats
    
    def display_usage_dashboard(self):
        """Display a usage dashboard in the sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Usage Statistics")
        
        stats = self.get_usage_stats()
        
        # Total requests
        st.sidebar.metric(
            "Total Requests",
            stats['total_requests'],
            help="Total API requests made in this session"
        )
        
        # Service-specific usage
        for service, data in stats['services'].items():
            if data['used'] > 0 or service == 'groq':  # Always show Groq, others only if used
                service_name = service.title()
                
                # Create a progress bar
                progress = data['percentage'] / 100
                color = "🟢" if progress < 0.7 else "🟡" if progress < 0.9 else "🔴"
                
                st.sidebar.markdown(f"**{color} {service_name}**")
                st.sidebar.progress(progress)
                st.sidebar.caption(f"{data['used']}/{data['limit']} requests ({data['remaining']} remaining)")
        
        # Session info
        if stats['session_start']:
            try:
                start_time = datetime.fromisoformat(stats['session_start'])
                duration = datetime.now() - start_time
                st.sidebar.caption(f"Session: {duration.seconds // 60}m {duration.seconds % 60}s")
            except:
                pass
    
    def get_demo_mode_message(self, service: str) -> str:
        """Get a user-friendly message for demo mode."""
        messages = {
            'groq': "🎯 **Demo Mode**: You've reached the session limit for AI requests. The app will continue with cached responses and limited functionality.",
            'openai': "🎯 **Demo Mode**: OpenAI requests are limited. Switching to alternative AI models.",
            'email': "📧 **Demo Mode**: Email sending is limited to prevent spam. Feature temporarily disabled.",
            'weather': "🌤️ **Demo Mode**: Weather requests are limited. Using cached weather data."
        }
        return messages.get(service, f"🎯 **Demo Mode**: {service.title()} service temporarily limited.")
    
    def should_show_upgrade_message(self) -> bool:
        """Determine if we should show an upgrade/API key message."""
        stats = self.get_usage_stats()
        total_usage = sum(data['used'] for data in stats['services'].values())
        return total_usage > 20  # Show after significant usage

# Global usage manager instance (lazy-loaded)
_usage_manager = None

def get_usage_manager():
    """Get or create the global usage manager instance."""
    global _usage_manager
    if _usage_manager is None:
        try:
            _usage_manager = UsageManager()
        except Exception as e:
            # If initialization fails, create a minimal fallback
            st.warning(f"⚠️ Usage manager initialization failed: {e}")
            _usage_manager = None
    return _usage_manager

def check_api_usage(service: str) -> bool:
    """
    Convenient function to check API usage before making requests.
    
    Args:
        service: Service name ('groq', 'openai', 'email', 'weather')
        
    Returns:
        bool: True if request is allowed, False otherwise
    """
    try:
        usage_manager = get_usage_manager()
        if usage_manager is None:
            return True  # Allow requests if usage manager is not available
            
        result = usage_manager.check_usage_limit(service)
        
        if not result['allowed']:
            if result['limit_type'] == 'rate':
                st.warning(f"⏱️ {result['reason']}")
                time.sleep(1)  # Brief pause for rate limiting
            else:
                st.info(usage_manager.get_demo_mode_message(service))
            return False
        
        return True
    except Exception as e:
        st.warning(f"⚠️ Usage check error: {e}")
        return True  # Allow requests if check fails

def record_api_usage(service: str, success: bool = True):
    """
    Record API usage after making a request.
    
    Args:
        service: Service name ('groq', 'openai', 'email', 'weather')
        success: Whether the request was successful
    """
    try:
        usage_manager = get_usage_manager()
        if usage_manager is not None:
            usage_manager.record_usage(service, success)
    except Exception as e:
        # Silently fail for usage recording to not break the main functionality
        pass

def display_usage_info():
    """Display usage information in the sidebar."""
    try:
        usage_manager = get_usage_manager()
        if usage_manager is not None:
            usage_manager.display_usage_dashboard()
            
            # Show upgrade message if appropriate
            if usage_manager.should_show_upgrade_message():
                st.sidebar.markdown("---")
                st.sidebar.info("""
                💡 **Want unlimited access?**
                
                This is a demo deployment with usage limits. For unlimited access:
                1. Get your own API keys
                2. Run locally with your keys
                3. Deploy your own instance
                
                [View Setup Instructions](https://github.com/your-repo)
                """)
        else:
            st.sidebar.info("📊 Usage tracking temporarily unavailable")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Usage tracking error: {e}")