"""
Configuration file for Voice Assistant 2.0 LiveKit Integration
============================================================

This file contains configuration settings for the Voice Assistant
launcher and LiveKit playground integration.
"""

# LiveKit Playground Configuration
LIVEKIT_PLAYGROUND_URL = "https://agents-playground.livekit.io/"

# Alternative URLs (uncomment to use)
# LIVEKIT_PLAYGROUND_URL = "https://your-custom-livekit-playground.com/"
# LIVEKIT_PLAYGROUND_URL = "http://localhost:3000/"  # For local development

# Assistant Configuration
ASSISTANT_NAME = "Alice"
ASSISTANT_VERSION = "2.0.0"
ASSISTANT_DESCRIPTION = "Advanced AI Assistant with LiveKit Integration"

# UI Configuration
THEME_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"

# Feature Flags
ENABLE_SYSTEM_CONTROL = True
ENABLE_WEB_INTEGRATION = True
ENABLE_MEDIA_CONTROL = True
ENABLE_COMMUNICATION = True

# Status Configuration
STATUS = "ready"
LAST_UPDATED = "2024-01-18"