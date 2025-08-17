import logging
import os
import asyncio
import webbrowser
import subprocess
from datetime import datetime
from typing import Optional
import requests
from ddgs import DDGS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from livekit.agents import function_tool, RunContext
import psutil
import time

# Global process tracking
_active_processes = []

def cleanup_old_processes():
    """Clean up old browser/application processes to prevent accumulation"""
    global _active_processes
    current_time = time.time()
    
    # Remove processes older than 30 seconds or already terminated
    _active_processes = [
        (proc, timestamp) for proc, timestamp in _active_processes
        if current_time - timestamp < 30 and proc.poll() is None
    ]
    
    print(f"[DEBUG] Active processes: {len(_active_processes)}")

def add_process_to_tracking(process):
    """Add a process to tracking list"""
    global _active_processes
    cleanup_old_processes()  # Clean up first
    _active_processes.append((process, time.time()))
    print(f"[DEBUG] Added process to tracking. Total: {len(_active_processes)}")

# Original working tools
@function_tool()
async def get_weather(context: RunContext, city: str) -> str:
    """Get current weather for a city."""
    try:
        response = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        if response.status_code == 200:
            return response.text.strip()
        return f"Weather unavailable for {city}"
    except Exception as e:
        return f"Weather service temporarily unavailable for {city}"

@function_tool()
async def search_web(context: RunContext, query: str) -> str:
    """Search the web for current information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                formatted_output = []
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'No title')
                    body = result.get('body', 'No description')
                    if len(body) > 100:
                        body = body[:100] + "..."
                    formatted_output.append(f"{i}. {title}: {body}")
                return "\n\n".join(formatted_output)
            return f"No results found for '{query}'"
    except Exception as e:
        return f"Search temporarily unavailable for '{query}'"

@function_tool()
async def send_email(context: RunContext, to_email: str, subject: str, message: str) -> str:
    """Send email through Gmail."""
    try:
        gmail_user = os.getenv("GMAIL_USER")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        
        if not gmail_user or not gmail_password:
            return "Email credentials not configured"
        
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, [to_email], msg.as_string())
        server.quit()
        
        return f"Email sent to {to_email}, Sir"
    except Exception as e:
        return f"Email failed: {str(e)[:50]}"

@function_tool()
async def open_website(context: RunContext, website: str) -> str:
    """Open any website in the default browser."""
    print(f"[DEBUG] open_website called with: {website}")
    logging.info(f"TOOL CALLED: open_website with {website}")
    
    # Clean up old processes first
    cleanup_old_processes()
    
    try:
        if not website.startswith(('http://', 'https://')):
            if '.' not in website:
                website = f"https://www.{website}.com"
            else:
                website = f"https://{website}"
        
        print(f"[DEBUG] Opening URL: {website}")
        
        # Try methods in order, stop on first success
        success = False
        
        # Method 1: webbrowser module (most reliable)
        try:
            webbrowser.open(website)
            success = True
            print(f"[DEBUG] Successfully opened with webbrowser module")
        except Exception as e:
            print(f"[DEBUG] webbrowser.open failed: {e}")
        
        # Method 2: Only if first method failed
        if not success:
            try:
                process = subprocess.run(['start', website], shell=True, check=True, timeout=5)
                success = True
                print(f"[DEBUG] Successfully opened with subprocess")
            except Exception as e:
                print(f"[DEBUG] subprocess failed: {e}")
        
        # Method 3: Only if both previous methods failed
        if not success:
            try:
                os.system(f'start {website}')
                success = True
                print(f"[DEBUG] Successfully opened with os.system")
            except Exception as e:
                print(f"[DEBUG] os.system failed: {e}")
        
        if success:
            result = f"Opened {website}, Sir"
            print(f"[DEBUG] Returning: {result}")
            logging.info(f"TOOL RESULT: {result}")
            return result
        else:
            error_msg = f"Could not open website: All methods failed"
            print(f"[DEBUG] Error: {error_msg}")
            logging.error(f"Error opening website {website}: All methods failed")
            return error_msg
            
    except Exception as e:
        error_msg = f"Could not open website: {str(e)[:50]}"
        print(f"[DEBUG] Error: {error_msg}")
        logging.error(f"Error opening website {website}: {e}")
        return error_msg

def _is_running_in_livekit_playground():
    """Detect if running in LiveKit playground environment"""
    # Check for LiveKit-specific environment variables or context
    livekit_indicators = [
        os.getenv('LIVEKIT_URL'),
        os.getenv('LIVEKIT_API_KEY'),
        os.getenv('LIVEKIT_API_SECRET')
    ]
    
    # If any LiveKit environment variables are set, we're likely in playground mode
    if any(livekit_indicators):
        return True
    
    # Additional checks for web-based environment
    try:
        # In a web environment, certain system operations might be restricted
        import platform
        if platform.system() == "Linux" and os.getenv('DISPLAY') is None:
            # Likely running in a headless server environment (like LiveKit playground)
            return True
    except:
        pass
    
    return False

@function_tool()
async def open_application(context: RunContext, app_name: str) -> str:
    """Open applications and websites."""
    print(f"[DEBUG] open_application called with: {app_name}")
    logging.info(f"TOOL CALLED: open_application with {app_name}")
    
    # Clean up old processes first
    cleanup_old_processes()
    
    # Check if running in LiveKit playground
    is_playground = _is_running_in_livekit_playground()
    print(f"[DEBUG] Running in LiveKit playground: {is_playground}")
    
    try:
        app_name = app_name.lower().strip()
        
        # Web applications - these always work
        web_apps = {
            'youtube': 'https://www.youtube.com',
            'instagram': 'https://www.instagram.com',
            'facebook': 'https://www.facebook.com',
            'twitter': 'https://www.twitter.com',
            'whatsapp': 'https://web.whatsapp.com',
            'gmail': 'https://mail.google.com',
            'google': 'https://www.google.com',
            'netflix': 'https://www.netflix.com',
            'amazon prime': 'https://www.primevideo.com',
            'prime video': 'https://www.primevideo.com',
            'disney plus': 'https://www.disneyplus.com',
            'hulu': 'https://www.hulu.com',
            'spotify': 'https://open.spotify.com',
            'amazon': 'https://www.amazon.com'
        }
        
        if app_name in web_apps:
            url = web_apps[app_name]
            print(f"[DEBUG] Opening web app: {url}")
            
            # Try methods in order, stop on first success
            success = False
            
            # Method 1: webbrowser module (most reliable)
            try:
                webbrowser.open(url)
                success = True
                print(f"[DEBUG] Successfully opened {app_name} with webbrowser")
            except Exception as e:
                print(f"[DEBUG] webbrowser.open failed for {app_name}: {e}")
            
            # Method 2: Only if first method failed
            if not success:
                try:
                    subprocess.run(['start', url], shell=True, check=True, timeout=5)
                    success = True
                    print(f"[DEBUG] Successfully opened {app_name} with subprocess")
                except Exception as e:
                    print(f"[DEBUG] subprocess failed for {app_name}: {e}")
            
            if success:
                result = f"Opened {app_name}, Sir"
                print(f"[DEBUG] Web app result: {result}")
                logging.info(f"TOOL RESULT: {result}")
                return result
            else:
                result = f"Failed to open {app_name}, Sir"
                print(f"[DEBUG] Web app failed: {result}")
                logging.error(f"TOOL ERROR: {result}")
                return result
        
        # System applications using Windows start command
        system_apps = {
            'calculator': 'calc',
            'notepad': 'notepad',
            'paint': 'mspaint',
            'camera': 'microsoft.windows.camera:',
            'chrome': 'chrome',
            'firefox': 'firefox',
            'edge': 'msedge'
        }
        
        if app_name in system_apps:
            # In LiveKit playground, system apps can't be opened directly
            if is_playground:
                result = f"I cannot open {app_name} directly in the web environment, Sir. System applications require local access. Please use the console version for system control."
                print(f"[DEBUG] Playground limitation: {result}")
                logging.info(f"TOOL RESULT (Playground): {result}")
                return result
            
            cmd = f'start {system_apps[app_name]}'
            print(f"[DEBUG] Running system command: {cmd}")
            
            # Try methods in order, stop on first success
            success = False
            
            # Method 1: Direct subprocess with shell
            try:
                result = subprocess.run(cmd, shell=True, check=True, timeout=10, 
                                      creationflags=subprocess.CREATE_NEW_CONSOLE)
                if result.returncode == 0:
                    success = True
                    print(f"[DEBUG] Successfully opened {app_name} with subprocess")
            except Exception as e:
                print(f"[DEBUG] subprocess failed for {app_name}: {e}")
            
            # Method 2: Only if first method failed
            if not success:
                try:
                    process = await asyncio.create_subprocess_shell(
                        cmd, 
                        stdout=asyncio.subprocess.PIPE, 
                        stderr=asyncio.subprocess.PIPE,
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
                    if process.returncode == 0:
                        success = True
                        print(f"[DEBUG] Successfully opened {app_name} with async subprocess")
                    else:
                        print(f"[DEBUG] Async subprocess failed: {stderr.decode() if stderr else 'Unknown error'}")
                except Exception as e:
                    print(f"[DEBUG] Async subprocess error for {app_name}: {e}")
            
            if success:
                result = f"Opened {app_name}, Sir"
                logging.info(f"TOOL RESULT: {result}")
                return result
            else:
                result = f"Failed to open {app_name}, Sir"
                logging.error(f"TOOL ERROR: {result}")
                return result
        
        # Try generic start command for unknown applications
        if is_playground:
            # In playground, we can't open arbitrary applications
            result = f"I cannot open '{app_name}' in the web environment, Sir. For system applications, please use the console version. For web services, try asking me to open the website instead."
            print(f"[DEBUG] Playground limitation for generic app: {result}")
            logging.info(f"TOOL RESULT (Playground): {result}")
            return result
        
        cmd = f'start {app_name}'
        print(f"[DEBUG] Running generic command: {cmd}")
        
        success = False
        
        # Method 1: Direct subprocess with shell
        try:
            result = subprocess.run(cmd, shell=True, check=True, timeout=10, 
                                  creationflags=subprocess.CREATE_NEW_CONSOLE)
            if result.returncode == 0:
                success = True
                print(f"[DEBUG] Successfully opened {app_name} with generic command")
        except Exception as e:
            print(f"[DEBUG] Generic subprocess failed for {app_name}: {e}")
        
        # Method 2: Only if first method failed
        if not success:
            try:
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
                if process.returncode == 0:
                    success = True
                    print(f"[DEBUG] Successfully opened {app_name} with async generic command")
                else:
                    print(f"[DEBUG] Async generic command failed: {stderr.decode() if stderr else 'Unknown error'}")
            except Exception as e:
                print(f"[DEBUG] Async generic command error for {app_name}: {e}")
        
        if success:
            result = f"Opened {app_name}, Sir"
            print(f"[DEBUG] Generic result: {result}")
            logging.info(f"TOOL RESULT: {result}")
            return result
        else:
            result = f"Could not open {app_name}, Sir"
            print(f"[DEBUG] Generic failed: {result}")
            logging.error(f"TOOL ERROR: {result}")
            return result
        
    except Exception as e:
        error_msg = f"Could not open {app_name}: {str(e)[:50]}"
        print(f"[DEBUG] Exception: {error_msg}")
        logging.error(f"Error opening {app_name}: {e}")
        return error_msg

@function_tool()
async def play_youtube_video(context: RunContext, search_query: str) -> str:
    """Search and play a video on YouTube."""
    print(f"[DEBUG] play_youtube_video called with: {search_query}")
    
    # Clean up old processes first
    cleanup_old_processes()
    
    try:
        search_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
        print(f"[DEBUG] YouTube URL: {search_url}")
        
        # Try methods in order, stop on first success
        success = False
        
        # Method 1: webbrowser module (most reliable)
        try:
            webbrowser.open(search_url)
            success = True
            print(f"[DEBUG] Successfully opened YouTube with webbrowser module")
        except Exception as e:
            print(f"[DEBUG] webbrowser.open failed for YouTube: {e}")
        
        # Method 2: Only if first method failed
        if not success:
            try:
                subprocess.run(['start', search_url], shell=True, check=True, timeout=5, 
                             creationflags=subprocess.CREATE_NEW_CONSOLE)
                success = True
                print(f"[DEBUG] Successfully opened YouTube with subprocess")
            except Exception as e:
                print(f"[DEBUG] subprocess failed for YouTube: {e}")
        
        # Method 3: Only if both previous methods failed
        if not success:
            try:
                os.system(f'start {search_url}')
                success = True
                print(f"[DEBUG] Successfully opened YouTube with os.system")
            except Exception as e:
                print(f"[DEBUG] os.system failed for YouTube: {e}")
        
        if success:
            result = f"Searching YouTube for '{search_query}', Sir"
            print(f"[DEBUG] YouTube result: {result}")
            return result
        else:
            error_msg = f"YouTube search failed: All methods failed"
            print(f"[DEBUG] YouTube error: {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"YouTube search failed: {str(e)[:50]}"
        print(f"[DEBUG] YouTube error: {error_msg}")
        return error_msg

@function_tool()
async def send_whatsapp_message(context: RunContext, contact_name: str, message: str) -> str:
    """Send WhatsApp message through WhatsApp Web."""
    print(f"[DEBUG] send_whatsapp_message called: {contact_name} - {message}")
    try:
        whatsapp_url = "https://web.whatsapp.com"
        print(f"[DEBUG] Opening WhatsApp Web: {whatsapp_url}")
        
        # Try methods in order, stop on first success
        success = False
        
        # Method 1: webbrowser module (most reliable)
        try:
            webbrowser.open(whatsapp_url)
            success = True
            print(f"[DEBUG] Successfully opened WhatsApp with webbrowser module")
        except Exception as e:
            print(f"[DEBUG] webbrowser.open failed for WhatsApp: {e}")
        
        # Method 2: Only if first method failed
        if not success:
            try:
                subprocess.run(['start', whatsapp_url], shell=True, check=True, timeout=5)
                success = True
                print(f"[DEBUG] Successfully opened WhatsApp with subprocess")
            except Exception as e:
                print(f"[DEBUG] subprocess failed for WhatsApp: {e}")
        
        if success:
            result = f"WhatsApp Web opened, Sir. Please find {contact_name} and send: '{message}'"
            print(f"[DEBUG] WhatsApp result: {result}")
            return result
        else:
            error_msg = f"WhatsApp failed: All methods failed"
            print(f"[DEBUG] WhatsApp error: {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"WhatsApp failed: {str(e)[:50]}"
        print(f"[DEBUG] WhatsApp error: {error_msg}")
        return error_msg

@function_tool()
async def system_command(context: RunContext, command: str) -> str:
    """Execute simple system commands."""
    print(f"[DEBUG] system_command called with: {command}")
    logging.info(f"TOOL CALLED: system_command with {command}")
    
    # Check if running in LiveKit playground
    is_playground = _is_running_in_livekit_playground()
    print(f"[DEBUG] Running in LiveKit playground: {is_playground}")
    
    if is_playground:
        result = f"I cannot execute system commands in the web environment, Sir. System control requires local access. Please use the console version for system commands like volume control, screen lock, etc."
        print(f"[DEBUG] Playground limitation for system command: {result}")
        logging.info(f"TOOL RESULT (Playground): {result}")
        return result
    
    try:
        command = command.lower().strip()
        
        if command in ['volume up', 'increase volume']:
            cmd = 'powershell -c "(New-Object -comObject WScript.Shell).SendKeys([char]175)"'
            print(f"[DEBUG] Running volume up command: {cmd}")
            for _ in range(3):
                process = await asyncio.create_subprocess_shell(cmd)
                await process.wait()
            result = "Volume increased, Sir"
            print(f"[DEBUG] Volume result: {result}")
            return result
        
        elif command in ['volume down', 'decrease volume']:
            cmd = 'powershell -c "(New-Object -comObject WScript.Shell).SendKeys([char]174)"'
            print(f"[DEBUG] Running volume down command: {cmd}")
            for _ in range(3):
                process = await asyncio.create_subprocess_shell(cmd)
                await process.wait()
            result = "Volume decreased, Sir"
            print(f"[DEBUG] Volume result: {result}")
            return result
        
        elif command in ['mute', 'volume mute']:
            cmd = 'powershell -c "(New-Object -comObject WScript.Shell).SendKeys([char]173)"'
            print(f"[DEBUG] Running mute command: {cmd}")
            process = await asyncio.create_subprocess_shell(cmd)
            await process.wait()
            result = "Volume muted, Sir"
            print(f"[DEBUG] Mute result: {result}")
            return result
        
        elif command in ['lock screen', 'lock']:
            cmd = 'rundll32.exe user32.dll,LockWorkStation'
            print(f"[DEBUG] Running lock command: {cmd}")
            process = await asyncio.create_subprocess_shell(cmd)
            await process.wait()
            result = "Screen locked, Sir"
            print(f"[DEBUG] Lock result: {result}")
            return result
        
        result = f"Command '{command}' not recognized, Sir"
        print(f"[DEBUG] Unrecognized command: {result}")
        return result
        
    except Exception as e:
        error_msg = f"System command failed: {str(e)[:50]}"
        print(f"[DEBUG] System command exception: {error_msg}")
        return error_msg

@function_tool()
async def take_screenshot(context: RunContext) -> str:
    """Take a screenshot using Windows built-in tool."""
    # Check if running in LiveKit playground
    is_playground = _is_running_in_livekit_playground()
    
    if is_playground:
        result = f"I cannot take screenshots in the web environment, Sir. Screenshot functionality requires local system access. Please use the console version for system control features."
        print(f"[DEBUG] Playground limitation for screenshot: {result}")
        logging.info(f"TOOL RESULT (Playground): {result}")
        return result
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        
        cmd = f'powershell -c "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait(\'%{{PRTSC}}\'); Start-Sleep 1"'
        process = await asyncio.create_subprocess_shell(cmd)
        await process.wait()
        
        return f"Screenshot taken, Sir. Use Ctrl+V to paste it."
    except Exception as e:
        return f"Screenshot failed: {str(e)[:50]}"

@function_tool()
async def get_system_info(context: RunContext) -> str:
    """Get basic system information."""
    try:
        import platform
        system = platform.system()
        version = platform.version()
        machine = platform.machine()
        return f"System: {system} {version}, Architecture: {machine}, Sir"
    except Exception as e:
        return f"System info failed: {str(e)[:50]}"