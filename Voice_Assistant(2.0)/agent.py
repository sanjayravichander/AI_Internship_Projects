from dotenv import load_dotenv
import asyncio
import logging
import aiohttp
import os

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import google
from livekit.plugins.noise_cancellation import BVC
from prompts_2 import JARVIS_INSTRUCTION, JARVIS_SESSION_INSTRUCTION
from tools_simple import (
    get_weather, search_web, send_email,
    open_application, open_website, 
    take_screenshot, play_youtube_video,
    system_command, get_system_info, send_whatsapp_message
)

# Enable debug logging for tools
logging.getLogger('tools_simple').setLevel(logging.DEBUG)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_network_connectivity():
    """Test basic network connectivity and Google API accessibility"""
    try:
        logger.info("Testing network connectivity...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get('https://www.google.com') as response:
                if response.status == 200:
                    logger.info("✓ Basic internet connectivity: OK")
                else:
                    logger.warning(f"⚠ Basic internet connectivity: HTTP {response.status}")
        
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            logger.info("✓ Google API key found in environment")
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={google_api_key}"
                async with session.get(test_url) as response:
                    if response.status == 200:
                        logger.info("✓ Google API accessibility: OK")
                    else:
                        logger.warning(f"⚠ Google API accessibility: HTTP {response.status}")
        else:
            logger.error("✗ Google API key not found in environment")
            
    except asyncio.TimeoutError:
        logger.error("✗ Network connectivity test timed out")
    except Exception as e:
        logger.error(f"✗ Network connectivity test failed: {e}")


class AliceAssistant(Agent):
    def __init__(self) -> None:
        logger.info("Initializing Alice - Advanced AI Assistant")
        llm_model = google.beta.realtime.RealtimeModel(
            voice="Aoede",
            temperature=0.7,
        )
        
        tools_list = [
            # Communication & Information
            get_weather,
            search_web,
            send_email,
            
            # System Control
            open_application,
            open_website,
            system_command,
            get_system_info,
            
            # Media & Capture
            take_screenshot,
            play_youtube_video,
            send_whatsapp_message,
        ]
        
        logger.info(f"Registering {len(tools_list)} tools: {[tool.__name__ for tool in tools_list]}")
            
        super().__init__(
            instructions=JARVIS_INSTRUCTION,
            llm=llm_model,
            tools=tools_list,
        )
        
        logger.info(f"Alice initialized with {len(self.tools)} tools")
    
    async def on_tool_calls_finished(self, tool_calls, speech_handle):
        """Ensure agent speaks after tool execution"""
        logger.info(f"Tool calls finished: {len(tool_calls) if tool_calls else 0} calls")
        for i, call in enumerate(tool_calls or []):
            logger.info(f"Tool call {i+1}: {call}")
        try:
            if speech_handle and not speech_handle.interrupted:
                await speech_handle.wait_for_playout()
        except Exception as e:
            logger.warning(f"Tool response handling warning: {e}")
    
    async def on_tool_call_start(self, tool_call):
        """Log when tool call starts"""
        logger.info(f"TOOL CALL STARTING: {tool_call}")
        print(f"[AGENT] Tool call starting: {tool_call}")
    
    async def on_tool_call_finish(self, tool_call, result):
        """Log when tool call finishes"""
        logger.info(f"TOOL CALL FINISHED: {tool_call} -> {result}")
        print(f"[AGENT] Tool call finished: {tool_call} -> {result}")


async def create_alice_with_retry(max_retries=3, retry_delay=5):
    """Create Alice with retry logic for connection issues"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Creating Alice (attempt {attempt + 1}/{max_retries})")
            return AliceAssistant()
        except Exception as e:
            logger.warning(f"Failed to create Alice on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Max retries exceeded for creating Alice")
                raise


async def entrypoint(ctx: agents.JobContext):
    # Test network connectivity first
    await test_network_connectivity()
    
    max_retries = 3
    retry_delay = 10
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Starting Alice session (attempt {attempt + 1}/{max_retries})")
            
            # Create Alice with retry logic
            alice = await create_alice_with_retry()
            
            session = AgentSession()

            await session.start(
                room=ctx.room,
                agent=alice,
                room_input_options=RoomInputOptions(
                    video_enabled=True,
                    noise_cancellation=BVC(),
                ),
            )

            await ctx.connect()

            logger.info("Generating initial reply...")
            await session.generate_reply(
                instructions=JARVIS_SESSION_INSTRUCTION,
            )
            logger.info("Initial reply generated")
            
            logger.info("Alice session completed successfully")
            break
            
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Max retries exceeded")
                raise


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))