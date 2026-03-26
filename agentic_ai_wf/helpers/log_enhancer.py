"""
LLM-powered log message enhancer for conversational, MDX-formatted logs.

This module transforms terse technical logs into detailed, user-friendly
messages in MDX format for real-time display on the frontend.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import os
logger = logging.getLogger(__name__)

# Initialize OpenAI client (reuse across calls)
_openai_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    """Get or create OpenAI async client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


async def enhance_log_message(
    log_data: Dict[str, Any],
    timeout: float = 3.0
) -> Optional[str]:
    """
    Transform a technical log message into conversational MDX format.
    
    Args:
        log_data: Complete log data dict with context
        timeout: Max seconds to wait for LLM response
    
    Returns:
        Enhanced MDX message or None if failed
    """
    try:
        client = get_openai_client()
        
        # Build context for LLM
        agent_name = log_data.get("agent_name", "System")
        message = log_data.get("log_message", "")
        step = log_data.get("step", "").replace("_", " ").title()
        log_level = log_data.get("log_level", "INFO")
        status = log_data.get("status", "running")
        current_step = log_data.get("current_step", 0)
        total_steps = log_data.get("total_steps", 0)
        error_msg = log_data.get("error_message", "")
        
        # Create prompt
        prompt = f"""Transform this technical log into a detailed, conversational message in MDX format for a scientific research portal.

            **Context:**
            - Agent: {agent_name}
            - Step: {step} ({current_step}/{total_steps})
            - Status: {status}
            - Level: {log_level}
            - Original: {message}
            {f"- Error: {error_msg}" if error_msg else ""}

            **Requirements:**
            1. Be conversational and reassuring
            2. Explain what's happening in scientific context
            3. Use MDX formatting (headers, bold, lists, code blocks)
            4. Keep it under 200 words
            5. Include emoji where appropriate
            6. If error, explain what went wrong and next steps

            **Output only the MDX message, no preamble:**
            """

        # Call LLM with timeout
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that transforms technical logs into user-friendly, detailed messages in MDX format for a bioinformatics research platform."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            ),
            timeout=timeout
        )
        
        enhanced = response.choices[0].message.content.strip()
        
        # Validate it's not empty
        if not enhanced or len(enhanced) < 10:
            logger.warning("LLM returned empty/short enhancement")
            return None
        
        return enhanced
        
    except asyncio.TimeoutError:
        # Timeout is expected in production - use debug level to avoid noise
        logger.debug(f"Log enhancement timed out after {timeout}s (non-critical)")
        return None
    except Exception as e:
        # Use debug level for production - enhancement failures shouldn't affect workflow
        logger.debug(f"Log enhancement failed (non-critical): {e}")
        return None


async def enhance_and_update_log(
    log_id: str,
    log_data: Dict[str, Any],
    update_callback: Optional[callable] = None
) -> None:
    """
    Enhance a log message and update it in DB + re-broadcast.
    
    This runs in background without blocking the original log flow.
    
    Args:
        log_id: DB log ID to update
        log_data: Original log data
        update_callback: Optional async function to call with (log_id, enhanced_msg)
    """
    try:
        # Enhance message
        enhanced_msg = await enhance_log_message(log_data)
        
        if not enhanced_msg:
            return  # Graceful failure - original message stays
        
        # Update DB via callback
        if update_callback:
            await update_callback(log_id, enhanced_msg)
        
        logger.debug(f"Enhanced log {log_id}: {len(enhanced_msg)} chars")
        
    except Exception as e:
        # Use debug level - enhancement failures are non-critical
        logger.debug(f"Background enhancement failed for {log_id} (non-critical): {e}")
        # Graceful failure - don't crash the workflow


def schedule_log_enhancement(
    log_id: str,
    log_data: Dict[str, Any],
    update_callback: Optional[callable] = None
) -> asyncio.Task:
    """
    Schedule log enhancement in background (fire-and-forget).
    
    Args:
        log_id: DB log ID
        log_data: Original log data
        update_callback: Function to update DB + re-broadcast
    
    Returns:
        asyncio.Task (can be ignored or awaited)
    """
    return asyncio.create_task(
        enhance_and_update_log(log_id, log_data, update_callback)
    )

