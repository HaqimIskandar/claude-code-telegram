"""Claude-mem integration - sends bot observations to claude-mem API.

The bot uses claude-agent-sdk directly, which doesn't write transcript files.
This module pushes tool observations to claude-mem's REST API for capture.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
import httpx

logger = structlog.get_logger()


@dataclass
class ClaudeMemConfig:
    """Claude-mem API configuration."""

    enabled: bool = True
    api_url: str = "http://127.0.0.1:37777"
    timeout_seconds: float = 2.0
    retry_attempts: int = 1


class ClaudeMemObserver:
    """Sends tool observations to claude-mem for persistent storage."""

    def __init__(
        self,
        config: Optional[ClaudeMemConfig] = None,
    ):
        """Initialize claude-mem observer.

        Args:
            config: Observer configuration. Defaults to standard localhost setup.
        """
        self.config = config or ClaudeMemConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._observations_queue: asyncio.Queue = asyncio.Queue()

        if not self.config.enabled:
            logger.info("Claude-mem integration disabled")
            return

        logger.info(
            "Claude-mem observer initialized",
            api_url=self.config.api_url,
            timeout=self.config.timeout_seconds,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def init_session(
        self,
        content_session_id: str,
        user_prompt: str,
        project: str = "unknown",
    ) -> bool:
        """Initialize a session in claude-mem with user prompt.

        This must be called before sending observations, otherwise
        observations will be rejected as "private".

        Args:
            content_session_id: Claude SDK session ID
            user_prompt: The user's prompt/message
            project: Project name for categorization

        Returns:
            True if successfully initialized, False otherwise
        """
        if not self.config.enabled:
            return False

        payload = {
            "contentSessionId": content_session_id,
            "project": project,
            "prompt": user_prompt,
        }

        try:
            client = await self._get_client()
            response = await client.post(
                "/api/sessions/init",
                json=payload,
            )

            if response.status_code == 200:
                result = response.json()
                # Init returns {sessionDbId, promptNumber, ...} not {status: "initialized"}
                session_db_id = result.get("sessionDbId")

                if session_db_id:
                    logger.info(
                        "Claude-mem session initialized",
                        session_id=content_session_id,
                        session_db_id=session_db_id,
                    )
                    return True
                else:
                    logger.warning(
                        "Unexpected init response",
                        result=result,
                    )
                    return False
            else:
                logger.warning(
                    "Claude-mem init API error",
                    status_code=response.status_code,
                    response=response.text[:200],
                )
                return False

        except Exception as e:
            logger.warning(
                "Claude-mem session init failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def complete_session(
        self,
        content_session_id: str,
    ) -> bool:
        """Mark a session as complete in claude-mem.

        This triggers summarization and cleanup for the session.

        Args:
            content_session_id: Claude SDK session ID

        Returns:
            True if successfully completed, False otherwise
        """
        if not self.config.enabled:
            return False

        payload = {
            "contentSessionId": content_session_id,
        }

        try:
            client = await self._get_client()
            response = await client.post(
                "/api/sessions/complete",
                json=payload,
            )

            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "")

                if status == "completed" or result.get("success") is True:
                    logger.info(
                        "Claude-mem session completed",
                        session_id=content_session_id,
                    )
                    return True
                else:
                    logger.warning(
                        "Session complete returned unexpected status",
                        status=status,
                        result=result,
                    )
                    return False
            else:
                logger.warning(
                    "Claude-mem complete API error",
                    status_code=response.status_code,
                    response=response.text[:200],
                )
                return False

        except Exception as e:
            logger.warning(
                "Claude-mem session complete failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def send_observation(
        self,
        content_session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_response: Dict[str, Any],
        cwd: str,
    ) -> bool:
        """Send a tool observation to claude-mem.

        Args:
            content_session_id: Claude SDK session ID
            tool_name: Name of the tool used
            tool_input: Tool input parameters
            tool_response: Tool execution result
            cwd: Current working directory

        Returns:
            True if successfully queued, False otherwise
        """
        if not self.config.enabled:
            return False

        payload = {
            "contentSessionId": content_session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "cwd": cwd,
        }

        for attempt in range(self.config.retry_attempts):
            try:
                client = await self._get_client()
                response = await client.post(
                    "/api/sessions/observations",
                    json=payload,
                )

                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "")

                    if status == "queued":
                        logger.debug(
                            "Observation queued to claude-mem",
                            session_id=content_session_id,
                            tool=tool_name,
                        )
                        return True
                    elif status == "skipped":
                        # Tool was filtered (meta-tool, private, etc.)
                        logger.debug(
                            "Observation skipped by claude-mem",
                            session_id=content_session_id,
                            tool=tool_name,
                            reason=result.get("reason", "unknown"),
                        )
                        return True
                    else:
                        logger.warning(
                            "Unexpected claude-mem response",
                            status=status,
                            result=result,
                        )
                        return False
                else:
                    logger.warning(
                        "Claude-mem API error",
                        status_code=response.status_code,
                        response=response.text[:200],
                    )
                    return False

            except httpx.ConnectError:
                logger.debug(
                    "Claude-mem API unavailable (connection refused)",
                    attempt=attempt + 1,
                    max_attempts=self.config.retry_attempts,
                )
                if attempt == self.config.retry_attempts - 1:
                    return False
                await asyncio.sleep(0.5 * (attempt + 1))

            except asyncio.TimeoutError:
                logger.debug(
                    "Claude-mem API timeout",
                    attempt=attempt + 1,
                    max_attempts=self.config.retry_attempts,
                )
                if attempt == self.config.retry_attempts - 1:
                    return False
                await asyncio.sleep(0.5 * (attempt + 1))

            except Exception as e:
                logger.warning(
                    "Claude-mem observation failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return False

        return False

    async def send_observation_non_blocking(
        self,
        content_session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_response: Dict[str, Any],
        cwd: str,
    ) -> None:
        """Send observation without blocking (fire-and-forget).

        Creates a background task that attempts delivery with timeout.
        Errors are logged but don't affect bot operation.

        Args:
            content_session_id: Claude SDK session ID
            tool_name: Name of the tool used
            tool_input: Tool input parameters
            tool_response: Tool execution result
            cwd: Current working directory
        """
        if not self.config.enabled:
            return

        async def _send_with_timeout():
            try:
                await asyncio.wait_for(
                    self.send_observation(
                        content_session_id=content_session_id,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_response=tool_response,
                        cwd=cwd,
                    ),
                    timeout=self.config.timeout_seconds + 0.5,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "Claude-mem send timed out (fire-and-forget)",
                    tool=tool_name,
                )
            except Exception as e:
                logger.debug(
                    "Claude-mem send failed (fire-and-forget)",
                    tool=tool_name,
                    error=str(e),
                )

        # Create background task - don't await
        asyncio.create_task(_send_with_timeout())


# Singleton instance for application-wide use
_observer_instance: Optional[ClaudeMemObserver] = None


def get_observer() -> Optional[ClaudeMemObserver]:
    """Get the global claude-mem observer instance.

    Returns:
        Observer instance if enabled, None otherwise
    """
    global _observer_instance
    return _observer_instance


def initialize_observer(config: Optional[ClaudeMemConfig] = None) -> ClaudeMemObserver:
    """Initialize the global claude-mem observer.

    Args:
        config: Observer configuration

    Returns:
        The initialized observer instance
    """
    global _observer_instance
    _observer_instance = ClaudeMemObserver(config)
    return _observer_instance


async def shutdown_observer() -> None:
    """Shutdown the global claude-mem observer."""
    global _observer_instance
    if _observer_instance:
        await _observer_instance.close()
        _observer_instance = None
