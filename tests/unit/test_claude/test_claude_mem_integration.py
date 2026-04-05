"""Unit tests for claude-mem integration module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from src.claude.claude_mem_integration import (
    ClaudeMemConfig,
    ClaudeMemObserver,
    get_observer,
    initialize_observer,
    shutdown_observer,
)


@pytest.fixture
def reset_observer():
    """Reset global observer state between tests."""
    from src.claude.claude_mem_integration import _observer_instance

    original = _observer_instance
    if _observer_instance:
        import asyncio

        async def cleanup():
            await _observer_instance.close()

        asyncio.run(cleanup())

    
    # Clear global instance
    import src.claude.claude_mem_integration as mod
    mod._observer_instance = None

    yield

    # Restore original if needed
    if original:
        mod._observer_instance = original


class TestClaudeMemConfig:
    """Test claude-mem configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ClaudeMemConfig()
        assert config.enabled is True
        assert config.api_url == "http://127.0.0.1:37777"
        assert config.timeout_seconds == 2.0
        assert config.retry_attempts == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ClaudeMemConfig(
            enabled=False,
            api_url="http://localhost:8080",
            timeout_seconds=5.0,
            retry_attempts=3,
        )
        assert config.enabled is False
        assert config.api_url == "http://localhost:8080"
        assert config.timeout_seconds == 5.0
        assert config.retry_attempts == 3


class TestClaudeMemObserver:
    """Test claude-mem observer functionality."""

    def test_observer_initialization_disabled(self):
        """Test observer logs disabled message when configured as disabled."""
        observer = ClaudeMemObserver(ClaudeMemConfig(enabled=False))
        assert observer.config.enabled is False

    @pytest.mark.asyncio
    async def test_send_observation_success(self, reset_observer):
        """Test successful observation sending."""
        observer = ClaudeMemObserver()

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "queued"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        observer._client = mock_client

        result = await observer.send_observation(
            content_session_id="test-session-123",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response={"output": "file1.txt\nfile2.txt"},
            cwd="/home/six/SIX",
        )

        assert result is True
        mock_client.post.assert_called_once_with(
            "/api/sessions/observations",
            json={
                "contentSessionId": "test-session-123",
                "tool_name": "Bash",
                "tool_input": {"command": "ls"},
                "tool_response": {"output": "file1.txt\nfile2.txt"},
                "cwd": "/home/six/SIX",
            },
        )

    @pytest.mark.asyncio
    async def test_send_observation_skipped(self, reset_observer):
        """Test handling of skipped observations (meta-tools)."""
        observer = ClaudeMemObserver()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "skipped", "reason": "tool_excluded"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        observer._client = mock_client

        result = await observer.send_observation(
            content_session_id="test-session-123",
            tool_name="AskUserQuestion",
            tool_input={"questions": []},
            tool_response={},
            cwd="/home/six/SIX",
        )

        assert result is True  # Skipped is still a success

    @pytest.mark.asyncio
    async def test_send_observation_connection_error(self, reset_observer):
        """Test handling of connection errors."""
        observer = ClaudeMemObserver()

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        observer._client = mock_client

        result = await observer.send_observation(
            content_session_id="test-session-123",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response={"output": ""},
            cwd="/home/six/SIX",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_observation_non_blocking(self, reset_observer):
        """Test non-blocking observation sending."""
        observer = ClaudeMemObserver()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "queued"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        observer._client = mock_client

        # Call non-blocking (doesn't await result)
        await observer.send_observation_non_blocking(
            content_session_id="test-session-123",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response={"output": "file1.txt"},
            cwd="/home/six/SIX",
        )

        # Give background task time to complete
        await asyncio.sleep(0.1)

        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, reset_observer):
        """Test observer cleanup."""
        observer = ClaudeMemObserver()

        mock_client = AsyncMock()
        observer._client = mock_client

        await observer.close()

        mock_client.aclose.assert_called_once()
        assert observer._client is None


class TestGlobalObserver:
    """Test global observer instance management."""

    def test_initialize_observer(self, reset_observer):
        """Test global observer initialization."""
        observer = initialize_observer(ClaudeMemConfig(enabled=True))

        assert observer is not None
        assert observer.config.enabled is True
        assert get_observer() is observer

    def test_shutdown_observer(self, reset_observer):
        """Test global observer shutdown."""
        initialize_observer(ClaudeMemConfig(enabled=True))

        async def run_shutdown():
            await shutdown_observer()
            assert get_observer() is None

        asyncio.run(run_shutdown())

    def test_get_observer_none_when_disabled(self, reset_observer):
        """Test get_observer returns None when not initialized."""
        observer = get_observer()
        assert observer is None
