#!/bin/bash
# Claude Code Telegram Bot - Start Script
# Run this from a terminal OUTSIDE of Claude Code

cd "$(dirname "$0")"
echo "Starting Claude Code Telegram Bot..."
echo "Press Ctrl+C to stop"
echo ""

/home/six/.local/bin/poetry run claude-telegram-bot
