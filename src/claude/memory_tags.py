"""Memory tag parser - extracts explicit memory tags from user prompts.

Supports syntax: <memory>key: value</memory>
Extracted tags are sent as structured facts to claude-mem.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

import structlog

logger = structlog.get_logger()

# Pattern: <memory>key: value</memory>
# Examples:
#   <memory>api_key: xyz123</memory>
#   <memory>decision: Use Python for CLI tools</memory>
#   <memory>note: User prefers detailed responses</memory>
MEMORY_TAG_PATTERN = re.compile(
    r"<memory>\s*(\w+(?:\s+\w+)*)\s*:\s*(.+?)\s*</memory>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class MemoryTag:
    """A parsed memory tag."""

    key: str
    value: str
    raw: str

    def __str__(self) -> str:
        return f"{self.key}: {self.value}"


def extract_memory_tags(prompt: str) -> Tuple[str, List[MemoryTag]]:
    """Extract memory tags from user prompt.

    Args:
        prompt: The user's input prompt

    Returns:
        Tuple of (cleaned_prompt, list_of_tags)
        - cleaned_prompt: Prompt with memory tags removed
        - tags: List of extracted MemoryTag objects
    """
    tags = []
    cleaned_prompt = prompt

    matches = list(MEMORY_TAG_PATTERN.finditer(prompt))
    if not matches:
        return cleaned_prompt, tags

    # Extract tags in reverse order to preserve indices when removing
    for match in reversed(matches):
        tag = MemoryTag(
            key=match.group(1).strip(),
            value=match.group(2).strip(),
            raw=match.group(0),
        )
        tags.append(tag)

        # Remove the tag from the prompt
        cleaned_prompt = (
            cleaned_prompt[: match.start()] +
            cleaned_prompt[match.end():]
        ).strip()

    if tags:
        logger.debug(
            "Memory tags extracted",
            tag_count=len(tags),
            tags=[str(tag) for tag in tags],
        )

    # Reverse tags to get original order
    tags.reverse()

    return cleaned_prompt, tags


def format_tags_as_facts(tags: List[MemoryTag]) -> List[str]:
    """Format memory tags as facts for claude-mem storage.

    Args:
        tags: List of MemoryTag objects

    Returns:
        List of fact strings
    """
    return [f"Remembered: {tag.key} = {tag.value}" for tag in tags]
