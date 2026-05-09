# mcp_tools.py
# ─────────────────────────────────────────────────────────────────────
# MCP-style tool registry.
#
# What "MCP-style" means here:
#   - Each tool is a standalone function with a clear description
#   - Tools are registered in a central registry (TOOL_REGISTRY)
#   - Agents load only the tools they are allowed to use
#   - The LLM reads tool descriptions to decide which tool to call
#
# Tools available:
#   1. wikipedia_search    — factual background info
#   2. web_search          — live web search via DuckDuckGo
#   3. youtube_transcript  — extract transcript from YouTube video
#   4. read_file           — read a local transcript/text file
# ─────────────────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass
from langchain_core.tools import tool, BaseTool

logger = logging.getLogger(__name__)


# ── Wikipedia Tool ────────────────────────────────────────────────────

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for factual background information about a topic,
    person, company, or technology mentioned in the transcript.

    Args:
        query: The topic or entity to search for.
    Returns:
        Wikipedia summary text (up to 800 characters).
    """
    logger.info(f"[TOOL] wikipedia_search: '{query}'")
    try:
        import wikipedia
        # search() returns a list of matching page titles
        results = wikipedia.search(query, results=2)
        if not results:
            return "No Wikipedia results found."

        # Get the summary of the first matching page
        summary = wikipedia.summary(results[0], sentences=4)
        return summary
    except Exception as e:
        logger.warning(f"[TOOL] Wikipedia failed: {e}")
        return f"Wikipedia search failed: {str(e)}"


# ── DuckDuckGo Web Search Tool ────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo for recent news, trends, or live
    information about topics mentioned in the transcript.
    Use this for current events, statistics, or topics Wikipedia may not have.

    Args:
        query: The search query string.
    Returns:
        Top web search results as text.
    """
    logger.info(f"[TOOL] web_search: '{query}'")
    try:
        from duckduckgo_search import DDGS

        # DDGS is the new DuckDuckGo search API class
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "No web search results found."

        # Format results as readable text
        formatted = []
        for r in results:
            title = r.get("title", "")
            body  = r.get("body", "")
            formatted.append(f"{title}: {body}")

        return "\n\n".join(formatted)
    except Exception as e:
        logger.warning(f"[TOOL] DuckDuckGo search failed: {e}")
        return f"Web search failed: {str(e)}"


# ── YouTube Transcript Tool ───────────────────────────────────────────

@tool
def youtube_transcript(video_url_or_id: str) -> str:
    """
    Extract the full spoken transcript from a YouTube video.
    Use this when the user provides a YouTube URL or video ID.

    Args:
        video_url_or_id: A YouTube URL (https://youtube.com/watch?v=...)
                         or just the video ID (e.g. dQw4w9WgXcQ).
    Returns:
        Full transcript text from the video (capped at 3000 characters).
    """
    logger.info(f"[TOOL] youtube_transcript: '{video_url_or_id}'")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        # Extract the video ID from a URL if a full URL was passed
        if "youtube.com" in video_url_or_id or "youtu.be" in video_url_or_id:
            if "v=" in video_url_or_id:
                video_id = video_url_or_id.split("v=")[1].split("&")[0]
            elif "youtu.be/" in video_url_or_id:
                video_id = video_url_or_id.split("youtu.be/")[1].split("?")[0]
            else:
                return "Could not parse YouTube video ID from URL."
        else:
            # Assume it's already a bare video ID
            video_id = video_url_or_id.strip()

        # v1.x API: instantiate the class then call .fetch(video_id).
        # The old class-method .get_transcript() was removed in v1.0.
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        full_text = " ".join([snippet.text for snippet in fetched])

        # Cap at 3000 chars to stay within LLM context limits
        return full_text[:3000]

    except Exception as e:
        logger.warning(f"[TOOL] YouTube transcript failed: {e}")
        return f"YouTube transcript failed: {str(e)}"


# ── File Reader Tool ──────────────────────────────────────────────────

@tool
def read_file(filepath: str) -> str:
    """
    Read a transcript or text file from the local filesystem.
    Use this when the user provides a file path to a transcript
    (.txt, .srt, .vtt, .md) that should be loaded for processing.

    Args:
        filepath: Absolute or relative path to the file.
    Returns:
        File contents as text (capped at 3000 characters).
    """
    logger.info(f"[TOOL] read_file: '{filepath}'")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Cap at 3000 chars to stay within LLM context window limits
        return content[:3000]
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"File read error: {str(e)}"


# ── Tool Registry ─────────────────────────────────────────────────────
# Central registry of all available tools.
# This is the "MCP-style" part — tools are registered here,
# and agents query the registry for the tools they're allowed to use.

@dataclass
class ToolManifest:
    """Metadata about a tool — name, category, and the tool object itself."""
    name:     str
    category: str      # "retrieval" | "web" | "file" | "output"
    tool:     BaseTool


# All tools registered in one place
TOOL_REGISTRY: dict[str, ToolManifest] = {
    "wikipedia_search": ToolManifest(
        name="wikipedia_search",
        category="web",
        tool=wikipedia_search,
    ),
    "web_search": ToolManifest(
        name="web_search",
        category="web",
        tool=web_search,
    ),
    "youtube_transcript": ToolManifest(
        name="youtube_transcript",
        category="file",
        tool=youtube_transcript,
    ),
    "read_file": ToolManifest(
        name="read_file",
        category="file",
        tool=read_file,
    ),
}

# ── Per-agent tool access control ─────────────────────────────────────
# Each agent type only gets the tools it needs.
# This is good security practice and prevents agents from calling
# tools that are irrelevant to their task.

AGENT_TOOL_ACCESS = {
    "summarize_agent": ["wikipedia_search", "web_search"],
    "highlight_agent": ["wikipedia_search", "web_search"],
    "social_agent":    ["web_search"],
    "research_agent":  ["wikipedia_search", "web_search", "youtube_transcript"],
}


def get_tools_for_agent(agent_type: str) -> list[BaseTool]:
    """
    Return the list of tool objects allowed for a given agent type.

    Args:
        agent_type: e.g. "summarize_agent", "social_agent"
    Returns:
        List of LangChain BaseTool objects the agent can call.
    """
    allowed_names = AGENT_TOOL_ACCESS.get(agent_type, [])
    tools = []
    for name in allowed_names:
        if name in TOOL_REGISTRY:
            tools.append(TOOL_REGISTRY[name].tool)
    return tools


# Convenience: flat list of all tools (used for direct tool lookup by name)
ALL_TOOLS  = [m.tool for m in TOOL_REGISTRY.values()]
TOOL_NAMES = {t.name: t for t in ALL_TOOLS}
