# agent.py
# ─────────────────────────────────────────────────────────────────────
# Multi-agent orchestration using LangGraph.
#
# Architecture:
#   1. Supervisor node — reads the query and routes to the right agent
#   2. Specialist agents — summarize / highlight / social
#   3. Tool calling node — each agent can call research tools
#   4. LangGraph manages the state machine so agents can hand off work
#
# Flow:
#   User query + RAG context
#     → Supervisor decides which agent to run
#     → Agent calls tools (wikipedia, web search)
#     → Agent runs its task using RAG context + tool results
#     → Final structured output returned
# ─────────────────────────────────────────────────────────────────────

import logging
import json
from typing import TypedDict, Annotated
import operator

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL_FAST, TEMP_PRECISE, LLM_PARAMS
from mcp_tools import get_tools_for_agent, TOOL_NAMES
from llm import call_llm, call_llm_for_json, get_llm
from prompts import (
    ROUTER_SYSTEM, ROUTER_PROMPT,
    SUMMARIZE_SYSTEM, SUMMARIZE_PROMPT,
    HIGHLIGHT_SYSTEM, HIGHLIGHT_PROMPT,
    SOCIAL_SYSTEM, SOCIAL_PROMPT,
)

logger = logging.getLogger(__name__)


# ── LangGraph State ───────────────────────────────────────────────────
# TypedDict defines the shared state that flows between all nodes.
# Every node reads from this state and returns an updated version of it.
# Annotated[list, operator.add] means messages are APPENDED (not replaced).

class AgentState(TypedDict):
    query:        str                         # original user query
    context:      str                         # RAG-retrieved transcript chunks
    num_chunks:   int                         # number of chunks indexed
    tool_results: str                         # tool enrichment text
    tool_calls:   Annotated[list, operator.add]  # log of tool calls made
    next_agent:   str                         # which agent to run next
    output:       dict                        # final result


# ── Groq with tools bound ─────────────────────────────────────────────
# We need a separate Groq instance for tool calling (bind_tools tells
# the LLM what tools exist so it can decide which ones to call)

def _build_tool_llm(agent_type: str) -> ChatGroq:
    """Build a Groq LLM with the tools for a specific agent bound to it."""
    tools = get_tools_for_agent(agent_type)
    llm = ChatGroq(
        model=GROQ_MODEL_FAST,
        api_key=GROQ_API_KEY,
        temperature=TEMP_PRECISE,   # deterministic for tool selection
        max_tokens=LLM_PARAMS["max_tokens"],
    )
    if tools:
        return llm.bind_tools(tools)
    return llm


# ── Supervisor Node ───────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> AgentState:
    """
    Reads the user query and decides which specialist agent should handle it.
    Returns the updated state with 'next_agent' set.
    """
    logger.info("[SUPERVISOR] Routing query...")

    prompt = ROUTER_PROMPT.format(query=state["query"])

    try:
        decision = call_llm(prompt, system=ROUTER_SYSTEM, mode="precise").strip().lower()

        # Validate the decision — fall back to summarize if unexpected
        valid_agents = {"summarize_agent", "highlight_agent", "social_agent"}
        if decision not in valid_agents:
            logger.warning(f"[SUPERVISOR] Unexpected routing '{decision}', defaulting to summarize_agent")
            decision = "summarize_agent"

        logger.info(f"[SUPERVISOR] Routed to: {decision}")
        return {**state, "next_agent": decision}

    except Exception as e:
        logger.error(f"[SUPERVISOR] Routing failed: {e}, defaulting to summarize_agent")
        return {**state, "next_agent": "summarize_agent"}


# ── Tool Calling Node (shared by all agents) ──────────────────────────

def run_tool_calling(state: AgentState, agent_type: str) -> tuple[str, list]:
    """
    Let Groq decide which research tools to call for enrichment.
    Returns the tool results text and a log of what was called.

    This is called inside each agent node before running the main task.
    """
    tool_llm = _build_tool_llm(agent_type)

    system_msg = SystemMessage(content="""
You are a media research assistant. Based on the user's query,
decide which research tools to call to enrich the content with facts.
Call 1-2 tools maximum. Be selective.
""")
    user_msg = HumanMessage(content=f"""
User query: {state['query']}

Transcript context (first 500 chars):
{state['context'][:500]}

Call the most relevant tools to enrich this content.
""")

    messages       = [system_msg, user_msg]
    tool_calls_log = []
    tool_results   = []

    try:
        response = tool_llm.invoke(messages)
        messages.append(response)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_name  = tc["name"]
                tool_input = tc["args"]
                tool_id    = tc["id"]

                logger.info(f"[TOOL] {agent_type} calling '{tool_name}' with {tool_input}")

                # Execute the tool
                if tool_name in TOOL_NAMES:
                    tool_fn = TOOL_NAMES[tool_name]
                    # Extract the first argument value (all our tools take one string arg)
                    arg_val = list(tool_input.values())[0] if tool_input else ""
                    output  = tool_fn.invoke(str(arg_val))
                else:
                    output = f"Unknown tool: {tool_name}"

                tool_calls_log.append({
                    "tool":   tool_name,
                    "input":  str(list(tool_input.values())[0]) if tool_input else "",
                    "output": str(output)[:300],  # truncate for display
                })
                tool_results.append(f"[{tool_name}]: {output}")

                # Feed tool result back into the conversation
                messages.append(
                    ToolMessage(content=str(output), tool_call_id=tool_id)
                )
        else:
            logger.info(f"[TOOL] {agent_type} chose not to call any tools.")

    except Exception as e:
        logger.warning(f"[TOOL] Tool calling failed for {agent_type}: {e}")

    tool_results_text = "\n\n".join(tool_results) if tool_results else "No external data retrieved."
    return tool_results_text, tool_calls_log


# ── Specialist Agent Nodes ────────────────────────────────────────────

def summarize_agent_node(state: AgentState) -> AgentState:
    """
    Runs the summarization task.
    Calls research tools first, then generates a structured summary
    using RAG context + tool enrichment.
    """
    logger.info("[AGENT] summarize_agent running...")

    # Step 1: Call research tools for enrichment
    tool_results_text, tool_calls_log = run_tool_calling(state, "summarize_agent")

    # Step 2: Build the prompt and call the LLM
    prompt = SUMMARIZE_PROMPT.format(
        context=state["context"],
        tool_results=tool_results_text,
        task=state["query"],
    )
    output = call_llm(prompt, system=SUMMARIZE_SYSTEM, mode="balanced")

    return {
        **state,
        "tool_results": tool_results_text,
        "tool_calls":   tool_calls_log,
        "output": {
            "task":    "summarize_agent",
            "content": output,
        },
        "next_agent": "__end__",  # signal to LangGraph that we're done
    }


def highlight_agent_node(state: AgentState) -> AgentState:
    """
    Runs the highlight extraction task.
    Returns a JSON list of key moments from the transcript.
    """
    logger.info("[AGENT] highlight_agent running...")

    # Step 1: Call research tools for enrichment
    tool_results_text, tool_calls_log = run_tool_calling(state, "highlight_agent")

    # Step 2: Build prompt and call the LLM (returns JSON)
    prompt = HIGHLIGHT_PROMPT.format(
        context=state["context"],
        tool_results=tool_results_text,
        task=state["query"],
    )

    # call_llm_for_json parses the JSON response automatically
    try:
        result = call_llm_for_json(prompt, system=HIGHLIGHT_SYSTEM)
        if isinstance(result, list):
            highlights = result
        elif isinstance(result, dict) and "highlights" in result:
            highlights = result["highlights"]
        else:
            highlights = [result]
    except ValueError as e:
        logger.error(f"[AGENT] Highlight JSON parse failed: {e}")
        highlights = [{"highlight": "Could not parse highlights", "importance": "high",
                       "category": "insight", "timestamp_hint": "mid"}]

    return {
        **state,
        "tool_results": tool_results_text,
        "tool_calls":   tool_calls_log,
        "output": {
            "task":    "highlight_agent",
            "content": highlights,
        },
        "next_agent": "__end__",
    }


def social_agent_node(state: AgentState) -> AgentState:
    """
    Runs the social media content generation task.
    Creates platform-specific posts (Twitter, LinkedIn, Instagram, YouTube).
    """
    logger.info("[AGENT] social_agent running...")

    # Step 1: Call research tools (web search for trends)
    tool_results_text, tool_calls_log = run_tool_calling(state, "social_agent")

    # Step 2: Build prompt and call the LLM
    prompt = SOCIAL_PROMPT.format(
        context=state["context"],
        tool_results=tool_results_text,
        task=state["query"],
    )
    output = call_llm(prompt, system=SOCIAL_SYSTEM, mode="creative")

    return {
        **state,
        "tool_results": tool_results_text,
        "tool_calls":   tool_calls_log,
        "output": {
            "task":    "social_agent",
            "content": output,
        },
        "next_agent": "__end__",
    }


# ── LangGraph Workflow Builder ────────────────────────────────────────

def _build_graph():
    """
    Build the LangGraph state machine.

    Graph structure:
      [supervisor] → conditional edge → [summarize | highlight | social]
      Each specialist agent ends the workflow (__end__)
    """
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)

        # Add all nodes
        graph.add_node("supervisor",      supervisor_node)
        graph.add_node("summarize_agent", summarize_agent_node)
        graph.add_node("highlight_agent", highlight_agent_node)
        graph.add_node("social_agent",    social_agent_node)

        # Entry point: always start at supervisor
        graph.set_entry_point("supervisor")

        # Conditional routing: supervisor's 'next_agent' field decides where to go
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["next_agent"],
            {
                "summarize_agent": "summarize_agent",
                "highlight_agent": "highlight_agent",
                "social_agent":    "social_agent",
            }
        )

        # All specialist agents end the workflow
        graph.add_edge("summarize_agent", END)
        graph.add_edge("highlight_agent", END)
        graph.add_edge("social_agent",    END)

        return graph.compile()

    except ImportError:
        logger.warning("[AGENT] LangGraph not installed. Falling back to simple routing.")
        return None


# Build the graph once at module load time
_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────

def run_agent(query: str, context: str, num_chunks: int) -> dict:
    """
    Main entry point — runs the full agent pipeline.

    Args:
        query      : The user's request.
        context    : RAG-retrieved transcript chunks.
        num_chunks : Number of chunks stored (for display).

    Returns:
        dict with keys:
          - task       : which agent handled this
          - tool_calls : list of tools called and their outputs
          - output     : the final content (string or list)
          - num_chunks : passed through for display
    """
    # Initial state for the LangGraph workflow
    initial_state: AgentState = {
        "query":        query,
        "context":      context,
        "num_chunks":   num_chunks,
        "tool_results": "",
        "tool_calls":   [],
        "next_agent":   "",
        "output":       {},
    }

    if _graph is not None:
        # ── Use LangGraph (preferred) ─────────────────────────────────
        try:
            final_state = _graph.invoke(initial_state)
            output_data = final_state.get("output", {})
            return {
                "task":       output_data.get("task", "unknown"),
                "tool_calls": final_state.get("tool_calls", []),
                "output":     output_data.get("content", "No output generated."),
                "num_chunks": num_chunks,
            }
        except Exception as e:
            logger.error(f"[AGENT] LangGraph invocation failed: {e}")
            # Fall through to simple routing below

    # ── Fallback: simple routing without LangGraph ────────────────────
    logger.info("[AGENT] Using fallback simple routing.")
    routed_state = supervisor_node(initial_state)
    next_agent   = routed_state.get("next_agent", "summarize_agent")

    if next_agent == "highlight_agent":
        final_state = highlight_agent_node(routed_state)
    elif next_agent == "social_agent":
        final_state = social_agent_node(routed_state)
    else:
        final_state = summarize_agent_node(routed_state)

    output_data = final_state.get("output", {})
    return {
        "task":       output_data.get("task", "summarize_agent"),
        "tool_calls": final_state.get("tool_calls", []),
        "output":     output_data.get("content", "No output generated."),
        "num_chunks": num_chunks,
    }
