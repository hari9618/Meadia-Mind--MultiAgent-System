# agent.py  (PRODUCTION UPGRADE — v2 with RAGAS evaluation)
# ─────────────────────────────────────────────────────────────────────
# Changes vs v1:
#   ✦ EvaluationService integrated — RAGAS scores logged after every response
#   ✦ Context chunks passed through state for evaluation
#   ✦ EvalDatasetCollector integration (optional, toggleable)
#   ✦ trace_id returned in result dict so app.py can pass to UI
#   ✦ All existing logic preserved — zero breaking changes
# ─────────────────────────────────────────────────────────────────────

import logging
import json
import time
import uuid
from typing import TypedDict, Annotated
import operator

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY, GROQ_MODEL_FAST, TEMP_PRECISE, LLM_PARAMS,
    EVAL_ENABLED, EVAL_COLLECT_DATASET, EVAL_DATASET_PATH,
)
from mcp_tools import get_tools_for_agent, TOOL_NAMES
from llm import call_llm, call_llm_for_json, get_llm
from prompts import (
    ROUTER_SYSTEM, ROUTER_PROMPT,
    SUMMARIZE_SYSTEM, SUMMARIZE_PROMPT,
    SUMMARIZE_CHAT_TEMPLATE,
    HIGHLIGHT_SYSTEM, HIGHLIGHT_PROMPT,
    HIGHLIGHT_CHAT_TEMPLATE,
    SOCIAL_SYSTEM, SOCIAL_PROMPT,
    SOCIAL_CHAT_TEMPLATE,
    QA_SYSTEM, QA_PROMPT,
    QA_CHAT_TEMPLATE,
    REACT_TOOL_SYSTEM, REACT_TOOL_CHAT_TEMPLATE,
    ROUTER_CHAT_TEMPLATE,
)
from schemas import HighlightOutput
from guardrails import run_output_pipeline
from tracer import get_tracer
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)
_parser = StrOutputParser()


# ── LangGraph State ───────────────────────────────────────────────────
# context_chunks is NEW — we need the raw chunk list (not the joined string)
# to pass to RAGAS, which expects list[str] not a single concatenated string.

class AgentState(TypedDict):
    query:          str
    context:        str           # joined context string for LLM prompt
    context_chunks: list[str]     # NEW: raw chunks for RAGAS evaluation
    num_chunks:     int
    tool_results:   str
    tool_calls:     Annotated[list, operator.add]
    next_agent:     str
    output:         dict
    trace_id:       str
    session_id:     str


# ── Groq with tools bound ─────────────────────────────────────────────

def _build_tool_llm(agent_type: str) -> ChatGroq:
    tools = get_tools_for_agent(agent_type)
    llm = ChatGroq(
        model=GROQ_MODEL_FAST,
        api_key=GROQ_API_KEY,
        temperature=TEMP_PRECISE,
        max_tokens=LLM_PARAMS["max_tokens"],
    )
    if tools:
        return llm.bind_tools(tools)
    return llm


# ── Supervisor Node ───────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> AgentState:
    logger.info("[SUPERVISOR] Routing query...")
    tracer = get_tracer()
    t0 = time.perf_counter()

    try:
        llm = get_llm(mode="precise")
        chain = ROUTER_CHAT_TEMPLATE | llm | _parser
        decision = chain.invoke({"query": state["query"]}).strip().lower()

        valid_agents = {"summarize_agent", "highlight_agent", "social_agent", "qa_agent"}
        if decision not in valid_agents:
            logger.warning(f"[SUPERVISOR] Unexpected routing '{decision}', defaulting")
            decision = "summarize_agent"

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[SUPERVISOR] Routed to: {decision} ({latency_ms:.0f}ms)")

        return {**state, "next_agent": decision}

    except Exception as e:
        logger.error(f"[SUPERVISOR] Routing failed: {e}")
        return {**state, "next_agent": "summarize_agent"}


# ── Tool Calling Node (ReAct-enhanced) ───────────────────────────────

def run_tool_calling(state: AgentState, agent_type: str) -> tuple[str, list]:
    tool_llm = _build_tool_llm(agent_type)
    react_messages = REACT_TOOL_CHAT_TEMPLATE.format_messages(
        query=state["query"],
        context_preview=state["context"][:500],
    )

    tool_calls_log = []
    tool_results   = []

    try:
        response = tool_llm.invoke(react_messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            messages = list(react_messages) + [response]
            for tc in response.tool_calls:
                tool_name  = tc["name"]
                tool_input = tc["args"]
                tool_id    = tc["id"]

                logger.info(f"[TOOL] {agent_type} → '{tool_name}' ({tool_input})")

                if tool_name in TOOL_NAMES:
                    tool_fn = TOOL_NAMES[tool_name]
                    arg_val = list(tool_input.values())[0] if tool_input else ""
                    output  = tool_fn.invoke(str(arg_val))
                else:
                    output = f"Unknown tool: {tool_name}"

                tool_calls_log.append({
                    "tool":   tool_name,
                    "input":  str(list(tool_input.values())[0]) if tool_input else "",
                    "output": str(output)[:300],
                })
                tool_results.append(f"[{tool_name}]: {output}")
                messages.append(ToolMessage(content=str(output), tool_call_id=tool_id))
        else:
            logger.info(f"[TOOL] {agent_type} chose not to call any tools (ReAct decided).")

    except Exception as e:
        logger.warning(f"[TOOL] Tool calling failed for {agent_type}: {e}")

    tool_results_text = "\n\n".join(tool_results) if tool_results else "No external data retrieved."
    return tool_results_text, tool_calls_log


# ── Specialist Agent Nodes ────────────────────────────────────────────

def summarize_agent_node(state: AgentState) -> AgentState:
    logger.info("[AGENT] summarize_agent running...")
    tool_results_text, tool_calls_log = run_tool_calling(state, "summarize_agent")

    try:
        llm = get_llm(mode="balanced")
        chain = SUMMARIZE_CHAT_TEMPLATE | llm | _parser
        output = chain.invoke({
            "context":      state["context"],
            "tool_results": tool_results_text,
            "task":         state["query"],
        })
    except Exception as e:
        logger.warning(f"[AGENT] ChatTemplate failed, falling back to call_llm: {e}")
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
        "output": {"task": "summarize_agent", "content": output},
        "next_agent": "__end__",
    }


def highlight_agent_node(state: AgentState) -> AgentState:
    logger.info("[AGENT] highlight_agent running...")
    tool_results_text, tool_calls_log = run_tool_calling(state, "highlight_agent")

    try:
        llm = get_llm(mode="precise")
        chain = HIGHLIGHT_CHAT_TEMPLATE | llm | _parser
        raw_text = chain.invoke({
            "context":      state["context"],
            "tool_results": tool_results_text,
            "task":         state["query"],
        })
        import json as _json
        clean = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        raw_list = _json.loads(clean)
        validated = HighlightOutput.from_raw(raw_list if isinstance(raw_list, list) else [raw_list])
        highlights = [h.model_dump() for h in validated.content]

    except Exception as e:
        logger.warning(f"[AGENT] Highlight chain failed, falling back: {e}")
        try:
            prompt = HIGHLIGHT_PROMPT.format(
                context=state["context"],
                tool_results=tool_results_text,
                task=state["query"],
            )
            result = call_llm_for_json(prompt, system=HIGHLIGHT_SYSTEM)
            raw_list = result if isinstance(result, list) else [result]
            validated = HighlightOutput.from_raw(raw_list)
            highlights = [h.model_dump() for h in validated.content]
        except Exception as e2:
            logger.error(f"[AGENT] Highlight fallback also failed: {e2}")
            highlights = [{"highlight": "Could not parse highlights", "importance": "high",
                           "category": "insight", "timestamp_hint": "mid"}]

    return {
        **state,
        "tool_results": tool_results_text,
        "tool_calls":   tool_calls_log,
        "output": {"task": "highlight_agent", "content": highlights},
        "next_agent": "__end__",
    }


def social_agent_node(state: AgentState) -> AgentState:
    logger.info("[AGENT] social_agent running...")
    tool_results_text, tool_calls_log = run_tool_calling(state, "social_agent")

    try:
        llm = get_llm(mode="creative")
        chain = SOCIAL_CHAT_TEMPLATE | llm | _parser
        output = chain.invoke({
            "context":      state["context"],
            "tool_results": tool_results_text,
            "task":         state["query"],
        })
    except Exception as e:
        logger.warning(f"[AGENT] Social chain failed, falling back: {e}")
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
        "output": {"task": "social_agent", "content": output},
        "next_agent": "__end__",
    }


def qa_agent_node(state: AgentState) -> AgentState:
    logger.info("[AGENT] qa_agent running...")
    tool_results_text, tool_calls_log = run_tool_calling(state, "summarize_agent")

    try:
        llm = get_llm(mode="balanced")
        chain = QA_CHAT_TEMPLATE | llm | _parser
        output = chain.invoke({
            "context":      state["context"],
            "tool_results": tool_results_text,
            "task":         state["query"],
        })
    except Exception as e:
        logger.warning(f"[AGENT] QA chain failed, falling back: {e}")
        prompt = QA_PROMPT.format(
            context=state["context"],
            tool_results=tool_results_text,
            task=state["query"],
        )
        output = call_llm(prompt, system=QA_SYSTEM, mode="balanced")

    return {
        **state,
        "tool_results": tool_results_text,
        "tool_calls":   tool_calls_log,
        "output": {"task": "qa_agent", "content": output},
        "next_agent": "__end__",
    }


# ── LangGraph Workflow Builder ────────────────────────────────────────

def _build_graph():
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)
        graph.add_node("supervisor",      supervisor_node)
        graph.add_node("summarize_agent", summarize_agent_node)
        graph.add_node("highlight_agent", highlight_agent_node)
        graph.add_node("social_agent",    social_agent_node)
        graph.add_node("qa_agent",        qa_agent_node)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["next_agent"],
            {
                "summarize_agent": "summarize_agent",
                "highlight_agent": "highlight_agent",
                "social_agent":    "social_agent",
                "qa_agent":        "qa_agent",
            }
        )
        graph.add_edge("summarize_agent", END)
        graph.add_edge("highlight_agent", END)
        graph.add_edge("social_agent",    END)
        graph.add_edge("qa_agent",        END)

        return graph.compile()
    except ImportError:
        logger.warning("[AGENT] LangGraph not installed. Falling back.")
        return None


_graph = _build_graph()


# ── Evaluation helper ─────────────────────────────────────────────────

def _run_evaluation_background(
    trace_id: str,
    query: str,
    answer: str,
    context_chunks: list[str],
) -> None:
    """
    Fire-and-forget evaluation after the agent responds.
    Runs in a daemon thread — never delays the user.

    Why fire-and-forget?
        RAGAS takes 3-8 seconds per query. The user should not wait
        for eval before seeing their answer. Scores appear in Langfuse
        a few seconds after the response is already rendered.
    """
    if not EVAL_ENABLED:
        return

    # Convert answer to string if it's a list (highlight agent returns list)
    if isinstance(answer, list):
        answer_str = " ".join(
            item.get("highlight", "") if isinstance(item, dict) else str(item)
            for item in answer
        )
    else:
        answer_str = str(answer)

    if not answer_str.strip() or not context_chunks:
        logger.debug("[EVAL] Skipping eval — empty answer or no context chunks")
        return

    try:
        from evaluation import get_eval_service
        svc = get_eval_service()
        svc.evaluate_fire_and_forget(
            trace_id=trace_id,
            query=query,
            answer=answer_str,
            contexts=context_chunks,
        )
    except Exception as e:
        # Evaluation must NEVER break the main pipeline
        logger.warning(f"[EVAL] Could not launch background evaluation: {e}")


def _collect_dataset_sample(
    query: str,
    answer: str,
    context_chunks: list[str],
) -> None:
    """
    Optionally save this Q/A/context to a JSONL file for offline evaluation.
    Controlled by EVAL_COLLECT_DATASET env var (default: false).
    """
    if not EVAL_COLLECT_DATASET:
        return
    try:
        from evaluation import EvalDatasetCollector
        collector = EvalDatasetCollector.load(EVAL_DATASET_PATH) if __import__("os").path.exists(EVAL_DATASET_PATH) else EvalDatasetCollector()
        answer_str = answer if isinstance(answer, str) else str(answer)
        collector.add(question=query, contexts=context_chunks, answer=answer_str)
        collector.save(EVAL_DATASET_PATH)
    except Exception as e:
        logger.debug(f"[EVAL] Dataset collection failed (non-critical): {e}")


# ── Public API ────────────────────────────────────────────────────────

def run_agent(
    query:          str,
    context:        str,
    num_chunks:     int,
    session_id:     str | None = None,
    context_chunks: list[str] | None = None,  # NEW: raw chunks for evaluation
) -> dict:
    """
    Main entry point — runs the full agent pipeline with:
      - Langfuse observability (auto-traces all LLM calls)
      - Output guardrails (validation + sanitization)
      - RAGAS evaluation (background, non-blocking)
      - Pydantic-validated results

    Args:
        query:          User query string.
        context:        Joined context string for LLM prompt (from rag.py).
        num_chunks:     Number of chunks retrieved (for logging).
        session_id:     Langfuse session ID for trace grouping.
        context_chunks: Raw list of retrieved chunks (for RAGAS evaluation).
                        If None, we split context by separator as fallback.
    """
    tracer    = get_tracer()
    trace_id  = str(uuid.uuid4())
    sid       = session_id or "default"

    # If caller didn't pass raw chunks, split the joined context string
    # This ensures backward compatibility with existing callers
    chunks_for_eval = context_chunks or [
        c.strip() for c in context.split("\n\n---\n\n") if c.strip()
    ]

    initial_state: AgentState = {
        "query":          query,
        "context":        context,
        "context_chunks": chunks_for_eval,
        "num_chunks":     num_chunks,
        "tool_results":   "",
        "tool_calls":     [],
        "next_agent":     "",
        "output":         {},
        "trace_id":       trace_id,
        "session_id":     sid,
    }

    with tracer.trace("mediamind_pipeline", query=query, session_id=sid) as trace:
        langfuse_handler = tracer.langchain_handler(
            trace_id=trace_id, session_id=sid
        )
        callbacks = [langfuse_handler] if langfuse_handler else []

        if _graph is not None:
            try:
                config = {"callbacks": callbacks} if callbacks else {}
                final_state = _graph.invoke(initial_state, config=config)
                output_data = final_state.get("output", {})

                result = {
                    "task":       output_data.get("task", "unknown"),
                    "tool_calls": final_state.get("tool_calls", []),
                    "output":     output_data.get("content", "No output generated."),
                    "num_chunks": num_chunks,
                    "trace_id":   trace_id,   # returned so app.py can link feedback
                }

                # Run output guardrails
                result = run_output_pipeline(result)

                # Log routing decision to Langfuse trace
                tracer.log_routing_decision(
                    trace,
                    query=query,
                    decision=output_data.get("task", "unknown"),
                    latency_ms=0,
                )

                # ── RAGAS Evaluation (fire-and-forget, non-blocking) ──
                # This launches in a daemon thread and NEVER blocks the response.
                # Scores appear in Langfuse ~5s after the response is rendered.
                _run_evaluation_background(
                    trace_id=trace_id,
                    query=query,
                    answer=result["output"],
                    context_chunks=chunks_for_eval,
                )

                # ── Optional: collect this sample for offline evaluation ──
                _collect_dataset_sample(
                    query=query,
                    answer=result["output"],
                    context_chunks=chunks_for_eval,
                )

                return result

            except Exception as e:
                logger.error(f"[AGENT] LangGraph invocation failed: {e}")

        # ── Fallback: simple routing ──────────────────────────────────
        logger.info("[AGENT] Using fallback routing.")
        routed_state = supervisor_node(initial_state)
        next_agent   = routed_state.get("next_agent", "summarize_agent")

        node_map = {
            "highlight_agent": highlight_agent_node,
            "social_agent":    social_agent_node,
            "qa_agent":        qa_agent_node,
        }
        final_state = node_map.get(next_agent, summarize_agent_node)(routed_state)
        output_data = final_state.get("output", {})

        result = {
            "task":       output_data.get("task", "summarize_agent"),
            "tool_calls": final_state.get("tool_calls", []),
            "output":     output_data.get("content", "No output generated."),
            "num_chunks": num_chunks,
            "trace_id":   trace_id,
        }
        result = run_output_pipeline(result)

        # Evaluation on fallback path too
        _run_evaluation_background(
            trace_id=trace_id,
            query=query,
            answer=result["output"],
            context_chunks=chunks_for_eval,
        )

        return result