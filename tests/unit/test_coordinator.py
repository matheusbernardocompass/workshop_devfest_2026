from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from pydantic import ValidationError

from app.agent import ProviderError
from app.agents.coordinator import CoordinatorAgent, DelegationBudget, schedule_query
from app.resilience import CircuitBreaker

pytestmark = pytest.mark.usefixtures("mcp_database")


@pytest.mark.anyio
async def test_multi_coordinator_uses_mcp_and_never_writes(sessions):
    agent = CoordinatorAgent(sessions, use_mcp=True)
    result = await agent.ask("Quero palestras de IA após 14h")
    assert result.error is None
    assert result.mode == "offline:multi"
    assert result.trace["tools"] == ["delegate:schedule_agent", "mcp.list_sessions"]
    assert [event["type"] for event in result.trace["adk_events"]] == [
        "transfer",
        "tool_call",
    ]
    assert result.trace["adk_events"][0]["target"] == "schedule_agent"
    assert result.trace["adk_events"][1]["name"] == "list_sessions"
    assert result.session_ids
    assert result.trace["reservation_writes"] == 0


@pytest.mark.anyio
async def test_multi_coordinator_routes_faq_with_verified_source(sessions):
    result = await CoordinatorAgent(sessions, use_mcp=False).ask(
        "Que recursos de acessibilidade existem?"
    )
    assert result.error is None
    assert result.sources[0]["document"] == "acessibilidade.md"
    assert result.trace["tools"] == ["delegate:faq_agent", "faq.search_policy"]
    assert result.trace["adk_events"][-1]["trace_tool"] == "faq.search_policy"


@pytest.mark.anyio
async def test_multi_coordinator_adk_combines_specialists_and_records_tools(sessions):
    result = await CoordinatorAgent(sessions, use_mcp=False).ask(
        "Palestras após 14h e quais são as opções de acessibilidade?"
    )
    assert result.error is None
    assert result.trace["tools"] == [
        "delegate:combined_agent",
        "schedule.list_sessions",
        "faq.search_policy",
    ]
    assert result.session_ids
    assert result.sources


@pytest.mark.anyio
async def test_eval_case_accepts_adk_event_trajectory(sessions):
    from pathlib import Path

    from app.evaluators import evaluate_case, load_cases

    result = await CoordinatorAgent(sessions, use_mcp=True).ask(
        "Quero três palestras de IA após 14h"
    )
    case_path = Path(__file__).parents[2] / "evals" / "cases.jsonl"
    case = next(item for item in load_cases(case_path) if item["id"] == "schedule-ai-after-14")
    verdict = evaluate_case(case, result, architecture="multi")
    assert verdict["passed"] is True


@pytest.mark.anyio
async def test_multi_coordinator_runs_adk_in_local_mode(sessions, monkeypatch):
    import app.agents.coordinator as coordinator

    kinds = iter(("schedule_agent", "faq_agent", "combined_agent", "coordinator"))
    monkeypatch.setattr(
        coordinator,
        "_ollama_adk_model",
        lambda model, _url, _timeout: coordinator._OfflineADKLlm(
            model, next(kinds), "Palestras de IA", sessions
        ),
    )
    result = await CoordinatorAgent(sessions, mode="local", use_mcp=False).ask("Palestras de IA")
    assert result.error is None
    assert result.mode == "local:multi"
    assert result.trace["status"] == "ok"
    assert result.trace["tools"] == ["delegate:schedule_agent", "schedule.list_sessions"]


@pytest.mark.anyio
async def test_local_ollama_adapter_runs_adk_tool_cycle(sessions):
    class OllamaStub(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tool_names = {item["function"]["name"] for item in request.get("tools", [])}
            has_tool_result = any(
                message.get("role") == "tool" for message in request.get("messages", [])
            )
            if "transfer_to_agent" in tool_names and not (
                {"list_sessions", "search_policy"} & tool_names
            ):
                reply = {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "transfer-call",
                                "function": {
                                    "name": "transfer_to_agent",
                                    "arguments": {
                                        "agent_name": "schedule_agent",
                                        "transfer_reason": "agenda request",
                                    },
                                },
                            }
                        ],
                    }
                }
            elif has_tool_result:
                reply = {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "answer": "Agenda consultada.",
                                "session_ids": ["S01"],
                                "sources": [],
                                "reservation_proposal": None,
                            }
                        ),
                    }
                }
            else:
                reply = {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "sessions-call",
                                "function": {
                                    "name": "list_sessions",
                                    "arguments": {"track": "IA", "limit": 3},
                                },
                            }
                        ],
                    }
                }
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaStub)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = await CoordinatorAgent(
            sessions,
            mode="local",
            model="stub-model",
            ollama_base_url=f"http://127.0.0.1:{server.server_port}",
            use_mcp=False,
        ).ask("Palestras de IA")
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert result.error is None
    assert result.answer == "Agenda consultada."
    assert result.trace["tools"] == ["delegate:schedule_agent", "schedule.list_sessions"]


@pytest.mark.anyio
async def test_open_mcp_circuit_returns_verified_partial_faq(sessions):
    breaker = CircuitBreaker()
    for _ in range(3):
        breaker.failure()
    result = await CoordinatorAgent(sessions, use_mcp=True, circuit_breaker=breaker).ask(
        "Palestras após 14h e quais são as opções de acessibilidade?"
    )
    assert result.error is not None
    assert "Agenda indisponível" in result.answer
    assert result.sources[0]["document"] == "acessibilidade.md"
    assert result.trace["partial"] is True
    assert result.session_ids == []


def test_shared_delegation_budget_stops_cycles():
    budget = DelegationBudget(max_delegations=3, max_tool_calls=3)
    for _ in range(3):
        budget.delegate("faq_agent")
    with pytest.raises(ProviderError, match="delegações"):
        budget.delegate("faq_agent")


def test_multi_schema_is_additive_and_single_remains_default():
    from app.schemas import ChatRequest

    assert ChatRequest(message="oi").architecture == "single"
    with pytest.raises(ValidationError):
        ChatRequest(message="oi", architecture="other")


def test_schedule_query_normalizes_track_aliases_and_level_inflections(sessions):
    query = schedule_query("Sugira palestra iniciante de Desenvolvimento", sessions)
    assert query["track"] == "Desenvolvimento"
    assert query["level"] == "iniciante"

    query = schedule_query("Sessões intermediárias de IA", sessions)
    assert query["track"] == "IA"
    assert query["level"] == "intermediario"
