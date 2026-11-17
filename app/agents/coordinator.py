"""Google ADK multi-agent orchestration for the read-only event assistant."""

from __future__ import annotations

import asyncio
import json
import re
import time
import unicodedata
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from app.agent import AgentResult, ProviderError
from app.mcp_client import ScheduleMCPClient
from app.resilience import CircuitBreaker
from app.schemas import Session
from app.tools import DemoTools

COORDINATOR_PROMPT_VERSION = "eventops-adk-coordinator-v2"


def schedule_query(message: str, sessions: list[Session]) -> dict[str, Any]:
    normalized = unicodedata.normalize("NFKD", message.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"\bdev\b", "desenvolvimento", normalized)

    def token(value: str) -> str:
        value = unicodedata.normalize("NFKD", value.casefold())
        return "".join(c for c in value if not unicodedata.combining(c))

    track = next(
        (s.track for s in sessions if re.search(rf"\b{re.escape(token(s.track))}\b", normalized)),
        None,
    )
    level_aliases = {
        "iniciante": r"\biniciant(?:e|es)\b",
        "intermediario": r"\bintermediari?(?:o|a|os|as)\b",
        "avancado": r"\bavancad[oa]s?\b",
    }
    level = next(
        (
            s.level
            for s in sessions
            if re.search(
                level_aliases.get(s.level.casefold(), rf"\b{re.escape(token(s.level))}\b"),
                normalized,
            )
        ),
        None,
    )
    m = re.search(r"(?:ap[oó]s|depois das?)\s*(\d{1,2})(?::(\d{2}))?", normalized)
    after = f"2026-11-07T{int(m.group(1)):02d}:{int(m.group(2) or 0):02d}:00-03:00" if m else None
    return {"track": track, "level": level, "after": after, "limit": 3}


@dataclass
class DelegationBudget:
    max_delegations: int = 3
    max_tool_calls: int = 3
    delegations: int = 0
    tool_calls: int = 0
    tools: list[str] = field(default_factory=list)

    def record(self, name: str) -> None:
        if name.startswith("delegate:"):
            if self.delegations >= self.max_delegations:
                raise ProviderError("O limite de três delegações foi atingido.")
            self.delegations += 1
        else:
            if self.tool_calls >= self.max_tool_calls:
                raise ProviderError("O limite global de chamadas foi atingido.")
            self.tool_calls += 1
        self.tools.append(name)

    def delegate(self, name: str) -> None:
        self.record(f"delegate:{name}")

    def record_tool(self, name: str) -> None:
        self.record(name)


class _OfflineADKLlm:
    """Deterministic local ADK model used for offline runs and evals."""

    def __new__(cls, model: str, kind: str, message: str, sessions: list[Session]) -> Any:
        from google.adk.models import BaseLlm
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        class OfflineModel(BaseLlm):
            kind: str
            message: str
            sessions: list[Any]

            async def generate_content_async(
                self, llm_request: Any, stream: bool = False
            ) -> AsyncGenerator[Any, None]:
                del stream
                contents = llm_request.contents or []
                if self.kind == "coordinator":
                    target = _offline_route(self.message)
                    call = types.FunctionCall(
                        name="transfer_to_agent",
                        args={"agent_name": target, "transfer_reason": "intent"},
                    )
                    yield LlmResponse(
                        content=types.Content(role="model", parts=[types.Part(function_call=call)]),
                        turn_complete=True,
                    )
                    return
                declarations = []
                for tool in llm_request.tools_dict:
                    declarations.append(tool)
                responses = {
                    part.function_response.name: part.function_response.response
                    for content in contents
                    for part in (content.parts or [])
                    if part.function_response
                }
                pending = [
                    n
                    for n in declarations
                    if n in {"list_sessions", "search_policy"} and n not in responses
                ]
                if pending:
                    name = pending[0]
                    if name == "list_sessions":
                        args = schedule_query(self.message, self.sessions)
                    else:
                        args = {"question": self.message}
                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(function_call=types.FunctionCall(name=name, args=args))
                            ],
                        ),
                        turn_complete=True,
                    )
                    return
                response = responses.get("list_sessions") or responses.get("search_policy")
                if self.kind == "schedule_agent":
                    sessions_found = (responses.get("list_sessions") or {}).get("sessions", [])
                    ids = [str(item["id"]) for item in sessions_found]
                    payload = {
                        "answer": "Encontrei sessões no catálogo."
                        if ids
                        else "Não encontrei sessões compatíveis com esses filtros.",
                        "session_ids": ids,
                        "sources": [],
                        "reservation_proposal": ids[0]
                        if ids
                        and any(w in self.message.casefold() for w in ("reserv", "vaga", "guardar"))
                        else None,
                    }
                elif self.kind == "combined_agent":
                    sessions_found = (responses.get("list_sessions") or {}).get("sessions", [])
                    ids = [str(item["id"]) for item in sessions_found]
                    faq = responses.get("search_policy") or {}
                    schedule_answer = (
                        "Encontrei sessões no catálogo."
                        if ids
                        else "Não encontrei sessões compatíveis com esses filtros."
                    )
                    payload = {
                        "answer": " ".join(
                            part for part in (schedule_answer, faq.get("answer", "")) if part
                        ),
                        "session_ids": ids,
                        "sources": faq.get("sources", []),
                        "reservation_proposal": ids[0]
                        if ids
                        and any(w in self.message.casefold() for w in ("reserv", "vaga", "guardar"))
                        else None,
                    }
                else:
                    payload = response or {
                        "answer": "Não encontrei evidência suficiente.",
                        "sources": [],
                    }
                    payload = {
                        "answer": payload.get("answer", "Não encontrei evidência suficiente."),
                        "session_ids": [],
                        "sources": payload.get("sources", []),
                        "reservation_proposal": None,
                    }
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=json.dumps(payload, ensure_ascii=False))],
                    ),
                    turn_complete=True,
                )

        return OfflineModel(model=model, kind=kind, message=message, sessions=sessions)


def _ollama_adk_model(model: str, base_url: str, timeout_seconds: int) -> Any:
    """Adapt Ollama's chat/tool-call API to the Google ADK BaseLlm contract."""
    from google.adk.models import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class OllamaADKLlm(BaseLlm):
        base_url: str
        timeout_seconds: int

        async def generate_content_async(
            self, llm_request: Any, stream: bool = False
        ) -> AsyncGenerator[Any, None]:
            del stream

            def text_parts(content: Any) -> str:
                return "".join(part.text or "" for part in (content.parts or []))

            messages: list[dict[str, Any]] = []
            system = getattr(llm_request.config, "system_instruction", None)
            if system:
                messages.append(
                    {
                        "role": "system",
                        "content": system if isinstance(system, str) else text_parts(system),
                    }
                )
            for content in llm_request.contents or []:
                calls = [part.function_call for part in (content.parts or []) if part.function_call]
                responses = [
                    part.function_response
                    for part in (content.parts or [])
                    if part.function_response
                ]
                if calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": text_parts(content) or None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": call.id or f"call_{index}",
                                    "function": {"name": call.name, "arguments": call.args or {}},
                                }
                                for index, call in enumerate(calls)
                            ],
                        }
                    )
                elif responses:
                    for response in responses:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": response.name,
                                "tool_call_id": response.id or "",
                                "content": json.dumps(response.response, ensure_ascii=False),
                            }
                        )
                else:
                    role = "assistant" if content.role == "model" else "user"
                    messages.append({"role": role, "content": text_parts(content)})
            tool_declarations = [
                declaration
                for tool in (getattr(llm_request.config, "tools", None) or [])
                for declaration in (getattr(tool, "function_declarations", None) or [])
            ]
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
            }
            if tool_declarations:
                body["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": declaration.name,
                            "description": declaration.description or "",
                            "parameters": declaration.parameters.model_dump(
                                mode="json", exclude_none=True
                            )
                            if declaration.parameters
                            else {"type": "object", "properties": {}},
                        },
                    }
                    for declaration in tool_declarations
                ]

            def invoke() -> dict[str, Any]:
                from urllib.request import Request, urlopen

                request = Request(
                    f"{self.base_url.rstrip('/')}/api/chat",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return cast(dict[str, Any], json.loads(response.read()))

            result = await asyncio.to_thread(invoke)
            message = result.get("message", {})
            parts = [types.Part(text=message["content"])] if message.get("content") else []
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                args = function.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call.get("id"), name=function.get("name"), args=args
                        )
                    )
                )
            if not parts:
                raise ProviderError("Ollama encerrou sem texto ou chamada de ferramenta.")
            usage = None
            if isinstance(result.get("prompt_eval_count"), int) and isinstance(
                result.get("eval_count"), int
            ):
                usage = types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=result["prompt_eval_count"],
                    candidates_token_count=result["eval_count"],
                )
            yield LlmResponse(
                content=types.Content(role="model", parts=parts),
                turn_complete=True,
                usage_metadata=usage,
            )

    return OllamaADKLlm(model=model, base_url=base_url, timeout_seconds=timeout_seconds)


def _offline_route(message: str) -> str:
    text = unicodedata.normalize("NFKD", message.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    policy = any(
        w in text
        for w in (
            "politica",
            "regra",
            "acessibilidade",
            "cancel",
            "inscricao",
            "alimentacao",
            "como faco",
            "quais sao",
            "qual e",
            "fuso",
            "faq",
            "preco",
            "empresa",
            "confirme",
            "confirmar",
            "token",
            "ignore",
            "instrucoes",
            "documento",
            "reserve_seat",
            "historico",
            "cookie",
            "vaga como confirmada",
        )
    )
    policy = policy or bool(re.search(r"\breservas?\b", text))
    schedule = any(
        w in text
        for w in (
            "agenda",
            "palestra",
            "sessao",
            "sessoes",
            "apos",
            "sugira",
            "recomende",
            "trilha",
            "horario",
            "iniciante",
            "desenvolvimento",
            "produto",
            "seguranca",
            "devops",
            "quero reservar",
            "guardar uma vaga",
            "mostre uma opcao",
            "reserve uma palestra",
        )
    )
    if policy:
        mixed = re.search(r"\be\s+(?:quais|qual|como|posso)\b", text)
        return "combined_agent" if schedule and mixed else "faq_agent"
    return "schedule_agent" if schedule else "faq_agent"


class CoordinatorAgent:
    """ADK Runner coordinator; ADK performs routing, delegation, and tool execution."""

    def __init__(
        self,
        sessions: list[Session],
        *,
        mode: str = "offline",
        model: str = "offline",
        timeout_seconds: int = 12,
        use_mcp: bool = True,
        circuit_breaker: CircuitBreaker | None = None,
        database: Path | str | None = None,
        api_key: str | None = None,
        ollama_base_url: str = "http://127.0.0.1:11434",
        vertex_ai: bool = False,
        google_project: str | None = None,
        google_location: str = "us-central1",
    ) -> None:
        self.sessions, self.mode, self.model = sessions, mode, model
        self.timeout_seconds, self.use_mcp, self.circuit_breaker = (
            timeout_seconds,
            use_mcp,
            circuit_breaker,
        )
        self.database, self.api_key, self.ollama_base_url = database, api_key, ollama_base_url
        self.vertex_ai, self.google_project, self.google_location = (
            vertex_ai,
            google_project,
            google_location,
        )

    async def ask(self, message: str, history: list[str] | None = None) -> AgentResult:
        started = time.monotonic()
        budget = DelegationBudget()
        adk_events: list[dict[str, str]] = []
        adk_tools: list[str] = []
        mcp: ScheduleMCPClient | None = None
        try:
            from google.adk.agents import Agent
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            from google.genai import types

            model_for: Callable[[str], Any]
            if self.mode == "live":
                if not self.api_key and not self.vertex_ai:
                    raise ProviderError("GOOGLE_API_KEY não está configurada para o coordenador.")
                from google import genai
                from google.adk.models.google_llm import Gemini

                client = (
                    genai.Client(
                        vertexai=True, project=self.google_project, location=self.google_location
                    )
                    if self.vertex_ai
                    else genai.Client(api_key=self.api_key)
                )
                model_backend: Any = Gemini(model=self.model, client=client)

                def model_for(_kind: str) -> Any:
                    return model_backend
            elif self.mode == "local":

                def model_for(_kind: str) -> Any:
                    return _ollama_adk_model(self.model, self.ollama_base_url, self.timeout_seconds)
            else:

                def model_for(_kind: str) -> Any:
                    return _OfflineADKLlm(self.model, _kind, message, self.sessions)

            should_prepare_mcp = self.use_mcp and (
                self.mode in {"live", "local"}
                or _offline_route(message) in {"schedule_agent", "combined_agent"}
            )
            if should_prepare_mcp:
                if self.circuit_breaker:
                    self.circuit_breaker.allow()
                mcp = ScheduleMCPClient(
                    timeout_seconds=min(self.timeout_seconds, 10),
                    database=self.database,
                    seed_catalog=self.database is None,
                )
                await mcp.start()

            async def list_sessions(
                track: str | None = None,
                level: str | None = None,
                after: str | None = None,
                limit: int = 3,
            ) -> dict[str, Any]:
                nonlocal mcp
                budget.record("mcp.list_sessions" if self.use_mcp else "schedule.list_sessions")
                if self.use_mcp:
                    assert mcp is not None
                    result = await mcp.call_tool(
                        "list_sessions",
                        {"track": track, "level": level, "after": after, "limit": limit},
                    )
                else:
                    result = DemoTools(self.sessions).list_sessions(
                        track=track, level=level, after=after
                    )
                return result

            async def search_policy(question: str) -> dict[str, Any]:
                budget.record("faq.search_policy")
                return await asyncio.to_thread(DemoTools([]).search_policy, question)

            tools = [list_sessions, search_policy]
            specialists: list[Any] = []
            for name, instruction in (
                (
                    "schedule_agent",
                    "Consulte list_sessions e responda somente JSON com answer, session_ids, "
                    "sources e reservation_proposal. IDs devem vir da ferramenta. "
                    "Nunca confirme reservas.",
                ),
                (
                    "faq_agent",
                    "Consulte search_policy e responda somente JSON com answer, session_ids, "
                    "sources e reservation_proposal. Use somente a evidência retornada.",
                ),
                (
                    "combined_agent",
                    "Para solicitação mista, use list_sessions e search_policy; responda JSON "
                    "com answer, session_ids, sources e reservation_proposal. "
                    "Nunca confirme reservas.",
                ),
            ):
                agent_model: Any = model_for(name)
                selected_tools: list[Any] = (
                    tools
                    if name == "combined_agent"
                    else ([list_sessions] if name == "schedule_agent" else [search_policy])
                )
                specialists.append(
                    Agent(
                        name=name,
                        model=agent_model,
                        description=instruction,
                        instruction=instruction,
                        tools=selected_tools,
                    )
                )
            root_model: Any = model_for("coordinator")
            root = Agent(
                name="event_coordinator",
                model=root_model,
                instruction=(
                    "Delegue para schedule_agent quando perguntarem sobre agenda ou palestras; "
                    "faq_agent para políticas e informações; combined_agent para pedidos "
                    "que pedem ambos. "
                    "Nunca responda diretamente, nunca confirme reservas."
                ),
                sub_agents=specialists,
            )
            # ADK's built-in transfer_to_agent is the only orchestration mechanism.
            session_service = InMemorySessionService()
            runner = Runner(agent=root, app_name="eventops-ai", session_service=session_service)
            user_id, session_id = "demo-user", str(uuid4())
            await session_service.create_session(
                app_name="eventops-ai", user_id=user_id, session_id=session_id
            )
            prompt = "\n".join(
                [*(f"Histórico: {x}" for x in (history or [])[-4:]), f"Pergunta atual: {message}"]
            )
            final_text = ""
            input_tokens = output_tokens = 0
            usage_seen = False
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                    ):
                        target = getattr(getattr(event, "actions", None), "transfer_to_agent", None)
                        if target:
                            budget.record(f"delegate:{target}")
                            adk_tools.append(f"delegate:{target}")
                            adk_events.append(
                                {
                                    "type": "transfer",
                                    "agent": str(getattr(event, "author", "")),
                                    "target": str(target),
                                }
                            )
                        for part in getattr(getattr(event, "content", None), "parts", None) or []:
                            call = getattr(part, "function_call", None)
                            if not call or call.name == "transfer_to_agent":
                                continue
                            name = str(call.name)
                            trace_name = {
                                "list_sessions": "mcp.list_sessions"
                                if self.use_mcp
                                else "schedule.list_sessions",
                                "check_schedule": "mcp.check_schedule"
                                if self.use_mcp
                                else "schedule.check_schedule",
                                "search_policy": "faq.search_policy",
                            }.get(name, name)
                            adk_tools.append(trace_name)
                            adk_events.append(
                                {
                                    "type": "tool_call",
                                    "agent": str(getattr(event, "author", "")),
                                    "name": name,
                                    "trace_tool": trace_name,
                                }
                            )
                        usage = getattr(event, "usage_metadata", None)
                        if usage:
                            p_tokens = getattr(usage, "prompt_token_count", None)
                            c_tokens = getattr(usage, "candidates_token_count", None)
                            if isinstance(p_tokens, int) and isinstance(c_tokens, int):
                                input_tokens += p_tokens
                                output_tokens += c_tokens
                                usage_seen = True
                        if event.is_final_response() and event.content:
                            final_text = "".join(p.text or "" for p in event.content.parts or [])
            finally:
                await runner.close()
            payload = _parse_result(final_text)
            known = {s.id for s in self.sessions}
            ids = payload.get("session_ids", [])
            if not isinstance(ids, list) or any(i not in known for i in ids):
                raise ProviderError("O ADK retornou IDs de sessão inválidos.")
            sources = payload.get("sources", [])
            if not isinstance(sources, list):
                sources = []
            if (
                self.circuit_breaker
                and self.use_mcp
                and any(t.endswith("list_sessions") for t in budget.tools)
            ):
                self.circuit_breaker.success()
            return AgentResult(
                str(payload.get("answer") or "Não encontrei evidência suficiente."),
                ids,
                sources,
                payload.get("reservation_proposal"),
                f"{self.mode}:multi",
                {
                    "request_id": f"multi-{time.time_ns()}",
                    "mode": f"{self.mode}:multi",
                    "model": self.model,
                    "dataset_version": COORDINATOR_PROMPT_VERSION,
                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                    "tool_calls": budget.tool_calls,
                    "delegations": budget.delegations,
                    "tools": adk_tools,
                    "adk_events": adk_events,
                    "input_tokens": input_tokens if usage_seen else None,
                    "output_tokens": output_tokens if usage_seen else None,
                    "status": "ok",
                    "raw_output": final_text,
                    "validated": True,
                    "fallback": self.mode == "offline",
                    "reservation_writes": 0,
                    "circuit_state": self.circuit_breaker.state.value
                    if self.circuit_breaker
                    else "unconfigured",
                },
            )
        except Exception as exc:
            if self.circuit_breaker and isinstance(exc, (TimeoutError, OSError)):
                self.circuit_breaker.failure()
            error = str(exc) or type(exc).__name__
            partial_answer = ""
            partial_sources: list[dict[str, Any]] = []
            if (
                isinstance(exc, Exception)
                and self.circuit_breaker
                and self.circuit_breaker.state.value == "open"
            ):
                try:
                    budget.delegate("faq_agent")
                    budget.record_tool("faq.search_policy")
                    partial = await asyncio.to_thread(DemoTools([]).search_policy, message)
                    partial_answer = str(partial.get("answer", ""))
                    partial_sources = partial.get("sources", [])
                except Exception:
                    partial_answer = ""
            return AgentResult(
                ("Agenda indisponível; resposta parcial baseada na FAQ: " + partial_answer)
                if partial_answer
                else "O coordenador ADK não conseguiu concluir. Tente novamente.",
                [],
                partial_sources,
                None,
                f"{self.mode}:multi",
                {
                    "request_id": f"multi-{time.time_ns()}",
                    "mode": f"{self.mode}:multi",
                    "model": self.model,
                    "dataset_version": COORDINATOR_PROMPT_VERSION,
                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                    "tool_calls": budget.tool_calls,
                    "delegations": budget.delegations,
                    "tools": adk_tools,
                    "adk_events": adk_events,
                    "input_tokens": None,
                    "output_tokens": None,
                    "status": "error",
                    "raw_output": None,
                    "validated": False,
                    "fallback": True,
                    "reservation_writes": 0,
                    "partial": bool(partial_answer),
                },
                error=error,
            )
        finally:
            if mcp is not None:
                await mcp.close()


def _parse_result(raw: str) -> dict[str, Any]:
    candidate = raw.strip().lstrip("\ufeff")
    candidate = re.sub(r"```(?:json)?\s*|\s*```", "", candidate, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise ProviderError("O agente ADK não retornou JSON válido.")
