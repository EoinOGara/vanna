"""
Azure OpenAI LLM service implementation.

This class adapts the OpenAI LlmService to work with Azure OpenAI endpoints.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from vanna.core.llm import (
    LlmService,
    LlmRequest,
    LlmResponse,
    LlmStreamChunk,
)
from vanna.core.tool import ToolCall, ToolSchema


class AzureOpenAILlmService(LlmService):
    """Azure OpenAI Chat Completions-backed LLM service.

    Args:
        deployment_name: The Azure OpenAI deployment name (maps to a specific model).
        api_key: Azure OpenAI API key.
        azure_endpoint: Azure OpenAI endpoint URL, e.g. 'https://my-resource.openai.azure.com/'.
        api_version: API version (default: '2024-06-01').
        **extra_client_kwargs: Optional extra arguments forwarded to `openai.AzureOpenAI()`.
    """

    def __init__(
        self,
        deployment_name: str,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-06-01",
        **extra_client_kwargs: Any,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except Exception as e:
            raise ImportError(
                "openai>=1.0.0 is required. Install with: pip install openai"
            ) from e

        self.deployment_name = deployment_name
        self.api_version = api_version
        self.azure_endpoint = (
            azure_endpoint
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_OPENAI_BASE_URL")
        )
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")

        if not self.azure_endpoint or not self.api_key:
            raise ValueError(
                "Both azure_endpoint and api_key are required for Azure OpenAI."
            )

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            **extra_client_kwargs,
        }

        self._client = AzureOpenAI(**client_kwargs)

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        """Send a non-streaming request to Azure OpenAI."""
        payload = self._build_payload(request)
        resp = self._client.chat.completions.create(**payload, stream=False)

        if not resp.choices:
            return LlmResponse(content=None, tool_calls=None, finish_reason=None)

        choice = resp.choices[0]
        content = getattr(choice.message, "content", None)
        tool_calls = self._extract_tool_calls_from_message(choice.message)

        usage: Dict[str, int] = {}
        if getattr(resp, "usage", None):
            usage = {
                k: int(v)
                for k, v in {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                    "total_tokens": getattr(resp.usage, "total_tokens", 0),
                }.items()
            }

        return LlmResponse(
            content=content,
            tool_calls=tool_calls or None,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage or None,
        )

    async def stream_request(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """Stream a request to Azure OpenAI."""
        payload = self._build_payload(request)
        stream = self._client.chat.completions.create(**payload, stream=True)

        tc_builders: Dict[int, Dict[str, Optional[str]]] = {}
        last_finish: Optional[str] = None

        for event in stream:
            if not getattr(event, "choices", None):
                continue

            choice = event.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                last_finish = getattr(choice, "finish_reason", last_finish)
                continue

            content_piece = getattr(delta, "content", None)
            if content_piece:
                yield LlmStreamChunk(content=content_piece)

            streamed_tool_calls = getattr(delta, "tool_calls", None)
            if streamed_tool_calls:
                for tc in streamed_tool_calls:
                    idx = getattr(tc, "index", 0) or 0
                    b = tc_builders.setdefault(
                        idx, {"id": None, "name": None, "arguments": ""}
                    )
                    if getattr(tc, "id", None):
                        b["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            b["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            b["arguments"] = (b["arguments"] or "") + fn.arguments

            last_finish = getattr(choice, "finish_reason", last_finish)

        final_tool_calls: List[ToolCall] = []
        for b in tc_builders.values():
            if not b.get("name"):
                continue
            args_raw = b.get("arguments") or "{}"
            try:
                loaded = json.loads(args_raw)
                args_dict = loaded if isinstance(loaded, dict) else {"args": loaded}
            except Exception:
                args_dict = {"_raw": args_raw}
            final_tool_calls.append(
                ToolCall(
                    id=b.get("id") or "tool_call",
                    name=b["name"] or "tool",
                    arguments=args_dict,
                )
            )

        if final_tool_calls:
            yield LlmStreamChunk(tool_calls=final_tool_calls, finish_reason=last_finish)
        else:
            yield LlmStreamChunk(finish_reason=last_finish or "stop")

    async def validate_tools(self, tools: List[ToolSchema]) -> List[str]:
        """Validate tool schemas."""
        errors: List[str] = []
        for t in tools:
            if not t.name or len(t.name) > 64:
                errors.append(f"Invalid tool name: {t.name!r}")
        return errors

    def _build_payload(self, request: LlmRequest) -> Dict[str, Any]:
        """Convert LlmRequest to Azure OpenAI chat.completions payload."""
        messages: List[Dict[str, Any]] = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for m in request.messages:
            msg = {"role": m.role, "content": m.content}
            if m.role == "tool" and m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            elif m.role == "assistant" and m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            messages.append(msg)

        tools_payload: Optional[List[Dict[str, Any]]] = None
        if request.tools:
            tools_payload = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]

        payload: Dict[str, Any] = {
            "model": self.deployment_name,  # Azure uses deployment as model identifier
            "messages": messages,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if tools_payload:
            payload["tools"] = tools_payload
            payload["tool_choice"] = "auto"

        return payload

    def _extract_tool_calls_from_message(self, message: Any) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        for tc in raw_tool_calls:
            fn = getattr(tc, "function", None)
            if not fn:
                continue
            args_raw = getattr(fn, "arguments", "{}")
            try:
                loaded = json.loads(args_raw)
                args_dict = loaded if isinstance(loaded, dict) else {"args": loaded}
            except Exception:
                args_dict = {"_raw": args_raw}
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", "tool_call"),
                    name=getattr(fn, "name", "tool"),
                    arguments=args_dict,
                )
            )
        return tool_calls
