from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from vanna.core.llm import LlmService, LlmRequest, LlmResponse, LlmStreamChunk
from vanna.core.tool import ToolCall, ToolSchema


class OpenAIResponsesService(LlmService):
    """Azure OpenAI Responses API implementation (async)."""

    def __init__(
        self,
        deployment_name: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-06-01",
    ) -> None:
        try:
            from openai import AsyncAzureOpenAI
            from openai.types.responses import Response
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "openai>=1.0.0 package is required. Install with: pip install openai"
            ) from e

        self.Response = Response
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version

        if not (self.api_key and self.azure_endpoint):
            raise ValueError("Both azure_endpoint and api_key are required for Azure OpenAI responses.")

        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        payload = self._payload(request)
        resp = await self.client.responses.create(**payload)
        self._debug_print("response", resp)
        text, tools, status, usage = self._extract(resp)
        return LlmResponse(
            content=text,
            tool_calls=tools or None,
            finish_reason=status,
            usage=usage or None,
            metadata={"request_id": getattr(resp, "id", None)},
        )

    async def stream_request(self, request: LlmRequest) -> AsyncGenerator[LlmStreamChunk, None]:
        payload = self._payload(request)
        async with self.client.responses.stream(**payload) as stream:
            async for event in stream:
                self._debug_print("stream_event", event)
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield LlmStreamChunk(content=delta)

            final = await stream.get_final_response()
            self._debug_print("final_response", final)

        _text, tools, status, _usage = self._extract(final)
        yield LlmStreamChunk(tool_calls=tools or None, finish_reason=status)

    async def validate_tools(self, tools: List[Any]) -> List[str]:
        return []  # minimal validation

    # ---- helpers ----

    def _payload(self, request: LlmRequest) -> Dict[str, Any]:
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        p: Dict[str, Any] = {
            "model": self.deployment_name,
            "input": msgs,
        }
        if request.system_prompt:
            p["instructions"] = request.system_prompt
        if request.max_tokens:
            p["max_output_tokens"] = request.max_tokens
        if request.tools:
            p["tools"] = [self._serialize_tool(t) for t in request.tools]
        return p

    def _debug_print(self, label: str, obj: Any) -> None:
        try:
            payload = obj.model_dump()
        except AttributeError:
            try:
                payload = obj.dict()
            except AttributeError:
                payload = obj
        print(f"[AzureOpenAIResponsesService] {label}: {payload}")

    def _extract(
        self, resp
    ) -> Tuple[Optional[str], Optional[List[ToolCall]], Optional[str], Optional[Dict[str, int]]]:
        text = getattr(resp, "output_text", None)
        tool_calls: List[ToolCall] = []

        for oc in getattr(resp, "output", []) or []:
            for item in getattr(oc, "content", []) or []:
                if getattr(item, "type", None) == "tool_call":
                    tc = getattr(item, "tool_call", None)
                    if tc and getattr(tc, "type", None) == "function":
                        fn = getattr(tc, "function", None)
                        if fn:
                            name = getattr(fn, "name", None)
                            args = getattr(fn, "arguments", None)
                            if not isinstance(args, (dict, list)):
                                try:
                                    args = json.loads(args) if args else {}
                                except Exception:
                                    args = {"_raw": args}
                            tool_calls.append(ToolCall(name=name, arguments=args))

        usage = None
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(resp.usage, "output_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", None)
                or ((getattr(resp.usage, "input_tokens", 0) or 0)
                    + (getattr(resp.usage, "output_tokens", 0) or 0)),
            }

        status = getattr(resp, "status", None)
        return text, (tool_calls or None), status, usage

    def _serialize_tool(self, tool: Any) -> Dict[str, Any]:
        if isinstance(tool, ToolSchema):
            return {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": False,
            }
        if hasattr(tool, "model_dump"):
            data = tool.model_dump()
            if all(key in data for key in ("name", "description", "parameters")):
                return {
                    "type": "function",
                    "name": data["name"],
                    "description": data["description"],
                    "parameters": data["parameters"],
                    "strict": data.get("strict", False),
                }
            return data
        if isinstance(tool, dict):
            if "type" in tool:
                return tool
            if all(k in tool for k in ("name", "description", "parameters")):
                return {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                    "strict": tool.get("strict", False),
                }
            return tool
        raise TypeError(f"Unsupported tool schema type: {type(tool)!r}")
