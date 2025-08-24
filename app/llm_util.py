import asyncio
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


@dataclass
class _LLMConfig:
    model: Optional[str] = None
    endpoint: Optional[str] = None
    api_key: Optional[str] = None

_cfg = _LLMConfig()


def init_llm(model: str, endpoint: Optional[str], api_key: str) -> None:
    if not model:
        raise RuntimeError("init_llm: model is required")
    _cfg.model = model
    _cfg.endpoint = endpoint
    _cfg.api_key = api_key


def ask_llm(prompt: str, use_calculator: bool = False) -> str:
    if not _cfg.model:
        raise RuntimeError("LLM is not initialized. Call init_llm(...) once at startup.")

    BASE_URL = _cfg.endpoint
    API_KEY = _cfg.api_key
    MODEL_NAME = _cfg.model

    async def _run(prompt_text: str) -> str:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.0,
        )

        tools = []
        if use_calculator:
            mcp_config = {
                "calculator": {
                    "command": "python",
                    "args": ["-m", "mcp_server_calculator"],
                    "transport": "stdio",
                }
            }
            mcp_client = MultiServerMCPClient(mcp_config)
            tools = await mcp_client.get_tools()
            if not tools:
                raise RuntimeError("Failed to load MCP tools (calculator). Check your mcp_server_calculator installation.")

        system_prompt = "You are a helping assistant."
        if use_calculator:
            system_prompt = "You are a helping assistant. If the task is about calculations, use the 'calculator' tool. "

        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )

        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt_text)]})
        msg = result["messages"][-1].content
        if isinstance(msg, str):
            return msg
        if isinstance(msg, list):
            parts = []
            for chunk in msg:
                if isinstance(chunk, dict) and "text" in chunk:
                    parts.append(str(chunk["text"]))
                else:
                    parts.append(str(chunk))
            return "\n".join(parts)
        return str(msg)

    return asyncio.run(_run(prompt))
