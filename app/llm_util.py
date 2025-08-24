import asyncio

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

def ask_llm(prompt: str, model: str, endpoint: str, api_key: str, use_calculator: bool = False) -> str:
    BASE_URL = endpoint
    API_KEY = api_key
    MODEL_NAME = model
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

        system_prompt = (
            "You are a helping assistant."
        )

        if use_calculator:
            system_prompt = (
                "You are a helping assistant. If the task is about calculations, use the 'calculator' tool. "
            )

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


