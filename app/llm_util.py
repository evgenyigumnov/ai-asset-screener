import asyncio

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

def ask_llm(prompt: str, model: str, endpoint: str, api_key: str, use_calculator: bool = False) -> str:
    """
    Синхронная функция-обёртка: создаёт ReAct-агента с MCP-тулзой `calculator`,
    дергает LLM и возвращает последний ответ строкой.

    ВНИМАНИЕ: URL, API-ключ и модель заданы внутри функции. Поменяй под себя.
    """

    # --- 1) Настройки LLM (заданы внутри функции, как просил) ---
    BASE_URL = endpoint  # при использовании OpenAI
    API_KEY = api_key
    MODEL_NAME = model  # или другой — например "gpt-4.1-mini"

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
                raise RuntimeError("Не удалось загрузить MCP-инструменты (calculator). Проверь установку mcp_server_calculator.")

        # 2.3) Небольшой системный промпт
        system_prompt = (
            "Ты — помогающий ассистент."
            "Язык ответа: русский."
        )

        if use_calculator:
            system_prompt = (
                "Ты — помогающий ассистент. Если задача про вычисления — используй инструмент 'calculator'. "
                "Язык ответа: русский."
            )


        # 2.4) ReAct-агент с тулзами
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )

        # 2.5) Запрос
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt_text)]})

        # 2.6) Достаём последний ответ
        msg = result["messages"][-1].content
        if isinstance(msg, str):
            return msg
        # На всякий случай, если LLM вернул структурированный контент
        if isinstance(msg, list):
            # склеим текстовые части
            parts = []
            for chunk in msg:
                if isinstance(chunk, dict) and "text" in chunk:
                    parts.append(str(chunk["text"]))
                else:
                    parts.append(str(chunk))
            return "\n".join(parts)
        return str(msg)

    # --- 3) Запуск асинхронной части синхронно ---
    return asyncio.run(_run(prompt))


