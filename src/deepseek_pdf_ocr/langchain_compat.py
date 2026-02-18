"""LangChain ↔ OpenAI 消息格式互转（需安装 ``langchain-core``）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def langchain_to_openai(messages: list[BaseMessage]) -> list[dict]:
    """将 LangChain ``BaseMessage`` 列表转为 OpenAI dict 格式。"""
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    return [{"role": role_map[m.type], "content": m.content} for m in messages]

def openai_to_langchain(messages: list[dict]) -> list[BaseMessage]:
    """将 OpenAI dict 格式转为 LangChain ``BaseMessage`` 列表。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    cls_map = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [cls_map[m["role"]](content=m["content"]) for m in messages]