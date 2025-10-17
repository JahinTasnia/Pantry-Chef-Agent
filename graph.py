from __future__ import annotations
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import TOOLS
import os


class AgentConfig(BaseModel):
    model: str = Field(default=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))


class GraphState(TypedDict):
    messages: List


SYSTEM_PROMPT = (
    "You are PantryChef, a tough-love but helpful recipe agent. "
    "Given a list of pantry items (comma-separated), find suitable dishes, "
    "list missing items, propose substitutions when possible, and output a concise recipe. "
    "Prefer simple, reliable methods. Ask ONE concise clarifying question only if absolutely required. "
    "When user provides servings, scale ingredients. Keep steps numbered, clear, and minimal."
)


def build_agent(config: AgentConfig):
    model = ChatGroq(model=config.model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "placeholder",
                "{messages}",
            ),  # <- required so the agent can inject the chat history
        ]
    )

    return create_react_agent(
        model=model,
        tools=TOOLS,
        prompt=prompt,
    )
