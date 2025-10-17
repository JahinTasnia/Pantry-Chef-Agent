from __future__ import annotations
import argparse, os, json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph import build_agent, AgentConfig
from tools import *
from tools import RECIPES_PATH
from pydantic import BaseModel

load_dotenv()

try:
    from fastapi import FastAPI
    import uvicorn

    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False


class Query(BaseModel):
    pantry: list[str]
    servings: int | None = None


def main_cli():
    agent = build_agent(AgentConfig())
    print("PantryChef ready. Example: eggs, onion, tomato | servings=2")
    while True:
        try:
            raw = input("Pantry> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit"}:
            break

        msg = raw
        result = agent.invoke({"messages": [HumanMessage(content=msg)]})
        final = result["messages"][-1].content
        print("\n" + str(final) + "\n")


def ensure_seed():
    if not os.path.exists(RECIPES_PATH):
        os.makedirs(os.path.dirname(RECIPES_PATH), exist_ok=True)
        seed = [
            {
                "id": "pasta_aglio_e_olio",
                "title": "Pasta Aglio e Olio",
                "ingredients": [
                    "spaghetti",
                    "garlic",
                    "olive oil",
                    "chili flakes",
                    "parsley",
                    "salt",
                ],
                "steps": [
                    "Boil pasta in salted water until al dente.",
                    "Gently sauté sliced garlic in olive oil; add chili flakes.",
                ],
                "servings": 2,
            }
        ]
        with open(RECIPES_PATH, "w", encoding="utf-8") as f:
            json.dump(seed, f, indent=2)


def main_api(host: str = "127.0.0.1", port: int = 8000):
    if not HAVE_FASTAPI:
        raise RuntimeError(
            "FastAPI/uvicorn not available; install dependencies or run with --cli"
        )
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="PantryChef API")

    @app.get("/")
    def read_root():
        return {"status": "PantryChef API running"}

    @app.post("/recommend")
    def recommend(q: Query):
        agent = build_agent(AgentConfig())
        text = ", ".join(q.pantry)
        if q.servings:
            text += f" | servings={q.servings}"
        from langchain_core.messages import HumanMessage

        result = agent.invoke({"messages": [HumanMessage(content=text)]})
        return {"result": result["messages"][-1].content}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Host for the API server")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for the API server"
    )
    parser.add_argument(
        "--cli", action="store_true", help="Run interactive CLI instead of API"
    )
    args = parser.parse_args()

    ensure_seed()

    if args.cli or not HAVE_FASTAPI:
        main_cli()
    else:
        main_api(args.host, args.port)
