"""A small LangGraph mathematics agent powered by OpenRouter."""

from __future__ import annotations

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class MathState(TypedDict, total=False):
    problem: str
    plan: str
    solution: str
    verification: str
    answer: str


def get_model() -> ChatOpenAI:
    """Create the OpenRouter-backed chat model from environment settings."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key.startswith("replace_with_"):
        raise RuntimeError(
            "Set OPENROUTER_API_KEY in .env before running the math agent."
        )

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/your-username/math-agent",
            "X-Title": "LangGraph Mathematics Agent",
        },
        temperature=0,
    )


def plan_problem(state: MathState) -> dict[str, str]:
    model = get_model()
    response = model.invoke(
        [
            (
                "system",
                "You are a mathematical planner. Identify the relevant concepts, "
                "known values, and a short sequence of steps. Do not solve yet.",
            ),
            ("human", state["problem"]),
        ]
    )
    return {"plan": response.content}


def solve_problem(state: MathState) -> dict[str, str]:
    model = get_model()
    response = model.invoke(
        [
            (
                "system",
                "You are a precise mathematics solver. Follow the supplied plan, "
                "show the important calculations, and state a candidate final answer. "
                "Use plain text and LaTeX where useful.",
            ),
            (
                "human",
                f"Problem:\n{state['problem']}\n\nPlan:\n{state['plan']}",
            ),
        ]
    )
    return {"solution": response.content}


def verify_solution(state: MathState) -> dict[str, str]:
    model = get_model()
    response = model.invoke(
        [
            (
                "system",
                "You are a rigorous mathematics verifier. Check the candidate solution "
                "for arithmetic, algebra, units, and whether it answers the problem. "
                "If there is an error, give a corrected result and explain the fix. "
                "End with either VERIFIED or CORRECTED.",
            ),
            (
                "human",
                f"Problem:\n{state['problem']}\n\nCandidate solution:\n{state['solution']}",
            ),
        ]
    )
    return {"verification": response.content}


def compose_answer(state: MathState) -> dict[str, str]:
    model = get_model()
    response = model.invoke(
        [
            (
                "system",
                "You are the final mathematics tutor. Give a concise, self-contained "
                "answer using the verified result. Include enough working to be useful. "
                "Do not mention internal agents or the workflow.",
            ),
            (
                "human",
                f"Problem:\n{state['problem']}\n\nWorking:\n{state['solution']}"
                f"\n\nVerification:\n{state['verification']}",
            ),
        ]
    )
    return {"answer": response.content}


def build_graph():
    graph = StateGraph(MathState)
    graph.add_node("planner", plan_problem)
    graph.add_node("solver", solve_problem)
    graph.add_node("verifier", verify_solution)
    graph.add_node("finalizer", compose_answer)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "solver")
    graph.add_edge("solver", "verifier")
    graph.add_edge("verifier", "finalizer")
    graph.add_edge("finalizer", END)
    return graph.compile()


def main() -> None:
    load_dotenv()
    problem = " ".join(sys.argv[1:]).strip()
    if not problem:
        problem = input("Math problem: ").strip()
    if not problem:
        raise SystemExit("Please provide a mathematics problem.")

    result = build_graph().invoke({"problem": problem})
    print("\n" + result["answer"])


if __name__ == "__main__":
    main()
