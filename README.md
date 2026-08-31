# LangGraph Mathematics Agent

A simple agentic mathematics workflow using LangGraph and an OpenRouter-compatible
chat model. Each problem passes through four focused steps:

1. **Planner** identifies the mathematical concepts and steps.
2. **Solver** works through the calculations.
3. **Verifier** checks the result and corrects errors.
4. **Finalizer** returns a concise tutor-style answer.

## Setup

Use Python 3.10 or newer, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open `.env` and replace `replace_with_your_openrouter_api_key` with your
OpenRouter API key. You can change `OPENROUTER_MODEL` to any model available on
OpenRouter.

## Run

```bash
python math_agent.py "Solve 3x + 7 = 22"
```

Or run it without an argument and enter the problem interactively:

```bash
python math_agent.py
```

The API key is loaded with `python-dotenv`, and `.env` is excluded from Git.
