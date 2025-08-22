# LABIB  Lightweight AgentBased Intelligent Builder

A minimal, pragmatic agent SDK for Python. Build agents that can call tools, orchestrate multistep workflows, and run via CLI or Streamlit UI.

- Backends: Mock (no keys), OpenAI, Ollama
- Tools: simple @tool decorator + external tool registry
- Orchestration: manual, chain/map_reduce, or autoplan
- UI: labib-ui Streamlit app; CLI: labib

> For the project story, goals, and philosophy, see ABOUT.md.

---

##  Quickstart (60 seconds)

`python
from labib import Agent, MockModel

agent = Agent(MockModel())
print(agent.run("Say hello in one sentence."))
`

Run with OpenAI:
`python
from labib import Agent, OpenAIModel
agent = Agent(OpenAIModel(api_key="YOUR_KEY", model_name="gpt-4o-mini"))
print(agent.run("What's the weather style of Paris?"))
`

Add a tool:
`python
from labib import Agent, MockModel, tool

@tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b

agent = Agent(MockModel(), tools=[add])
print(agent.run("Use add to compute 7 and 5."))
`

---

##  Install

Windows (PowerShell):
`powershell
python -m venv .venv
.\ .venv\Scripts\Activate
pip install -U pip
pip install -e .[all]
`

macOS/Linux (bash):
`ash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[all]'
`

Minimal install:
`ash
pip install -e .
`

---

##  Examples

- Manual orchestration (Mock):
`ash
python examples/orchestration.py
`
- Ollama orchestration (no tool-calls):
`ash
python examples/orchestration_ollama.py
`
- Auto orchestration:
`ash
python examples/auto_orchestration.py --goal "Summarize recent LLM evals" --model-type mock --mode sequential
`
- Autoplan (model proposes agents):
`ash
python examples/auto_orchestration.py --goal "Compare 3 frameworks" --model-type openai --openai-model gpt-4o-mini --auto-plan
`
- MCP (optional):
`ash
pip install -e '.[mcp]'
python examples/auto_orchestration.py --goal "Research X" --model-type openai --sources builtin,mcp --mcp-cmd "your_mcp_server --flag"
`

Notes:
- Ollama backend here does not make toolcalls; MCP tools wont be invoked with Ollama.
- APIs are reexported via labib for a smooth rename from majdk. See src/labib/__init__.py.

---

##  CLI and UI

CLI:
`ash
labib --model-type mock --interactive
labib --model-type openai --api-key YOUR_KEY --model gpt-4o-mini --query "Hello"
labib --model-type ollama --base-url http://localhost:11434 --model gemma3:1b --interactive
`

Streamlit UI:
`ash
pip install -e '.[ui]'
labib-ui
`

---

##  Features

- Simple tool API: @tool decorator with autoschema
- External tools with explicit JSON schema + handler
- Memory logging with compact step traces
- Orchestrator helpers: chain, map_reduce
- Autoplan mode (optional)

---

##  Roadmap

- Better planner JSON validation and error messages
- Tool invocation with Ollama (when available upstream)
- More builtin tools and integrations

---

##  Project URLs

- Repo: https://github.com/majdaitech/labib
- About: https://github.com/majdaitech/labib/blob/main/ABOUT.md

##  License

Apache2.0
