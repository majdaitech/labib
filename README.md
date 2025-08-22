# LABIB  Lightweight Agent-Based Intelligent Builder

Lightweight agent SDK with optional tools and multi-agent orchestration.

- Works with Mock (no keys), OpenAI, and Ollama backends
- Optional MCP tool integration
- CLI and Streamlit UI shims via labib during rename from majdk

## Install

- Editable + all extras:
`
pip install -U pip
pip install -e .[all]
`
- Minimal:
`
pip install -e .
`

## Quickstart

Run the example:
`
python examples/quickstart.py
`
Uses labib.Agent, labib.tool, and labib.MockModel. No API keys required.

## CLI

Entrypoint: labib (proxies to original CLI)

Examples:
`
labib --model-type mock --interactive
labib --model-type openai --api-key YOUR_KEY --model gpt-4o-mini --query "Hello"
labib --model-type ollama --base-url http://localhost:11434 --model gemma3:1b --interactive
`

## Examples

- Manual orchestration (Mock):
`
python examples/orchestration.py
`
- Ollama orchestration (no tool-calls):
`
python examples/orchestration_ollama.py
`
- Auto orchestration:
`
python examples/auto_orchestration.py --goal "Summarize recent LLM evals" --model-type mock --mode sequential
`
- Auto-plan (lets the model propose agents):
`
python examples/auto_orchestration.py --goal "Compare 3 frameworks" --model-type openai --openai-model gpt-4o-mini --auto-plan
`
- MCP (optional):
`
pip install -e .[mcp]
python examples/auto_orchestration.py --goal "Research X" --model-type openai --sources builtin,mcp --mcp-cmd "your_mcp_server --flag"
`

Notes:
- Ollama backend here does not make tool-calls; MCP tools wont be invoked with Ollama.
- APIs are re-exported via labib for a smooth rename from majdk. See src/labib/__init__.py.

## Dev and Build

`
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[dev]
pytest -q
`
Build and verify:
`
python -m pip install build twine
python -m build
twine check dist\*
`

## Project URLs

- Repo: https://github.com/majdaitech/labib

## License

Apache-2.0
