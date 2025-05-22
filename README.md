# Setup:

1. create a .env file using this template:
```
OPENAI_API_KEY=
XLS_PATH=
TAVILY_API_KEY
```
where the XLS_PATH is the xls data file you want to look into.

2. we recommend to use uv to manage the project
```bash
uv venv
source .venv/bin/activate
uv sync
```
then the .venv is updated as the `pyproject.toml` file.

3. choose a set of prompts you want to use in `src/llm_analyst.py`, and run it.