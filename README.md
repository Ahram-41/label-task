## 🚀 Quick Setup

### 1. Environment Configuration

Create a `.env` file using this template:
```
OPENAI_API_KEY=your_openai_api_key_here
XLS_PATH=path/to/your/data.xls
TAVILY_API_KEY=your_tavily_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

### 2. Python Environment Setup

We recommend using `uv` to manage the project:
```bash
uv venv
source .venv/bin/activate
uv sync
```

The virtual environment will be updated according to the `pyproject.toml` file.

## 📁 Project Structure

```
├── src/
│   ├── agent.py              # Main AI agent for data collection
│   ├── llm_analyst.py        # LLM-based data analysis
│   ├── utils/
│   │   ├── universal_transform.py   # Universal data transformation (replaces transform_founders_improved.py)
│   │   └── merge_csv_files.py       # Merge multiple CSV outputs
│   └── prompt/               # Prompt templates and mappings
└── outputs/                 # Generated data files
```

## 🔧 Core Components

### 1. Data Collection Agent (`src/agent.py`)

The main AI agent that researches companies and extracts structured data using web search and LLMs.

**Key Features:**
- Multi-threaded processing with configurable batch sizes
- Automatic error handling and retry logic
- Support for multiple data types: founders, executives, products, technology, partners
- Token usage tracking and cost monitoring
- Rate limiting and API error management


### 2. Universal Data Transformer (`src/utils/universal_transform.py`)

**Replaces the deprecated `transform_founders_improved.py`**

A data transformation utility that processes CSV data with embedded JSON-like structures.

**Key Features:**
- Proper CSV parsing with `csv.reader()` for complex quoted fields
- Balanced parentheses parsing for nested structures
- BOM character handling
- Smart column detection for mixed data structures
- `is_top10_university` feature for founder analysis

### 3. CSV Merger (`src/utils/merge_csv_files.py`)

Merges multiple CSV outputs from different data collection tasks into a single comprehensive dataset.

**Key Features:**
- Merges all data types: founder, executive, product, technology, partner, ipo_ma
- Token usage aggregation across all tasks
- Intelligent column handling and deduplication
- Data cleaning and validation
- Handles IPO/M&A data extraction from founder columns


## 🎯 LLM Analysis Tools

### LLM Analyst (`src/llm_analyst.py`)

Choose a set of prompts for analysis and run:
```python
from src.llm_analyst import LabelTask
from src.prompt.prompts import base_prompt

iea = LabelTask(base_prompt)
iea.process_xls("data.xls", "output.csv")
```

## 📈 Output Files

The pipeline generates several key output files:

- **Raw Data**: `full_{type}.csv` - Direct agent outputs
- **Expanded Data**: `{type}_expanded_clean.csv` - With additional analysis
- **Merged Data**: `full_merged_data.csv` - Combined all data types

