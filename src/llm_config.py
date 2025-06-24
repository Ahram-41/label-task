import os
from typing import Any, Dict

from langchain_openai import ChatOpenAI

model_configurations: Dict[str, Dict[str, Any]] = {
    "deepseek-chat": {
        "class": ChatOpenAI,
        "kwargs": {
            "temperature": 0.0,
            "model": "deepseek-chat",
            "openai_api_base": "https://api.deepseek.com",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
        },
    },
    "gpt-4o": {
        "class": ChatOpenAI,
        "kwargs": {
            "temperature": 0.0,
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 300,
        },
    },
}


def get_llm_model(llm_model: str):
    # Get the configuration for the specified model, or use a default configuration
    config = model_configurations.get(
        llm_model,
        {"class": ChatOpenAI, "kwargs": {"temperature": 0.0, "model": llm_model}},
    )

    # Instantiate and return the language model with the specified configuration
    return config["class"](**config["kwargs"])


if __name__ == "__main__":
    llm = get_llm_model("deepseek-chat")
    print(llm)
    response = llm.invoke("Hello, which model are you?")
    print(response)
