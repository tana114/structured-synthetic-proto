import warnings
import os
from typing import Optional, Literal

from langchain_core._api.beta_decorator import LangChainBetaWarning

from logging import getLogger, NullHandler

# InMemoryRateLimiter使用時の警告を消す
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from dotenv import load_dotenv

load_dotenv()

valid_model_names = Literal[
    "meta-llama/Llama-Guard-4-12B",
    "gemma2-9b-it",
    "llama3-8b-8192",
    "llama3-70b-8192",
    'llama-3.1-8b-instant',
    'llama-3.3-70b-versatile',
]

from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter


class GroqChatBase(ChatGroq):
    def __init__(
            self,
            model_name: Literal[valid_model_names],
            requests_per_second: Optional[float] = None,
            **kwargs,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        shared_kwargs: dict = dict(
            api_key=api_key,
            model_name=model_name,
            **kwargs,
        )

        if requests_per_second:
            r_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second)
            shared_kwargs["rate_limiter"] = r_limiter

        super().__init__(**shared_kwargs)


if __name__ == "__main__":
    """
    python -m model.groq_llm
    """

    llm = GroqChatBase(
        model_name="llama-3.1-8b-instant",
        requests_per_second=0.32,
    )

    res = llm.invoke("hello")
    print(res)
