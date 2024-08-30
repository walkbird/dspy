import logging
from typing import Any, Literal, Optional

import qianfan

from dsp.modules.lm import LM


class QianfanLM(LM):
    """Wrapper around Baidu's Qianfan API."""

    def __init__(
        self,
        model: str = "ERNIE-4.0-Turbo-8K",
        endpoint: Optional[str] = None,
        model_type: Literal["chat", "completion"] = "chat",
        retry_count: Optional[int] = None,
        request_timeout: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "qianfan"
        self.model_type = model_type
        if model_type == "chat":
            self.client = qianfan.ChatCompletion()
        else:
            self.client = qianfan.Completion()

        self.kwargs = {"temperature": 0.7, "top_p": 1, "stream": False, **kwargs}

        self.kwargs["model"] = model
        if endpoint is not None:
            self.kwargs["endpoint"] = endpoint

        if retry_count is not None:
            self.kwargs["retry_count"] = retry_count
        if request_timeout is not None:
            self.kwargs["request_timeout"] = request_timeout
        if backoff_factor is not None:
            self.kwargs["backoff_factor"] = backoff_factor

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "chat":
            messages = [{"role": "user", "content": prompt}]
            kwargs["messages"] = messages
        else:
            kwargs["prompt"] = prompt

        response = self.client.do(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Qianfan whilst handling rate limiting and caching."""
        return self.basic_request(prompt, **kwargs)

    def __call__(self, prompt: str, only_completed: bool = True, return_sorted: bool = False, **kwargs) -> list[str]:
        """Retrieves completions from Qianfan.

        Args:
            prompt (str): prompt to send to Qianfan
            only_completed (bool, optional): return only completed responses. Defaults to True.
            return_sorted (bool, optional): sort the completion choices. Defaults to False.

        Returns:
            list[str]: list of completion choices
        """
        assert only_completed, "Qianfan does not support incomplete responses"
        assert not return_sorted, "Sorting not implemented for Qianfan"

        response = self.request(prompt, **kwargs)

        self.log_usage(response)

        return [response["body"]["result"]]

    def log_usage(self, response):
        """Log the total tokens from the Qianfan API response."""
        usage_data = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logging.debug(f"Qianfan Response Token Usage: {total_tokens}")
