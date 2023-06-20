from typing import Dict, Union, Optional, List

from pydantic import root_validator

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoModel, AutoTokenizer



class ChatGLM(LLM):
    model: str = 'THUDM/chatglm-6b'
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    client: object = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs['model']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        client = AutoModel.from_pretrained(self.model, trust_remote_code=True).half().to('mps')
        self.client = client.eval()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response, _ = self.client.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response]]
        return response
