"""Optional actual model adapters. Loading is explicit; no automatic downloads on import."""
from __future__ import annotations
import json
import os
from urllib.request import Request, urlopen


class ChatEndpoint:
    """Non-streaming chat-completions compatible endpoint (e.g. your vLLM server)."""
    def __init__(self, base_url: str, model: str, *, timeout: int = 60,
                 api_key_env: str = "AIGT_API_KEY"):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.timeout, self.api_key_env = model, timeout, api_key_env

    def generate(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": 0, "max_tokens": 256}).encode()
        headers = {"Content-Type": "application/json"}
        key = os.getenv(self.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        with urlopen(Request(self.url, data=body, headers=headers), timeout=self.timeout) as r:
            data = json.load(r)
        return data["choices"][0]["message"]["content"]


class QwenEncoder:
    def __init__(self, model_path: str, *, device="cpu", batch_size=8, max_length=4096):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.device, self.batch_size, self.max_length = device, batch_size, max_length

    def encode(self, texts, *, query=False):
        results = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start+self.batch_size]
            if query:
                batch = ["Instruct: Retrieve evidence for the query respecting entity and date.\nQuery: " + t
                         for t in batch]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                # Last non-padding index works for both left/right padding.
                positions = self.torch.arange(hidden.shape[1], device=hidden.device)
                last = (inputs["attention_mask"] * positions).max(dim=1).values
                vectors = hidden[self.torch.arange(hidden.shape[0], device=hidden.device), last]
                results.extend(self.torch.nn.functional.normalize(vectors.float(), dim=-1).cpu().tolist())
        return results


class QwenReranker:
    """Qwen yes/no scoring with reserved suffix, following official input protocol."""
    def __init__(self, model_path: str, *, device="cpu", max_length=4096, batch_size=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        if self.yes == self.no or None in (self.yes, self.no):
            raise ValueError("yes/no token mapping is not compatible")
        self.prefix = self.tokenizer.encode(
            '<|im_start|>system\nJudge whether the Document satisfies the Query. '
            'Answer only "yes" or "no".<|im_end|>\n<|im_start|>user\n',
            add_special_tokens=False)
        self.suffix = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False)
        if max_length <= len(self.prefix) + len(self.suffix):
            raise ValueError("max_length leaves no room for query and document")

    def score(self, query, documents):
        scores = []
        capacity = self.max_length - len(self.prefix) - len(self.suffix)
        for start in range(0, len(documents), self.batch_size):
            ids = []
            for doc in documents[start:start+self.batch_size]:
                body = self.tokenizer.encode(
                    "<Instruct>: Retrieve evidence respecting entity and date.\n<Query>: "
                    + query + "\n<Document>: " + doc, add_special_tokens=False)
                ids.append(self.prefix + body[:capacity] + self.suffix)
            inputs = self.tokenizer.pad({"input_ids": ids}, padding=True,
                                        return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, [self.no, self.yes]].float()
                scores.extend(logits.softmax(-1)[:, 1].cpu().tolist())
        return scores
