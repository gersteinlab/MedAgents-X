import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class RetrieverConfig:
    query_encoder_name: str = "ncbi/MedCPT-Query-Encoder"
    cross_encoder_name: str = "ncbi/MedCPT-Cross-Encoder"
    max_length: int = 512
    batch_size: int = 16


class MedCPTRetriever:
    def __init__(self, 
                 client: Any,
                 device: str = "cpu",
                 config: RetrieverConfig = RetrieverConfig()):
        self.client = client
        self.device = torch.device(device)
        self.config = config
        self._cache: Dict[str, Tensor] = {}

    @property
    def query_tokenizer(self):
        if not hasattr(self, "_q_tok"):
            self._q_tok = AutoTokenizer.from_pretrained(self.config.query_encoder_name)
        return self._q_tok

    @property
    def query_model(self):
        if not hasattr(self, "_q_mod"):
            self._q_mod = AutoModel.from_pretrained(self.config.query_encoder_name) \
                                  .to(self.device).eval()
        return self._q_mod

    @property
    def cross_tokenizer(self):
        if not hasattr(self, "_c_tok"):
            self._c_tok = AutoTokenizer.from_pretrained(self.config.cross_encoder_name)
        return self._c_tok

    @property
    def cross_model(self):
        if not hasattr(self, "_c_mod"):
            self._c_mod = AutoModelForSequenceClassification.from_pretrained(
                self.config.cross_encoder_name
            ).to(self.device).eval()
        return self._c_mod

    def _batch_encode(self, texts: List[str]) -> Tensor:
        inputs = self.query_tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            embeds = self.query_model(**inputs).last_hidden_state[:, 0, :]
        return embeds  # (batch, dim)

    def encode(self, query: str) -> Tensor:
        if query not in self._cache:
            emb = self._batch_encode([query])[0]
            self._cache[query] = emb.cpu()
        return self._cache[query].to(self.device)

    def _search(self, collection: str, embedding: List[float], topk: int):
        return self.client.search(
            collection_name=collection,
            data=[embedding],
            limit=topk,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text", "source"],
        )[0]

    def retrieve(
        self,
        query: str,
        topk: int = 100,
        sources: Optional[List[str]] = None
    ) -> List[str]:
        emb = self.encode(query).cpu().tolist()
        collections = sources if sources is not None else self.client.list_collections()
        docs: List[str] = []

        for coll in collections:
            try:
                hits = self._search(coll, emb, topk)
                docs.extend(hit["entity"]["text"] for hit in hits[:topk])
            except Exception as e:
                logger.warning(f"Search failed on {coll}: {e}")

        return list(dict.fromkeys(docs))

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        all_scores: List[float] = []
        pairs = [[query, d] for d in docs]
        for i in range(0, len(pairs), self.config.batch_size):
            batch = pairs[i : i + self.config.batch_size]
            inputs = self.cross_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.cross_model(**inputs).logits.squeeze(-1).cpu().tolist()
            all_scores.extend(logits)

        ranked = [d for _, d in sorted(zip(all_scores, docs), reverse=True)]
        return ranked