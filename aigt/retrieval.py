"""Independent BM25 + vector retrieval + weighted reciprocal rank fusion."""
from __future__ import annotations
import hashlib
import math
import re
from collections import Counter
from typing import Protocol
from .schema import Document, Query, instant


def terms(text: str) -> list[str]:
    # Baseline lexical segmentation, NOT a production Chinese tokenizer.
    latin = re.findall(r"[a-z0-9]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    return latin + [t for word in chinese for t in
                    list(word) + [word[i:i + 2] for i in range(len(word) - 1)]]


class Encoder(Protocol):
    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class HashEncoder:
    """Offline deterministic lexical vector baseline; not a semantic model."""
    def __init__(self, dimensions: int = 256):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, texts, *, query=False):
        result = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token, count in Counter(terms(text)).items():
                idx = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
                vector[idx % self.dimensions] += count
            norm = math.sqrt(sum(v * v for v in vector)) or 1
            result.append([v / norm for v in vector])
        return result


class HybridRetriever:
    def __init__(self, documents: list[Document], encoder=None, reranker=None):
        if len({d.id for d in documents}) != len(documents):
            raise ValueError("duplicate document ids")
        self.docs = documents
        self.encoder = encoder or HashEncoder()
        self.reranker = reranker
        self.counts = [Counter(terms(d.text + " " + (d.city or ""))) for d in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.mean_length = sum(self.lengths) / max(1, len(documents)) or 1
        self.df = Counter(t for c in self.counts for t in c)
        self.vectors = self.encoder.encode([d.text for d in documents])
        if len(self.vectors) != len(documents) or (self.vectors and
                (not self.vectors[0] or any(len(v) != len(self.vectors[0]) for v in self.vectors))):
            raise ValueError("encoder returned invalid document vectors")

    def eligible(self, doc: Document, query: Query) -> bool:
        if instant(doc.published_at) > instant(query.as_of):
            return False
        # Strict filters are justified only when these metadata are trusted.
        if query.city and doc.city != query.city:
            return False
        if query.target_date and doc.valid_on != query.target_date:
            return False
        return not query.topic or doc.topic == query.topic

    def bm25(self, text: str) -> list[float]:
        out = []
        for counts, length in zip(self.counts, self.lengths):
            score = 0.0
            for token in set(terms(text)):
                freq = counts[token]
                df = self.df[token]
                idf = math.log(1 + (len(self.docs) - df + 0.5) / (df + 0.5))
                score += idf * freq * 2.2 / (freq + 1.2 *
                         (0.25 + 0.75 * length / self.mean_length))
            out.append(score)
        return out

    def search(self, query: Query, *, top_k=5, candidate_k=20,
               keyword_weight=0.5, filter_metadata=True) -> list[dict]:
        if not 0 <= keyword_weight <= 1 or top_k < 1 or candidate_k < top_k:
            raise ValueError("invalid retrieval parameters")
        eligible = [i for i, d in enumerate(self.docs)
                    if not filter_metadata or self.eligible(d, query)]
        if not eligible:
            return []
        bm = self.bm25(query.search_text)
        qv = self.encoder.encode([query.search_text], query=True)[0]
        if len(qv) != len(self.vectors[0]):
            raise ValueError("query/document vector dimensions differ")
        dense = [sum(a*b for a, b in zip(qv, dv)) for dv in self.vectors]
        lexical = sorted(eligible, key=lambda i: (-bm[i], self.docs[i].id))[:candidate_k]
        semantic = sorted(eligible, key=lambda i: (-dense[i], self.docs[i].id))[:candidate_k]
        scores: dict[int, float] = {}
        for weight, ranking in ((keyword_weight, lexical), (1-keyword_weight, semantic)):
            if not weight:
                continue
            for rank, i in enumerate(ranking, 1):
                scores[i] = scores.get(i, 0) + weight / (60 + rank)
        fused = sorted(scores, key=lambda i: (-scores[i], self.docs[i].id))
        reranked = {}
        if self.reranker:
            values = self.reranker.score(query.search_text, [self.docs[i].text for i in fused])
            if len(values) != len(fused):
                raise ValueError("reranker returned mismatched scores")
            reranked = dict(zip(fused, values))
            fused.sort(key=lambda i: (-reranked[i], -scores[i], self.docs[i].id))
        return [{"document": self.docs[i].visible(), "fusion_score": scores[i],
                 "rerank_score": reranked.get(i)} for i in fused[:top_k]]
