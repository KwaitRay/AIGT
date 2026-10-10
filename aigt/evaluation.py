"""Independent objective checks for curated fixtures, not a semantic truth judge."""
from __future__ import annotations
import math


def evaluate(answer: dict, verifier: dict) -> dict:
    text = answer["text"]
    required = verifier.get("required", [])
    forbidden = verifier.get("forbidden", [])
    checks = {
        "required_items": all(part in text for part in required),
        "no_forbidden_items": not any(part in text for part in forbidden),
        "evidence_ids": set(verifier.get("evidence_ids", [])) <=
                        {e["document"]["id"] for e in answer["evidence"]},
    }
    if "city" in verifier:
        checks["city"] = answer["query"]["city"] == verifier["city"]
    if "target_date" in verifier:
        checks["target_date"] = answer["query"]["target_date"] == verifier["target_date"]
    length = answer["display"]["compliant"]
    if length is not None:
        checks["display_length"] = length
    # Unmeasured semantics remain None; missing judgements are not automatic full marks.
    return {"objective_pass": all(checks.values()), "checks": checks,
            "dimensions": {
                "language_logic": None,
                "factual_accuracy": None,
                "multi_turn_consistency": checks.get("city"),
                "timeliness": checks.get("target_date"),
                "need_fulfillment": checks["required_items"],
                "memory_use": None,
            },
            "limitations": "Fixture predicates only; semantic quality and Chinese word limit need review."}


def retrieval_metrics(ranked_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    if k < 1 or len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("k must be positive and ranked ids unique")
    if not relevant_ids:
        return {"hit": None, "recall": None, "ndcg": None}
    hit = [int(x in relevant_ids) for x in ranked_ids[:k]]
    dcg = sum(v / math.log2(i+2) for i, v in enumerate(hit))
    ideal = sum(1 / math.log2(i+2) for i in range(min(k, len(relevant_ids))))
    return {"hit": int(any(hit)), "recall": sum(hit)/len(relevant_ids), "ndcg": dcg/ideal}
