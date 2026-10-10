"""Training exports: shared visible context, grouped split, private verifier excluded."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .pipeline import assemble_messages
from .schema import Turn, Document, instant


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in records:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_name(group_id: str, *, salt="aigt-v1", validation_fraction=0.2):
    if not 0 < validation_fraction < 1 or not group_id:
        raise ValueError("invalid split configuration")
    score = int.from_bytes(hashlib.sha256((salt + ":" + group_id).encode()).digest()[:8], "big") / 2**64
    return "validation" if score < validation_fraction else "train"


def visible_messages(task):
    current = Turn("user", task["query"], task["as_of"])
    history = [Turn(**m) for m in task.get("history", [])]
    documents = [Document(**d) for d in task.get("evidence", [])]
    if any(instant(d.published_at) > instant(current.created_at) for d in documents):
        raise ValueError("future evidence is not model-visible")
    context = [{"document": d.visible()} for d in documents]
    return assemble_messages(history, current, context, task.get("memory", []))


def export_datasets(tasks, output, *, validation_fraction=.2):
    output = Path(output)
    groups = {}
    records = {(kind, split): [] for kind in ("sft", "dpo", "opd")
               for split in ("train", "validation")}
    manifest = []
    seen = set()
    for task in tasks:
        if task["id"] in seen:
            raise ValueError("duplicate task id")
        seen.add(task["id"])
        group = task["source_group"]
        split = split_name(group, validation_fraction=validation_fraction)
        groups[group] = split
        messages = visible_messages(task)
        if not isinstance(task["chosen"], str) or not task["chosen"]:
            raise ValueError("empty chosen answer")
        records["sft", split].append({"messages": messages + [
            {"role": "assistant", "content": task["chosen"]}]})
        rejected = task.get("rejected")
        if rejected:
            if rejected == task["chosen"]:
                raise ValueError("preference pair cannot be identical")
            records["dpo", split].append({"messages": messages + [
                {"role": "assistant", "content": task["chosen"]}],
                "rejected_response": rejected})
        records["opd", split].append({"messages": messages})
        manifest.append({"id": task["id"], "source_group": group, "split": split})
    for (kind, split), rows in records.items():
        write_jsonl(output / f"{kind}.{split}.jsonl", rows)
    write_jsonl(output / "manifest.jsonl", manifest)
    return {f"{kind}.{split}": len(rows) for (kind, split), rows in records.items()}
