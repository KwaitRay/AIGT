from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Literal


def instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must contain a UTC offset")
    return parsed


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str
    created_at: str

    def __post_init__(self):
        if self.role not in ("user", "assistant"):
            raise ValueError("history accepts user/assistant messages only")
        instant(self.created_at)


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str
    published_at: str
    city: str | None = None
    valid_on: str | None = None
    topic: str = "knowledge"

    def __post_init__(self):
        instant(self.published_at)
        if self.valid_on:
            date.fromisoformat(self.valid_on)
        if not self.id or not self.source:
            raise ValueError("document requires id and source")

    def visible(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Query:
    text: str
    as_of: str
    city: str | None = None
    target_date: str | None = None
    topic: str | None = None

    def __post_init__(self):
        instant(self.as_of)
        if self.target_date:
            date.fromisoformat(self.target_date)

    @property
    def search_text(self) -> str:
        return " ".join(x for x in (self.text, self.city, self.target_date, self.topic) if x)
