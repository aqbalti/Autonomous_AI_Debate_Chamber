"""
Memory Manager — stores every debate turn chronologically.

Why memory matters: without it, each agent generates a disconnected
monologue. With it, each prompt includes the last N turns so agents
can rebut, reference, and build on prior arguments — making the debate
coherent and realistic.
"""
from datetime import datetime


class MemoryManager:
    """Maintains the full debate transcript and formats it for LLM prompts."""

    def __init__(self):
        self.turns: list[dict] = []

    def add_turn(self, agent: str, text: str):
        self.turns.append({
            "agent": agent,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "round": len(self.turns) + 1
        })

    def get_context(self, last_n: int = 6) -> str:
        """Return last N turns as formatted text for prompt injection."""
        recent = self.turns[-last_n:]
        if not recent:
            return "This is the opening statement — no prior exchanges yet."
        return "\n\n".join(f"{t['agent']}: {t['text']}" for t in recent)

    def get_all(self) -> list[dict]:
        return self.turns

    def count(self) -> int:
        return len(self.turns)

    def total_words(self) -> int:
        return sum(len(t["text"].split()) for t in self.turns)

    def reset(self):
        self.turns = []
