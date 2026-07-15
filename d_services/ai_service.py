"""
AI Service — Manages Agent A (Advocate) and Agent B (Challenger).

Each agent has a persona injected via system prompt:
- Agent A: Scientific, logical, evidence-based (Zenvyro Advocate style)
- Agent B: Philosophical, critical, challenges assumptions (Challenger style)

Memory is passed into every prompt so agents reference each other's
prior arguments rather than generating isolated statements.
"""
from .llm import LLMService
from .memory_manager import MemoryManager


ADVOCATE_PERSONA = """You are Agent A — The Advocate. You defend and support the debate topic.
Style: scientific, data-driven, logical. Use statistics, studies, and evidence.
Rules: reference the previous conversation directly. Be concise (3-5 sentences).
Never repeat what was already said — advance the argument with new evidence."""

CHALLENGER_PERSONA = """You are Agent B — The Challenger. You attack and challenge the topic.
Style: philosophical, critical, skeptical. Challenge assumptions and use counter-examples.
Rules: directly rebut Agent A's most recent argument. Ask rhetorical questions.
Be concise (3-5 sentences). Use thought experiments and logical deconstruction."""


class DebateConductor:
    """Orchestrates both debate agents with shared memory."""

    def __init__(self, model: str = "llama3.2"):
        self.llm = LLMService(model=model)
        self.memory = MemoryManager()

    def _prompt(self, persona: str, topic: str, name: str) -> str:
        context = self.memory.get_context(last_n=6)
        return (
            f"{persona}\n\n"
            f"Debate Topic: {topic}\n\n"
            f"Recent Exchange:\n{context}\n\n"
            f"Your response as {name} (3-5 sentences, stay in character):"
        )

    def generate_agent_a_response(self, topic: str) -> str:
        """Agent A defends the topic using logic and evidence."""
        response = self.llm.generate(
            self._prompt(ADVOCATE_PERSONA, topic, "Agent A (The Advocate)"),
            max_tokens=250
        )
        self.memory.add_turn("Agent A", response)
        return response

    def generate_agent_b_response(self, topic: str) -> str:
        """Agent B challenges the topic with philosophical critique."""
        response = self.llm.generate(
            self._prompt(CHALLENGER_PERSONA, topic, "Agent B (The Challenger)"),
            max_tokens=250
        )
        self.memory.add_turn("Agent B", response)
        return response

    def is_ready(self) -> bool:
        return self.llm.is_available()

    def reset(self):
        self.memory.reset()
