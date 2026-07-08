"""LLM abstraction layer: interface, mock, and Ollama backend.

This file defines the `LLMClient` interface and two implementations. All LLM calls in the
pipeline go through this interface, so the backend can be swapped without changing anything
else. This is important for running PRISM in environments where a real model is not available
(e.g. testing, GEL genomics environment).

`LLMClient` (abstract) — two methods:
  - `resolve_match()` — given a patient term and a disease feature, classify their
    relationship as matched / partial / absent_expected / unexplained / contradiction.
    Used in C1 (broader-term case) and C4 (silent discriminator inference).
  - `synthesise_narrative()` — given a disease name and fit summary, write a 2–3 sentence
    plain-English clinical narrative. Purely cosmetic — does not affect ranking.

`MockLLMClient` — deterministic, no model required.
  - resolve_match: exact HPO ID match → "matched", anything else → "partial".
  - synthesise_narrative: returns a bracketed placeholder string.
  Used for all tests and for runs where --llm=mock (the default).

`OllamaLLMClient` — calls a locally-running Ollama instance via HTTP.
  - Checks at construction that Ollama is reachable and the model is available.
  - Prompts are plain text; the response is parsed for the first valid label word.
  - Use with: prism run ... --llm ollama --llm-model qwen2.5:7b
"""
import sys
from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel


MatchLabel = Literal["matched", "partial", "absent_expected", "unexplained", "contradiction"]


class LLMMatch(BaseModel):
    """Structured judgement returned by the LLM for one patient-term / disease-feature pair.

    The deterministic scorer (C2) consumes these labels — same labels in,
    same score out, always. The LLM never emits a rank-affecting number.
    """
    patient_term_id: str
    disease_feature_id: str
    label: MatchLabel
    provenance: str  # verbatim retrieved KB sentence grounding this judgement


class LLMClient(ABC):
    """Abstract LLM backend — swap between mock, local weights, or approved API
    without changing any caller code (§2: GEL environment constraint)."""

    @abstractmethod
    def resolve_match(
        self,
        patient_term_id: str,
        patient_term_label: str,
        disease_feature_id: str,
        disease_feature_label: str,
        context: str,
    ) -> LLMMatch:
        """Resolve a single patient-term / disease-feature pair into a labelled match."""
        ...

    @abstractmethod
    def synthesise_narrative(
        self,
        disease_name: str,
        fit_summary: dict,
    ) -> str:
        """Produce a human-readable fit narrative for a disease candidate."""
        ...


class MockLLMClient(LLMClient):
    """Deterministic mock for Phase 0–2 development.

    Exact HPO ID match → 'matched'. Anything else → 'partial'.
    Enables full pipeline runs and exact test assertions without a live model.
    """

    def resolve_match(
        self,
        patient_term_id: str,
        patient_term_label: str,
        disease_feature_id: str,
        disease_feature_label: str,
        context: str,
    ) -> LLMMatch:
        label: MatchLabel = (
            "matched" if patient_term_id == disease_feature_id else "partial"
        )
        return LLMMatch(
            patient_term_id=patient_term_id,
            disease_feature_id=disease_feature_id,
            label=label,
            provenance=f"[MOCK] {context[:120] if context else 'no context provided'}",
        )

    def synthesise_narrative(self, disease_name: str, fit_summary: dict) -> str:
        return f"[MOCK narrative for {disease_name}] {fit_summary}"


class OllamaLLMClient(LLMClient):
    """LLM backend using a locally-running Ollama instance (e.g. qwen2.5:7b).

    No third-party packages required — uses stdlib urllib only.

    Usage::

        llm = OllamaLLMClient(model="qwen2.5:7b")
        # or via CLI: prism ... --llm ollama --llm-model qwen3:8b
    """

    _ORDERED_LABELS: tuple[str, ...] = (
        "matched", "partial", "contradiction", "absent_expected", "unexplained"
    )

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self._call_count = 0
        self._check_connection()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def resolve_match(
        self,
        patient_term_id: str,
        patient_term_label: str,
        disease_feature_id: str,
        disease_feature_label: str,
        context: str,
    ) -> LLMMatch:
        self._call_count += 1
        if self.verbose:
            print(
                f"[Ollama #{self._call_count}] resolve_match: "
                f"{patient_term_label} ({patient_term_id}) ↔ "
                f"{disease_feature_label} ({disease_feature_id})",
                file=sys.stderr,
            )
        prompt = (
            "You are a clinical genomics assistant.\n"
            "Classify the relationship between a patient HPO term and a disease feature.\n\n"
            f"Patient term  : {patient_term_label} ({patient_term_id})\n"
            f"Disease feature: {disease_feature_label} ({disease_feature_id})\n"
            f"Context: {context}\n\n"
            "Choose exactly ONE label from the list below and output it alone on a line:\n"
            "  matched          — clinically equivalent\n"
            "  partial          — related but not identical\n"
            "  absent_expected  — disease feature typically present but absent here\n"
            "  unexplained      — patient term has no counterpart in this disease\n"
            "  contradiction    — disease lists this feature as ABSENT but patient has it\n\n"
            "Label:"
        )
        raw = self._generate(prompt)
        label = self._extract_label(raw)
        if self.verbose:
            print(f"  → {label}  (raw: {raw[:80]!r})", file=sys.stderr)
        return LLMMatch(
            patient_term_id=patient_term_id,
            disease_feature_id=disease_feature_id,
            label=label,
            provenance=f"[Ollama/{self.model}] {raw[:200]}",
        )

    def synthesise_narrative(self, disease_name: str, fit_summary: dict) -> str:
        import json as _json
        self._call_count += 1
        if self.verbose:
            print(
                f"[Ollama #{self._call_count}] synthesise_narrative: {disease_name}",
                file=sys.stderr,
            )
        prompt = (
            "You are a clinical genomics assistant.\n"
            "Write a 2–3 sentence clinical narrative explaining why this disease "
            "is (or is not) a strong match for the patient. "
            "Plain English, no markdown, no bullet points.\n\n"
            f"Disease: {disease_name}\n"
            f"Evidence:\n{_json.dumps(fit_summary, indent=2)}\n\n"
            "Narrative:"
        )
        result = self._generate(prompt)
        if self.verbose:
            print(f"  → {result[:120]!r}", file=sys.stderr)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_connection(self) -> None:
        """Verify Ollama is reachable and the requested model is available."""
        import json as _json
        import urllib.request
        import urllib.error
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=5
            ) as resp:
                body = _json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url} — is it running?\n"
                f"  Start it with: ollama serve\n"
                f"  Error: {exc}"
            ) from exc

        available = [m["name"] for m in body.get("models", [])]
        base_name = self.model.split(":")[0]
        exact_match = self.model in available
        prefix_match = any(m.startswith(base_name) for m in available)

        if not exact_match and not prefix_match:
            names = "\n    ".join(available) if available else "(none pulled yet)"
            raise RuntimeError(
                f"Model '{self.model}' not found in Ollama.\n"
                f"  Available models:\n    {names}\n"
                f"  Pull it with: ollama pull {self.model}\n"
                f"  Or pass one of the available names with --llm-model"
            )

        if not exact_match and prefix_match:
            # Found a match by prefix — tell the user the exact name to use
            exact = next(m for m in available if m.startswith(base_name))
            print(
                f"[PRISM] Model '{self.model}' matched as '{exact}'. "
                f"Use --llm-model {exact} to be explicit.",
                file=sys.stderr,
            )
            self.model = exact  # use the name Ollama actually knows

    def _generate(self, prompt: str) -> str:
        """POST to /api/generate and return the response text."""
        import json as _json
        import urllib.request
        import urllib.error
        payload = _json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = _json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    f"Ollama returned 404 for model '{self.model}'. "
                    f"Run 'ollama list' to see pulled models, "
                    f"or 'ollama pull {self.model}' to fetch it."
                ) from exc
            raise RuntimeError(
                f"Ollama request failed (HTTP {exc.code}): {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Lost connection to Ollama at {self.base_url}: {exc}"
            ) from exc
        return body.get("response", "").strip()

    def _extract_label(self, text: str) -> MatchLabel:
        """Find the first valid label word in the model's response."""
        lower = text.lower()
        for label in self._ORDERED_LABELS:
            if label in lower:
                return label  # type: ignore[return-value]
        return "partial"  # safe fallback