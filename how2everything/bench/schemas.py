from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

#
# Canonical schemas for the release `how2bench` pipeline.
#

SCHEMA_VERSION_EXAMPLE = "how2bench.example.v1"
SCHEMA_VERSION_GENERATION = "how2bench.generation.v1"
SCHEMA_VERSION_JUDGMENT = "how2bench.judgment.v1"


class BenchExample(BaseModel):
    """
    Canonical example schema used by how2bench.

    This is the internal contract that adapters should produce regardless of
    upstream source (how2mine export vs legacy JSONL).
    """

    schema_version: str = Field(default=SCHEMA_VERSION_EXAMPLE)
    source_example_id: str
    url: str = ""
    source_text: str = ""
    topic: str = ""
    goal: str
    resources: List[str]
    steps: List[str]


class GeneratedSteps(BaseModel):
    """Structured output for generation."""

    steps: List[str] = Field(default_factory=list)


class CriticalFailure(BaseModel):
    failure: str
    L1_steps: List[int] = Field(default_factory=list)
    L2_steps: List[int] = Field(default_factory=list)


class JudgeResult(BaseModel):
    """
    Structured output for the judge prompt in `prompts/judge.txt`.
    """

    reasoning: str = ""
    critical_failures: List[CriticalFailure] = Field(default_factory=list)


class LLMInfo(BaseModel):
    backend: str = "deluge"  # deluge|vllm
    provider: str = ""  # informational
    model: str = ""


class GenerationRecord(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION_GENERATION)
    generator_id: str = ""
    generation_prompt_sha256: str = ""
    source_example_id: str
    topic: str = ""
    goal: str
    steps: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)
    model_completion: str = ""
    predicted_steps: List[str] = Field(default_factory=list)
    prompt: str = ""
    generator: LLMInfo = Field(default_factory=LLMInfo)


class JudgmentRecord(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION_JUDGMENT)
    judge_id: str = ""
    judge_prompt_sha256: str = ""
    source_example_id: str
    topic: str = ""
    goal: str
    steps: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)
    predicted_steps: List[str] = Field(default_factory=list)
    judge: LLMInfo = Field(default_factory=LLMInfo)
    reasoning: str = ""
    critical_failures: List[CriticalFailure] = Field(default_factory=list)

    # Convenience fields for downstream aggregation.
    has_failure: bool = False
    n_failures: int = 0
    parse_failed: bool = False

