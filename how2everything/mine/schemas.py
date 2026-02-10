from __future__ import annotations

from typing import Literal, List

from pydantic import BaseModel, Field


class ProcessExtraction(BaseModel):
    has_valid_process: bool
    goal: str
    steps: List[str]


class FilterJudgment(BaseModel):
    judgment: bool
    reason: str = ""


class PostprocessResult(BaseModel):
    rewritten_goal: str
    rewritten_steps: List[str]


class ToolsResult(BaseModel):
    resources: List[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    answer: Literal["yes", "no"]
    reason: str = ""


class FinalFilterResult(BaseModel):
    correctness: VerificationResult
    sequential: VerificationResult
    no_specific_entity: VerificationResult
    goal_steps_alignment: VerificationResult

