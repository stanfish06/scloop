# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from pydantic import model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from .types import Diameter_t, Index_t


@dataclass
class LoopClass:
    rank: Index_t
    birth: Diameter_t = 0.0
    death: Diameter_t = 0.0
    cocycles: list | None = None
    representatives: list[list[Index_t]] | None = None

    @model_validator(mode="after")
    def check_birth_death(self) -> Self:
        if self.birth > self.death:
            raise ValueError("loop dies before its birth")
        return self

    @property
    def lifetime(self):
        return self.death - self.birth
