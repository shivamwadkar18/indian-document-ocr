"""Fictional identity data for synthetic licences.

TODO(owner): decide the source of fictional names/addresses (curated word lists
vs. a faker-style library) and how licence numbers are made clearly non-real
(e.g. a reserved state-code prefix such as "XX" that no RTO issues).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass
class FictionalIdentity:
    name: str
    father_or_spouse_name: str
    address: str
    date_of_birth: date
    issue_date: date
    valid_until: date
    licence_number: str
    vehicle_categories: list[str]
    blood_group: str | None = None


class IdentityGenerator(ABC):
    @abstractmethod
    def generate(self, rng: np.random.Generator) -> FictionalIdentity:
        """Return a new, entirely fictional identity."""
