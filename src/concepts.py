from __future__ import annotations

from typing import List

CONCEPT_DEFINITIONS = {
    "SHOOTING": "elite three point shooter deep range high percentage catch and shoot",
    "PLAYMAKING": "elite passer high assist turnover ratio court vision floor general",
    "DEFENSE": "lockdown defender steals blocks active hands stops the ball",
    "REBOUNDING": "relentless rebounder crashes glass boxes out high motor",
    "ATHLETICISM": "elite vertical leap explosive speed quick first step dunker",
}


def get_active_concepts(selected_tags: List[str]) -> List[str]:
    return [CONCEPT_DEFINITIONS[t] for t in selected_tags if t in CONCEPT_DEFINITIONS]
