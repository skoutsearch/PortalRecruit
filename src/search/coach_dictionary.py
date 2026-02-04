"""Coach-speak dictionary → intent mapping.

Buckets map phrases to trait boosts and required tags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class IntentBoost:
    traits: Dict[str, int] = field(default_factory=dict)  # trait -> min boost
    tags: Set[str] = field(default_factory=set)
    exclude_tags: Set[str] = field(default_factory=set)


@dataclass
class IntentHit:
    intent: IntentBoost
    weight: float = 1.0
    role_hints: Set[str] = field(default_factory=set)  # guard/wing/big


# Canonical intents (extensible)
INTENTS: Dict[str, IntentBoost] = {
    "paint_presence_offense": IntentBoost(
        traits={"rim": 30},
        tags={"drive", "rim_pressure"},
    ),
    "paint_presence_defense": IntentBoost(
        traits={"menace": 25, "tough": 20},
        tags={"block", "rim_protection"},
    ),
    "shooting_spacing": IntentBoost(
        traits={"shot": 30},
        tags={"jumpshot", "3pt"},
    ),
    "gravity_well": IntentBoost(
        traits={"gravity": 30},
        tags={"assist", "3pt", "pnr", "handoff"},
    ),
    "unselfish_connectivity": IntentBoost(
        traits={"unselfish": 30},
        tags={"assist"},
    ),
    "finishing": IntentBoost(
        traits={"rim": 25},
        tags={"rim_finish", "layup", "dunk", "rim_pressure", "drive", "made"},
    ),
    "defensive_menace": IntentBoost(
        traits={"menace": 35},
        tags={"steal", "block", "deflection"},
    ),
    "toughness_winning": IntentBoost(
        traits={"tough": 25, "dog": 20},
        tags={"loose_ball", "charge_taken"},
    ),
    "iq_feel": IntentBoost(
        traits={"unselfish": 20},
        tags=set(),
    ),
    "character_stability": IntentBoost(
        traits={},
        tags=set(),
    ),
    "leadership": IntentBoost(
        traits={"unselfish": 35, "tough": 25, "dog": 25, "menace": 20},
        tags={"assist", "deflection", "charge_taken", "loose_ball"},
        exclude_tags={"turnover", "non_possession", "ft"},
    ),
    "resilience": IntentBoost(
        traits={"tough": 25, "dog": 20},
        tags={"drive", "rim_pressure"},
        exclude_tags={"non_possession"},
    ),
    "defensive_big": IntentBoost(
        traits={"menace": 30, "tough": 25},
        tags={"block", "charge_taken", "loose_ball"},
        exclude_tags={"non_possession"},
    ),
    "clutch": IntentBoost(
        traits={"tough": 25, "dog": 20, "shot": 20},
        tags={"made", "score", "assist"},
        exclude_tags={"non_possession"},
    ),
    "undervalued": IntentBoost(
        traits={"unselfish": 25, "tough": 15, "menace": 10},
        tags={"assist", "deflection", "loose_ball"},
        exclude_tags={"non_possession"},
    ),
    "size_measurables": IntentBoost(
        traits={},
        tags=set(),
    ),
    "negative_filters": IntentBoost(
        traits={"unselfish": 20, "shot": 20},
        tags=set(),
        exclude_tags={"turnover", "non_possession"},
    ),
}


# Phrase dictionary (seeded from Jesse's list)
PHRASES: Dict[str, List[str]] = {
    "paint_presence_offense": [
        "paint touches",
        "downhill driver",
        "downhill guard",
        "rim pressure",
        "two feet in the paint",
        "collapses the defense",
        "draws two defenders",
        "finishes through contact",
        "lives at the line",
        "attacks closeouts",
        "first step explosion",
        "blow-by speed",
        "straight line driver",
        "force at the rim",
        "drive and kick",
        "body control",
        "absorbs contact",
        "slasher",
        "penetrator",
        "rim attacker",
        "north-south attacker",
        "turns the corner",
        "get in the lane",
        "touch the paint",
        "touch paint",
        "paint touch guy",
        "rim runner guard",
    ],
    "paint_presence_defense": [
        "keep people out",
        "rim protector",
        "anchor",
        "verticality",
        "walls up",
        "drop coverage big",
        "shot blocker",
        "paint enforcer",
        "rotator",
        "clean rebounder",
        "ends possessions",
        "alter shots",
        "post defense",
        "holds ground",
        "physical interior defender",
        "protects the basket",
        "low post wall",
        "glass cleaner",
        "box out discipline",
    ],
    "shooting_spacing": [
        "floor spacer",
        "knockdown shooter",
        "catch and shoot",
        "off the bounce shooting",
        "range extender",
        "shooting clip",
        "efficient volume",
        "zone buster",
        "clean mechanics",
        "quick release",
        "deep range",
        "movement shooter",
        "runs off screens",
        "shot hunter",
        "microwave scorer",
        "heat check",
        "corner specialist",
        "stretch big",
        "pick and pop",
        "three-level scorer",
        "perimeter threat",
        "elite stroke",
        "pure shooter",
        "automatic from deep",
        "spacing equity",
        "shot creator",
        "floor stretcher",
        "transition 3s",
        "pull-up jumper",
        "mid-range assassin",
        "bucket getter",
        "bucket-getter",
    ],
    "gravity_well": [
        "gravity well",
        "gravity",
        "magnet",
        "defensive bend",
        "bends the defense",
        "warps the defense",
        "draws two",
        "draws double",
        "double team magnet",
        "face-guarded",
        "denied the ball",
        "decoy",
        "off-ball gravity",
        "spacing threat",
        "pulls help",
        "help magnet",
        "creates space",
        "warps spacing",
        "defense shifts",
        "zone bender",
        "flare screen",
        "pin-down",
        "staggered screens",
        "ghost screen",
        "screen slip",
        "pull-up gravity",
        "gravity assist",
    ],
    "unselfish_connectivity": [
        "high assist-to-turnover ratio",
        "ball mover",
        "hockey assist",
        "extra pass",
        "willing passer",
        "connector",
        "glue guy",
        "facilitator",
        "processing speed",
        "reads the floor",
        "team-first mentality",
        "low usage efficiency",
        "keeps the ball popping",
        "play connector",
        "makes teammates better",
        "screens for others",
        "dimer",
        "vision",
        "spray passer",
        "unselfish",
        "share the rock",
        "flow offense",
        "chain mover",
        "role acceptance",
        "chemistry booster",
        "servant leader",
        "pass-first",
        "connective tissue",
        "moves it early",
        "0.5 second decision maker",
    ],
    "finishing": [
        "finisher",
        "finishing",
        "finish at the rim",
        "finishes",
        "rim finisher",
        "rim finisher",
        "rim runner",
        "lob threat",
        "plays above the rim",
        "dunks",
        "dunker",
        "paint finisher",
        "around the rim",
        "at the rim",
        "strong finisher",
    ],
    "defensive_menace": [
        "point of attack defender",
        "lockdown",
        "clamps",
        "on-ball pest",
        "disruptor",
        "passing lanes",
        "deflections",
        "steal percentage",
        "pick-pocket",
        "switchability",
        "blows up screens",
        "navigates screens",
        "shoots the gap",
        "ball pressure",
        "full court press",
        "takes charges",
        "active hands",
        "closeout speed",
        "isolation defender",
        "screen navigation",
        "takes away strong hand",
        "sits down on defense",
        "lateral quickness",
        "defensive stopper",
        "defensive menace",
        "3-and-d",
        "two-way",
        "two way",
        "dog on defense",
        "guard who can guard",
        "tone setter on defense",
        "sets the tone defensively",
        "heat check on defense",
    ],
    "toughness_winning": [
        "winner",
        "championship dna",
        "winning program",
        "state champion",
        "prep champion",
        "playoff experience",
        "clutch gene",
        "big shot maker",
        "mental toughness",
        "gritty",
        "dog",
        "dawg",
        "junkyard dog",
        "scrappy",
        "50/50 balls",
        "diving on the floor",
        "winning plays",
        "closer",
        "resilient",
        "handles adversity",
        "bounces back",
        "competitor",
        "hates to lose",
        "physical toughness",
        "enforcer",
        "battle tested",
        "grind it out",
        "dirty work",
        "chip on shoulder",
        "heart over height",
        "impact player",
        "energy bunny",
        "motor never stops",
    ],
    "iq_feel": [
        "basketball iq",
        "feel for the game",
        "cerebral player",
        "high awareness",
        "spatial awareness",
        "clock management",
        "situational basketball",
        "read and react",
        "processing ability",
        "visionary",
        "play anticipation",
        "manipulates defense",
        "smart cuts",
        "understanding spacing",
        "pace control",
        "change of speeds",
        "decision making",
        "low turnover",
        "court mapping",
        "instinctive",
        "second level thinking",
        "playbook",
        "tempo setter",
    ],
    "character_stability": [
        "culture setter",
        "locker room guy",
        "high character",
        "zero baggage",
        "academic standing",
        "coachable",
        "eye contact",
        "body language",
        "emotional stability",
        "steady hand",
        "even-keeled",
        "stoic",
        "vocal leader",
        "extension of the coach",
        "floor general",
        "accountability",
        "professional approach",
        "mature",
        "no red flags",
        "self-starter",
        "practice habits",
        "first in last out",
        "gym rat",
        "mentor",
        "commits to the program",
        "trustworthy",
        "reliable",
        "consistent effort",
        "positive reinforcement",
        "energy giver",
        "energy vampire",
        "command of the huddle",
    ],
    "leadership": [
        "leader",
        "team leader",
        "vocal leader",
        "floor general",
        "coach on the floor",
        "extension of the coach",
        "captain",
        "steady hand",
        "command of the huddle",
        "accountability",
        "organizes the team",
        "directs traffic",
        "tone setter",
        "lead by example",
        "go-to guy",
        "go to guy",
        "primary option",
        "alpha",
        "franchise",
        "star",
    ],
    "resilience": [
        "resilience",
        "resilient",
        "bounce back",
        "bounce-back",
        "overcomes adversity",
        "overcome adversity",
        "perseverance",
        "persevere",
        "adversity",
        "tough-minded",
        "short memory",
        "next play",
        "next-play",
    ],
    "defensive_big": [
        "rim protector",
        "lane clogger",
        "paint presence",
        "anchor",
        "shot blocker",
        "beast in the lane",
        "no-fly zone",
        "rim deterrent",
        "paint enforcer",
        "lane warden",
        "restricted area",
        "verticality",
        "wall at the rim",
        "paint defender",
    ],
    "clutch": [
        "clutch",
        "closer",
        "big shot",
        "go-to late",
        "late game",
        "game winner",
        "dagger",
        "fourth quarter",
        "4th quarter",
        "end of game",
        "crunch time",
        "pressure",
        "ice cold",
        "buzzer beater",
    ],
    "undervalued": [
        "undervalued",
        "overlooked",
        "hidden gem",
        "low maintenance",
        "high yield",
        "low usage",
        "glue guy",
        "dirty work",
        "impact without scoring",
        "role player",
        "connector",
        "unsung",
        "under the radar",
    ],
    "size_measurables": [
        "positional size",
        "length",
        "wingspan",
        "standing reach",
        "frame",
        "physical profile",
        "big guard",
        "switchable size",
        "nba body",
        "athletic tools",
        "measurables",
        "vertical pop",
        "wide shoulders",
        "physically imposing",
        "overpowering",
        "mismatches",
        "post-up guard",
        "see over defense",
        "recoverable length",
        "catch radius",
        "stride length",
        "functional strength",
        "prototypical size",
        "big body",
        "long levered",
        "big wing",
        "big body guard",
        "long athlete",
        "plus length",
    ],
    "negative_filters": [
        "empty calories",
        "volume scorer",
        "black hole",
        "ball stopper",
        "energy vampire",
        "me first",
        "low iq",
        "bad shot",
        "wild",
        "takes bad shots",
        "stat chaser",
        "no feel",
        "low motor",
    ],
}


ROLE_PHRASES: Dict[str, List[str]] = {
    "guard": ["pg", "point guard", "lead guard", "floor general"],
    "wing": ["wing", "3-and-d", "two-way wing"],
    "big": ["big", "rim protector", "anchor", "post player", "center"],
}

WEIGHTED_PHRASES: Dict[str, List[Tuple[str, float]]] = {
    # core phrases (1.0) + support phrases (0.6)
    "paint_presence_offense": [(p, 1.0) for p in PHRASES["paint_presence_offense"]]
    + [("downhill", 0.6), ("paint touch", 0.6)],
    "paint_presence_defense": [(p, 1.0) for p in PHRASES["paint_presence_defense"]]
    + [("rim deterrent", 0.6), ("paint patrol", 0.6)],
    "shooting_spacing": [(p, 1.0) for p in PHRASES["shooting_spacing"]]
    + [("deep shooter", 0.6), ("sniper", 0.6), ("laser", 0.6)],
    "gravity_well": [(p, 1.0) for p in PHRASES["gravity_well"]]
    + [("magnet", 1.0), ("gravity", 1.0), ("defense bends", 0.6)],
    "unselfish_connectivity": [(p, 1.0) for p in PHRASES["unselfish_connectivity"]]
    + [("quick decision", 0.6), ("hit ahead", 0.6)],
    "finishing": [(p, 1.0) for p in PHRASES["finishing"]]
    + [("finish", 0.8), ("rim finish", 0.8), ("paint finisher", 0.8)],
    "defensive_menace": [(p, 1.0) for p in PHRASES["defensive_menace"]]
    + [("heat", 0.6), ("pest", 0.6)],
    "toughness_winning": [(p, 1.0) for p in PHRASES["toughness_winning"]]
    + [("junkyard", 0.6), ("grinder", 0.6)],
    "iq_feel": [(p, 1.0) for p in PHRASES["iq_feel"]]
    + [("processor", 0.6), ("connector iq", 0.6)],
    "character_stability": [(p, 1.0) for p in PHRASES["character_stability"]],
    "leadership": [(p, 1.0) for p in PHRASES["leadership"]]
    + [("leader", 1.0), ("vocal", 0.6), ("captain", 0.6)],
    "resilience": [(p, 1.0) for p in PHRASES["resilience"]]
    + [("bounce back", 0.8), ("next play", 0.8)],
    "defensive_big": [(p, 1.0) for p in PHRASES["defensive_big"]]
    + [("rim protector", 1.0), ("lane clogger", 0.8)],
    "clutch": [(p, 1.0) for p in PHRASES["clutch"]]
    + [("clutch", 1.0), ("dagger", 0.8), ("game winner", 0.8)],
    "undervalued": [(p, 1.0) for p in PHRASES["undervalued"]]
    + [("hidden gem", 1.0), ("overlooked", 0.8)],
    "size_measurables": [(p, 1.0) for p in PHRASES["size_measurables"]]
    + [("long", 0.6), ("big framed", 0.6)],
    "negative_filters": [(p, 1.0) for p in PHRASES["negative_filters"]],
}


def infer_intents(query: str) -> Dict[str, IntentHit]:
    q = (query or "").lower()
    hits: Dict[str, IntentHit] = {}

    # role hints
    role_hints: Set[str] = set()
    for role, phrases in ROLE_PHRASES.items():
        if any(p in q for p in phrases):
            role_hints.add(role)

    for bucket, phrases in WEIGHTED_PHRASES.items():
        for p, w in phrases:
            if p in q:
                hit = IntentHit(intent=INTENTS[bucket], weight=w, role_hints=role_hints)
                hits[bucket] = hit
                break

    return hits


def infer_intents_verbose(query: str) -> Dict[str, tuple[IntentHit, str]]:
    """Return intent hits with the matched phrase for explainability."""
    q = (query or "").lower()
    hits: Dict[str, tuple[IntentHit, str]] = {}

    role_hints: Set[str] = set()
    for role, phrases in ROLE_PHRASES.items():
        if any(p in q for p in phrases):
            role_hints.add(role)

    for bucket, phrases in WEIGHTED_PHRASES.items():
        for p, w in phrases:
            if p in q:
                hit = IntentHit(intent=INTENTS[bucket], weight=w, role_hints=role_hints)
                hits[bucket] = (hit, p)
                break
    return hits
