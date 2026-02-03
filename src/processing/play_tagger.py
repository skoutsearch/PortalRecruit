def tag_play(description: str, clock_seconds: int = None) -> list[str]:
    """
    Analyzes a play description and game clock to assign tactical tags.
    """
    tags = set()
    desc = (description or "").lower()

    # --- OFFENSIVE ACTIONS ---
    if "screen" in desc or "pick" in desc or "p&r" in desc:
        tags.add("pnr")  # Pick and Roll
    if "isolation" in desc or "iso" in desc:
        tags.add("iso")
    if "handoff" in desc or "dho" in desc:
        tags.add("handoff")
    if "post" in desc:
        tags.add("post_up")
    if "drive" in desc or "to basket" in desc or "to the basket" in desc:
        tags.add("drive")
    if "cut" in desc:
        tags.add("cut")
    if "at basket" in desc or "at the basket" in desc or "rim" in desc:
        tags.add("rim_pressure")
    if "pull up" in desc or "pull-up" in desc:
        tags.add("pull_up")

    # --- SHOT TYPES ---
    if "3pt" in desc or "3-pt" in desc or "three" in desc or "make 3" in desc or "miss 3" in desc:
        tags.add("3pt")
        tags.add("jumpshot")
    elif "dunk" in desc:
        tags.add("dunk")
        tags.add("rim_finish")
    elif "layup" in desc:
        tags.add("layup")
        tags.add("rim_finish")
    elif "jumper" in desc or "jump shot" in desc:
        tags.add("jumpshot")

    # --- OUTCOMES ---
    if "made" in desc or "make" in desc:
        tags.add("made")
        tags.add("score")
    elif "missed" in desc or "miss" in desc:
        tags.add("missed")

    # Synergy taxonomy outcomes
    if "make 2 pts" in desc or "make 3 pts" in desc:
        tags.add("made")
        tags.add("score")
    if "miss 2 pts" in desc or "miss 3 pts" in desc:
        tags.add("missed")

    if (
        "assist" in desc
        or " ast " in desc
        or "assisted" in desc
        or "ball delivered" in desc
        or "kick out" in desc
        or "drive & kick" in desc
        or "drive and kick" in desc
        or "pass leads to" in desc
    ):
        tags.add("assist")

    if "free throw" in desc or "ft" in desc:
        tags.add("ft")

    if "turnover" in desc:
        tags.add("turnover")
        if "steal" in desc:
            tags.add("live_ball_turnover")
            tags.add("steal")
            
    if "rebound" in desc:
        tags.add("rebound")
        if "offensive" in desc:
            tags.add("oreb")
        else:
            tags.add("dreb")
            
    if "foul" in desc:
        tags.add("foul")
        if "charge" in desc:
            tags.add("charge_taken")

    # --- DEFENSIVE EVENTS ---
    if "block" in desc:
        tags.add("block")
        tags.add("rim_protection")
    if "deflection" in desc:
        tags.add("deflection")
    if "loose ball" in desc:
        tags.add("loose_ball")

    # --- TRANSITION / CONTEXT ---
    if "fast break" in desc or "transition" in desc:
        tags.add("transition")
    
    # --- CLOCK SITUATIONS ---
    # Assuming clock_seconds is seconds remaining in the period
    if clock_seconds is not None:
        if 0 < clock_seconds <= 5:
            tags.add("late_clock")
        if 0 < clock_seconds <= 2:
            tags.add("buzzer_beater_scenario")

    return sorted(list(tags))
