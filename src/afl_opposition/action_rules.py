"""Explicit annotation rules for the first ball-control sequence dataset.

These are project conventions, not the provider's complete possession rules.
Unknown descriptions remain visible with role ``review``.
"""

RULE_VERSION = "0.1.0"

# Only these observations enter the model event stream. Subtypes are preserved
# in description; action_type provides a smaller, explicit vocabulary.
ACTION_TYPES = {
    "Kick": "kick",
    "Ground Kick": "ground_kick",
    "Handball": "handball",
    "Bounce": "bounce",
    "Handball Received": "receive",
    "Contested Mark": "mark",
    "Uncontested Mark": "mark",
    "Mark On Lead": "mark",
    "Gather": "gather",
    "Gather From Hitout": "gather",
    "Gather from Opposition": "gather",
    "Hard Ball Get": "hard_ball_get",
    "Hard Ball Get Crumb": "hard_ball_get",
    "Loose Ball Get": "loose_ball_get",
    "Loose Ball Get Crumb": "loose_ball_get",
    "Ruck Hard Ball Get": "hard_ball_get",
    "Free For": "free_kick",
    "Free For: In Possession": "free_kick",
    "Free For: Off The Ball": "free_kick",
    "Free For: Before the Bounce": "free_kick",
    "Free Advantage": "free_kick",
    "OOF Kick In": "free_kick",
    "Kickin play on": "kick_in_play_on",
    "Kickin short": "kick_in",
    "Kickin long": "kick_in",
}

STOPPAGES = {
    "Ball Up Call", "Out of Bounds", "Goal", "Behind", "Rushed",
    "Out On Full", "Out On Full After Kick",
}
RESTARTS = {"Centre Bounce"}
CONTEXT = {
    "Spoil", "Mark Fumbled", "Mark Dropped", "Dropped Mark", "Tackle",
    "Tackled By", "Missed Tackle", "Smothered", "Smother After Disposal",
    "Contested Knock On", "Knock On", "Hitout", "Shepherd",
}
ANNOTATIONS = {
    "Kick Inside 50 Result", "Contest Target", "Shot At Goal",
    "Inside 50", "Rebound 50", "Clearance", "Clearance (Operator)",
    "Credit", "Debit", "Pressure Credit (O)",
}
# Rare or insufficiently understood descriptions are deliberately quarantined.
REVIEW = {
    "50m Penalty For", "Baulked", "Break Operator", "Broken Tackle",
    "CB Hitout To Advantage", "Dispossessed", "Effective Knock On",
    "First Possession", "Hold", "Kick In Clanger", "Kick In Ineffective",
    "Kick In Long To Adv.", "Kick Long To Adv.", "No Pressure Error",
    "Pack Mark (O)", "Pack Mark (P)", "Play On",
    "Run Down Tackle On Disposal", "Spoil gaining possession",
    "Spoil ineffective",
}


def action_role(description: str) -> str:
    """Return a declared role; never silently treat an unknown as an action."""
    if description in ACTION_TYPES:
        return "action"
    if description == "Kick Into F50":
        return "kick_annotation"
    for role, names in (
        ("stoppage", STOPPAGES), ("restart", RESTARTS),
        ("context", CONTEXT), ("annotation", ANNOTATIONS),
    ):
        if description in names:
            return role
    return "review"


KNOWN_DESCRIPTIONS = (
    set(ACTION_TYPES) | STOPPAGES | RESTARTS | CONTEXT | ANNOTATIONS | REVIEW
    | {"Kick Into F50"}
)
