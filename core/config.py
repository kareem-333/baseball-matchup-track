"""
core/config.py — Global constants, lookup tables, and team colours.

No external imports beyond the standard library.  Every other module
in the project should pull shared constants from here rather than
defining them locally.
"""

# ── MLB ───────────────────────────────────────────────────────────────────────

ASTROS_ID = 117

PITCH_NAMES = {
    "FF": "4-Seam FB", "SI": "Sinker",   "FC": "Cutter",
    "SL": "Slider",    "ST": "Sweeper",  "CH": "Changeup",
    "CU": "Curveball", "KC": "K-Curve",  "FS": "Splitter",
    "SV": "Slurve",    "KN": "Knuckler", "CS": "Slow Curve",
    "FO": "Forkball",  "EP": "Eephus",   "SC": "Screwball",
}

PITCH_COLORS = {
    "FF": "#e41a1c", "SI": "#ff7f00", "FC": "#984ea3",
    "SL": "#377eb8", "ST": "#4daf4a", "CH": "#f781bf",
    "CU": "#a65628", "KC": "#999999", "FS": "#ffff33",
    "SV": "#17becf", "KN": "#bcbd22", "CS": "#7f7f7f",
}

# Notable batters — {display name: MLB player ID}
STAR_BATTERS: dict[str, int] = {
    "Shohei Ohtani":          660271,
    "Aaron Judge":            592450,
    "Bobby Witt Jr.":         677951,
    "Yordan Alvarez":         670541,
    "Juan Soto":              665742,
    "Fernando Tatis Jr.":     665487,
    "Mookie Betts":           605141,
    "Freddie Freeman":        518692,
    "Gunnar Henderson":       683002,
    "Ronald Acuña Jr.":       660670,
    "Pete Alonso":            624413,
    "Jose Ramirez":           608070,
    "Corey Seager":           608369,
    "Rafael Devers":          646240,
    "Matt Olson":             621566,
    "Kyle Tucker":            663656,
    "Vladimir Guerrero Jr.":  665489,
    "Jose Altuve":            514888,
    "Carlos Correa":          621043,
    "Isaac Paredes":          670623,
    "Christian Walker":       572233,
}

# Strike zone boundaries (feet, catcher's view)
SZ_L, SZ_R = -0.831,  0.831
SZ_B, SZ_T =  1.50,   3.50

_cw = (SZ_R - SZ_L) / 3
_ch = (SZ_T - SZ_B) / 3

# Zone 1-9 → (x_left, x_right, z_bot, z_top)
ZONE_BOXES: dict[int, tuple] = {}
ZONE_CENTERS: dict[int, tuple] = {}
_ZONE_LAYOUT = {
    1: (0, 2), 2: (0, 1), 3: (0, 0),
    4: (1, 2), 5: (1, 1), 6: (1, 0),
    7: (2, 2), 8: (2, 1), 9: (2, 0),
}
for _z, (_r, _c) in _ZONE_LAYOUT.items():
    _xl = SZ_L + _c * _cw
    _xr = SZ_L + (_c + 1) * _cw
    _zb = SZ_T - (_r + 1) * _ch
    _zt = SZ_T - _r * _ch
    ZONE_BOXES[_z]   = (_xl, _xr, _zb, _zt)
    ZONE_CENTERS[_z] = ((_xl + _xr) / 2, (_zb + _zt) / 2)

# ── MLB team hex colours (primary / accent) ───────────────────────────────────
MLB_TEAM_COLORS: dict[int, dict] = {
    108: {"primary": "#BA0021", "accent": "#FFFFFF"},   # Angels
    109: {"primary": "#A71930", "accent": "#E3D4AD"},   # Diamondbacks
    110: {"primary": "#DF4601", "accent": "#000000"},   # Orioles
    111: {"primary": "#BD3039", "accent": "#0C2340"},   # Red Sox
    112: {"primary": "#0E3386", "accent": "#CC3433"},   # Cubs
    113: {"primary": "#C6011F", "accent": "#000000"},   # Reds
    114: {"primary": "#00385D", "accent": "#E50022"},   # Guardians
    115: {"primary": "#333366", "accent": "#C4CED4"},   # Rockies
    116: {"primary": "#002D62", "accent": "#EB6E1F"},   # Astros (away)
    117: {"primary": "#002D62", "accent": "#EB6E1F"},   # Astros
    118: {"primary": "#004687", "accent": "#C09A5B"},   # Royals
    119: {"primary": "#005A9C", "accent": "#EF3E42"},   # Dodgers
    120: {"primary": "#AB0003", "accent": "#14225A"},   # Nationals
    121: {"primary": "#002D72", "accent": "#FF5910"},   # Mets
    133: {"primary": "#003831", "accent": "#EFB21E"},   # Athletics
    134: {"primary": "#FDB827", "accent": "#27251F"},   # Pirates
    135: {"primary": "#2F241D", "accent": "#FFC425"},   # Padres
    136: {"primary": "#0C2C56", "accent": "#005C5C"},   # Mariners
    137: {"primary": "#FD5A1E", "accent": "#000000"},   # Giants
    138: {"primary": "#C41E3A", "accent": "#0C2340"},   # Cardinals
    139: {"primary": "#092C5C", "accent": "#8FBCE6"},   # Rays
    140: {"primary": "#003278", "accent": "#C0111F"},   # Rangers
    141: {"primary": "#134A8E", "accent": "#E8291C"},   # Blue Jays
    142: {"primary": "#002B5C", "accent": "#D31145"},   # Twins
    143: {"primary": "#E81828", "accent": "#002D72"},   # Phillies
    144: {"primary": "#CE1141", "accent": "#13274F"},   # Braves
    145: {"primary": "#27251F", "accent": "#C4CED4"},   # White Sox
    146: {"primary": "#00A3E0", "accent": "#FF6600"},   # Marlins
    147: {"primary": "#003087", "accent": "#E4002C"},   # Yankees
    158: {"primary": "#12284B", "accent": "#FFC52F"},   # Brewers
}

# ── NBA team hex colours ──────────────────────────────────────────────────────
NBA_TEAM_COLORS: dict[str, dict] = {
    "ATL": {"primary": "#E03A3E", "accent": "#C1D32F"},
    "BOS": {"primary": "#007A33", "accent": "#BA9653"},
    "BKN": {"primary": "#000000", "accent": "#FFFFFF"},
    "CHA": {"primary": "#1D1160", "accent": "#00788C"},
    "CHI": {"primary": "#CE1141", "accent": "#000000"},
    "CLE": {"primary": "#860038", "accent": "#041E42"},
    "DAL": {"primary": "#00538C", "accent": "#002B5E"},
    "DEN": {"primary": "#0E2240", "accent": "#FEC524"},
    "DET": {"primary": "#C8102E", "accent": "#006BB6"},
    "GSW": {"primary": "#1D428A", "accent": "#FFC72C"},
    "HOU": {"primary": "#CE1141", "accent": "#000000"},
    "IND": {"primary": "#002D62", "accent": "#FDBB30"},
    "LAC": {"primary": "#C8102E", "accent": "#1D428A"},
    "LAL": {"primary": "#552583", "accent": "#FDB927"},
    "MEM": {"primary": "#5D76A9", "accent": "#12173F"},
    "MIA": {"primary": "#98002E", "accent": "#F9A01B"},
    "MIL": {"primary": "#00471B", "accent": "#EEE1C6"},
    "MIN": {"primary": "#0C2340", "accent": "#236192"},
    "NOP": {"primary": "#0C2340", "accent": "#C8102E"},
    "NYK": {"primary": "#006BB6", "accent": "#F58426"},
    "OKC": {"primary": "#007AC1", "accent": "#EF3B24"},
    "ORL": {"primary": "#0077C0", "accent": "#C4CED4"},
    "PHI": {"primary": "#006BB6", "accent": "#ED174C"},
    "PHX": {"primary": "#1D1160", "accent": "#E56020"},
    "POR": {"primary": "#E03A3E", "accent": "#000000"},
    "SAC": {"primary": "#5A2D81", "accent": "#63727A"},
    "SAS": {"primary": "#C4CED4", "accent": "#000000"},
    "TOR": {"primary": "#CE1141", "accent": "#000000"},
    "UTA": {"primary": "#002B5C", "accent": "#F9A01B"},
    "WAS": {"primary": "#002B5C", "accent": "#E31837"},
}
