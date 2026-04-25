# Layer 4: Shared feature engineering for pitcher scoring.
# Pure functions — no I/O, no API calls. Computes derived Statcast features used by
# Stuff+, Command+, and Deception+.

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.config import SZ_L, SZ_R, SZ_B, SZ_T

_FEET_TO_INCHES = 12.0
_4_INCHES_FT = 4.0 / 12.0      # 4-inch break threshold in feet
_PLATE_Y = 0.0                  # y=0 is the front of the plate
_STATCAST_REF_Y = 50.0          # Statcast vx0/vy0/vz0 defined at y=50 ft from plate


# ── Strike-zone edge distance ─────────────────────────────────────────────────

def edge_distance_inches(plate_x: float, plate_z: float) -> float:
    """
    Signed distance from pitch location to nearest strike-zone edge, in inches.
    Negative = inside zone (strike), positive = outside zone (ball).
    NaN if either coordinate is NaN.
    """
    if math.isnan(plate_x) or math.isnan(plate_z):
        return float("nan")

    inside_x = SZ_L <= plate_x <= SZ_R
    inside_z = SZ_B <= plate_z <= SZ_T

    if inside_x and inside_z:
        dx = min(plate_x - SZ_L, SZ_R - plate_x)
        dz = min(plate_z - SZ_B, SZ_T - plate_z)
        return -min(dx, dz) * _FEET_TO_INCHES
    else:
        dx = max(SZ_L - plate_x, 0.0, plate_x - SZ_R)
        dz = max(SZ_B - plate_z, 0.0, plate_z - SZ_T)
        return math.sqrt(dx ** 2 + dz ** 2) * _FEET_TO_INCHES


def compute_edge_distances(df: pd.DataFrame) -> pd.Series:
    """Vectorized edge distance for all pitches. Returns Series aligned to df.index."""
    px = df["plate_x"].to_numpy(dtype=float)
    pz = df["plate_z"].to_numpy(dtype=float)

    in_x = (px >= SZ_L) & (px <= SZ_R)
    in_z = (pz >= SZ_B) & (pz <= SZ_T)
    inside = in_x & in_z

    result = np.full(len(df), np.nan)

    # Inside zone → negative distance
    dx_in = np.minimum(px - SZ_L, SZ_R - px)
    dz_in = np.minimum(pz - SZ_B, SZ_T - pz)
    result[inside] = -np.minimum(dx_in[inside], dz_in[inside]) * _FEET_TO_INCHES

    # Outside zone → positive distance
    dx_out = np.maximum(np.maximum(SZ_L - px, 0.0), px - SZ_R)
    dz_out = np.maximum(np.maximum(SZ_B - pz, 0.0), pz - SZ_T)
    result[~inside] = np.sqrt(dx_out[~inside] ** 2 + dz_out[~inside] ** 2) * _FEET_TO_INCHES

    # NaN where inputs were NaN
    nan_mask = np.isnan(px) | np.isnan(pz)
    result[nan_mask] = np.nan

    return pd.Series(result, index=df.index, name="edge_distance_in")


# ── Vertical Approach Angle (VAA) ─────────────────────────────────────────────

def compute_vaa(vy0: float, vz0: float, ay: float, az: float) -> float:
    """
    Vertical approach angle at the front of the plate, in degrees. Negative = downward.

    Statcast defines vy0, vz0, ay, az at y=50 ft from plate.
    Solves y(t)=0 for t (time to reach plate), then computes angle from velocity components.
    """
    # Solve 50 + vy0*t + 0.5*ay*t^2 = 0 for positive t
    disc = vy0 ** 2 - 100.0 * ay   # vy0^2 + 100*|ay| (always positive since ay < 0)
    if disc < 0:
        return float("nan")
    t = (-vy0 - math.sqrt(disc)) / ay if ay != 0 else -_STATCAST_REF_Y / vy0
    if t <= 0:
        return float("nan")
    vz_plate = vz0 + az * t
    vy_plate = vy0 + ay * t
    if vy_plate == 0:
        return float("nan")
    return math.degrees(math.atan2(vz_plate, abs(vy_plate)))


def compute_vaa_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized VAA for a DataFrame. Requires vy0, vz0, ay, az columns."""
    required = {"vy0", "vz0", "ay", "az"}
    if not required.issubset(df.columns):
        return pd.Series(np.nan, index=df.index, name="vaa")
    return df.apply(
        lambda r: compute_vaa(r["vy0"], r["vz0"], r["ay"], r["az"])
        if pd.notna(r["vy0"]) and pd.notna(r["vz0"]) and pd.notna(r["ay"]) and pd.notna(r["az"])
        else float("nan"),
        axis=1,
    ).rename("vaa")


# ── Break-point distance ───────────────────────────────────────────────────────

def compute_break_point_distance(
    ax: float, az: float, vy0: float, ay: float, threshold_inches: float = 4.0
) -> float:
    """
    Distance from plate (feet) at which the pitch first deviates 'threshold_inches'
    from a straight-line projection. Computed from Statcast fields at y=50.

    Deviation from straight line (xz plane only):
        dev(t) = 0.5 * t^2 * sqrt(ax^2 + az^2)

    Solving for t when dev = threshold_inches / 12:
        t = sqrt(threshold_ft / (0.5 * total_accel))

    Returns inf if pitch barely curves (never reaches threshold before plate).
    Returns nan on bad inputs.
    """
    if any(math.isnan(v) for v in (ax, az, vy0, ay)):
        return float("nan")

    total_accel = math.sqrt(ax ** 2 + az ** 2)
    if total_accel == 0:
        return float("inf")

    threshold_ft = threshold_inches / 12.0
    t_break = math.sqrt(threshold_ft / (0.5 * total_accel))

    # y-position at t_break (measured from ref point y=50)
    y_at_break = _STATCAST_REF_Y + vy0 * t_break + 0.5 * ay * t_break ** 2
    return max(y_at_break, 0.0)   # clamp to 0 (at or past plate)


def compute_break_point_distances(df: pd.DataFrame, threshold_inches: float = 4.0) -> pd.Series:
    """Vectorized break-point distance. Requires ax, az, vy0, ay columns."""
    required = {"ax", "az", "vy0", "ay"}
    if not required.issubset(df.columns):
        return pd.Series(np.nan, index=df.index, name="break_pt_dist_ft")
    return df.apply(
        lambda r: compute_break_point_distance(r["ax"], r["az"], r["vy0"], r["ay"], threshold_inches)
        if pd.notna(r["ax"]) and pd.notna(r["az"])
        else float("nan"),
        axis=1,
    ).rename("break_pt_dist_ft")


# ── Normalization helper ───────────────────────────────────────────────────────

def normalize_to_100(raw: float, league_mean: float, league_std: float, higher_is_better: bool = True) -> float:
    """Scale a raw score to the 100 = league average convention."""
    if league_std == 0:
        return 100.0
    direction = 1 if higher_is_better else -1
    return 100.0 + direction * (raw - league_mean) / league_std * 10.0
