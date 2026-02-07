from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.processing.play_tagger import tag_play

from config.ncaa_di_mens_basketball import NCAA_DI_MENS_BASKETBALL
from src.ingestion.db import db_path
from src.ingestion.synergy_client import SynergyClient


@dataclass
class TrainResult:
    model: XGBRegressor
    features: list[str]
    metrics: dict[str, float]
    train_rows: int
    test_rows: int


def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = name.replace("st.", "saint").replace("st ", "saint ")
    name = re.sub(r"[^a-z0-9 ]+", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _build_team_to_conference() -> dict[str, str]:
    team_to_conf: dict[str, str] = {}
    for conf, teams in NCAA_DI_MENS_BASKETBALL.items():
        for team in teams:
            team_to_conf[_normalize_name(team)] = conf
    return team_to_conf


def _conference_strength_map() -> dict[str, float]:
    # Adjustable weights: power conferences highest, then strong mid-majors, then others.
    return {
        "ACC": 1.0,
        "SEC": 1.0,
        "Big Ten": 1.0,
        "Big 12": 1.0,
        "Big East": 0.98,
        "Pac-12": 0.95,
        "AAC": 0.9,
        "Mountain West": 0.9,
        "A-10": 0.88,
        "WCC": 0.88,
        "MAC": 0.84,
        "Sun Belt": 0.84,
        "C-USA": 0.84,
        "Ivy League": 0.83,
        "Patriot League": 0.82,
        "SoCon": 0.82,
        "Big Sky": 0.82,
        "ASUN": 0.8,
    }


def _season_map_from_api(client: SynergyClient) -> dict[str, int]:
    payload = client.get_seasons("ncaamb")
    items = []
    if isinstance(payload, dict):
        items = payload.get("data") or payload.get("items") or payload.get("seasons") or []
    elif isinstance(payload, list):
        items = payload

    season_map: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        season_id = item.get("id") or item.get("seasonId") or item.get("season_id")
        year = item.get("year") or item.get("seasonYear") or item.get("season_year")
        name = item.get("name") or item.get("label") or ""
        if year is None and isinstance(name, str):
            m = re.search(r"(20\d{2})", name)
            if m:
                year = int(m.group(1))
        if season_id and year:
            season_map[str(season_id)] = int(year)
    return season_map


def _team_map_from_api(client: SynergyClient, season_id: str) -> dict[str, str]:
    payload = client.get_teams("ncaamb", season_id)
    items = []
    if isinstance(payload, dict):
        items = payload.get("data") or payload.get("items") or payload.get("teams") or []
    elif isinstance(payload, list):
        items = payload

    team_map: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        team_id = item.get("id") or item.get("teamId") or item.get("team_id")
        name = item.get("name") or item.get("teamName") or item.get("displayName")
        if team_id and name:
            team_map[str(team_id)] = str(name)
    return team_map


def _load_tables(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = pd.read_sql_query("SELECT * FROM player_season_stats", conn)
    players = pd.read_sql_query("SELECT player_id, height_in, weight_lb, class_year, team_id FROM players", conn)
    traits = pd.read_sql_query("SELECT * FROM player_traits", conn)
    return stats, players, traits


def _class_year_to_num(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().upper()
    mapping = {
        "FR": 1,
        "SO": 2,
        "JR": 3,
        "SR": 4,
        "RS": 5,
        "GR": 5,
        "GRAD": 5,
    }
    return mapping.get(s)


def _compute_eff_per_40(df: pd.DataFrame, minutes_col: str | None, possessions_col: str | None) -> pd.Series:
    pts = df.get("points", 0)
    reb = df.get("reb", 0)
    ast = df.get("ast", 0)
    stl = df.get("stl", 0)
    blk = df.get("blk", 0)

    fga = df.get("fg_attempt", df.get("fga", 0))
    fgm = df.get("fg_made", df.get("fgm", 0))
    fta = df.get("ft_attempt", df.get("fta", 0))
    ftm = df.get("ft_made", df.get("ftm", 0))
    tov = df.get("turnover", df.get("tov", 0))

    raw_eff = (pts + reb + ast + stl + blk) - ((fga - fgm) + (fta - ftm) + tov)

    if minutes_col and minutes_col in df.columns:
        denom = df[minutes_col].replace(0, np.nan)
    elif possessions_col and possessions_col in df.columns:
        denom = df[possessions_col].replace(0, np.nan)
    else:
        denom = pd.Series(np.nan, index=df.index)

    return (raw_eff / denom) * 40


def _build_within_season_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    # Fallback: split each season into early/late using median game date
    plays = pd.read_sql_query(
        """
        SELECT p.game_id, g.season_id, g.date, p.description, p.clock_seconds,
               p.player_id, p.assist_player_id, p.r_player_id, p.d_player_id,
               p.duration, p.offensive_lineup
        FROM plays p
        JOIN games g ON g.game_id = p.game_id
        """,
        conn,
    )

    if plays.empty:
        return pd.DataFrame()

    # Determine split per season
    plays["date"] = pd.to_datetime(plays["date"], errors="coerce")
    season_median = plays.groupby("season_id")["date"].transform("median")
    plays["split"] = (plays["date"] > season_median).map({True: "late", False: "early"})

    def _lineup_player_ids(raw) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            lineup = raw
        else:
            try:
                lineup = json.loads(raw)
                if not isinstance(lineup, list):
                    lineup = []
            except Exception:
                lineup = []
        ids = []
        for item in lineup:
            if isinstance(item, dict):
                pid = item.get("id") or item.get("playerId") or item.get("player_id")
                if pid:
                    ids.append(str(pid))
            elif isinstance(item, str):
                ids.append(item)
        return ids

    rows = []
    for _, r in plays.iterrows():
        tags = tag_play(r.get("description") or "", r.get("clock_seconds"))
        lineup_ids = _lineup_player_ids(r.get("offensive_lineup"))
        duration = r.get("duration")
        involved = set(filter(None, [r.get("player_id"), r.get("assist_player_id"), r.get("r_player_id"), r.get("d_player_id")]))

        minutes_map = {}
        if duration is not None:
            if lineup_ids:
                for pid in lineup_ids:
                    minutes_map[pid] = minutes_map.get(pid, 0.0) + float(duration) / 60.0
            else:
                for pid in involved:
                    minutes_map[pid] = minutes_map.get(pid, 0.0) + float(duration) / 60.0

        def _add(pid, field, val=1):
            if not pid:
                return
            rows.append({
                "player_id": pid,
                "season_id": r.get("season_id"),
                "split": r.get("split"),
                "field": field,
                "value": val,
            })

        # Minutes
        for pid, mins in minutes_map.items():
            _add(pid, "minutes", mins)

        # Possessions proxy
        if r.get("player_id"):
            _add(r.get("player_id"), "possessions", 1)

        # Rebounds
        if "rebound" in tags:
            _add(r.get("r_player_id") or r.get("player_id"), "reb", 1)

        # Assists
        if "assist" in tags:
            _add(r.get("assist_player_id"), "ast", 1)

        # Steals
        if "steal" in tags:
            _add(r.get("d_player_id") or r.get("player_id"), "stl", 1)

        # Blocks
        if "block" in tags:
            _add(r.get("d_player_id") or r.get("player_id"), "blk", 1)

        # Turnovers
        if "turnover" in tags:
            _add(r.get("player_id"), "turnover", 1)

        # FG/FT attempts + makes
        if "ft" in tags:
            _add(r.get("player_id"), "ft_attempt", 1)
            if "made" in tags:
                _add(r.get("player_id"), "ft_made", 1)
            else:
                _add(r.get("player_id"), "ft_miss", 1)
        elif "3pt" in tags or "jumpshot" in tags or "dunk" in tags or "layup" in tags:
            _add(r.get("player_id"), "fg_attempt", 1)
            if "made" in tags:
                _add(r.get("player_id"), "fg_made", 1)
            else:
                _add(r.get("player_id"), "fg_miss", 1)
            if "3pt" in tags:
                _add(r.get("player_id"), "shot3_attempt", 1)
                if "made" in tags:
                    _add(r.get("player_id"), "shot3_made", 1)
                else:
                    _add(r.get("player_id"), "shot3_miss", 1)
            else:
                _add(r.get("player_id"), "shot2_attempt", 1)
                if "made" in tags:
                    _add(r.get("player_id"), "shot2_made", 1)
                else:
                    _add(r.get("player_id"), "shot2_miss", 1)

        # Points
        pts = 0
        if "ft" in tags:
            pts = 1 if "made" in tags else 0
        elif "3pt" in tags:
            pts = 3 if "made" in tags else 0
        elif "made" in tags:
            pts = 2
        if pts > 0:
            _add(r.get("player_id"), "points", pts)

    if not rows:
        return pd.DataFrame()

    agg = pd.DataFrame(rows)
    agg = agg.groupby(["player_id", "season_id", "split", "field"])["value"].sum().unstack(fill_value=0).reset_index()

    # Compute derived percents
    agg["fg_attempt"] = agg.get("fg_attempt", 0)
    agg["fg_made"] = agg.get("fg_made", 0)
    agg["fg_miss"] = agg.get("fg_miss", agg["fg_attempt"] - agg["fg_made"])
    agg["shot2_attempt"] = agg.get("shot2_attempt", 0)
    agg["shot2_made"] = agg.get("shot2_made", 0)
    agg["shot2_miss"] = agg.get("shot2_miss", agg["shot2_attempt"] - agg["shot2_made"])
    agg["shot3_attempt"] = agg.get("shot3_attempt", 0)
    agg["shot3_made"] = agg.get("shot3_made", 0)
    agg["shot3_miss"] = agg.get("shot3_miss", agg["shot3_attempt"] - agg["shot3_made"])
    agg["ft_attempt"] = agg.get("ft_attempt", 0)
    agg["ft_made"] = agg.get("ft_made", 0)
    agg["ft_miss"] = agg.get("ft_miss", agg["ft_attempt"] - agg["ft_made"])

    def _pct(made, att):
        return np.where(att > 0, made / att, np.nan)

    agg["fg_percent"] = _pct(agg["fg_made"], agg["fg_attempt"])
    agg["shot2_percent"] = _pct(agg["shot2_made"], agg["shot2_attempt"])
    agg["shot3_percent"] = _pct(agg["shot3_made"], agg["shot3_attempt"])
    agg["ft_percent"] = _pct(agg["ft_made"], agg["ft_attempt"])
    agg["fg_percent_effective"] = np.where(
        agg["fg_attempt"] > 0,
        (agg["fg_made"] + 0.5 * agg["shot3_made"]) / agg["fg_attempt"],
        np.nan,
    )

    # Merge physicals + traits
    players = pd.read_sql_query("SELECT player_id, height_in, weight_lb, class_year, team_id FROM players", conn)
    traits = pd.read_sql_query("SELECT * FROM player_traits", conn)
    players = players.drop_duplicates(subset=["player_id"]).rename(columns={"team_id": "player_team_id"})
    df = agg.merge(players, on="player_id", how="left")
    if not traits.empty:
        df = df.merge(traits, on="player_id", how="left", suffixes=("", "_trait"))

    df["class_year_num"] = df["class_year"].apply(_class_year_to_num)

    # Conference strength (best-effort via teams endpoint for latest season)
    client = SynergyClient()
    season_map = _season_map_from_api(client)
    team_to_conf = _build_team_to_conference()
    conf_strength = _conference_strength_map()
    team_map = {}
    if season_map:
        latest_season_id = max(season_map.items(), key=lambda x: x[1])[0]
        team_map = _team_map_from_api(client, latest_season_id)

    def _team_strength(team_id: Any) -> float | None:
        if team_id is None:
            return None
        name = team_map.get(str(team_id))
        if not name:
            return None
        conf = team_to_conf.get(_normalize_name(name))
        if not conf:
            return None
        return conf_strength.get(conf, 0.85)

    df["conference_strength"] = df["player_team_id"].apply(_team_strength)

    minutes_col = "minutes" if "minutes" in df.columns and df["minutes"].sum() > 0 else None
    possessions_col = "possessions" if "possessions" in df.columns else None
    df["eff_per_40"] = _compute_eff_per_40(df, minutes_col, possessions_col)

    early = df[df["split"] == "early"].copy()
    late = df[df["split"] == "late"].copy()

    early = early.rename(columns={"eff_per_40": "eff_per_40_early"})
    late = late.rename(columns={"eff_per_40": "target_next_eff_per_40"})

    merged = early.merge(
        late[["player_id", "season_id", "target_next_eff_per_40"]],
        on=["player_id", "season_id"],
        how="inner",
    )

    merged["season_year"] = None
    return merged


def build_training_frame(db_file: str | None = None, allow_fallback: bool = True) -> pd.DataFrame:
    path = db_file or db_path()
    conn = sqlite3.connect(path)

    stats, players, traits = _load_tables(conn)

    if stats.empty:
        conn.close()
        raise RuntimeError("player_season_stats is empty. Run ingestion first.")

    # Merge physicals + traits
    players = players.drop_duplicates(subset=["player_id"]).rename(columns={"team_id": "player_team_id"})
    df = stats.merge(players, on="player_id", how="left")
    if not traits.empty:
        df = df.merge(traits, on="player_id", how="left", suffixes=("", "_trait"))

    df["class_year_num"] = df["class_year"].apply(_class_year_to_num)

    # Season mapping
    client = SynergyClient()
    season_map = _season_map_from_api(client)
    df["season_year"] = df["season_id"].map(season_map)

    # Team -> conference strength mapping
    team_to_conf = _build_team_to_conference()
    conf_strength = _conference_strength_map()

    team_map = {}
    if season_map:
        latest_season_id = max(season_map.items(), key=lambda x: x[1])[0]
        team_map = _team_map_from_api(client, latest_season_id)

    def _team_strength(team_id: Any) -> float | None:
        if team_id is None:
            return None
        name = team_map.get(str(team_id))
        if not name:
            return None
        conf = team_to_conf.get(_normalize_name(name))
        if not conf:
            return None
        return conf_strength.get(conf, 0.85)

    df["conference_strength"] = df["team_id"].apply(_team_strength)

    # Compute EFF/40 (minutes if available, else possessions proxy)
    minutes_col = "minutes" if "minutes" in df.columns else None
    possessions_col = "possessions" if "possessions" in df.columns else None
    df["eff_per_40"] = _compute_eff_per_40(df, minutes_col, possessions_col)

    # Build target by next-season shift (only consecutive years)
    df = df.sort_values(["player_id", "season_year"])
    df["next_season_year"] = df.groupby("player_id")["season_year"].shift(-1)
    df["target_next_eff_per_40"] = df.groupby("player_id")["eff_per_40"].shift(-1)

    df = df[df["season_year"].notna() & df["next_season_year"].notna()]
    df = df[df["next_season_year"] == df["season_year"] + 1]

    # Filter low-sample players (minutes or possessions proxy)
    if minutes_col and minutes_col in df.columns:
        df = df[df[minutes_col] > 100]
    elif possessions_col and possessions_col in df.columns:
        df = df[df[possessions_col] > 100]

    if df.empty and allow_fallback:
        df = _build_within_season_frame(conn)

    conn.close()
    return df


def train_model(df: pd.DataFrame) -> TrainResult:
    feature_cols = [
        "gp",
        "possessions",
        "points",
        "fg_made",
        "fg_miss",
        "fg_attempt",
        "fg_percent",
        "fg_percent_effective",
        "shot2_made",
        "shot2_miss",
        "shot2_attempt",
        "shot2_percent",
        "shot3_made",
        "shot3_miss",
        "shot3_attempt",
        "shot3_percent",
        "ft_made",
        "ft_miss",
        "ft_attempt",
        "ft_percent",
        "plus_one",
        "shot_foul",
        "score",
        "turnover",
        "height_in",
        "weight_lb",
        "class_year_num",
        "conference_strength",
        "dog_index",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = pd.to_numeric(df["target_next_eff_per_40"], errors="coerce")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    return TrainResult(
        model=model,
        features=feature_cols,
        metrics=metrics,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )


def save_artifacts(result: TrainResult, model_path: str, meta_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    result.model.save_model(model_path)

    meta = {
        "features": result.features,
        "metrics": result.metrics,
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def predict_latest(df: pd.DataFrame, model: XGBRegressor, features: list[str]) -> pd.DataFrame:
    if "season_year" not in df.columns or df["season_year"].notna().sum() == 0:
        return pd.DataFrame()

    latest_year = df["season_year"].max()
    latest = df[df["season_year"] == latest_year].copy()
    if latest.empty:
        return pd.DataFrame()

    X = latest[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    latest["pred_next_eff_per_40"] = model.predict(X)
    return latest
