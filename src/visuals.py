from __future__ import annotations

from typing import List, Tuple, Dict, Any


def generate_pca_coordinates(embeddings_list: List[List[float]]) -> List[Tuple[float, float]]:
    try:
        if embeddings_list is None or len(embeddings_list) == 0:
            return []
    except Exception:
        return []
    try:
        from sklearn.decomposition import PCA
        import numpy as np

        X = np.array(embeddings_list)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        return [(float(x), float(y)) for x, y in coords]
    except Exception:
        return [(0.0, 0.0) for _ in embeddings_list]


def generate_radar_chart(player_a: Dict[str, Any], player_b: Dict[str, Any], query: str = "Big Guard"):
    import plotly.graph_objects as go
    from src.position_calibration import calculate_percentile, map_db_to_canonical, score_positions, topk

    def _pos_for(player):
        pos = player.get("position") or ""
        mapped = map_db_to_canonical(pos)
        return mapped[0] if mapped else pos

    def _fit_score(player):
        h = player.get("height_in")
        w = player.get("weight_lb")
        scores = score_positions(query, height_in=h, weight_lb=w)
        top = topk(scores, k=1)
        if not top:
            return 0.0
        max_score = max(scores.values()) or 1.0
        return max(0.0, min(1.0, top[0][1] / max_score)) * 100.0

    def _overall(player):
        val = player.get("score") or player.get("Recruit Score") or 0.0
        try:
            val = float(val)
        except Exception:
            val = 0.0
        return max(0.0, min(100.0, val))

    pos_a = _pos_for(player_a)
    pos_b = _pos_for(player_b)
    h_pct_a = calculate_percentile(player_a.get("height_in"), pos_a, metric="h")
    w_pct_a = calculate_percentile(player_a.get("weight_lb"), pos_a, metric="w")
    h_pct_b = calculate_percentile(player_b.get("height_in"), pos_b, metric="h")
    w_pct_b = calculate_percentile(player_b.get("weight_lb"), pos_b, metric="w")

    categories = ["Height %", "Weight %", "Vector Match", "Scout Score"]
    a_vals = [h_pct_a, w_pct_a, _fit_score(player_a), _overall(player_a)]
    b_vals = [h_pct_b, w_pct_b, _fit_score(player_b), _overall(player_b)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill="toself", name=player_a.get("name", "Player A"), line=dict(color="#31d0ff")))
    fig.add_trace(go.Scatterpolar(r=b_vals, theta=categories, fill="toself", name=player_b.get("name", "Player B"), line=dict(color="#7f8ba3")))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#f3f6ff"),
    )
    return fig


def generate_zone_chart(locations: dict):
    import plotly.graph_objects as go
    if not locations:
        return go.Figure()
    labels = list(locations.keys())
    values = list(locations.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#f3f6ff"),
    )
    return fig


def generate_tendency_comparison(player_a_clips, player_b_clips, label_a: str = "Player A", label_b: str = "Player B"):
    import plotly.graph_objects as go
    from src.film import analyze_tendencies

    a_t = analyze_tendencies(player_a_clips)
    b_t = analyze_tendencies(player_b_clips)
    keys = sorted(set(a_t.keys()) | set(b_t.keys()))
    if not keys:
        return go.Figure()
    a_vals = [a_t.get(k, 0) for k in keys]
    b_vals = [b_t.get(k, 0) for k in keys]

    fig = go.Figure()
    fig.add_bar(name=label_a, x=keys, y=a_vals, marker_color="#31d0ff")
    fig.add_bar(name=label_b, x=keys, y=b_vals, marker_color="#f6c453")
    fig.update_layout(
        barmode="group",
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#f3f6ff"),
        yaxis=dict(title="%"),
    )
    return fig


def generate_archetype_map(players: list[dict], clusters: dict, selected_player: str | None = None):
    import plotly.graph_objects as go
    try:
        from sklearn.decomposition import PCA
        import numpy as np
    except Exception:
        return go.Figure()

    if not players:
        return go.Figure()

    embeddings = [p.get("embedding") for p in players if p.get("embedding") is not None]
    if not embeddings:
        return go.Figure()

    X = np.array(embeddings)
    coords = PCA(n_components=2).fit_transform(X)

    labels = []
    names = []
    ppgs = []
    xs = []
    ys = []
    pids = []
    for i, p in enumerate(players):
        pid = p.get("player_id")
        if p.get("embedding") is None:
            continue
        x, y = coords[i]
        cluster = clusters.get(pid) or clusters.get(str(pid))
        labels.append(cluster or "Unknown")
        names.append(p.get("name") or "Unknown")
        ppgs.append(p.get("ppg") or 0)
        xs.append(float(x))
        ys.append(float(y))
        pids.append(pid)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=8),
            text=[f"{n} | {l} | PPG {ppg}" for n, l, ppg in zip(names, labels, ppgs)],
            hoverinfo="text",
        )
    )

    if selected_player and selected_player in pids:
        idx = pids.index(selected_player)
        fig.add_trace(
            go.Scatter(
                x=[xs[idx]],
                y=[ys[idx]],
                mode="markers",
                marker=dict(size=16, symbol="star", color="#f6c453"),
                text=[f"{names[idx]} | {labels[idx]}"],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#f3f6ff"),
    )
    return fig
