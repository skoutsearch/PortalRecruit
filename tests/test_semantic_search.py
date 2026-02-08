from src.search.semantic import semantic_search


class DummyCollection:
    def query(self, **kwargs):
        return {
            "ids": [["p1", "p2", "p3"]],
            "documents": [[
                "high motor wing gets deflections and loose balls",
                "spot up shooter and catch and shoot scoring",
                "rim protection and blocks in drop coverage",
            ]],
            "distances": [[0.25, 0.15, 0.35]],
            "metadatas": [[
                {"tags": "deflection,loose_ball,wing"},
                {"tags": "3pt,jumpshot"},
                {"tags": "block,rim_protection"},
            ]],
        }


def test_semantic_search_fallback_without_cross_encoder(monkeypatch):
    def boom():
        raise RuntimeError("no model")

    monkeypatch.setattr("src.search.semantic.get_cross_encoder", boom)
    monkeypatch.setattr("src.search.semantic.encode_query", lambda q: [0.1, 0.2, 0.3])
    results = semantic_search(
        DummyCollection(),
        query="high motor wing deflections",
        n_results=2,
        required_tags=["deflection"],
    )
    assert results == ["p1"]


def test_semantic_search_uses_rerank(monkeypatch):
    class DummyCross:
        def predict(self, pairs, batch_size=16):
            mapping = {
                "high motor wing gets deflections and loose balls": 0.2,
                "spot up shooter and catch and shoot scoring": 0.1,
                "rim protection and blocks in drop coverage": 1.0,
            }
            return [mapping[p[1]] for p in pairs]

    monkeypatch.setattr("src.search.semantic.get_cross_encoder", lambda: DummyCross())
    monkeypatch.setattr("src.search.semantic.encode_query", lambda q: [0.1, 0.2, 0.3])
    results = semantic_search(
        DummyCollection(),
        query="drop coverage rim protector",
        n_results=2,
        required_tags=["block", "rim_protection"],
    )
    assert results[0] == "p3"


def test_semantic_search_respects_required_tags(monkeypatch):
    class DummyCross:
        def predict(self, pairs, batch_size=16):
            return [0.9 for _ in pairs]

    monkeypatch.setattr("src.search.semantic.get_cross_encoder", lambda: DummyCross())
    monkeypatch.setattr("src.search.semantic.encode_query", lambda q: [0.1, 0.2, 0.3])

    class TagCollection:
        def query(self, **kwargs):
            return {
                "ids": [["p1", "p2"]],
                "documents": [["rim pressure finish", "spot up 3"]],
                "distances": [[0.1, 0.1]],
                "metadatas": [[
                    {"tags": "rim_finish,drive", "player_id": "10"},
                    {"tags": "3pt,jumpshot", "player_id": "11"},
                ]],
            }

    results = semantic_search(
        TagCollection(),
        query="rim finisher",
        n_results=2,
        required_tags=["rim_finish"],
    )
    assert results == ["p1"]


def test_semantic_search_diversifies_players(monkeypatch):
    class DummyCross:
        def predict(self, pairs, batch_size=16):
            return [1.0, 0.95, 0.9]

    monkeypatch.setattr("src.search.semantic.get_cross_encoder", lambda: DummyCross())
    monkeypatch.setattr("src.search.semantic.encode_query", lambda q: [0.1, 0.2, 0.3])

    class DiverseCollection:
        def query(self, **kwargs):
            return {
                "ids": [["a1", "a2", "b1"]],
                "documents": [["elite rim pressure", "elite downhill", "great spacing wing"]],
                "distances": [[0.05, 0.06, 0.07]],
                "metadatas": [[
                    {"tags": "drive", "player_id": "100"},
                    {"tags": "drive", "player_id": "100"},
                    {"tags": "3pt", "player_id": "200"},
                ]],
            }

    results = semantic_search(DiverseCollection(), query="best athlete", n_results=2)
    assert results == ["a1", "b1"]


def test_semantic_search_required_tags_fallback_to_partial(monkeypatch):
    class DummyCross:
        def predict(self, pairs, batch_size=16):
            return [0.7 for _ in pairs]

    monkeypatch.setattr("src.search.semantic.get_cross_encoder", lambda: DummyCross())
    monkeypatch.setattr("src.search.semantic.encode_query", lambda q: [0.1, 0.2, 0.3])

    class SparseTagCollection:
        def query(self, **kwargs):
            return {
                "ids": [["p1", "p2", "p3"]],
                "documents": [["rim pressure", "drive and kick", "spot up shot"]],
                "distances": [[0.1, 0.12, 0.15]],
                "metadatas": [[
                    {"tags": "rim_finish", "player_id": "10"},
                    {"tags": "drive", "player_id": "11"},
                    {"tags": "3pt", "player_id": "12"},
                ]],
            }

    results = semantic_search(
        SparseTagCollection(),
        query="rim pressure",
        n_results=3,
        required_tags=["rim_finish", "drive"],
    )
    assert "p1" in results and "p2" in results
