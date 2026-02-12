from __future__ import annotations

from typing import List, Tuple


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
