"""
Recherche exhaustive vectorisee avec numpy.

Au lieu de boucler en Python sur chaque image (lent),
on empile toutes les signatures en une matrice numpy
et on calcule toutes les distances en une seule operation vectorisee.

Speedup typique : 10-50x par rapport a la boucle Python.
"""

import numpy as np
from typing import Callable


def search(
    query_vector: np.ndarray,
    index: dict,
    distance_fn: Callable,
    k: int = 10,
) -> list[tuple[str, float]]:
    """
    Recherche exhaustive vectorisee.

    Args:
        query_vector : vecteur numpy 1D de la requete.
        index        : {chemin_image: vecteur_numpy} pour UN descripteur.
        distance_fn  : fonction de distance (chi2, l1, l2).
        k            : nombre de resultats a retourner.

    Returns:
        Liste triee de (chemin_image, distance), du plus similaire au moins.
    """
    paths = list(index.keys())
    vectors = np.array(list(index.values()), dtype=np.float32)
    query = query_vector.astype(np.float32)

    # Calcul vectorise selon le type de distance
    fn_name = getattr(distance_fn, "__name__", "")

    if fn_name == "l2":
        # Euclidienne vectorisee : sqrt(sum((q - v)^2, axis=1))
        diffs = vectors - query[np.newaxis, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))

    elif fn_name == "l1":
        # Manhattan vectorisee : sum(|q - v|, axis=1)
        dists = np.sum(np.abs(vectors - query[np.newaxis, :]), axis=1)

    elif fn_name == "chi2":
        # Chi2 vectorisee : sum((q-v)^2 / (q+v+eps), axis=1)
        eps = 1e-10
        diff = vectors - query[np.newaxis, :]
        summ = vectors + query[np.newaxis, :] + eps
        dists = np.sum(diff ** 2 / summ, axis=1)

    else:
        # Fallback : boucle Python (pour des distances custom)
        dists = np.array([distance_fn(query, v) for v in vectors], dtype=np.float32)

    # Top-k par argpartition (O(n) au lieu de O(n log n) pour le tri complet)
    if k < len(dists):
        top_k_idx = np.argpartition(dists, k)[:k]
        # Trier seulement les k elements
        sorted_idx = top_k_idx[np.argsort(dists[top_k_idx])]
    else:
        sorted_idx = np.argsort(dists)

    return [(paths[i], float(dists[i])) for i in sorted_idx]
