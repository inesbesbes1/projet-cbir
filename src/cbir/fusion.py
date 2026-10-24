"""Module de fusion multimodale (Texte ET/OU Image) pour le CBIR (Tâche B5).

Combine les résultats issus de la recherche textuelle (Elasticsearch)
et de la recherche visuelle (signatures / descripteurs d'images).
"""

from typing import Any, Dict, List, Optional


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    weights: Optional[List[float]] = None,
    k_constant: float = 60.0,
) -> Dict[str, float]:
    """Calcule le score RRF (Reciprocal Rank Fusion) pour chaque identifiant d'image.

    Formule pour un élément d :
        Score_RRF(d) = Σ_m [ w_m / (k_constant + rang_m(d)) ]
    où rang_m(d) est la position (1-indexée) de d dans la liste classée m.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = {}
    for r_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(r_list, start=1):
            img_id = item["id"]
            scores[img_id] = scores.get(img_id, 0.0) + weight / (k_constant + rank)
    return scores


def fuse_results(
    text_results: List[Dict[str, Any]],
    visual_results: List[Dict[str, Any]],
    operator: str = "AND",
    k: int = 20,
) -> List[Dict[str, Any]]:
    """Fusionne les résultats de la recherche textuelle et de la recherche visuelle.

    Args:
        text_results: Liste de dicts [{"id": str, "score": float, "category": str}]
        visual_results: Liste de dicts [{"id": str, "distance": float}]
        operator: 'AND' (intersection avec reranking) ou 'OR' (union avec RRF)
        k: Nombre maximum de résultats à retourner

    Returns:
        Liste de dicts [{"id": str, "category": str, "score": float, "distance": Optional[float]}]
    """
    operator = operator.upper().strip()

    # Indexer les métadonnées pour enrichir les résultats finaux
    text_map = {item["id"]: item for item in text_results}
    visual_map = {item["id"]: item for item in visual_results}

    text_ids = set(text_map.keys())
    visual_ids = set(visual_map.keys())

    # Calcul des scores RRF combinés
    rrf_scores = reciprocal_rank_fusion([text_results, visual_results])

    if operator == "AND":
        # Intersection stricte : les images présentes dans les DEUX modalités
        candidate_ids = text_ids & visual_ids
        
        # Si l'intersection stricte donne des résultats, on les trie par score RRF
        if candidate_ids:
            sorted_ids = sorted(candidate_ids, key=lambda x: rrf_scores[x], reverse=True)[:k]
        else:
            # Fallback : si aucun chevauchement direct dans les top résultats,
            # on privilégie les résultats textuels (garantissant la pertinence du concept)
            sorted_ids = [item["id"] for item in text_results][:k]
    else:  # OR
        # Union : toutes les images qui matchent soit le texte, soit l'image
        candidate_ids = text_ids | visual_ids
        sorted_ids = sorted(candidate_ids, key=lambda x: rrf_scores[x], reverse=True)[:k]

    # Construction de la liste finale
    fused_hits: List[Dict[str, Any]] = []
    for img_id in sorted_ids:
        category = ""
        if img_id in text_map and text_map[img_id].get("category"):
            category = text_map[img_id]["category"]
        elif "/" in img_id:
            category = img_id.split("/")[0]

        vis_item = visual_map.get(img_id)
        txt_item = text_map.get(img_id)

        fused_hits.append({
            "id": img_id,
            "category": category,
            "score": float(rrf_scores.get(img_id, 0.0)),
            "distance": float(vis_item["distance"]) if vis_item and "distance" in vis_item else None,
            "text_score": float(txt_item["score"]) if txt_item and "score" in txt_item else None,
        })

    return fused_hits
