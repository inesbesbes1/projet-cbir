"""
Orchestrateur CBIR — Version v5 (qualite + vitesse).

Ameliorations vs v4 :
  - 4 descripteurs : HSV (128d), RGB (512d), LBP (26d), HOG (8100d)
  - Fusion par score combine normalise (pas RRF) :
    1. Chaque descripteur calcule ses distances
    2. Normalisation min-max par descripteur -> [0,1]
    3. Score final = somme ponderee des distances normalisees
    4. HSV et RGB (couleur) comptent DOUBLE vs LBP et HOG
       -> 2 descripteurs couleur x poids 2 + 2 autres x poids 1 = 6
  - Le query_image est TOUJOURS re-extrait (pas de lookup dans l'index)
    -> memes resultats pour upload et search par id
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from cbir.descriptors.color import HSVDescriptor, RGBDescriptor
from cbir.descriptors.texture import LBPDescriptor
from cbir.descriptors.shape import HOGDescriptor
from cbir.index.store import load_index


# ── Configuration ─────────────────────────────────────────────────────────────

# (index_key, descriptor_instance, distance_type, weight)
DESCRIPTOR_CONFIG = [
    ("hsv_8_4_4",          HSVDescriptor(),   "chi2", 2.0),
    ("rgb_8_8_8",          RGBDescriptor(),   "chi2", 2.0),
    ("lbp_P24_R3",         LBPDescriptor(),   "chi2", 1.0),
    ("hog_o9_c8x8_b2x2",  HOGDescriptor(),   "l2",   1.0),
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


# ── Cache ─────────────────────────────────────────────────────────────────────

class _IndexCache:
    def __init__(self):
        self.loaded = False
        self.descriptors = []   # list of dicts
        self.all_paths = []     # ordered image IDs
        self.n = 0

    def ensure_loaded(self):
        if self.loaded:
            return
        raw_index = load_index()

        # Determine which descriptors are available in the index
        available = []
        for desc_key, descriptor, dist_type, weight in DESCRIPTOR_CONFIG:
            if desc_key in raw_index and len(raw_index[desc_key]) > 0:
                available.append((desc_key, descriptor, dist_type, weight))

        if not available:
            self.loaded = True
            return

        # Find common image paths across all available descriptors
        path_sets = []
        for desc_key, _, _, _ in available:
            clean = [_clean_id(p) for p in raw_index[desc_key].keys()]
            path_sets.append(set(clean))

        common = path_sets[0]
        for s in path_sets[1:]:
            common &= s

        self.all_paths = sorted(common)
        self.n = len(self.all_paths)

        # Build aligned matrices for each descriptor
        for desc_key, descriptor, dist_type, weight in available:
            clean_map = {_clean_id(p): v for p, v in raw_index[desc_key].items()}
            vectors = [clean_map[p] for p in self.all_paths]
            matrix = np.array(vectors, dtype=np.float32)

            info = {
                "key": desc_key,
                "matrix": matrix,
                "dist_type": dist_type,
                "descriptor": descriptor,
                "weight": weight,
            }
            # Precompute squared norms for L2
            if dist_type == "l2":
                info["norms_sq"] = np.sum(matrix ** 2, axis=1)

            self.descriptors.append(info)

        self.loaded = True


_cache = _IndexCache()


def _clean_id(raw_path: str) -> str:
    p = raw_path.replace("\\", "/")
    if p.startswith("objects/"):
        p = p[len("objects/"):]
    return p


# ── Vectorized distance computation ──────────────────────────────────────────

def _compute_distances(query: np.ndarray, info: dict) -> np.ndarray:
    q = query.astype(np.float32)
    matrix = info["matrix"]
    dt = info["dist_type"]

    if dt == "chi2":
        diff = matrix - q
        summ = matrix + q + 1e-10
        return np.sum(diff ** 2 / summ, axis=1)
    elif dt == "l1":
        return np.sum(np.abs(matrix - q), axis=1)
    else:  # l2 via BLAS
        q_nsq = np.dot(q, q)
        dots = matrix @ q
        dsq = info["norms_sq"] + q_nsq - 2.0 * dots
        np.maximum(dsq, 0.0, out=dsq)
        return np.sqrt(dsq)


# ── Main search ──────────────────────────────────────────────────────────────

def _optimized_search(image: Image.Image, k: int) -> list[dict]:
    """
    Multi-descriptor weighted fusion search.

    For each descriptor:
      1. Extract query vector
      2. Compute distances to all images (vectorized)
      3. Normalize distances to [0, 1] using min-max

    Fusion:
      4. Weighted sum of normalized distances
      5. Top-k by ascending fused score (0 = best match)
    """
    _cache.ensure_loaded()
    if not _cache.descriptors:
        raise RuntimeError("Index vide. Lance : python scripts/build_index.py")

    n = _cache.n
    fused = np.zeros(n, dtype=np.float64)
    total_weight = 0.0

    for info in _cache.descriptors:
        query_vec = info["descriptor"].extract(image)
        dists = _compute_distances(query_vec, info)

        # Min-max normalization to [0, 1]
        d_min = dists.min()
        d_max = dists.max()
        span = d_max - d_min
        if span > 1e-10:
            normed = (dists - d_min) / span
        else:
            normed = np.zeros(n, dtype=np.float32)

        w = info["weight"]
        fused += w * normed
        total_weight += w

    # Average by total weight
    if total_weight > 0:
        fused /= total_weight

    # Top-k
    actual_k = min(k, n)
    top_idx = np.argpartition(fused, actual_k)[:actual_k]
    sorted_idx = top_idx[np.argsort(fused[top_idx])]

    return [
        {"id": _cache.all_paths[i], "distance": round(float(fused[i]), 6)}
        for i in sorted_idx
    ]


# ── Public API ───────────────────────────────────────────────────────────────

def search(
    image_path: str,
    descriptor_name: str = "auto",
    distance_name: str = "auto",
    k: int = 10,
    index_file: str = "index.pkl",
) -> list[tuple[str, float]]:
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    image = Image.open(img_path).convert("RGB")
    results = _optimized_search(image, k)
    return [(r["id"], r["distance"]) for r in results]


def visual_search(
    dataset: str,
    query_image_id: str,
    descriptors: list[str],
    k: int = 20,
) -> list[dict]:
    img_path = DATA_DIR / dataset / query_image_id
    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    image = Image.open(img_path).convert("RGB")
    return _optimized_search(image, k)


def visual_search_image(
    dataset: str,
    image_bytes: bytes,
    descriptors: list[str],
    k: int = 20,
) -> list[dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _optimized_search(image, k)
