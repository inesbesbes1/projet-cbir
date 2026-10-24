"""Simule exactement ce que le navigateur envoie a l'API."""
import sys, json
sys.path.insert(0, "src")

# Simuler le endpoint POST /search tel que le JS l'appelle
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

import cbir.search as search_module
from cbir.search import visual_search

DATASETS = {"objects": ROOT / "data" / "objects"}

def image_url(dataset, image_id):
    return "/images/%s/%s" % (dataset, image_id)

def category_of(image_id):
    return image_id.split("/")[0] if "/" in image_id else ""

# Cas 1: Recherche image depuis la base (clic sur un resultat)
print("=== Cas 1: Clic sur image de la base ===")
# Le JS envoie query_image_id = hit.id = "butterfly/image_0001.jpg"
req_data = {
    "dataset": "objects",
    "query_image_id": "butterfly/image_0001.jpg",
    "descriptors": ["color"],
    "k": 10,
}
print("  Requete:", json.dumps(req_data))
try:
    found = visual_search(
        req_data["dataset"],
        req_data["query_image_id"],
        req_data["descriptors"],
        k=req_data["k"]
    )
    for h in found[:5]:
        url = image_url(req_data["dataset"], h["id"])
        cat = category_of(h["id"])
        print("  -> id=%s  cat=%s  url=%s" % (h["id"], cat, url))
        # Verifier que l'image existe
        phys = DATASETS["objects"] / h["id"].replace("/", "\\")
        print("     physique=%s  existe=%s" % (phys, phys.is_file()))
except Exception as e:
    print("  ERREUR: %s" % e)
    import traceback
    traceback.print_exc()

# Cas 2: Upload image
print("\n=== Cas 2: Upload image ===")
try:
    with open("data/objects/butterfly/image_0001.jpg", "rb") as f:
        img_bytes = f.read()
    fn = getattr(search_module, "visual_search_image", None)
    if fn is None:
        print("  ERREUR: visual_search_image n'existe pas!")
    else:
        found2 = fn("objects", img_bytes, ["color"], k=5)
        for h in found2[:5]:
            url = image_url("objects", h["id"])
            cat = category_of(h["id"])
            print("  -> id=%s  cat=%s  url=%s" % (h["id"], cat, url))
            phys = DATASETS["objects"] / h["id"].replace("/", "\\")
            print("     physique=%s  existe=%s" % (phys, phys.is_file()))
except Exception as e:
    print("  ERREUR: %s" % e)
    import traceback
    traceback.print_exc()

# Cas 3: Re-recherche (clic sur un resultat avec id retourne)
print("\n=== Cas 3: Re-recherche depuis un resultat ===")
# Le user clique sur garfield/image_0016.jpg dans les resultats
# Le JS fait queryImage = hit.id = "garfield/image_0016.jpg"
try:
    found3 = visual_search("objects", "garfield/image_0016.jpg", ["color"], k=5)
    for h in found3[:5]:
        cat = category_of(h["id"])
        url = image_url("objects", h["id"])
        print("  -> [%s] url=%s" % (cat, url))
except Exception as e:
    print("  ERREUR: %s" % e)
    import traceback
    traceback.print_exc()

print("\n=== FIN DIAGNOSTIC ===")
