"""Diagnostic complet du moteur de recherche."""
import sys, os, pickle
sys.path.insert(0, "src")

print("=" * 60)
print("DIAGNOSTIC COMPLET")
print("=" * 60)

# 1. Verifier l'index
print("\n[1] INDEX")
idx_path = "index/index.pkl"
if not os.path.exists(idx_path):
    print("  ERREUR: index/index.pkl introuvable!")
    sys.exit(1)

with open(idx_path, "rb") as f:
    index = pickle.load(f)

for desc_key, desc_idx in index.items():
    keys = list(desc_idx.keys())
    print("  %s : %d signatures" % (desc_key, len(desc_idx)))
    print("    Exemple cle : %s" % keys[0])
    print("    Exemple cle : %s" % keys[1])
    print("    Exemple cle : %s" % keys[2])

# 2. Verifier les chemins d'images
print("\n[2] CHEMINS")
first_key = list(list(index.values())[0].keys())[0]
print("  Cle brute dans l'index : '%s'" % first_key)
print("  Contient backslash?    : %s" % ("\\" in first_key))
print("  Commence par objects?  : %s" % first_key.startswith("objects"))

# Verifier que le fichier image existe vraiment
data_path = os.path.join("data", first_key.replace("/", os.sep))
print("  Chemin physique teste  : %s" % data_path)
print("  Fichier existe?        : %s" % os.path.exists(data_path))

# Aussi tester sans le prefixe objects
clean = first_key.replace("\\", "/")
if clean.startswith("objects/"):
    without = clean[len("objects/"):]
    data_path2 = os.path.join("data", "objects", without.replace("/", os.sep))
    print("  Sans prefixe objects   : %s" % without)
    print("  Chemin physique teste  : %s" % data_path2)
    print("  Fichier existe?        : %s" % os.path.exists(data_path2))

# 3. Tester la recherche
print("\n[3] RECHERCHE")
from cbir.search import visual_search, visual_search_image

# Test avec image de la base
print("\n  --- visual_search(butterfly) ---")
try:
    results = visual_search("objects", "butterfly/image_0001.jpg", ["color"], k=10)
    for i, r in enumerate(results[:10], 1):
        cat = r["id"].split("/")[0]
        print("    %2d. [%s]  dist=%.4f  id=%s" % (i, cat, r["distance"], r["id"]))
except Exception as e:
    print("  ERREUR: %s" % e)

# Test upload
print("\n  --- visual_search_image(butterfly upload) ---")
try:
    with open("data/objects/butterfly/image_0001.jpg", "rb") as f:
        img_bytes = f.read()
    results2 = visual_search_image("objects", img_bytes, ["color"], k=10)
    for i, r in enumerate(results2[:10], 1):
        cat = r["id"].split("/")[0]
        print("    %2d. [%s]  dist=%.4f  id=%s" % (i, cat, r["distance"], r["id"]))
except Exception as e:
    print("  ERREUR: %s" % e)

# Test avec d'autres categories
print("\n  --- visual_search(airplanes) ---")
try:
    results3 = visual_search("objects", "airplanes/image_0001.jpg", ["color"], k=5)
    for i, r in enumerate(results3[:5], 1):
        cat = r["id"].split("/")[0]
        print("    %2d. [%s]  id=%s" % (i, cat, r["id"]))
except Exception as e:
    print("  ERREUR: %s" % e)

# 4. Verifier que l'image URL est correcte
print("\n[4] URL DES IMAGES")
if results:
    sample_id = results[0]["id"]
    url = "/images/objects/%s" % sample_id
    phys = os.path.join("data", "objects", sample_id.replace("/", os.sep))
    print("  ID retourne       : %s" % sample_id)
    print("  URL generee       : %s" % url)
    print("  Chemin physique   : %s" % phys)
    print("  Image existe?     : %s" % os.path.exists(phys))

# 5. Tester le endpoint /images/ comme le ferait le navigateur
print("\n[5] SIMULATION ENDPOINT /images/")
from pathlib import Path
ROOT = Path(".").resolve()
DATASETS = {"objects": ROOT / "data" / "objects"}
if results:
    dataset = "objects"
    image_id = results[0]["id"]
    root = DATASETS.get(dataset)
    path = (root / image_id).resolve()
    print("  root     : %s" % root)
    print("  image_id : %s" % image_id)
    print("  resolved : %s" % path)
    print("  is_file? : %s" % path.is_file())
    print("  root in parents? : %s" % (root.resolve() in path.parents))
