"""Test de performance et qualite de la recherche optimisee."""
import sys, time
sys.path.insert(0, "src")
from PIL import Image
from cbir.search import _optimized_search, _cache

# Precharger l'index
print("Chargement de l'index...")
t0 = time.perf_counter()
_cache.ensure_loaded()
t_load = time.perf_counter() - t0
print("  Index charge en %.2fs" % t_load)
for key, info in _cache.descs.items():
    print("  %s : matrice %s" % (key, info["matrix"].shape))

# Test 1 : butterfly
print("\n=== Butterfly (image_0001) ===")
img = Image.open("data/objects/butterfly/image_0001.jpg").convert("RGB")
t0 = time.perf_counter()
results = _optimized_search(img, k=15)
t_search = time.perf_counter() - t0
print("  Temps de recherche : %.1fms" % (t_search * 1000))
correct = 0
for i, r in enumerate(results, 1):
    cat = r["id"].split("/")[0]
    if cat == "butterfly":
        correct += 1
    ok = " <-- OK" if cat == "butterfly" else ""
    print("  %2d. [%s] dist=%.4f%s" % (i, cat, r["distance"], ok))
print("  Precision butterfly: %d/15" % correct)

# Test 2 : airplane
print("\n=== Airplanes (image_0001) ===")
img2 = Image.open("data/objects/airplanes/image_0001.jpg").convert("RGB")
t0 = time.perf_counter()
results2 = _optimized_search(img2, k=15)
t_search2 = time.perf_counter() - t0
print("  Temps de recherche : %.1fms" % (t_search2 * 1000))
correct2 = 0
for i, r in enumerate(results2, 1):
    cat = r["id"].split("/")[0]
    if cat == "airplanes":
        correct2 += 1
    ok = " <-- OK" if cat == "airplanes" else ""
    print("  %2d. [%s] dist=%.4f%s" % (i, cat, r["distance"], ok))
print("  Precision airplanes: %d/15" % correct2)

# Test 3 : sunflower
print("\n=== Sunflower (image_0001) ===")
img3 = Image.open("data/objects/sunflower/image_0001.jpg").convert("RGB")
t0 = time.perf_counter()
results3 = _optimized_search(img3, k=15)
t_search3 = time.perf_counter() - t0
print("  Temps de recherche : %.1fms" % (t_search3 * 1000))
correct3 = 0
for i, r in enumerate(results3, 1):
    cat = r["id"].split("/")[0]
    if cat == "sunflower":
        correct3 += 1
    ok = " <-- OK" if cat == "sunflower" else ""
    print("  %2d. [%s] dist=%.4f%s" % (i, cat, r["distance"], ok))
print("  Precision sunflower: %d/15" % correct3)

# Resume
print("\n" + "=" * 50)
print("RESUME")
print("=" * 50)
print("  Temps moyen par recherche : %.1fms" % ((t_search + t_search2 + t_search3) / 3 * 1000))
print("  Precision butterfly  : %d/15" % correct)
print("  Precision airplanes  : %d/15" % correct2)
print("  Precision sunflower  : %d/15" % correct3)
print("  Precision moyenne    : %.0f%%" % ((correct + correct2 + correct3) / 45 * 100))
