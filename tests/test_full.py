import sys; sys.path.insert(0, "src")
from cbir.search import visual_search, visual_search_image

# Test 1: visual_search (comme l'API POST /search)
print("=== Test visual_search ===")
results = visual_search("objects", "butterfly/image_0001.jpg", ["color"], k=10)
for i, r in enumerate(results[:10], 1):
    cat = r["id"].split("/")[0]
    ok = " <-- OK" if cat == "butterfly" else ""
    print("  %2d. dist=%.4f  [%s]%s" % (i, r["distance"], cat, ok))

# Test 2: visual_search_image (comme l'API POST /search/upload)
print()
print("=== Test visual_search_image ===")
with open("data/objects/butterfly/image_0001.jpg", "rb") as f:
    img_bytes = f.read()
results2 = visual_search_image("objects", img_bytes, ["color"], k=5)
for i, r in enumerate(results2[:5], 1):
    cat = r["id"].split("/")[0]
    print("  %2d. dist=%.4f  [%s]" % (i, r["distance"], cat))

# Test 3: check image IDs format
print()
print("=== Test ID format ===")
sample_id = results[0]["id"]
print("  ID format:", sample_id)
starts = sample_id.startswith("objects/")
print("  Starts with objects/?", starts, " (must be False)")
print("  URL would be: /images/objects/" + sample_id)
