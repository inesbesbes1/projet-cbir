import sys
sys.path.insert(0, "src")

from cbir.fusion import fuse_results

# Données simulées
text_results = [
    {"id": "accordion/image_0001.jpg", "score": 12.5, "category": "accordion"},
    {"id": "accordion/image_0002.jpg", "score": 10.2, "category": "accordion"},
    {"id": "piano/image_0001.jpg", "score": 8.0, "category": "piano"},
]

visual_results = [
    {"id": "accordion/image_0001.jpg", "distance": 0.05},
    {"id": "harmonica/image_0003.jpg", "distance": 0.12},
    {"id": "accordion/image_0002.jpg", "distance": 0.20},
]

print("=== Test Fusion Mode AND ===")
res_and = fuse_results(text_results, visual_results, operator="AND", k=5)
for r in res_and:
    print(f"  {r['id']} | score RRF: {r['score']:.4f} | dist: {r['distance']} | cat: {r['category']}")
assert len(res_and) == 2, f"Attendu: 2 résultats dans l'intersection, obtenu: {len(res_and)}"
assert res_and[0]["id"] == "accordion/image_0001.jpg"

print()
print("=== Test Fusion Mode OR ===")
res_or = fuse_results(text_results, visual_results, operator="OR", k=5)
for r in res_or:
    print(f"  {r['id']} | score RRF: {r['score']:.4f} | dist: {r['distance']} | cat: {r['category']}")
assert len(res_or) == 4, f"Attendu: 4 résultats dans l'union, obtenu: {len(res_or)}"
assert res_or[0]["id"] == "accordion/image_0001.jpg"

print()
print("Tous les tests de fusion sont validés avec succès !")
