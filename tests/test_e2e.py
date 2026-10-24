"""
Test E2E v2 : lance serveur, teste recherche par image de la base + upload.
"""
import sys, os, json, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from urllib.request import Request, urlopen

PORT = 8766

def start_server():
    from api.main import app
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="error")

t = threading.Thread(target=start_server, daemon=True)
t.start()
for _ in range(30):
    try:
        urlopen("http://127.0.0.1:%d/health" % PORT)
        break
    except Exception:
        time.sleep(0.5)
else:
    print("ERREUR: serveur ne demarre pas!")
    sys.exit(1)

print("Serveur OK sur port %d" % PORT)

def api_post(path, body_dict):
    data = json.dumps(body_dict).encode()
    req = Request(
        "http://127.0.0.1:%d%s" % (PORT, path),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urlopen(req)
    return json.loads(resp.read())

def upload_search(img_path, k=15):
    boundary = "----FormBoundary7MA4YWxk"
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    parts = []
    parts.append(("--%s\r\n" % boundary).encode())
    parts.append(b'Content-Disposition: form-data; name="file"; filename="q.jpg"\r\n')
    parts.append(b'Content-Type: image/jpeg\r\n\r\n')
    parts.append(img_bytes)
    parts.append(b'\r\n')
    for name, val in [("dataset", "objects"), ("k", str(k)), ("descriptors", "auto"), ("text", "")]:
        parts.append(("--%s\r\n" % boundary).encode())
        parts.append(('Content-Disposition: form-data; name="%s"\r\n\r\n%s\r\n' % (name, val)).encode())
    parts.append(("--%s--\r\n" % boundary).encode())
    body = b"".join(parts)
    req = Request(
        "http://127.0.0.1:%d/search/upload" % PORT,
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary},
        method="POST",
    )
    resp = urlopen(req)
    return json.loads(resp.read())

def test_search(label, expected_cat, img_id, k=15, mode="base"):
    t0 = time.time()
    if mode == "base":
        resp = api_post("/search", {
            "dataset": "objects",
            "query_image_id": img_id,
            "descriptors": ["auto"],
            "k": k,
        })
    else:
        img_path = os.path.join("data", "objects", img_id.replace("/", os.sep))
        resp = upload_search(img_path, k=k)
    elapsed = (time.time() - t0) * 1000
    hits = resp["results"]
    n_ok = sum(1 for h in hits[:k] if h.get("category", "") == expected_cat)
    cats = [h.get("category", "?") for h in hits[:k]]
    print("  [%s] %s  prec=%d/%d  time=%.0fms  %s" % (
        label, mode.upper(), n_ok, k, elapsed,
        "cats=%s" % cats
    ))
    return n_ok, k

# ── Tests ─────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST COMPLET - RECHERCHE PAR IMAGE (BASE) + UPLOAD")
print("=" * 70)

total_ok, total_k = 0, 0

tests = [
    ("butterfly",  "butterfly/image_0001.jpg"),
    ("airplanes",  "airplanes/image_0001.jpg"),
    ("sunflower",  "sunflower/image_0001.jpg"),
    ("Faces",      "Faces/image_0001.jpg"),
    ("Motorbikes", "Motorbikes/image_0001.jpg"),
    ("car_side",   "car_side/image_0001.jpg"),
    ("accordion",  "accordion/image_0001.jpg"),
    ("garfield",   "garfield/image_0001.jpg"),
]

print("\n--- Recherche depuis la base (image connue) ---")
for cat, img_id in tests:
    path = os.path.join("data", "objects", img_id.replace("/", os.sep))
    if not os.path.exists(path):
        print("  [%s] SKIP (not found)" % cat)
        continue
    ok, k = test_search(cat, cat, img_id, k=15, mode="base")
    total_ok += ok
    total_k += k

print("\n--- Recherche par UPLOAD (image differente) ---")
# Use image_0005 for upload tests (different from image_0001 used in base)
upload_tests = [
    ("butterfly",  "butterfly/image_0005.jpg"),
    ("airplanes",  "airplanes/image_0010.jpg"),
    ("sunflower",  "sunflower/image_0005.jpg"),
    ("Faces",      "Faces/image_0005.jpg"),
    ("Motorbikes", "Motorbikes/image_0005.jpg"),
    ("car_side",   "car_side/image_0005.jpg"),
    ("accordion",  "accordion/image_0005.jpg"),
]
for cat, img_id in upload_tests:
    path = os.path.join("data", "objects", img_id.replace("/", os.sep))
    if not os.path.exists(path):
        print("  [%s] SKIP (not found)" % cat)
        continue
    ok, k = test_search(cat, cat, img_id, k=15, mode="upload")
    total_ok += ok
    total_k += k

print()
print("=" * 70)
print("PRECISION GLOBALE : %d/%d = %.0f%%" % (total_ok, total_k, 100 * total_ok / total_k if total_k > 0 else 0))
print("=" * 70)
