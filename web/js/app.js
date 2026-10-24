// Client web : appelle l'API et affiche les resultats.
const $ = (id) => document.getElementById(id);
const MAX_BYTES = 10 * 1024 * 1024;
let queryImage = null; // id d'une image de la base (clic sur un resultat)
let uploadFile = null; // fichier envoye depuis l'ordinateur
let previewUrl = null;

function setStatus(message, isError = false) {
    $("status").textContent = message;
    $("status").classList.toggle("error", isError);
}

function applyMode() {
    const mode = $("mode").value;
    $("textbox").hidden = mode === "image";
    $("imagebox").hidden = mode === "text";
    $("descs").hidden = mode === "text";
}

function showQuery(src, label) {
    $("qimg").src = src;
    $("qname").textContent = label;
    $("query").hidden = false;
    $("drop").hidden = true;
}

function clearQuery() {
    queryImage = null;
    uploadFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    $("file").value = "";
    $("query").hidden = true;
    $("drop").hidden = false;
}

function handleFile(file) {
    if (!file) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) {
        setStatus("Format non accepte : utilisez PNG ou JPEG.", true);
        return;
    }
    if (file.size > MAX_BYTES) {
        setStatus("Image trop lourde (10 Mo maximum).", true);
        return;
    }
    clearQuery();
    uploadFile = file;
    previewUrl = URL.createObjectURL(file);
    showQuery(previewUrl, file.name);
    setStatus("");
}

function render(results) {
    const grid = $("results");
    grid.replaceChildren();
    for (const hit of results) {
        const card = document.createElement("figure");
        card.className = "item";

        const btn = document.createElement("button");
        btn.type = "button";
        btn.title = "Rechercher les images similaires";
        const img = document.createElement("img");
        img.src = hit.url;
        img.alt = hit.id;
        img.loading = "lazy";
        btn.appendChild(img);
        btn.addEventListener("click", () => {
            clearQuery();
            queryImage = hit.id;
            showQuery(hit.url, hit.category || hit.id);
            // Basculer en mode image si on etait en texte
            if ($("mode").value === "text") {
                $("mode").value = "image";
                applyMode();
            }
            runSearch();
            window.scrollTo({ top: 0, behavior: "smooth" });
        });

        const info = document.createElement("div");
        const cat = document.createElement("strong");
        cat.textContent = hit.category || hit.id;
        const detail = document.createElement("span");
        if (hit.distance !== null && hit.distance !== undefined) {
            detail.textContent = "distance " + hit.distance.toFixed(4);
        } else if (hit.score !== null && hit.score !== undefined) {
            detail.textContent = "score " + hit.score.toFixed(2);
        }
        info.append(cat, detail);
        card.append(btn, info);
        grid.appendChild(card);
    }
}

async function runSearch() {
    const mode = $("mode").value;
    const k = Number($("k").value) || 20;
    const dataset = $("dataset").value;
    const descriptors = Array.from(
        document.querySelectorAll("#descs input:checked"),
        (input) => input.value
    );

    if (mode === "text") {
        // Recherche textuelle
        const text = $("text").value.trim();
        if (!text) return setStatus("Saisissez des mots-cles.", true);

        setStatus("Recherche en cours...");
        try {
            const res = await fetch("/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dataset, text, k }),
            });
            const data = await res.json();
            if (!res.ok) {
                render([]);
                return setStatus(typeof data.detail === "string" ? data.detail : "Erreur.", true);
            }
            render(data.results);
            setStatus(data.count + " resultat(s)");
        } catch (err) {
            render([]);
            setStatus("Impossible de joindre l'API. Est-elle demarree ?", true);
        }

    } else {
        // Recherche visuelle
        if (!queryImage && !uploadFile) {
            return setStatus("Choisissez une image : upload ou clic sur un resultat.", true);
        }
        if (descriptors.length === 0) {
            return setStatus("Selectionnez au moins un descripteur.", true);
        }

        setStatus("Recherche en cours...");
        try {
            let res;
            if (uploadFile) {
                // Upload depuis l'ordinateur
                const fd = new FormData();
                fd.append("file", uploadFile);
                fd.append("dataset", dataset);
                fd.append("k", String(k));
                fd.append("descriptors", descriptors.join(","));
                fd.append("text", "");
                res = await fetch("/search/upload", { method: "POST", body: fd });
            } else {
                // Image de la base (clic sur un resultat)
                res = await fetch("/search", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        dataset,
                        query_image_id: queryImage,
                        descriptors,
                        k,
                    }),
                });
            }
            const data = await res.json();
            if (!res.ok) {
                render([]);
                return setStatus(typeof data.detail === "string" ? data.detail : "Erreur.", true);
            }
            render(data.results);
            setStatus(data.count + " resultat(s)");
        } catch (err) {
            render([]);
            setStatus("Impossible de joindre l'API. Est-elle demarree ?", true);
        }
    }
}

// Event listeners
$("mode").addEventListener("change", applyMode);
$("k").addEventListener("input", () => ($("kout").textContent = $("k").value));
$("form").addEventListener("submit", (e) => {
    e.preventDefault();
    runSearch();
});
$("clear").addEventListener("click", clearQuery);
$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    handleFile(e.dataTransfer.files[0]);
});
applyMode();