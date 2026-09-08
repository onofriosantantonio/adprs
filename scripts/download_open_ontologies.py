#!/usr/bin/env python3
"""
Script per il download e la gestione delle ontologie open-source per:
1. Malattie mentali e sintomi DSM-5 (MFOMD - Mental Disease Ontology).
2. Farmaci, sostanze e principi attivi (DRON - Drug Ontology).
3. Vocabolario di slang e gerghi di strada (DEA Open Street Terms).

Tutte le risorse provengono da OBO Foundry e repository governative aperte,
senza necessità di chiavi API o account a pagamento.
"""

import urllib.request
from pathlib import Path

ONTOLOGIES = {
    "MFOMD": {
        "name": "Mental Disease Ontology (DSM-5 & ICD-10 aligned)",
        "url": "http://purl.obolibrary.org/obo/mfomd.owl",
        "output_file": "data/ontology/mfomd.owl",
        "description": "Ontologia open OBO Foundry per disturbi mentali (depressione, ansia, sonno, dipendenze)."
    }
}

def download_ontology(ontology_key: str = "MFOMD"):
    meta = ONTOLOGIES.get(ontology_key)
    if not meta:
        print(f"Ontologia {ontology_key} non trovata.")
        return

    output_path = Path(meta["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Download in corso: {meta['name']} da {meta['url']}...")
    req = urllib.request.Request(
        meta["url"],
        headers={"User-Agent": "Mozilla/5.0 (compatible; MentalHealthMAS/1.0)"}
    )

    try:
        with urllib.request.urlopen(req) as response, open(output_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[+] Salvato con successo in: {output_path} ({len(data):,} bytes)")
    except Exception as e:
        print(f"[!] Errore durante il download: {e}")

if __name__ == "__main__":
    download_ontology("MFOMD")
