"""
Load extracted KnowledgeUnit JSONL into LanceDB with metadata columns so
retrieval can filter by knowledge_type / session / instrument / testability /
epistemic_status BEFORE semantic ranking.

Embeddings via Ollama (nomic-embed-text or bge-*) — local, free, the "routine"
work that belongs local.

Run after ingest.py. Requires: pip install lancedb pyarrow
"""

import json
import glob
from pathlib import Path
from typing import List

import requests


def embed_ollama(texts: List[str], model: str = "nomic-embed-text",
                 url: str = "http://localhost:11434") -> List[List[float]]:
    out = []
    for t in texts:
        r = requests.post(f"{url}/api/embeddings",
                          json={"model": model, "prompt": t}, timeout=120)
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out


def load_units(units_dirs):
    """Load all units from one or more dirs. Accepts a str or list of dirs."""
    if isinstance(units_dirs, str):
        units_dirs = [units_dirs]
    rows = []
    for units_dir in units_dirs:
        for fp in glob.glob(str(Path(units_dir) / "*.jsonl")):
            for line in Path(fp).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def build_lancedb(units_dirs, db_path: str = "./knowledge.lancedb",
                  table: str = "knowledge", embed_model: str = "nomic-embed-text"):
    # embed_model should match config.OllamaConfig.embed_model. Pull it first:
    #   ollama pull nomic-embed-text   (or bge-m3)
    import lancedb

    rows = load_units(units_dirs)
    if not rows:
        print("No units found.")
        return

    # flatten to columns LanceDB can filter on
    records, texts = [], []
    for r in rows:
        meta, prov = r["metadata"], r["provenance"]
        # reconstruct retrieval text (summary + concepts + payload)
        payload = next((r[k] for k in
                        ("setup", "contextual", "framework", "tip", "psychology", "anecdote")
                        if r.get(k)), None)
        rt = r["summary"]
        if meta.get("concepts_canonical"):
            rt += " | " + ", ".join(meta["concepts_canonical"])
        if payload:
            rt += " | " + json.dumps(payload)
        texts.append(rt)

        records.append({
            "unit_id": r["unit_id"],
            "summary": r["summary"],
            "verbatim_anchor": r.get("verbatim_anchor") or "",
            "knowledge_type": meta["knowledge_type"],
            "testability": meta["testability"],
            "epistemic_status": meta["epistemic_status"],
            "sessions": ",".join(meta.get("session_applicability", [])),
            "instruments": ",".join(meta.get("instrument_applicability", [])),
            "concepts": ",".join(meta.get("concepts_canonical", [])),
            "confidence": meta.get("extraction_confidence", 0.0),
            "source_file": prov["source_file"],
            "session_date": str(prov.get("session_date") or ""),
            "timestamp_range": prov.get("timestamp_range") or "",
            "retrieval_text": rt,
            "full_json": json.dumps(r),
        })

    print(f"Embedding {len(texts)} units with {embed_model} ...")
    vectors = embed_ollama(texts, model=embed_model)
    for rec, v in zip(records, vectors):
        rec["vector"] = v

    db = lancedb.connect(db_path)
    if table in db.table_names():
        db.drop_table(table)
    db.create_table(table, data=records)
    print(f"Wrote {len(records)} units to {db_path}::{table}")


def search(query: str, db_path: str = "./knowledge.lancedb", table: str = "knowledge",
           embed_model: str = "nomic-embed-text", k: int = 8,
           knowledge_type: str = None, session: str = None,
           testability: str = None, min_confidence: float = 0.0):
    """Metadata-filtered semantic search. Filters applied BEFORE vector ranking."""
    import lancedb
    db = lancedb.connect(db_path)
    tbl = db.open_table(table)
    qvec = embed_ollama([query], model=embed_model)[0]

    q = tbl.search(qvec).metric("cosine")
    conds = []
    if knowledge_type:
        conds.append(f"knowledge_type = '{knowledge_type}'")
    if session:
        conds.append(f"(sessions LIKE '%{session}%' OR sessions LIKE '%any%')")
    if testability:
        conds.append(f"testability = '{testability}'")
    if min_confidence:
        conds.append(f"confidence >= {min_confidence}")
    if conds:
        q = q.where(" AND ".join(conds), prefilter=True)
    return q.limit(k).to_list()
