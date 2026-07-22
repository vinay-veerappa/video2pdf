"""
Knowledge Base API server — exposes the ICT knowledge base to any LLM or client.

This is the "hook" that connects your knowledge base to any LLM:
  - Ollama (local/cloud models you already use)
  - Copilot Chat (via MCP or HTTP)
  - Open WebUI / any chat frontend
  - Any Python script or agent

Run it:
  python -m knowledge_ingest.serve --db "C:\\ICT_Videos\\Testing\\_v4_lancedb"
  python -m knowledge_ingest.serve --port 8900

Then any LLM can query it:
  POST http://localhost:8900/ask
  {"question": "What is the Sharp Turn entry model?", "k": 8}

  POST http://localhost:8900/search
  {"query": "CSD order flow", "k": 5, "knowledge_type": "setup"}

  GET  http://localhost:8900/stats
"""

import os, sys, json, argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

CHART_DB = r"C:\ICT_Videos\Testing\_v4_lancedb"
LLM_MODEL = "deepseek-v4-flash:cloud"


class KBHandler(BaseHTTPRequestHandler):
    db_path = CHART_DB
    llm_model = LLM_MODEL

    def _send_json(self, code, data):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/stats":
            import lancedb
            db = lancedb.connect(self.db_path)
            tbl = db.open_table("knowledge")
            data = tbl.to_pandas()
            self._send_json(200, {
                "total_units": tbl.count_rows(),
                "by_type": data["knowledge_type"].value_counts().to_dict(),
                "by_source": data["source_file"].value_counts().head(10).to_dict(),
                "db_path": self.db_path,
            })
            return

        if path == "/health":
            self._send_json(200, {"status": "ok", "db": self.db_path})
            return

        self._send_json(404, {"error": f"Unknown endpoint: {path}"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        if path == "/search":
            # Raw semantic search — returns units, no LLM
            from knowledge_ingest.pipeline.vector_store import search
            results = search(
                body.get("query", ""),
                db_path=self.db_path,
                k=body.get("k", 8),
                knowledge_type=body.get("knowledge_type"),
                min_confidence=body.get("min_confidence", 0.0),
            )
            # Trim retrieval_text for API response
            for r in results:
                r["retrieval_text"] = r.get("retrieval_text", "")[:500]
            self._send_json(200, {"results": results, "count": len(results)})
            return

        if path == "/ask":
            # Full RAG: retrieve + LLM synthesize
            from knowledge_ingest.tests.ask_kb import retrieve, format_context, synthesize
            question = body.get("question", "")
            if not question:
                self._send_json(400, {"error": "Missing 'question'"})
                return

            results = retrieve(
                question, db_path=self.db_path,
                k=body.get("k", 8),
                knowledge_type=body.get("knowledge_type"),
                min_confidence=body.get("min_confidence", 0.5),
            )

            if not results:
                self._send_json(200, {
                    "answer": "No relevant units found in the knowledge base.",
                    "sources": [],
                    "question": question,
                })
                return

            context = format_context(results)
            answer = synthesize(question, context, model=self.llm_model)

            self._send_json(200, {
                "answer": answer,
                "question": question,
                "sources": [
                    {
                        "knowledge_type": r.get("knowledge_type"),
                        "source_file": r.get("source_file"),
                        "speaker": r.get("speaker"),
                        "confidence": r.get("confidence"),
                        "concepts": r.get("concepts"),
                        "chunk_id": r.get("chunk_id"),
                    }
                    for r in results
                ],
            })
            return

        self._send_json(404, {"error": f"Unknown endpoint: {path}"})

    def log_message(self, format, *args):
        # Simple logging
        print(f"  {args[0] if args else ''}")


def main():
    ap = argparse.ArgumentParser(description="ICT Knowledge Base API Server")
    ap.add_argument("--db", default=CHART_DB, help="LanceDB path")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=LLM_MODEL, help="LLM for /ask synthesis")
    args = ap.parse_args()

    KBHandler.db_path = args.db
    KBHandler.llm_model = args.model

    server = HTTPServer((args.host, args.port), KBHandler)

    print(f"\n{'='*60}")
    print(f"  ICT Knowledge Base API Server")
    print(f"{'='*60}")
    print(f"  DB:   {args.db}")
    print(f"  LLM:  {args.model}")
    print(f"  URL:  http://{args.host}:{args.port}")
    print(f"{'='*60}")
    print(f"\nEndpoints:")
    print(f"  GET  /health       — health check")
    print(f"  GET  /stats        — database statistics")
    print(f"  POST /search       — semantic search (raw units)")
    print(f"  POST /ask          — RAG (retrieve + LLM answer)")
    print(f"\nExample:")
    print(f'  curl -X POST http://{args.host}:{args.port}/ask \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"question": "What is the Sharp Turn entry?"}}\'')
    print(f"\nPress Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()