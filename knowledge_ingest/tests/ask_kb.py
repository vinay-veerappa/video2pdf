"""
RAG (Retrieval-Augmented Generation) interface for the ICT knowledge base.

This is the practical payoff: instead of re-explaining ICT concepts or re-watching
videos, you ask a question and get an answer grounded in the actual source material
with citations back to the exact transcript/PDF.

Usage:
  python -m knowledge_ingest.tests.ask_kb "What is the Sharp Turn entry model?"
  python -m knowledge_ingest.tests.ask_kb "How does Kish use CSD for entries?"
  python -m knowledge_ingest.tests.ask_kb "Explain the Power of Three" --sources
  python -m knowledge_ingest.tests.ask_kb "What are the 7 Rules?" --no-llm

The flow:
  1. Your question → embedded → semantic search against LanceDB
  2. Top-K relevant units retrieved (with provenance: who said it, where, when)
  3. Units + your question → sent to LLM (deepseek-v4-flash:cloud)
  4. LLM synthesizes an answer using ONLY the retrieved context
  5. Answer includes citations: "Kish said this in the May 11 2023 transcript (00:15:00)"

This means:
  - You don't re-watch 335 videos to find "that one thing about the 9:12 macro"
  - You don't manually explain ICT concepts — the KB already has them
  - Answers are GROUNDED (not hallucinated) — every claim traces to a source
  - An LLM assistant can use this same interface to answer YOUR questions
"""

import os, sys, json, argparse, requests, textwrap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CHART_DB = r"C:\ICT_Videos\Testing\_v4_lancedb"
LLM_MODEL = "deepseek-v4-flash:cloud"
LLM_URL = "http://localhost:11434/api/generate"


def retrieve(question, db_path=CHART_DB, k=8, knowledge_type=None, min_confidence=0.5):
    """Retrieve relevant knowledge units for a question."""
    from knowledge_ingest.pipeline.vector_store import search
    results = search(
        question, db_path=db_path, k=k,
        knowledge_type=knowledge_type,
        min_confidence=min_confidence,
    )
    return results


def format_context(results):
    """Format retrieved units as context for the LLM."""
    ctx_parts = []
    for i, r in enumerate(results, 1):
        ktype = r.get("knowledge_type", "?")
        src = r.get("source_file", "?")
        conf = r.get("confidence", 0)
        text = r.get("retrieval_text", "")
        speaker = r.get("speaker", "")
        chunk = r.get("chunk_id", "")

        # Extract payload details if available
        concepts = r.get("concepts", "")
        ctx_parts.append(
            f"[Source {i}] ({ktype}, confidence={conf:.1f}, speaker={speaker}, "
            f"source={src}, location={chunk})\n"
            f"Concepts: {concepts}\n"
            f"Content: {text[:500]}\n"
        )
    return "\n---\n".join(ctx_parts)


def synthesize(question, context, model=LLM_MODEL):
    """Use LLM to synthesize an answer from retrieved context."""
    system = (
        "You are an ICT/Smart Money Concepts trading knowledge assistant. "
        "Answer the user's question using ONLY the provided source material. "
        "If the sources don't contain enough information, say so explicitly. "
        "Cite sources by number: [Source 1], [Source 2], etc. "
        "Be precise — use the exact terminology from the sources. "
        "When a source mentions a specific time, session, or setup name, include it."
    )

    prompt = f"""QUESTION: {question}

SOURCE MATERIAL (retrieved from knowledge base):

{context}

INSTRUCTIONS:
- Answer using ONLY the above source material
- Cite sources as [Source N]
- If multiple sources say different things, note the difference
- Include specific setup names, time references, and educator attributions
- If the sources don't fully answer the question, say what IS available and what's missing

ANSWER:"""

    r = requests.post(LLM_URL, json={
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")


def ask(question, db_path=CHART_DB, k=8, knowledge_type=None,
        min_confidence=0.5, use_llm=True, show_sources=False, model=LLM_MODEL):
    """Full RAG pipeline: retrieve → synthesize → answer."""

    # Step 1: Retrieve
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    results = retrieve(question, db_path, k=k, knowledge_type=knowledge_type,
                      min_confidence=min_confidence)

    print(f"\nRetrieved {len(results)} units from knowledge base:")

    if show_sources or not use_llm:
        for i, r in enumerate(results, 1):
            ktype = r.get("knowledge_type", "?")
            src = r.get("source_file", "?")[:40]
            conf = r.get("confidence", 0)
            speaker = r.get("speaker", "")
            print(f"  [{i}] [{ktype:12s}] {src:40s} conf={conf:.1f} speaker={speaker}")
            print(f"      {r.get('retrieval_text', '')[:120]}")

    if not use_llm:
        print(f"\n(--no-llm: showing retrieved sources only, no synthesis)")
        return results

    if not results:
        print("\nNo relevant units found. Try lowering --min-conf or rephrasing.")
        return []

    # Step 2: Format context
    context = format_context(results)

    # Step 3: Synthesize
    print(f"\nSynthesizing answer with {model}...")
    answer = synthesize(question, context, model)

    # Step 4: Output
    print(f"\n{'='*60}")
    print(f"ANSWER:")
    print(f"{'='*60}")
    print()
    # Wrap for readability
    for line in answer.split("\n"):
        print(textwrap.fill(line, width=80) if line.strip() else line)

    print(f"\n{'='*60}")
    print(f"Sources used: {len(results)} units from {db_path}")
    print(f"{'='*60}")

    return results


def interactive(db_path=CHART_DB, model=LLM_MODEL):
    """Interactive REPL."""
    print(f"\nICT Knowledge Base — RAG Interface")
    print(f"DB: {db_path}")
    print(f"Model: {model}")
    print(f"Type your questions. 'sources' to toggle source display. 'quit' to exit.\n")

    show_sources = False
    k = 8

    while True:
        try:
            q = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q in ("quit", "exit", "q"):
            break
        if q == "sources":
            show_sources = not show_sources
            print(f"  show_sources: {show_sources}")
            continue
        if q.startswith("k "):
            k = int(q[2:])
            print(f"  k={k}")
            continue

        ask(q, db_path=db_path, k=k, show_sources=show_sources, model=model)
        print()


def main():
    ap = argparse.ArgumentParser(description="Ask the ICT knowledge base")
    ap.add_argument("question", nargs="?", help="your question")
    ap.add_argument("--db", default=CHART_DB)
    ap.add_argument("--k", type=int, default=8, help="number of units to retrieve")
    ap.add_argument("--type", help="filter by knowledge_type")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--no-llm", action="store_true", help="show sources only, no LLM synthesis")
    ap.add_argument("--sources", action="store_true", help="show retrieved sources")
    ap.add_argument("--model", default=LLM_MODEL)
    args = ap.parse_args()

    if args.question:
        ask(args.question, db_path=args.db, k=args.k,
            knowledge_type=args.type, min_confidence=args.min_conf,
            use_llm=not args.no_llm, show_sources=args.sources, model=args.model)
    else:
        interactive(args.db, args.model)


if __name__ == "__main__":
    main()