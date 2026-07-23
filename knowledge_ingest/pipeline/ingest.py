"""
Ingestion orchestrator.

Borrows the user's proven skeleton: batch loop, resume/skip with sanity check,
per-stage artifact saving, retry. Replaces the "extract everything into prose"
core with: SEGMENT -> CLASSIFY -> (typed) EXTRACT -> validated KnowledgeUnit JSON.

Two primary outputs per source, both fed from one pass:
  - units/<stem>.jsonl       : all KnowledgeUnit records (feeds vector store + registry)
  - notes/<stem>.md          : human-readable rollup (nice-to-have, like the old output)

Plus intermediate artifacts (segments/, classified/) for debugging + resume.
"""

import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, date
from typing import List, Optional

from pydantic import ValidationError

from ..config.config import PipelineConfig
from ..schema.models import (
    KnowledgeUnit, KnowledgeMetadata, Provenance,
    SetupPayload, ContextualPayload, FrameworkPayload,
    TipPayload, PsychologyPayload, AnecdotePayload,
    KnowledgeType, Testability, EpistemicStatus, Session, Instrument,
)
from ..vocab.ict_vocabulary import map_to_canonical
from .ollama_client import OllamaClient
from . import prompts
from .prompt_builder import resolve_active_profile

_TS = re.compile(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]')

_PAYLOAD_CLASSES = {
    KnowledgeType.SETUP: SetupPayload,
    KnowledgeType.CONTEXTUAL: ContextualPayload,
    KnowledgeType.FRAMEWORK: FrameworkPayload,
    KnowledgeType.TIP: TipPayload,
    KnowledgeType.PSYCHOLOGY: PsychologyPayload,
    KnowledgeType.ANECDOTE: AnecdotePayload,
}
_PAYLOAD_ATTR = {
    KnowledgeType.SETUP: "setup",
    KnowledgeType.CONTEXTUAL: "contextual",
    KnowledgeType.FRAMEWORK: "framework",
    KnowledgeType.TIP: "tip",
    KnowledgeType.PSYCHOLOGY: "psychology",
    KnowledgeType.ANECDOTE: "anecdote",
}


class IngestPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.client = OllamaClient(cfg.ollama)
        self.out = Path(cfg.output_dir)
        for sub in ("segments", "classified", "units", "notes", "raw"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)

        # Prompt profile resolution (DESIGN.md §9 Phase 2). The active Profile
        # supplies the classify/extract system+user prompts and the domain tag
        # stamped onto every extracted KnowledgeMetadata.domains. Resolution
        # prefers cfg.profile, then falls back to the legacy cfg.ict_aware flag,
        # then to the "ict" default. See pipeline/prompt_builder.py.
        self._profile = resolve_active_profile(cfg)
        self._classify_prompt = self._profile.classify_prompt
        self._classify_system = self._profile.classify_system
        self._classify_batch_prompt_fn = self._profile.classify_batch_prompt_fn
        self._extract_system = self._profile.extract_system
        self._extract_prompt_fn = self._profile.extract_prompt_fn
        self._extract_batch_prompt_fn = self._profile.extract_batch_prompt_fn
        # the domains this run stamps onto every unit (used in _build_unit)
        self._domains = list(self._profile.domains)
        print(f"  prompt profile: {self._profile.name} (domains={self._profile.domains})")

    # ---- resume (borrowed pattern, adapted to JSONL output) -------------- #
    def _already_done(self, stem: str) -> bool:
        f = self.out / "units" / f"{stem}.jsonl"
        if not f.exists() or f.stat().st_size < 50:
            return False
        # ensure last line parses -> not a half-written file
        try:
            last = f.read_text(encoding="utf-8").strip().splitlines()[-1]
            json.loads(last)
            return True
        except Exception:
            return False

    # ---- helpers --------------------------------------------------------- #
    @staticmethod
    def _parse_date_from_name(stem: str) -> Optional[date]:
        m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', stem)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = re.search(r'(\d{2})_?([A-Za-z]{3})[_ ](\d{1,2})[_ ](\d{4})', stem)
        if m:
            try:
                return datetime.strptime(f"{m.group(2)} {m.group(3)} {m.group(4)}", "%b %d %Y").date()
            except ValueError:
                return None
        return None

    def _char_prechunk(self, text: str) -> List[str]:
        """Coarse split before segmentation so we never send a huge doc to the LM.
        Splits on timestamp lines, respecting max_segment_chars*2 windows."""
        if len(text) <= self.cfg.max_segment_chars * 2:
            return [text]
        lines = text.split("\n")
        chunks, cur, size = [], [], 0
        window = self.cfg.max_segment_chars * 2
        for ln in lines:
            if size + len(ln) > window and cur:
                chunks.append("\n".join(cur))
                cur, size = [], 0
            cur.append(ln)
            size += len(ln) + 1
        if cur:
            chunks.append("\n".join(cur))
        return chunks

    def _uid(self, stem: str, chunk_i: int, ordinal: int) -> str:
        h = hashlib.sha1(f"{stem}:{chunk_i}:{ordinal}".encode()).hexdigest()[:10]
        return f"{stem}__{chunk_i:02d}_{ordinal:02d}_{h}"

    # ---- stage 1: segment ------------------------------------------------ #
    def _segment(self, chunk: str) -> List[dict]:
        # prose sources (blog/pdf/own_doc/markdown) segment on headers/topics;
        # transcripts segment on timestamps.
        prose = self.cfg.source_type in ("blog", "pdf", "own_doc", "markdown")
        prompt = (prompts.SEGMENT_PROSE_PROMPT if prose
                  else prompts.SEGMENT_PROMPT).format(chunk=chunk)
        raw = self.client.generate(
            prompt,
            model=self.cfg.ollama.segmenter_model,
            system=prompts.SEGMENT_SYSTEM,
            temperature=0.0,
            num_ctx=self.cfg.ollama.segmenter_num_ctx,
            json_mode=True,
        )
        try:
            segs = self.client.parse_json(raw)
            return segs if isinstance(segs, list) else []
        except Exception:
            # fallback: treat whole chunk as one segment rather than lose it
            tss = _TS.findall(chunk)
            return [{"start_ts": tss[0] if tss else None,
                     "end_ts": tss[-1] if tss else None,
                     "text": chunk}]

    # ---- stage 2: classify ---------------------------------------------- #
    def _classify(self, unit_text: str) -> Optional[dict]:
        raw = self.client.generate(
            self._classify_prompt.format(unit=unit_text),
            model=self.cfg.ollama.classifier_model,
            system=self._classify_system,
            temperature=self.cfg.ollama.classifier_temperature,
            num_ctx=self.cfg.ollama.classifier_num_ctx,
            json_mode=True,
        )
        try:
            return self.client.parse_json(raw)
        except Exception:
            return None

    # ---- stage 3: extract ------------------------------------------------ #
    def _extract(self, ktype: str, unit_text: str) -> Optional[dict]:
        raw = self.client.generate(
            self._extract_prompt_fn(ktype, unit_text),
            model=self.cfg.ollama.extractor_model,
            system=self._extract_system,
            temperature=self.cfg.ollama.extractor_temperature,
            num_ctx=self.cfg.ollama.extractor_num_ctx,
            num_predict=self.cfg.ollama.extractor_num_predict,
            json_mode=True,
        )
        try:
            return self.client.parse_json(raw)
        except Exception:
            return None

    # ---- stage 2b: classify BATCH (all segments of a file, one call) ----- #
    def _classify_batch(self, unit_texts: List[str]) -> List[Optional[dict]]:
        numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(unit_texts))
        raw = self.client.generate(
            self._classify_batch_prompt_fn(numbered),
            model=self.cfg.ollama.classifier_model,
            system=self._classify_system,
            temperature=self.cfg.ollama.classifier_temperature,
            num_ctx=self.cfg.ollama.classifier_num_ctx,
            num_predict=self.cfg.ollama.extractor_num_predict,
            json_mode=True,
        )
        out: List[Optional[dict]] = [None] * len(unit_texts)
        try:
            arr = self.client.parse_json(raw)
            for obj in arr:
                idx = obj.get("idx")
                if isinstance(idx, int) and 0 <= idx < len(out):
                    out[idx] = obj
        except Exception:
            pass
        # fill any gaps with single-unit fallback so nothing is silently dropped
        for i, o in enumerate(out):
            if o is None:
                out[i] = self._classify(unit_texts[i])
        return out

    # ---- stage 3b: extract BATCH PER TYPE (shared session context) ------- #
    def _extract_batch(self, ktype: str, unit_texts: List[str]) -> List[Optional[dict]]:
        """Extract a batch of SAME-TYPE units; chunked by extract_batch_size."""
        results: List[Optional[dict]] = [None] * len(unit_texts)
        bs = self.cfg.ollama.extract_batch_size
        for start in range(0, len(unit_texts), bs):
            sub = unit_texts[start:start + bs]
            numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(sub))
            raw = self.client.generate(
                self._extract_batch_prompt_fn(ktype, numbered),
                model=self.cfg.ollama.extractor_model,
                system=self._extract_system,
                temperature=self.cfg.ollama.extractor_temperature,
                num_ctx=self.cfg.ollama.extractor_num_ctx,
                num_predict=self.cfg.ollama.extractor_num_predict,
                json_mode=True,
            )
            try:
                arr = self.client.parse_json(raw)
                if isinstance(arr, list):
                    extra = sum(1 for obj in arr
                                if not (isinstance(obj, dict)
                                        and isinstance(obj.get("idx"), int)
                                        and 0 <= obj.get("idx") < len(sub)))
                    if extra:
                        print(f"      ! {ktype} batch returned {extra} objects with "
                              f"out-of-range/missing idx (hallucinated) — ignored")
                for obj in arr if isinstance(arr, list) else [arr]:
                    if not isinstance(obj, dict):
                        continue
                    idx = obj.get("idx")
                    if isinstance(idx, int) and 0 <= idx < len(sub):
                        results[start + idx] = obj
            except Exception:
                pass
            # per-unit fallback for any gaps in this sub-batch
            for j in range(len(sub)):
                if results[start + j] is None:
                    results[start + j] = self._extract(ktype, sub[j])
        return results
    def _build_unit(self, stem, chunk_i, ordinal, seg, cls, ext, sess_date) -> Optional[KnowledgeUnit]:
        try:
            ktype = KnowledgeType(cls["knowledge_type"])
        except (KeyError, ValueError):
            return None

        raw_concepts = cls.get("concepts_raw", []) or []
        canon = map_to_canonical(raw_concepts)

        def _enum_list(vals, enum, default):
            out = []
            for v in (vals or []):
                try:
                    out.append(enum(v))
                except ValueError:
                    pass
            return out or [default]

        meta = KnowledgeMetadata(
            knowledge_type=ktype,
            testability=Testability(cls.get("testability", "not_testable"))
                if cls.get("testability") in {t.value for t in Testability} else Testability.NOT_TESTABLE,
            epistemic_status=EpistemicStatus.UNVALIDATED,
            domains=self._domains,
            session_applicability=_enum_list(cls.get("session_applicability"), Session, Session.ANY),
            instrument_applicability=_enum_list(cls.get("instrument_applicability"), Instrument, Instrument.ANY),
            concepts_raw=raw_concepts,
            concepts_canonical=canon,
            inferred_fields=(ext or {}).get("inferred_fields", []) if ext else [],
            extraction_confidence=float((ext or {}).get("extraction_confidence", 0.0)) if ext else 0.0,
        )

        prov = Provenance(
            source_file=f"{stem}",
            source_type=self.cfg.source_type,
            source_credibility=self.cfg.source_credibility,
            session_date=sess_date,
            speaker=self.cfg.default_speaker,
            chunk_id=f"{stem}:{chunk_i}",
            timestamp_range=(f"{seg.get('start_ts')} - {seg.get('end_ts')}"
                             if seg.get("start_ts") else None),
            extractor_model=self.cfg.ollama.extractor_model,
            extracted_at=datetime.now().isoformat(timespec="seconds"),
        )

        unit_kwargs = dict(
            unit_id=self._uid(stem, chunk_i, ordinal),
            summary=(ext or {}).get("summary") or seg["text"][:160],
            verbatim_anchor=(ext or {}).get("verbatim_anchor"),
            metadata=meta,
            provenance=prov,
        )

        # attach typed payload
        if ext and ext.get("payload") is not None:
            payload_cls = _PAYLOAD_CLASSES[ktype]
            try:
                payload = payload_cls(**{k: v for k, v in ext["payload"].items()
                                         if k in payload_cls.model_fields})
                unit_kwargs[_PAYLOAD_ATTR[ktype]] = payload
            except ValidationError as e:
                print(f"      ! payload validation failed for {ktype}: {str(e)[:100]}")
                print(f"        payload keys: {list(ext['payload'].keys())}")
        else:
            print(f"      ! no payload for {ktype} — skipping unit")
            return None

        try:
            return KnowledgeUnit(**unit_kwargs)
        except ValidationError as e:
            print(f"      ! unit validation failed: {str(e)[:120]}")
            return None

    # ---- process one file ------------------------------------------------ #
    def process_file(self, fp: Path) -> dict:
        stem = fp.stem
        if self.cfg.skip_existing and self._already_done(stem):
            print(f"  skip (done): {fp.name}")
            return {"file": fp.name, "skipped": True}

        text = fp.read_text(encoding="utf-8", errors="replace")
        if len(text) < 100:
            return {"file": fp.name, "skipped": True, "reason": "too short"}

        (self.out / "raw" / f"{stem}.txt").write_text(text, encoding="utf-8")
        sess_date = self._parse_date_from_name(stem)

        # --- Stage 1: segment. Large cloud window -> whole file in one call. #
        prechunks = self._char_prechunk(text)
        segments: List[dict] = []
        for chunk in prechunks:
            segments.extend(self._segment(chunk))

        # --- collapse guard: a long file yielding too-few segments means the
        #     segmenter gave up and returned one blob. Force windowed re-seg. -- #
        total_chars = len(text)
        expected_min = max(4, total_chars // self.cfg.resegment_chars_per_unit)
        if total_chars > self.cfg.resegment_min_chars and len(segments) < expected_min:
            print(f"  ! segmenter collapse ({len(segments)} segs for {total_chars} "
                  f"chars, expected >={expected_min}) — forcing windowed re-segment")
            segments = []
            win = self.cfg.resegment_forced_window
            buf, size, windows = [], 0, []
            for ln in text.split("\n"):
                buf.append(ln); size += len(ln) + 1
                if size >= win:
                    windows.append("\n".join(buf)); buf, size = [], 0
            if buf:
                windows.append("\n".join(buf))
            for w in windows:
                segments.extend(self._segment(w))

        # keep only substantive segments, remember original ordinal
        seg_items = [(oi, s) for oi, s in enumerate(segments, 1)
                     if len(s.get("text", "").strip()) >= 40]
        seg_texts = [s["text"].strip() for _, s in seg_items]
        print(f"  {len(seg_texts)} segments")

        # --- Stage 2: classify all segments in one batched call ----------- #
        classifications = self._classify_batch(seg_texts) if seg_texts else []

        # --- group units by type for batched, context-sharing extraction -- #
        # index i refers to position in seg_items / seg_texts / classifications
        by_type: dict = {}          # ktype -> list of indices to extract
        skipped_idx: set = set()    # indices we won't extract (still become units)
        for i, cls in enumerate(classifications):
            if not cls:
                skipped_idx.add(i)
                continue
            ktype = cls.get("knowledge_type")
            worthwhile = cls.get("extraction_worthwhile", True)
            if (ktype in self.cfg.skip_extract_types) or (not worthwhile) or (ktype is None):
                skipped_idx.add(i)
            else:
                by_type.setdefault(ktype, []).append(i)

        # --- Stage 3: batched extraction per type ------------------------- #
        extractions: List[Optional[dict]] = [None] * len(seg_texts)
        for ktype, idxs in by_type.items():
            print(f"  extract {ktype}: {len(idxs)} units")
            texts = [seg_texts[i] for i in idxs]
            ext_batch = self._extract_batch(ktype, texts)
            for local_i, global_i in enumerate(idxs):
                extractions[global_i] = ext_batch[local_i]

        # --- assemble validated KnowledgeUnits ---------------------------- #
        units: List[KnowledgeUnit] = []
        cls_log = []
        for i, (oi, seg) in enumerate(seg_items):
            cls = classifications[i]
            if not cls:
                continue
            cls_log.append({"ordinal": oi, "cls": cls})
            unit = self._build_unit(stem, 1, oi, seg, cls, extractions[i], sess_date)
            if unit:
                units.append(unit)

        # write artifacts
        (self.out / "segments" / f"{stem}.json").write_text(
            json.dumps(segments, indent=2), encoding="utf-8")
        (self.out / "classified" / f"{stem}.json").write_text(
            json.dumps(cls_log, indent=2), encoding="utf-8")

        jsonl = self.out / "units" / f"{stem}.jsonl"
        # confidence gate: low-conf units are likely segmentation fragments.
        # Log them for review but keep them out of the vector-store JSONL.
        thr = self.cfg.min_unit_confidence
        low_conf = [u for u in units if u.metadata.extraction_confidence < thr]
        keep = [u for u in units if u.metadata.extraction_confidence >= thr]
        if low_conf:
            (self.out / "classified" / f"{stem}_lowconf.json").write_text(
                json.dumps([u.model_dump(mode="json") for u in low_conf],
                           indent=2, default=str),
                encoding="utf-8")
            print(f"  {len(low_conf)} low-confidence units held back for review")
        with open(jsonl, "w", encoding="utf-8") as f:
            for u in keep:
                f.write(u.model_dump_json() + "\n")

        self._write_notes(stem, keep)
        print(f"  done: {len(keep)} units ({len(low_conf)} held back) -> {jsonl.name}")
        return {"file": fp.name, "skipped": False,
                "units": len(keep), "held_back": len(low_conf)}

    def _write_notes(self, stem, units: List[KnowledgeUnit]):
        by_type = {}
        for u in units:
            by_type.setdefault(u.metadata.knowledge_type.value, []).append(u)
        lines = [f"# {stem} — extracted knowledge\n",
                 f"_{len(units)} units, generated {datetime.now():%Y-%m-%d %H:%M}_\n"]
        for t, us in by_type.items():
            lines.append(f"\n## {t} ({len(us)})\n")
            for u in us:
                conf = u.metadata.extraction_confidence
                canon = ", ".join(u.metadata.concepts_canonical) or "—"
                lines.append(f"- **{u.summary}**  \n"
                             f"  _{canon}_ · conf {conf:.2f} · "
                             f"{u.provenance.timestamp_range or ''}")
        (self.out / "notes" / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- batch ----------------------------------------------------------- #
    def run(self):
        in_dir = Path(self.cfg.input_dir)
        if self.cfg.file_filter:
            files = sorted(in_dir.glob(self.cfg.file_filter))
        else:
            files = sorted(list(in_dir.glob("*.txt")) + list(in_dir.glob("*.md")))
        if self.cfg.max_files is not None:
            files = files[:self.cfg.max_files]
            print(f"Ingesting {len(files)} files (limited by max_files) from {in_dir}")
        else:
            print(f"Ingesting {len(files)} files from {in_dir}")
        summary = []
        for i, fp in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {fp.name}")
            try:
                summary.append(self.process_file(fp))
            except Exception as e:
                print(f"  ERROR: {str(e)[:120]}")
                summary.append({"file": fp.name, "error": str(e)})
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        done = sum(1 for s in summary if s.get("units"))
        print(f"\nComplete. {done} files produced units. Summary -> {self.out/'summary.json'}")
        return summary
