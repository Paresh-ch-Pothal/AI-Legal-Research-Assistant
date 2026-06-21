import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from document_extractor import LegalDocumentExtractor
from embeddings import InLegalBERTEmbeddings
from indian_kannon_client import IndianKanoonClient
from rag_chains import (
    build_combined_chain,
    build_qa_chain,
    build_similar_verdicts_chain,
    build_summary_chain,
    format_docs_for_prompt,
)
from vector_store import DualVectorStore
import numpy as np
import re

def _clean_snippet(raw_html: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', raw_html or "")
    return re.sub(r'\s+', ' ', text).strip()

def _cosine_sim(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom else 0.0

def _build_reference_text(reference_doc: Dict) -> str:
    return (
        f"{reference_doc.get('court_name','')} "
        f"{reference_doc.get('petitioner','')} vs {reference_doc.get('respondent','')} "
        f"{' '.join(reference_doc.get('sections_cited', []))} "
        f"{' '.join(reference_doc.get('acts_cited', []))} "
        f"{reference_doc.get('verdict','')}"
    ).strip()

class LegalAIPipeline:
    """
    Single entry point for the entire Legal AI system.
    """

    def __init__(
        self,
        google_api_key: str,
        kanoon_api_token: str = "",
        db_path: str = "./legal_db",
        gemini_model: str = "gemini-1.5-flash",
        device: str = "cpu",
        fetch_kanoon_full_text: bool = True,
    ):
        self.google_api_key = google_api_key
        self.kanoon_api_token = kanoon_api_token
        self.gemini_model = gemini_model
        self.fetch_kanoon_full_text = fetch_kanoon_full_text

        self.embeddings = InLegalBERTEmbeddings(device=device)
        self.vector_store = DualVectorStore(
            embeddings=self.embeddings,
            judgments_path=os.path.join(db_path, "judgments"),
            user_session_path=os.path.join(db_path, "user_sessions"),
        )

        if kanoon_api_token:
            self.kanoon = IndianKanoonClient(kanoon_api_token)
        else:
            self.kanoon = None

        self.qa_chain = build_qa_chain(google_api_key, gemini_model)
        self.similar_chain = build_similar_verdicts_chain(google_api_key, gemini_model)
        self.combined_chain = build_combined_chain(google_api_key, gemini_model)
        self.summary_chain = build_summary_chain(google_api_key, gemini_model)

        self._sessions: Dict[str, List[Dict]] = {}

    # ── Session management ──────────────────────────────────────────────────────

    def new_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        return session_id

    def end_session(self, session_id: str):
        self.vector_store.clear_session(session_id)
        self._sessions.pop(session_id, None)

    def ensure_session(self, session_id: str):
        """Re-register a session_id that already exists (e.g. after page reload)."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

    # ── Document upload ─────────────────────────────────────────────────────────

    def upload_document(self, filepath: str, session_id: str) -> Dict:
        self.ensure_session(session_id)

        extracted = LegalDocumentExtractor.detect_and_extract(filepath)
        if extracted["status"] == "failed":
            return {
                "error": extracted["error"],
                "filename": extracted["filename"],
                "chunks_added": 0,
                "session_id": session_id,
            }

        chunks_added = self.vector_store.add_user_document(extracted, session_id)
        self._sessions[session_id].append(extracted)

        try:
            summary = self.summary_chain.invoke(extracted)
        except Exception as e:
            summary = f"Summary generation failed: {e}"

        return {
            "filename": extracted["filename"],
            "extraction_method": extracted["extraction_method"],
            "pages": extracted["page_count"],
            "words": extracted["word_count"],
            "court": extracted["court_name"],
            "case_number": extracted["case_number"],
            "date": extracted["judgment_date"],
            "petitioner": extracted["petitioner"],
            "respondent": extracted["respondent"],
            "sections_cited": extracted["sections_cited"],
            "acts_cited": extracted["acts_cited"],
            "verdict": extracted["verdict"],
            "summary": summary,
            "chunks_added": chunks_added,
            "session_id": session_id,
            "status": "success",
        }

    # ── Ask ─────────────────────────────────────────────────────────────────────

    def ask(
        self, question: str, session_id: str, n_results: int = 6, threshold: float = 0.25
    ) -> Dict:
        self.ensure_session(session_id)
        user_docs = self.vector_store.search_user_documents(question, session_id, n_results, threshold)
        user_context = format_docs_for_prompt(user_docs, "uploaded documents")

        if not user_docs:
            return {
                "answer": "No relevant content found in your uploaded documents for this question. Please upload documents first.",
                "sources": [],
                "session_id": session_id,
            }

        answer = self.qa_chain.invoke({"question": question, "user_context": user_context})
        return {
            "answer": answer,
            "sources": [
                {
                    "filename": d.metadata.get("filename"),
                    "case_number": d.metadata.get("case_number"),
                    "court": d.metadata.get("court"),
                    "date": d.metadata.get("date"),
                    "score": d.metadata.get("similarity_score"),
                }
                for d in user_docs
            ],
            "session_id": session_id,
        }

    # ── Find similar verdicts ───────────────────────────────────────────────────
 
 
    def find_similar_verdicts(
        self,
        session_id: str,
        n_results: int = 8,          # how many cases to LIST (cheap, snippet-only)
        full_text_top_k: int = 3,    # how many to FETCH FULL TEXT + ground analysis on
        threshold: float = 0.0,      # set >0 to drop low-relevance snippets from the list
        court_filter: Optional[str] = None,
        fetch_from_kanoon: bool = True,
    ) -> Dict:
        self.ensure_session(session_id)
        docs_in_session = self._sessions.get(session_id, [])
        if not docs_in_session:
            return {"error": "No documents uploaded in this session yet.", "similar_cases": []}
    
        reference_doc = docs_in_session[-1]
    
        if not fetch_from_kanoon or not self.kanoon:
            return {
                "analysis": "Precedent search is disabled or no Indian Kanoon token is configured.",
                "similar_cases": [],
                "kanoon_fetched": 0,
                "session_id": session_id,
            }
    
        # 1. Build query from extracted case facts.
        from indian_kannon_client import IndianKanoonClient
        query = IndianKanoonClient.build_query_from_document(reference_doc)
    
        # 2. ONE search call. headline snippets only — no full-text fetch yet.
        raw_results = self.kanoon.search(query, max_results=n_results)
        if court_filter:
            raw_results = [
                r for r in raw_results
                if court_filter.lower() in str(r.get("docsource", "")).lower()
            ]
    
        if not raw_results:
            return {
                "analysis": (
                    f"No precedents were found on Indian Kanoon for the query `{query}`. "
                    f"Check that sections_cited / acts_cited were extracted from the "
                    f"uploaded document — an empty query usually means extraction failed."
                ),
                "similar_cases": [],
                "kanoon_fetched": 0,
                "session_id": session_id,
            }
    
        # 3. Rank by embedding similarity of the SHORT snippets only — cheap.
        snippets = [_clean_snippet(r.get("headline") or r.get("title", "")) for r in raw_results]
        reference_text = _build_reference_text(reference_doc)
    
        try:
            ref_vec = self.embeddings.embed_query(reference_text)
            snippet_vecs = self.embeddings.embed_documents(snippets)
            scores = [_cosine_sim(ref_vec, v) for v in snippet_vecs]
        except Exception as e:
            print(f"⚠️  Snippet ranking failed, falling back to Kanoon's own order: {e}")
            scores = [1.0 - (i * 0.01) for i in range(len(raw_results))]  # preserve original order
    
        ranked_idx = sorted(range(len(raw_results)), key=lambda i: scores[i], reverse=True)
        if threshold > 0:
            ranked_idx = [i for i in ranked_idx if scores[i] >= threshold] or ranked_idx[:1]
    
        # 4. Build the FULL list of matched cases (lightweight — this is what shows in the UI).
        from document_extractor import LegalDocumentExtractor
        similar_cases = []
        for i in ranked_idx:
            r = raw_results[i]
            doc_id = str(r.get("tid", ""))
            similar_cases.append({
                "title": r.get("title", "Unknown"),
                "court": r.get("docsource", "Unknown"),
                "date": r.get("publishdate", "Unknown"),
                "case_number": LegalDocumentExtractor._get_case_number(snippets[i]) or "",
                "kanoon_url": f"https://indiankanoon.org/doc/{doc_id}/",
                "snippet": snippets[i][:300],
                "relevance_score": round(scores[i], 4),
            })
    
        # 5. Fetch FULL TEXT only for the top-k — this is the expensive step, now bounded.
        top_raw = [raw_results[i] for i in ranked_idx[:full_text_top_k]]
        judgment_dicts = [
            self.kanoon.kanoon_result_to_judgment_dict(r, fetch_full_text=True) for r in top_raw
        ]
    
        # 6. Persist only what was actually fetched (dedup via ids — see vector_store_PATCH.py).
        if judgment_dicts:
            self.vector_store.add_judgments(judgment_dicts)
    
        # 7. Grounded synthesis — only from the full text just fetched, never from memory.
        from rag_chains import format_docs_for_prompt
        judgment_context = format_docs_for_prompt(judgment_dicts, "candidate precedents")
        case_summary = (
            f"Court: {reference_doc.get('court_name','Unknown')}\n"
            f"Parties: {reference_doc.get('petitioner','?')} vs {reference_doc.get('respondent','?')}\n"
            f"Sections: {', '.join(reference_doc.get('sections_cited', [])) or 'Not extracted'}\n"
            f"Acts: {', '.join(reference_doc.get('acts_cited', [])) or 'Not extracted'}\n"
            f"Verdict: {reference_doc.get('verdict','Not found')}"
        )
    
        try:
            analysis = self.similar_chain.invoke({
                "case_summary": case_summary,
                "judgment_context": judgment_context,
            })
        except Exception as e:
            analysis = f"⚠️ Precedent synthesis failed at model runtime: {str(e)}"
    
        return {
            "analysis": analysis,
            "similar_cases": similar_cases,   # ALL matched cases, ranked
            "kanoon_fetched": len(raw_results),
            "full_text_fetched": len(judgment_dicts),
            "session_id": session_id,
        }

    # ── Ask with precedents ─────────────────────────────────────────────────────

    def ask_with_precedents(
        self, question: str, session_id: str, n_results: int = 5, threshold: float = 0.25
    ) -> Dict:
        self.ensure_session(session_id)
        results = self.vector_store.search_combined(question, session_id, n_results, threshold)
        user_docs = results["user_docs"]
        judgments = results["judgments"]

        user_context = format_docs_for_prompt(user_docs, "uploaded documents")
        judgment_context = format_docs_for_prompt(judgments, "similar past judgments")

        answer = self.combined_chain.invoke({
            "question": question,
            "user_context": user_context,
            "judgment_context": judgment_context,
        })

        return {
            "answer": answer,
            "user_doc_sources": [
                {"filename": d.metadata.get("filename"), "case_number": d.metadata.get("case_number"), "score": d.metadata.get("similarity_score")}
                for d in user_docs
            ],
            "judgment_sources": [
                {"case_number": d.metadata.get("case_number"), "title": d.metadata.get("title"),
                 "court": d.metadata.get("court"), "date": d.metadata.get("date"),
                 "score": d.metadata.get("similarity_score"), "kanoon_url": d.metadata.get("kanoon_url", "")}
                for d in judgments
            ],
            "session_id": session_id,
        }

    # ── Load judgment folder ────────────────────────────────────────────────────

    def load_judgments_from_folder(self, folder_path: str):
        folder = Path(folder_path)
        supported = {".pdf", ".txt", ".jpg", ".jpeg", ".png"}
        files = [f for f in folder.iterdir() if f.suffix.lower() in supported]
        judgment_dicts = []
        for filepath in files:
            extracted = LegalDocumentExtractor.detect_and_extract(str(filepath))
            if extracted["status"] == "success" and extracted["raw_text"]:
                judgment_dicts.append(extracted)
        if judgment_dicts:
            self.vector_store.add_judgments(judgment_dicts)

    # ── Status ──────────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "judgment_chunks": self.vector_store.judgment_count,
            "user_session_chunks": self.vector_store.user_doc_count,
            "active_sessions": len(self._sessions),
            "kanoon_available": self.kanoon is not None,
            "gemini_model": self.gemini_model,
        }