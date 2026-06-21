import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import InLegalBERTEmbeddings


def judgment_dict_to_documents(judgment_dict: Dict[str, Any]) -> List[Document]:
    raw_text = judgment_dict.get("raw_text", "").strip()
    if not raw_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_text(raw_text)

    base_meta = {
        "filename": judgment_dict.get("filename", "unknown"),
        "title": judgment_dict.get("title", judgment_dict.get("pdf_title", "unknown")),
        "court": judgment_dict.get("court_name", "unknown"),
        "date": judgment_dict.get("judgment_date", "unknown"),
        "case_number": judgment_dict.get("case_number", "unknown"),
        "petitioner": judgment_dict.get("petitioner", "unknown"),
        "respondent": judgment_dict.get("respondent", "unknown"),
        "verdict": judgment_dict.get("verdict", "unknown"),
        "citation": judgment_dict.get("citation", ""),
        "outcome": judgment_dict.get("outcome", ""),
        "source": judgment_dict.get("source", "local"),
        "kanoon_url": judgment_dict.get("kanoon_url", ""),
        "sections_cited": json.dumps(judgment_dict.get("sections_cited", [])),
        "acts_cited": json.dumps(judgment_dict.get("acts_cited", [])),
    }

    docs = []
    for i, chunk_text in enumerate(text_chunks):
        if len(chunk_text.strip()) < 50:
            continue
        doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.splitext(base_meta["filename"])[0])[:60]
        meta = {**base_meta, "chunk_id": f"{doc_id}_chunk_{i}", "chunk_num": i}
        docs.append(Document(page_content=chunk_text.strip(), metadata=meta))
    return docs


def build_documents_from_judgments(judgment_dicts: List[Dict]) -> List[Document]:
    all_docs = []
    for jd in judgment_dicts:
        all_docs.extend(judgment_dict_to_documents(jd))
    return all_docs


class DualVectorStore:
    """
    Manages two ChromaDB collections:
      - judgments_db    : historical judgments (persistent)
      - user_session_db : per-session uploaded documents
    """

    def __init__(
        self,
        embeddings: InLegalBERTEmbeddings,
        judgments_path: str = "./legal_db/judgments",
        user_session_path: str = "./legal_db/user_sessions",
    ):
        self.embeddings = embeddings
        self.judgments_path = judgments_path
        self.user_session_path = user_session_path

        self.judgments_store = self._init_store(judgments_path, "judgments_db")
        self.user_session_store = self._init_store(user_session_path, "user_session_db")

    def _init_store(self, path: str, collection_name: str) -> Chroma:
        os.makedirs(path, exist_ok=True)
        return Chroma(
            persist_directory=path,
            embedding_function=self.embeddings,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_judgments(self, judgment_dicts: List[Dict]):
        docs = build_documents_from_judgments(judgment_dicts)
        if docs:
            ids = [doc.metadata["chunk_id"] for doc in docs]
            self.judgments_store.add_documents(docs, ids=ids)

    def add_user_document(self, extracted_dict: Dict, session_id: str) -> int:
        extracted_dict_tagged = {**extracted_dict, "session_id": session_id, "source": "user_upload"}
        docs = judgment_dict_to_documents(extracted_dict_tagged)
        if not docs:
            return 0
        for doc in docs:
            doc.metadata["session_id"] = session_id
        self.user_session_store.add_documents(docs)
        return len(docs)

    def add_kanoon_results(self, kanoon_judgment_dicts: List[Dict]):
        docs = build_documents_from_judgments(kanoon_judgment_dicts)
        if docs:
            ids = [doc.metadata["chunk_id"] for doc in docs]
            self.judgments_store.add_documents(docs, ids=ids)

    def search_user_documents(
        self, query: str, session_id: str, n_results: int = 5, threshold: float = 0.25
    ) -> List[Document]:
        try:
            results = self.user_session_store.similarity_search_with_relevance_scores(
                query, k=n_results, filter={"session_id": session_id}
            )
            docs = []
            for doc, score in results:
                if score >= threshold:
                    doc.metadata["similarity_score"] = round(score, 4)
                    doc.metadata["result_source"] = "user_document"
                    docs.append(doc)
            return docs
        except Exception as e:
            print(f"⚠️  User doc search error: {e}")
            return []

    def search_judgments(
        self, query: str, n_results: int = 5, threshold: float = 0.25,
        court_filter: Optional[str] = None
    ) -> List[Document]:
        try:
            search_kwargs: Dict[str, Any] = {"k": n_results}
            if court_filter:
                search_kwargs["filter"] = {"court": court_filter}
            results = self.judgments_store.similarity_search_with_relevance_scores(
                query, **search_kwargs
            )
            docs = []
            for doc, score in results:
                if score >= threshold:
                    doc.metadata["similarity_score"] = round(score, 4)
                    doc.metadata["result_source"] = "judgment_db"
                    docs.append(doc)
            return docs
        except Exception as e:
            print(f"⚠️  Judgment search error: {e}")
            return []

    def search_combined(
        self, query: str, session_id: str, n_per_source: int = 5,
        threshold: float = 0.25, court_filter: Optional[str] = None
    ) -> Dict[str, List[Document]]:
        user_docs = self.search_user_documents(query, session_id, n_per_source, threshold)
        judgments = self.search_judgments(query, n_per_source, threshold, court_filter)
        return {"user_docs": user_docs, "judgments": judgments}

    def clear_session(self, session_id: str):
        try:
            self.user_session_store._collection.delete(where={"session_id": session_id})
        except Exception as e:
            print(f"⚠️  Could not clear session {session_id}: {e}")

    @property
    def judgment_count(self) -> int:
        return self.judgments_store._collection.count()

    @property
    def user_doc_count(self) -> int:
        return self.user_session_store._collection.count()