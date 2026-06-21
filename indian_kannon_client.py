import re
import requests
from datetime import datetime
from typing import Dict, List, Optional

from document_extractor import LegalDocumentExtractor


class IndianKanoonClient:
    """
    Client for the Indian Kanoon REST API.
    Docs: https://api.indiankanoon.org/
    """

    BASE_URL = "https://api.indiankanoon.org"

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

    def search(self, query: str, page_num: int = 0, max_results: int = 10) -> List[Dict]:
        endpoint = f"{self.BASE_URL}/search/"
        payload = {"formInput": query, "pagenum": page_num}
        try:
            response = requests.post(endpoint, data=payload, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get("docs", [])[:max_results]
        except requests.exceptions.Timeout:
            print("⚠️  Indian Kanoon API timeout")
            return []
        except Exception as e:
            print(f"⚠️  Indian Kanoon API error: {e}")
            return []

    def fetch_document(self, doc_id: str) -> Optional[Dict]:
        endpoint = f"{self.BASE_URL}/doc/{doc_id}/"
        try:
            response = requests.post(endpoint, headers=self.headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️  Error fetching document {doc_id}: {e}")
            return None

    @staticmethod
    def build_query_from_document(extracted_dict: Dict) -> str:
        parts = []
        sections = extracted_dict.get("sections_cited", [])
        if sections:
            parts.extend(sections[:5])
        acts = extracted_dict.get("acts_cited", [])
        if acts:
            for act in acts[:3]:
                parts.append(act[:50])
        verdict = extracted_dict.get("verdict", "")
        if verdict and verdict != "Not found":
            verdict_words = re.findall(
                r'\b(?:dismissed|allowed|acquitted|convicted|bail)\b', verdict, re.IGNORECASE
            )
            parts.extend(verdict_words[:2])
        court = extracted_dict.get("court_name", "")
        if court and court != "Unknown":
            parts.append(court)
        if not parts:
            raw = extracted_dict.get("raw_text", "")[:500]
            legal_keywords = re.findall(
                r'\b(?:murder|theft|cheating|fraud|bail|custody|writ|injunction|'
                r'contempt|divorce|maintenance|dowry|rape|assault|forgery)\b',
                raw, re.IGNORECASE
            )
            parts.extend(legal_keywords[:5])
        return " ".join(parts) if parts else "Indian court judgment"

    def kanoon_result_to_judgment_dict(self, kanoon_doc: Dict, fetch_full_text: bool = True) -> Dict:
        doc_id = str(kanoon_doc.get("tid", ""))
        title = kanoon_doc.get("title", "Unknown")
        headline = kanoon_doc.get("headline", "")
        source = kanoon_doc.get("docsource", "Unknown")
        pub_date = kanoon_doc.get("publishdate", "Unknown")

        raw_text = headline
        if fetch_full_text and doc_id:
            full_doc = self.fetch_document(doc_id)
            if full_doc:
                html_text = full_doc.get("doc", "")
                raw_text = re.sub(r'<[^>]+>', ' ', html_text)
                raw_text = LegalDocumentExtractor.clean_text(raw_text)

        return {
            "filename": f"kanoon_{doc_id}.txt",
            "filepath": f"https://indiankanoon.org/doc/{doc_id}/",
            "extraction_method": "indian_kanoon_api",
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "indian_kanoon",
            "kanoon_doc_id": doc_id,
            "kanoon_url": f"https://indiankanoon.org/doc/{doc_id}/",
            "pdf_title": title, "title": title,
            "court_name": source, "judgment_date": pub_date,
            "case_number": LegalDocumentExtractor._get_case_number(raw_text),
            "petitioner": LegalDocumentExtractor._get_petitioner(raw_text),
            "respondent": LegalDocumentExtractor._get_respondent(raw_text),
            "judges": LegalDocumentExtractor._get_judges(raw_text),
            "sections_cited": LegalDocumentExtractor._get_sections(raw_text),
            "acts_cited": LegalDocumentExtractor._get_acts(raw_text),
            "verdict": LegalDocumentExtractor._get_verdict(raw_text),
            "citation": f"Indian Kanoon Doc #{doc_id}",
            "outcome": "", "raw_text": raw_text,
            "word_count": len(raw_text.split()),
            "char_count": len(raw_text),
            "status": "success", "error": "",
        }