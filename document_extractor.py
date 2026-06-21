import os
import re
from datetime import datetime
from typing import Dict, List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path


class LegalDocumentExtractor:
    """
    Extracts text and legal metadata from:
      .txt   — plain text files
      .pdf   — digital (typed) or scanned PDFs via OCR
      .jpg / .jpeg / .png — photographed documents via OCR
    """

    @staticmethod
    def detect_and_extract(filepath: str) -> Dict:
        if not os.path.exists(filepath):
            return LegalDocumentExtractor._error_result(filepath, "File not found")

        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".txt":
            return LegalDocumentExtractor.extract_from_text(filepath)
        elif ext == ".pdf":
            if LegalDocumentExtractor._is_digital_pdf(filepath):
                return LegalDocumentExtractor.extract_from_digital_pdf(filepath)
            else:
                return LegalDocumentExtractor.extract_from_scanned_pdf(filepath)
        elif ext in [".jpg", ".jpeg", ".png"]:
            return LegalDocumentExtractor.extract_from_image(filepath)
        else:
            return LegalDocumentExtractor._error_result(
                filepath, f"Unsupported file type: {ext}. Supported: .txt .pdf .jpg .jpeg .png"
            )

    @staticmethod
    def extract_from_text(filepath: str) -> Dict:
        result = LegalDocumentExtractor._base_result(filepath, "plain_text")
        try:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except UnicodeDecodeError:
                with open(filepath, "r", encoding="latin-1") as f:
                    raw_text = f.read()

            result["page_count"] = raw_text.count("\n")
            cleaned = LegalDocumentExtractor.clean_text(raw_text)
            result["raw_text"] = cleaned
            result["word_count"] = len(cleaned.split())
            result["char_count"] = len(cleaned)
            result = LegalDocumentExtractor._extract_content_metadata(result, cleaned)
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
        return result

    @staticmethod
    def extract_from_digital_pdf(filepath: str) -> Dict:
        result = LegalDocumentExtractor._base_result(filepath, "digital_pdf")
        try:
            with fitz.open(filepath) as doc:
                meta = doc.metadata
                result["pdf_title"] = meta.get("title", "").strip()
                result["pdf_author"] = meta.get("author", "").strip()
                result["pdf_created_date"] = meta.get("creationDate", "").strip()
                result["pdf_format"] = meta.get("format", "").strip()
                result["page_count"] = len(doc)

                full_text = ""
                for page_num, page in enumerate(doc):
                    full_text += f"\n--- Page {page_num + 1} ---\n"
                    full_text += page.get_text()

                cleaned = LegalDocumentExtractor.clean_text(full_text)
                result["raw_text"] = cleaned
                result["word_count"] = len(cleaned.split())
                result["char_count"] = len(cleaned)
                result = LegalDocumentExtractor._extract_content_metadata(result, cleaned)
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
        return result

    @staticmethod
    def extract_from_scanned_pdf(filepath: str, dpi: int = 300, lang: str = "eng") -> Dict:
        result = LegalDocumentExtractor._base_result(filepath, "scanned_pdf")
        try:
            images = convert_from_path(filepath, dpi=dpi, fmt="jpeg", thread_count=2)
            result["page_count"] = len(images)

            full_text = ""
            for page_num, image in enumerate(images):
                custom_config = r"--oem 3 --psm 3"
                page_text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
                full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
                del image

            cleaned = LegalDocumentExtractor.clean_text(full_text)
            result["raw_text"] = cleaned
            result["word_count"] = len(cleaned.split())
            result["char_count"] = len(cleaned)
            result["ocr_dpi"] = dpi
            result["ocr_lang"] = lang
            result = LegalDocumentExtractor._extract_content_metadata(result, cleaned)
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
        return result

    @staticmethod
    def extract_from_image(filepath: str, lang: str = "eng") -> Dict:
        result = LegalDocumentExtractor._base_result(filepath, "image")
        try:
            image = Image.open(filepath)
            result["page_count"] = 1
            result["image_size"] = f"{image.width} x {image.height} px"
            result["image_mode"] = image.mode
            result["image_format"] = image.format

            if image.mode != "RGB":
                image = image.convert("RGB")

            custom_config = r"--oem 3 --psm 3"
            raw_text = pytesseract.image_to_string(image, lang=lang, config=custom_config)

            cleaned = LegalDocumentExtractor.clean_text(raw_text)
            result["raw_text"] = cleaned
            result["word_count"] = len(cleaned.split())
            result["char_count"] = len(cleaned)
            result["ocr_lang"] = lang
            result = LegalDocumentExtractor._extract_content_metadata(result, cleaned)
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
        return result

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\x0c", "\n")
        text = re.sub(r'[\x00-\x08\x0b\x0e-\x1f\x7f]', '', text)
        text = re.sub(r'^\s*[-–]?\s*\d+\s*[-–]?\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[Pp]age\s+\d+\s*(?:of\s+\d+)?\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-=_]{4,}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)
        lines = [l for l in text.splitlines() if len(l.strip()) > 2 or l.strip() == ""]
        return "\n".join(lines).strip()

    @staticmethod
    def _is_digital_pdf(filepath: str) -> bool:
        try:
            with fitz.open(filepath) as doc:
                text_found = ""
                for i in range(min(5, len(doc))):
                    text_found += doc[i].get_text()
                return len(text_found.strip()) > 100
        except Exception:
            return False

    @staticmethod
    def _base_result(filepath: str, method: str) -> Dict:
        return {
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "file_size_kb": round(os.path.getsize(filepath) / 1024, 2),
            "file_extension": os.path.splitext(filepath)[1].lower(),
            "extraction_method": method,
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pdf_title": "", "pdf_author": "", "pdf_created_date": "",
            "pdf_format": "", "image_size": "", "image_mode": "",
            "image_format": "", "ocr_dpi": "", "ocr_lang": "",
            "page_count": 0, "word_count": 0, "char_count": 0,
            "court_name": "", "case_number": "", "judgment_date": "",
            "petitioner": "", "respondent": "", "judges": [],
            "sections_cited": [], "acts_cited": [], "verdict": "",
            "raw_text": "", "status": "success", "error": ""
        }

    @staticmethod
    def _extract_content_metadata(result: Dict, text: str) -> Dict:
        result["court_name"] = LegalDocumentExtractor._get_court_name(text)
        result["case_number"] = LegalDocumentExtractor._get_case_number(text)
        result["judgment_date"] = LegalDocumentExtractor._get_date(text)
        result["petitioner"] = LegalDocumentExtractor._get_petitioner(text)
        result["respondent"] = LegalDocumentExtractor._get_respondent(text)
        result["judges"] = LegalDocumentExtractor._get_judges(text)
        result["sections_cited"] = LegalDocumentExtractor._get_sections(text)
        result["acts_cited"] = LegalDocumentExtractor._get_acts(text)
        result["verdict"] = LegalDocumentExtractor._get_verdict(text)
        return result

    @staticmethod
    def _get_court_name(text: str) -> str:
        header = text[:500].upper()
        courts = [
            "SUPREME COURT OF INDIA", "HIGH COURT OF DELHI", "HIGH COURT OF BOMBAY",
            "HIGH COURT OF MADRAS", "HIGH COURT OF CALCUTTA", "HIGH COURT OF ALLAHABAD",
            "HIGH COURT OF KERALA", "HIGH COURT OF GUJARAT", "HIGH COURT OF KARNATAKA",
            "HIGH COURT OF TELANGANA", "HIGH COURT OF ANDHRA PRADESH",
            "HIGH COURT OF RAJASTHAN", "HIGH COURT OF MADHYA PRADESH",
            "HIGH COURT OF PUNJAB AND HARYANA", "HIGH COURT OF ORISSA",
            "NATIONAL CONSUMER DISPUTES REDRESSAL COMMISSION",
            "NATIONAL COMPANY LAW TRIBUNAL", "CENTRAL ADMINISTRATIVE TRIBUNAL",
            "ARMED FORCES TRIBUNAL", "INCOME TAX APPELLATE TRIBUNAL",
            "DISTRICT COURT", "SESSIONS COURT", "FAMILY COURT",
        ]
        for court in courts:
            if court in header:
                return court.title()
        return "Unknown"

    @staticmethod
    def _get_case_number(text: str) -> str:
        patterns = [
            r'W\.P\.?\s*\(C(?:ivil)?\)?\s*No\.?\s*\d+\s*(?:of|/)\s*\d{4}',
            r'Civil Appeal No\.?\s*\d+\s*of\s*\d{4}',
            r'Criminal Appeal No\.?\s*\d+\s*of\s*\d{4}',
            r'SLP\s*\(?Civil\)?\s*No\.?\s*\d+\s*(?:of|/)\s*\d{4}',
            r'Crl\.?\s*M\.?A\.?\s*No\.?\s*\d+\s*(?:of|/)\s*\d{4}',
            r'Case No\.?\s*\d+\s*(?:of|/)\s*\d{4}',
            r'Petition No\.?\s*\d+\s*(?:of|/)\s*\d{4}',
            r'F\.I\.R\.?\s*No\.?\s*\d+\s*(?:of|/)\s*\d{4}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                return match.group().strip()
        return "Not found"

    @staticmethod
    def _get_date(text: str) -> str:
        patterns = [
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'[Dd]ated?\s*:?\s*\d{1,2}\.\d{1,2}\.\d{4}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                return match.group().strip()
        return "Not found"

    @staticmethod
    def _get_petitioner(text: str) -> str:
        patterns = [
            r'(?:Petitioner|Appellant|Plaintiff)\s*[:\-]\s*([A-Z][A-Za-z\s\.]+?)(?:\n|Vs?\.)',
            r'([A-Z][A-Za-z\s\.]+?)\s*\n\s*(?:\.{3,})?(?:Petitioner|Appellant|Plaintiff)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:3000])
            if match:
                return match.group(1).strip()
        return "Not found"

    @staticmethod
    def _get_respondent(text: str) -> str:
        patterns = [
            r'(?:Respondent|Defendant|Opposite Party)\s*[:\-]\s*([A-Z][A-Za-z\s\.]+?)(?:\n|$)',
            r'([A-Z][A-Za-z\s\.]+?)\s*\n\s*(?:\.{3,})?(?:Respondent|Defendant)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:3000])
            if match:
                return match.group(1).strip()
        return "Not found"

    @staticmethod
    def _get_judges(text: str) -> List[str]:
        patterns = [
            r'(?:HON\'?BLE|HONOURABLE)\s+(?:MR\.?\s+)?JUSTICE\s+([A-Z][A-Z\s\.]+)',
            r'CORAM\s*:\s*([A-Z][A-Za-z\s\.,]+?)(?:\n\n|\Z)',
            r'Before\s*:\s*([A-Z][A-Za-z\s\.,]+?)(?:\n\n|\Z)',
        ]
        judges = []
        for pattern in patterns:
            matches = re.findall(pattern, text[:2000], re.IGNORECASE)
            for m in matches:
                name = m.strip().rstrip(".,")
                if name and name not in judges:
                    judges.append(name)
        return judges[:5]

    @staticmethod
    def _get_sections(text: str) -> List[str]:
        patterns = [
            r'[Ss]ection\s+\d+[A-Za-z]?(?:\(\d+\))?\s+(?:of\s+the\s+)?(?:IPC|CPC|CrPC|I\.P\.C|C\.P\.C)',
            r'(?:IPC|CPC|CrPC)\s+[Ss]ec(?:tion)?\.?\s*\d+[A-Za-z]?',
            r'Article\s+\d+[A-Za-z]?\s+of\s+the\s+Constitution',
            r'Order\s+[IVXLCM]+\s+Rule\s+\d+\s+(?:of\s+the\s+)?CPC',
        ]
        found = []
        for pattern in patterns:
            found.extend(re.findall(pattern, text))
        seen, unique = set(), []
        for s in found:
            s = s.strip()
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique[:15]

    @staticmethod
    def _get_acts(text: str) -> List[str]:
        matches = re.findall(r'(?:The\s+)?[A-Z][A-Za-z\s]+?Act,?\s*\d{4}', text)
        acts = [m.strip() for m in matches if 10 < len(m.strip()) < 80]
        seen, unique = set(), []
        for a in acts:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique[:10]

    @staticmethod
    def _get_verdict(text: str) -> str:
        patterns = [
            r'(?:the\s+)?(?:petition|appeal|suit|application)\s+is\s+(?:hereby\s+)?(?:dismissed|allowed|disposed\s+of|rejected|partly\s+allowed)',
            r'(?:we\s+)?(?:dismiss|allow|dispose\s+of|reject)\s+the\s+(?:petition|appeal|suit)',
            r'(?:DISMISSED|ALLOWED|DISPOSED\s+OF|REJECTED|PARTLY\s+ALLOWED)',
        ]
        tail = text[-2000:]
        for pattern in patterns:
            match = re.search(pattern, tail, re.IGNORECASE)
            if match:
                return match.group().strip()[:300]
        return "Not found"

    @staticmethod
    def _error_result(filepath: str, error_message: str) -> Dict:
        return {
            "filename": os.path.basename(filepath), "filepath": filepath,
            "file_size_kb": 0, "extraction_method": "none",
            "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "failed", "error": error_message,
            "raw_text": "", "word_count": 0, "char_count": 0, "page_count": 0,
        }