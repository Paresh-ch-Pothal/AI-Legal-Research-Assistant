from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI


# ── Prompts ────────────────────────────────────────────────────────────────────

USER_DOC_QA_PROMPT = ChatPromptTemplate.from_template("""
You are an expert Indian legal research assistant helping a lawyer or judge.
Answer the question based ONLY on the uploaded case documents provided below.
Be precise. Cite the document filename and page section for every claim.

QUESTION:
{question}

UPLOADED DOCUMENT EXCERPTS:
{user_context}

Provide your answer in this format:
1. **Direct Answer** — one clear sentence
2. **Details from Document** — specific findings with references
3. **Relevant Sections/Acts Identified** — list any IPC/CPC/CrPC sections found
4. **Document Info** — filename, court, date, parties
""")

SIMILAR_VERDICTS_PROMPT = ChatPromptTemplate.from_template("""
You are an expert Indian legal research assistant.
The user has uploaded a case document. Based ONLY on the retrieved similar past
judgments below, identify the most relevant precedents and explain how they apply.
 
USER'S CASE SUMMARY:
{case_summary}
 
SIMILAR PAST JUDGMENTS RETRIEVED:
{judgment_context}
 
IMPORTANT: If the section above says "[No candidate precedents found]" or
otherwise contains no real judgment excerpts, do NOT invent, recall from
memory, or guess at any case. Instead respond only with:
"No relevant precedents were found in the retrieved judgments for this case."
Do not fabricate a case name, citation, court, or ratio decidendi under any
circumstances — every case you mention must come from the text in
SIMILAR PAST JUDGMENTS RETRIEVED above, not from your training data.
 
If real judgments ARE present above, provide your analysis in this format:
1. **Most Relevant Precedent** — case name, court, date, and why it matches
   (quote/paraphrase only from the excerpt provided, do not add outside facts)
2. **Other Similar Cases** — brief list with case numbers, from the excerpts only
3. **Common Legal Pattern** — what legal principle connects these cases
4. **How These Precedents Apply** — practical guidance for the current case
5. **Key Citations** — [Case No. | Court | Date | Similarity Score]
""")

COMBINED_LEGAL_PROMPT = ChatPromptTemplate.from_template("""
You are an expert Indian legal research assistant helping with case analysis.

QUESTION:
{question}

SECTION A — UPLOADED CASE DOCUMENTS:
{user_context}

SECTION B — SIMILAR PAST JUDGMENTS (from Indian Kanoon & local DB):
{judgment_context}

Provide a comprehensive legal analysis:
1. **Direct Answer** — based on the uploaded documents
2. **Legal Reasoning** — applying relevant law from the documents
3. **Supporting Precedents** — from similar past judgments
4. **Relevant Sections & Acts** — all legal provisions identified
5. **Full Citations** — [Case No. | Court | Date | Source | Score]
""")

DOCUMENT_SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are a legal analyst. Summarize this Indian legal document in 5-6 sentences.
Include: document type, court/authority, parties, key legal issues, and outcome (if present).

Filename : {filename}
Court    : {court}
Date     : {date}

DOCUMENT TEXT:
{text}

Summary:
""")


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_gemini_llm(
    google_api_key: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=google_api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
        convert_system_message_to_human=True,
    )


# ── Chain builders ─────────────────────────────────────────────────────────────

def build_qa_chain(google_api_key: str, model: str = "gemini-1.5-flash"):
    llm = get_gemini_llm(google_api_key, model)
    return USER_DOC_QA_PROMPT | llm | StrOutputParser()


def build_similar_verdicts_chain(google_api_key: str, model: str = "gemini-1.5-flash"):
    llm = get_gemini_llm(google_api_key, model)
    return SIMILAR_VERDICTS_PROMPT | llm | StrOutputParser()


def build_combined_chain(google_api_key: str, model: str = "gemini-1.5-flash"):
    llm = get_gemini_llm(google_api_key, model)
    return COMBINED_LEGAL_PROMPT | llm | StrOutputParser()


def build_summary_chain(google_api_key: str, model: str = "gemini-1.5-flash"):
    llm = get_gemini_llm(google_api_key, model)
    chain = (
        RunnableLambda(lambda j: {
            "filename": j.get("filename", "Unknown"),
            "court": j.get("court_name", "Unknown"),
            "date": j.get("judgment_date", "Unknown"),
            "text": j.get("raw_text", "")[:4000],
        })
        | DOCUMENT_SUMMARY_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── Formatting helper ──────────────────────────────────────────────────────────

def format_docs_for_prompt(docs: List[Document], label: str = "") -> str:
    if not docs:
        return f"[No {label or 'documents'} found]"

    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        score = m.get("similarity_score", "N/A")
        url = m.get("kanoon_url", "")
        url_line = f"\n    URL     : {url}" if url else ""
        parts.append(
            f"[{i}] Case     : {m.get('case_number', 'Unknown')}\n"
            f"    Title   : {m.get('title', 'Unknown')}\n"
            f"    Court   : {m.get('court', 'Unknown')}\n"
            f"    Date    : {m.get('date', 'Unknown')}\n"
            f"    Parties : {m.get('petitioner', '?')} vs {m.get('respondent', '?')}\n"
            f"    Verdict : {m.get('verdict', 'Unknown')}\n"
            f"    Source  : {m.get('source', 'local')} | Score: {score}"
            f"{url_line}\n"
            f"    Excerpt :\n{doc.page_content}"
        )
    return "\n\n" + ("─" * 60 + "\n\n").join(parts)