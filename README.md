# ⚖️ Legal AI Research Assistant

A Retrieval-Augmented Generation (RAG) powered Streamlit application that lets you upload legal documents — FIRs, contracts, legal notices, petitions, orders, and more — and have a conversation with them. Ask questions in plain English, get grounded answers with citations back to the exact document, export the whole conversation as a polished PDF, and look up related case law or terminology on the open web without ever leaving the app.

Built with an Indian legal context in mind (IPC sections, Indian Kanoon references, InLegalBERT embeddings), but works equally well for any contract, notice, or case file in PDF, image, or text form.

---

## ✨ Features

### 📂 Multi-format document ingestion
Upload any number of case files in **PDF, TXT, or image** format (PNG, JPG, JPEG, TIFF). The app automatically extracts text from each file — including OCR for scanned/photographed documents — so even non-digital paperwork becomes searchable.

### 🧠 Legal-domain embeddings + vector search
Extracted text is chunked and converted into embeddings using **InLegalBERT** (a BERT model pre-trained specifically on Indian legal text), then stored in a **ChromaDB** vector database. This means retrieval understands legal terminology and phrasing far better than a general-purpose embedding model would.

### 💬 Unlimited Q&A over your documents (RAG)
Ask as many questions as you like about your uploaded documents — facts, dates, monetary figures, clauses, parties, sections of law, obligations, comparisons across multiple documents, and more. A LangChain-orchestrated RAG pipeline retrieves the most relevant chunks from ChromaDB and feeds them to **Google Gemini** to generate accurate, grounded answers, with the source document(s) cited for every claim.

### 📥 Export the conversation as a PDF
Every chat — questions, answers, and cited sources — can be downloaded as a clean, professionally formatted PDF transcript (built with **ReportLab**), styled like an actual chat conversation with a branded header and page numbers. Useful for case notes, client records, or simply keeping a copy for later.

### 🌐 Independent DuckDuckGo web search
A dedicated search box (separate from the document chat) lets you look up legal terms, recent judgments, or any other reference material directly from the sidebar — helpful for cross-checking facts or finding precedents that aren't in your uploaded files.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / App framework | Streamlit |
| LLM | Google Gemini (`gemini-2.5-flash`) via Gemini API |
| Orchestration | LangChain (RAG pipeline) |
| Embeddings | InLegalBERT (legal-domain BERT model) |
| Vector store | ChromaDB |
| Web search | DuckDuckGo Search |
| PDF generation | ReportLab |
| Core language | Python |
| AI paradigm | Generative AI (GenAI) |

---

## ⚙️ How It Works

```
 Upload (PDF / TXT / Image)
          │
          ▼
   Text Extraction (incl. OCR for scanned images)
          │
          ▼
   Chunking + InLegalBERT Embeddings
          │
          ▼
     Stored in ChromaDB (vector store)
          │
          ▼
   User asks a question  ──────────────►  Relevant chunks retrieved
          │                                        │
          │                                        ▼
          │                          Google Gemini generates a grounded
          │                          answer, citing the source document(s)
          ▼
   Answer shown in chat  ──► (optional) Export full chat as PDF
```

Separately, the **web search panel** queries DuckDuckGo directly and shows results in the sidebar — it does not touch the document pipeline, so your indexed files and your open-web lookups stay cleanly separated.

---

## 🙋 Why It's Useful

Legal documents are dense, repetitive, and full of jargon — finding one specific clause, date, or figure inside a 10-page contract or FIR usually means reading the whole thing. This tool removes that friction:

- **Lawyers & paralegals** can quickly pull facts, clauses, or sections of law out of case files instead of manually re-reading them for every query.
- **Law students & researchers** get a fast way to interrogate FIRs, judgments, contracts, or notices while studying or preparing case briefs.
- **Individuals** dealing with a personal legal matter (a rental dispute, a cheque-bounce notice, an employment contract) can understand what their own documents actually say, in plain language.
- **Anyone managing multiple related documents** (e.g., a contract plus its amendments, or an FIR plus witness statements) can ask cross-document questions like *"compare the notice periods across all the documents"* and get a single consolidated answer instead of manually cross-referencing each file.
- The **PDF export** turns every research session into a reusable, shareable record — useful for client files, case diaries, or simply not having to redo the same research twice.
- The **web search panel** keeps quick reference lookups (terminology, recent judgments, context) one click away, without breaking focus on the document conversation.

In short: it turns a stack of static legal paperwork into something you can actually talk to.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A [Google Gemini API key](https://aistudio.google.com)
- (Optional) An Indian Kanoon API token — the app falls back to a free scraper mode if left blank

### Installation
```bash
git clone <your-repo-url>
cd legal-ai-research-assistant
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

### Configure
1. Open the sidebar and enter your **Gemini API Key** (and optionally your Indian Kanoon token).
2. Choose your embedding device (`cpu` or `cuda`).
3. Click **🚀 Load Pipeline**.
4. Upload your case files (PDF, image, or TXT) — they'll be processed and indexed automatically.
5. Start asking questions in the chat box.
6. Use **📥 Download Chat as PDF** anytime to export the conversation.
7. Use the **🌐 Web Search** box in the sidebar for independent lookups.

---

## 📁 Project Structure

```
.
├── app.py            # Streamlit UI — chat, sidebar, document upload, layout
├── pipeline.py        # LegalAIPipeline — document processing, embeddings, RAG logic
├── pdf_export.py       # Chat transcript → downloadable PDF (ReportLab)
├── web_search.py       # Standalone DuckDuckGo search helper
├── requirements.txt
└── README.md
```

---

## ⚠️ Disclaimer

This tool is intended as a **research and document-comprehension aid**, not a substitute for advice from a qualified, licensed advocate. Always verify critical facts, dates, and legal interpretations against the original documents and consult a legal professional before relying on any output for real-world decisions.
