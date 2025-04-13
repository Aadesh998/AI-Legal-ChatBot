# ⚖️ AI Legal Chatbot

An intelligent and interactive legal chatbot that can read legal PDF documents and answer questions based on their contents. Built using **Streamlit**, **Groq’s LLM API**, and **MiniCPM Vision Model**, this tool offers a seamless way to chat with legal documents or simply ask general legal questions without needing a document.


## 🚀 Purpose

The AI Legal Chatbot is designed to:

- Extract text from scanned or digital legal PDFs using **OCR (Optical Character Recognition)**.
- Understand legal language and summarize or answer questions from uploaded documents.
- Support **context-aware conversation** powered by **LLMs** via Groq.
- Function as a general legal assistant when no document is uploaded.


## 📦 Installation

Before running the app, make sure the following libraries are installed:

```bash
pip install streamlit
pip install python-dotenv
pip install groq
pip install pymupdf
pip install torch
pip install pillow
pip install transformers
pip install sentencepiece
pip install accelerate
pip install bitsandbytes
pip install faiss-cpu
pip install sentence-transformers
```

## 🧠 Functionality

### ✅ With PDF Upload (Legal Document Analysis)
- Upload a legal PDF from the **sidebar**.
- The app performs **OCR on all pages** using the **MiniCPM-LLaMA3 Vision model**.
- The extracted text is **split and embedded** using `sentence-transformers`.
- **FAISS** is used to build a **retrieval index**.
- Any user question will **search this index** and **query Groq's LLM** using the most relevant content.

### 💬 Without PDF (General Legal Chat)
- If no document is uploaded, you can still **chat freely**.
- The bot responds using its **general legal knowledge** via the LLM.

### 🔁 Session Memory
- The chatbot **remembers previous messages** to make the conversation more coherent.
- **OCR is performed only once per document** using a session flag (`ocr_done`) to prevent repetition.

## 🧪 How to Run

```bash
streamlit run app.py
```

## 🌐 Deployment Options

You can deploy this app on platforms like:

- 🧠 **Google Colab**
- 🚀 **Render** – for production-ready deployment


## 👨‍⚖️ Ideal Use Cases

- 🧑‍🔬 Legal researchers looking to **extract and query large legal documents**
- 🎓 Law students exploring **Indian Penal Code** or other acts
- 💻 Developers prototyping **LLM + OCR-based document QA systems**

## 🙏 Acknowledgements

- 🔗 [Groq API](https://groq.com)
- 🧠 [MiniCPM Vision LLM](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4)
- 📺 [Streamlit](https://streamlit.io)
