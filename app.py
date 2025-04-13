import streamlit as st
import os
import fitz  # PyMuPDF
import shutil
import torch
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from textwrap import dedent


# ========== Setup ==========
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=api_key)

st.set_page_config(page_title="AI Legal Chatbot", layout="wide")
st.title("⚖️ AI Legal Chatbot")



# ========== Load Models ==========
@st.cache_resource
def load_vision_model():
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
    model.eval()
    return model, tokenizer

model, tokenizer = load_vision_model()

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedder()

# ========== Session State ==========
if "history" not in st.session_state:
    st.session_state.history = []

if "pdf_mode" not in st.session_state:
    st.session_state.pdf_mode = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

# ========== Sidebar Upload ==========
with st.sidebar:
    st.header("📎 Upload PDF")
    uploaded_pdf = st.file_uploader("Upload your legal PDF", type=["pdf"])

    if uploaded_pdf:
        temp_path = "temp_uploaded.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.read())

        def pdf_to_images(pdf_path, output_folder="pdf_images"):
            pdf_doc = fitz.open(pdf_path)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder)

            for page_number in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_number)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.save(os.path.join(output_folder, f"page_{page_number+1}.png"))

            return output_folder

        def ocr_image(img_path):
            img = Image.open(img_path).convert('RGB')
            msgs = [{"role": "user", "content": "Extract the text from the image. Just give only image text."}]
            system_role = "You are an AI that extracts and transcribes text from images."

            res = model.chat(image=img, msgs=msgs, tokenizer=tokenizer, sampling=True,
                             temperature=0.7, system_prompt=system_role)
            return ''.join(res)

        def extract_pdf_text(pdf_file):
            folder = pdf_to_images(pdf_file)
            full_text = ""
            for file in sorted(os.listdir(folder)):
                full_text += ocr_image(os.path.join(folder, file)) + "\n"
            return full_text

        with st.spinner("🔍 Performing OCR..."):
            extracted_text = extract_pdf_text(temp_path)

            def split_chunks(text, chunk_size=500):
                sentences = text.split('. ')
                chunks, chunk = [], ""
                for sent in sentences:
                    if len(chunk) + len(sent) < chunk_size:
                        chunk += sent + ". "
                    else:
                        chunks.append(chunk.strip())
                        chunk = sent + ". "
                if chunk: chunks.append(chunk.strip())
                return chunks

            chunks = split_chunks(extracted_text)
            embeddings = embedding_model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))

            st.session_state.pdf_mode = True
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.success("PDF processed successfully!")

# ========== Chat Section ==========
st.divider()
st.subheader("💬 Chat Interface")

# Display history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# User Input
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if st.session_state.pdf_mode:
        # If PDF is uploaded, use context-based retrieval
        q_embed = embedding_model.encode([user_input])
        _, top_idxs = st.session_state.index.search(np.array(q_embed), 3)
        retrieved = "\n".join([st.session_state.chunks[i] for i in top_idxs[0]])

        context = f"Answer the question based on the following:\n{retrieved}"
        messages = [
            {"role": "system", "content": "You are a helpful AI that answers legal questions from given context."},
            {"role": "user", "content": f"{context}\n\nQuestion: {user_input}"}
        ]
    else:
        # Free-form chat
        messages = st.session_state.history

    with st.spinner("AI is thinking..."):
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        st.chat_message("assistant").write(reply)
        st.session_state.history.append({"role": "assistant", "content": reply})


