import streamlit as st
import os
import shutil
import base64
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from mistralai import Mistral
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF

# ========== Setup ==========
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
mistral_api_key = os.getenv('MISTRAL_API_KEY')

client = Groq(api_key=groq_api_key)
mistral_client = Mistral(api_key=mistral_api_key)

st.set_page_config(page_title="AI Legal Chatbot", layout="wide")
st.title("⚖️ AI Legal Chatbot")

# ========== Load Embedding Model ==========
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

# ========== OCR Functions ==========
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def mistral_image_ocr(image_path):
    base64_image = encode_image(image_path)
    response = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        }
    )
    return "\n\n".join(page.markdown for page in response.pages)

def pdf_to_images(pdf_path, output_folder="pdf_images"):
    import fitz  # PyMuPDF
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    pdf_doc = fitz.open(pdf_path)

    for page_number in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(os.path.join(output_folder, f"page_{page_number + 1}.png"))

    return output_folder

def split_chunks(text, chunk_size=500):
    sentences = text.split('. ')
    chunks, chunk = [], ""
    for sent in sentences:
        if len(chunk) + len(sent) < chunk_size:
            chunk += sent + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sent + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# ========== Sidebar Upload ==========
with st.sidebar:
    st.header("📎 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload your legal documents (PDF or images)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    def process_uploaded_file(uploaded_file):
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.lower().endswith(".pdf"):
            image_folder = pdf_to_images(file_path)
            extracted_text = ""
            for img_name in os.listdir(image_folder):
                img_path = os.path.join(image_folder, img_name)
                extracted_text += mistral_image_ocr(img_path) + "\n"
            return extracted_text
        else:
            return mistral_image_ocr(file_path)

    if uploaded_files and "last_uploaded_files" not in st.session_state or \
       [f.name for f in uploaded_files] != st.session_state.get("last_uploaded_files", []):

        with st.spinner("🔍 Extracting and embedding text..."):
            all_extracted_text = ""
            for file in uploaded_files:
                all_extracted_text += process_uploaded_file(file) + "\n"

            chunks = split_chunks(all_extracted_text)
            embeddings = embedding_model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))

            st.session_state.pdf_mode = True
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.last_uploaded_files = [f.name for f in uploaded_files]
            st.success("Files processed successfully!")

        shutil.rmtree("temp", ignore_errors=True)

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
        q_embed = embedding_model.encode([user_input])
        _, top_idxs = st.session_state.index.search(np.array(q_embed), 3)
        retrieved = "\n".join([st.session_state.chunks[i] for i in top_idxs[0]])

        context = f"Answer the question based on the following:\n{retrieved}"
        messages = [
            {"role": "system", "content": "You are a helpful AI that answers legal questions from given context."},
            {"role": "user", "content": f"{context}\n\nQuestion: {user_input}"}
        ]
    else:
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
