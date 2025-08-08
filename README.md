# 📄 PDF Question Answering App using RAG, LangChain, Groq, and Streamlit

This project is an interactive PDF-based Question Answering (QnA) application built using **Retrieval-Augmented Generation (RAG)** with the **LangChain** framework. It allows users to upload a PDF, ask questions about its content, and get accurate, context-aware responses using **Groq's Gemma-2 9B-IT** model and **Hugging Face embeddings**.

---

## 🚀 Features

- 📄 Upload any PDF document
- 💬 Ask natural language questions based on PDF content
- ⚡ Real-time response generation using **Groq (Gemma-2 9B-IT)**
- 🔍 Contextual retrieval powered by **FAISS** and **Hugging Face embeddings**
- 🧠 Prompt engineering with LangChain's chat prompt templates
- 🖥️ Intuitive **Streamlit UI**

---

## 🧰 Tech Stack

| Layer        | Tools Used |
|--------------|------------|
| LLM Inference | [Groq API](https://groq.com) with Gemma-2 9B-IT |
| Embeddings   | [Hugging Face Transformers](https://huggingface.co) (`all-MiniLM-L6-v2`) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| Framework    | [LangChain](https://www.langchain.com/) |
| UI           | [Streamlit](https://streamlit.io/) |
| PDF Parsing  | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) |

---

## 📂 Project Structure

├── app.py # Streamlit frontend & core logic
├── .env # API keys and environment variables
├── requirements.txt # Python dependencies
└── README.md # Project documentation



## 🔑 Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token

## 🛠️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/shravanssr11/pdf-qna-rag-app.git
cd pdf-qna-rag-app

2. **Create virtual envoirnment**
python -m venv venv
source venv/bin/activate 

3. **Install Dependencies**
pip install -r requirements.txt

4. **Run the Streamlit app**
streamlit run app.py




