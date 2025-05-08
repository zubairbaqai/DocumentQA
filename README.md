
# 📄 Document Intelligence API


A REST API built with Node.js and Express to upload PDF documents, process their contents, and answer questions using advanced AI techniques powered by OpenAI and LangChain.

## 🎯 Objective

This project demonstrates REST API development, JavaScript proficiency, and the design and implementation of AI-powered solutions that intelligently answer questions based on document content.

## 🛠️ Tech Stack

- **Backend:** Node.js, Express.js
- **AI & NLP:** OpenAI GPT-4, LangChain.js, FAISS Vector Search
- **Document Parsing:** PDF2JSON
- **Frontend (Bonus):** Vanilla JavaScript, HTML, CSS
- **Containerization:** Docker, Docker Compose

## ✅ Features

- **Document Upload:** Supports PDF files, indexes content into a vector store.
- **Intelligent Question-Answering:** Answers questions using GPT-4 based on document contents.
- **Robust Error Handling:** Handles invalid uploads and file types gracefully.
- **Document Management:** Lists uploaded documents.
- **Frontend Interface (Bonus):** Simple, intuitive UI for document upload and question submission.

## 🚀 Getting Started

### Prerequisites

- Docker
- Docker Compose
- OpenAI API Key

### Clone Repository

```bash
git clone https://github.com/your_username/document-intelligence-api.git
cd document-intelligence-api
```

## ⚙️ Setup & Run (Docker)

### Environment Setup

Create `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
PORT=8000
```

### Run with Docker Compose

```bash
docker compose up --build
```

## 🌐 API Endpoints

### Upload Document

```http
POST /upload
```

```bash
curl -F "file=@/path/to/document.pdf" http://localhost:8000/upload
```

### Ask a Question

```http
POST /ask
```

```bash
curl -X POST http://localhost:8000/ask \
  -F "question=Your question" \
  -F "doc_id=document_id"
```

### List Documents

```http
GET /documents
```

```bash
curl http://localhost:8000/documents
```

## 🖥️ Frontend

Open `index.html` in your browser to interact.

