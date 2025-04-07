import express from 'express';
import multer from 'multer';
import cors from 'cors';
import dotenv from 'dotenv';
import fs from 'fs/promises';
import path from 'path';
import PDFParser from 'pdf2json';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import OpenAI from 'openai';

// Load environment variables
dotenv.config();
const { OPENAI_API_KEY, PORT } = process.env;

// Initialize Express server
const app = express();
const upload = multer({ dest: './uploads/' });

app.use(express.json());
app.use(cors());

// Initialize embeddings
const embeddings = new OpenAIEmbeddings({
  apiKey: OPENAI_API_KEY,
  model: 'text-embedding-3-small',
});

// Initialize OpenAI LLM client
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// Initialize or load Faiss vector store
const storePath = './faiss_store';
const metadataPath = './faiss_store/metadata.json';
let vectorStore;
let metadata = {};

async function initVectorStore() {
  try {
    vectorStore = await FaissStore.load(storePath, embeddings);
    console.log('✅ Loaded existing Faiss store.');
  } catch {
    vectorStore = new FaissStore(embeddings, {});
    console.log('✅ Initialized new Faiss store.');
  }

  // Load existing metadata if available
  try {
    metadata = JSON.parse(await fs.readFile(metadataPath, 'utf-8'));
    console.log('✅ Loaded metadata.');
  } catch {
    metadata = {};
    console.log('✅ Initialized new metadata store.');
  }
}

await initVectorStore();

// PDF text extraction
async function extractText(filePath) {
  const pdfParser = new PDFParser();

  return new Promise((resolve, reject) => {
    pdfParser.on("pdfParser_dataError", err => reject(err.parserError));
    pdfParser.on("pdfParser_dataReady", pdfData => {
      const text = pdfData.Pages
        .flatMap(page => page.Texts)
        .map(t => decodeURIComponent(t.R[0].T))
        .join(' ');
      resolve(text);
    });

    pdfParser.loadPDF(filePath);
  });
}

// Endpoint to handle PDF file upload
app.post('/upload', upload.single('file'), async (req, res) => {
  const { path: tempPath, originalname } = req.file;
  const ext = path.extname(originalname).toLowerCase();

  if (ext !== '.pdf') {
    await fs.unlink(tempPath);
    return res.status(400).json({ detail: 'Only PDF files are supported.' });
  }

  const doc_id = Date.now().toString();

  try {
    const text = await extractText(tempPath);
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const chunks = await splitter.splitText(text);

    const docs = chunks.map(chunk => ({
      pageContent: chunk,
      metadata: { doc_id, filename: originalname },
    }));

    await vectorStore.addDocuments(docs);
    await vectorStore.save(storePath);

    metadata[doc_id] = originalname;
    await fs.writeFile(metadataPath, JSON.stringify(metadata));

    await fs.unlink(tempPath);

    res.json({ message: 'Document uploaded and indexed.', doc_id });
  } catch (err) {
    await fs.unlink(tempPath);
    res.status(500).json({ detail: err.message });
  }
});

// Endpoint for querying documents (real QA using GPT-4)
app.post('/ask', upload.none(), async (req, res) => {
  const { question, doc_id } = req.body;

  try {
    let results;
    if (doc_id && metadata[doc_id]) {
      results = await vectorStore.similaritySearch(question, 10);
      results = results.filter(doc => doc.metadata.doc_id === doc_id);
    } else {
      results = await vectorStore.similaritySearch(question, 5);
    }

    const context = results.map(doc => doc.pageContent).join('\n\n');

    if (!context.trim()) {
      return res.json({ answer: "Sorry, I couldn’t find a good answer in the documents." });
    }

    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        { role: "system", content: "You are an assistant answering questions based solely on provided context." },
        { role: "user", content: `Context:\n${context}\n\nQuestion:\n${question}` },
      ],
    });

    res.json({ answer: completion.choices[0].message.content.trim() });
  } catch (err) {
    res.status(500).json({ detail: err.message });
  }
});

// Endpoint to list documents with doc_id
app.get('/documents', async (req, res) => {
  const docs = Object.entries(metadata).map(([doc_id, filename]) => ({ doc_id, filename }));
  res.json(docs);
});

// Start server
app.listen(PORT || 8000, () => {
  console.log(`🚀 Server running on port ${PORT || 8000}`);
});
