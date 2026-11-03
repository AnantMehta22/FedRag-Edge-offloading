"""
=============================================================
  RAG Pipeline - Demo Runner (Non-Interactive)
  Runs 3 demo queries to showcase the complete pipeline
=============================================================
"""

import os
import sys
import time
import textwrap
import warnings
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
DOCUMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(DOCUMENTS_DIR, "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# Demo queries to run
DEMO_QUERIES = [
    "What is RAG and why was it created?",
    "Explain the different chunking strategies for RAG pipelines",
    "What is the difference between BM25 sparse search and dense vector search?",
]

print("=" * 70)
print("  RAG PIPELINE — Demo Runner")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# STEP 1: LOAD DOCUMENTS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading PDF documents...")
from langchain_community.document_loaders import PyPDFLoader

pdf_files = list(Path(DOCUMENTS_DIR).glob("*.pdf"))
all_documents = []
for pdf_path in pdf_files:
    print(f"  Loading: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    all_documents.extend(docs)
print(f"  -> Total pages loaded: {len(all_documents)}")

# ══════════════════════════════════════════════════════════════
# STEP 2: CHUNK DOCUMENTS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 2] Chunking documents...")
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
all_chunks = text_splitter.split_documents(all_documents)
print(f"  -> Created {len(all_chunks)} text chunks (avg {sum(len(c.page_content) for c in all_chunks) // len(all_chunks)} chars)")

# ══════════════════════════════════════════════════════════════
# STEP 3: LOAD EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════
print("\n[STEP 3] Loading embedding model...")
from langchain_community.embeddings import HuggingFaceEmbeddings

t0 = time.time()
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print(f"  -> Loaded in {time.time()-t0:.1f}s | Dimensions: {len(embeddings.embed_query('test'))}")

# ══════════════════════════════════════════════════════════════
# STEP 4: BUILD VECTOR INDEX
# ══════════════════════════════════════════════════════════════
print("\n[STEP 4] Building ChromaDB vector index...")
from langchain_community.vectorstores import Chroma

t0 = time.time()
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_DIR,
    collection_name="rag_demo",
    collection_metadata={"hnsw:space": "cosine"}
)
print(f"  -> Indexed {len(all_chunks)} chunks in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════
# STEP 5: SETUP RETRIEVERS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 5] Setting up Hybrid Retriever...")

# Dense retriever (MMR)
dense_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.7}
)

# Sparse BM25 retriever
from langchain_community.retrievers import BM25Retriever
bm25_retriever = BM25Retriever.from_documents(all_chunks)
bm25_retriever.k = TOP_K

# Hybrid Ensemble (RRF)
from langchain_classic.retrievers import EnsembleRetriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]
)
print("  -> Hybrid Retriever ready (BM25=0.4 + Dense=0.6)")

# ══════════════════════════════════════════════════════════════
# STEP 6: CONNECT LLM
# ══════════════════════════════════════════════════════════════
print(f"\n[STEP 6] Connecting to Ollama ({OLLAMA_MODEL})...")
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1, num_predict=512)
print("  -> Connected!")

# ══════════════════════════════════════════════════════════════
# STEP 7: BUILD RAG CHAIN
# ══════════════════════════════════════════════════════════════
print("\n[STEP 7] Building RAG Chain...")
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

RAG_PROMPT = """You are a helpful research assistant.
Answer the question based ONLY on the following context from research documents.
If the context doesn't have enough info, say so. Be concise and specific.

--- CONTEXT ---
{context}
--- END CONTEXT ---

QUESTION: {question}

ANSWER:"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        src = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Doc {i} | {src} | Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("  -> RAG Chain ready!")

# ══════════════════════════════════════════════════════════════
# STEP 8: RUN DEMO QUERIES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  RUNNING DEMO QUERIES")
print("=" * 70)

for q_idx, query in enumerate(DEMO_QUERIES, 1):
    print(f"\n{'='*70}")
    print(f"  QUERY {q_idx}/{len(DEMO_QUERIES)}: {query}")
    print(f"{'='*70}")
    
    # Retrieve
    t0 = time.time()
    retrieved_docs = hybrid_retriever.invoke(query)
    retrieval_time = time.time() - t0
    print(f"\n  [Retrieval] Found {len(retrieved_docs)} chunks in {retrieval_time:.2f}s")
    
    # Show retrieved sources
    for i, doc in enumerate(retrieved_docs, 1):
        src = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"    Chunk {i}: [{src} p.{page}] {preview}...")
    
    # Generate
    print(f"\n  [Generation] Generating answer with {OLLAMA_MODEL}...")
    t0 = time.time()
    answer = rag_chain.invoke(query)
    gen_time = time.time() - t0
    
    print(f"  [Generation] Completed in {gen_time:.1f}s")
    print(f"\n  {'~'*60}")
    print(f"  ANSWER:")
    print(f"  {'~'*60}")
    for line in answer.strip().split("\n"):
        wrapped = textwrap.fill(line, width=65, initial_indent="  ", subsequent_indent="  ")
        print(wrapped)
    print(f"  {'~'*60}")

print(f"\n{'='*70}")
print(f"  ALL {len(DEMO_QUERIES)} DEMO QUERIES COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")
print(f"\n  Your RAG pipeline is fully operational.")
print(f"  To run interactively, execute: python rag_pipeline.py")
print(f"  To add more documents, place PDFs in: {DOCUMENTS_DIR}")
