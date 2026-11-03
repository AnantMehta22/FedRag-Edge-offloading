"""
=============================================================
  RAG Pipeline - Complete Working Prototype
  File: rag_pipeline.py
  Directory: c:\\Users\\admin\\Desktop\\RAG PIPELINE
  
  This script implements a full Retrieval-Augmented Generation
  pipeline using:
    - PyPDF for document loading
    - RecursiveCharacterTextSplitter for chunking
    - HuggingFace sentence-transformers for embeddings
    - ChromaDB for vector storage & similarity search
    - Ollama (llama3) for local LLM generation
    - BM25 for sparse keyword retrieval
    - Hybrid search with Reciprocal Rank Fusion
=============================================================
"""

import os
import sys
import time
import textwrap
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ══════════════════════════════════════════════════════════════
# STEP 1: CONFIGURATION
# ══════════════════════════════════════════════════════════════
DOCUMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(DOCUMENTS_DIR, "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

print("=" * 70)
print("  🚀 RAG PIPELINE — Complete Working Prototype")
print("=" * 70)
print(f"  Documents Directory : {DOCUMENTS_DIR}")
print(f"  Vector DB Directory : {CHROMA_DB_DIR}")
print(f"  Embedding Model     : {EMBEDDING_MODEL}")
print(f"  LLM Model (Ollama)  : {OLLAMA_MODEL}")
print(f"  Chunk Size          : {CHUNK_SIZE} chars")
print(f"  Chunk Overlap       : {CHUNK_OVERLAP} chars")
print(f"  Top-K Results       : {TOP_K}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# STEP 2: DOCUMENT LOADING
# ══════════════════════════════════════════════════════════════
print("\n📂 [STEP 2] Loading PDF documents...")

from langchain_community.document_loaders import PyPDFLoader

pdf_files = list(Path(DOCUMENTS_DIR).glob("*.pdf"))
if not pdf_files:
    print("  ❌ No PDF files found in the directory!")
    print(f"  Please add PDF files to: {DOCUMENTS_DIR}")
    sys.exit(1)

all_documents = []
for pdf_path in pdf_files:
    print(f"  📄 Loading: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    all_documents.extend(docs)
    print(f"     → Extracted {len(docs)} pages")

print(f"\n  ✅ Total pages loaded: {len(all_documents)}")

# ══════════════════════════════════════════════════════════════
# STEP 3: TEXT CHUNKING
# ══════════════════════════════════════════════════════════════
print("\n✂️  [STEP 3] Chunking documents...")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

all_chunks = text_splitter.split_documents(all_documents)

print(f"  ✅ Created {len(all_chunks)} text chunks")
print(f"  📊 Average chunk length: {sum(len(c.page_content) for c in all_chunks) // len(all_chunks)} chars")
print(f"  📏 Largest chunk: {max(len(c.page_content) for c in all_chunks)} chars")
print(f"  📏 Smallest chunk: {min(len(c.page_content) for c in all_chunks)} chars")

# Show a sample chunk
print("\n  --- Sample Chunk (first 300 chars) ---")
print(f"  {all_chunks[0].page_content[:300]}...")
print(f"  Metadata: {all_chunks[0].metadata}")

# ══════════════════════════════════════════════════════════════
# STEP 4: EMBEDDING MODEL LOADING
# ══════════════════════════════════════════════════════════════
print("\n🔢 [STEP 4] Loading embedding model...")

from langchain_community.embeddings import HuggingFaceEmbeddings

start_time = time.time()
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
embed_time = time.time() - start_time

# Test embedding
test_vector = embeddings.embed_query("test")
print(f"  ✅ Embedding model loaded in {embed_time:.2f}s")
print(f"  📐 Vector dimensions: {len(test_vector)}")

# ══════════════════════════════════════════════════════════════
# STEP 5: VECTOR DATABASE (ChromaDB) — INDEXING
# ══════════════════════════════════════════════════════════════
print("\n🗄️  [STEP 5] Building ChromaDB vector index...")

from langchain_community.vectorstores import Chroma

start_time = time.time()
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_DIR,
    collection_name="rag_research_papers",
    collection_metadata={"hnsw:space": "cosine"}
)
index_time = time.time() - start_time

print(f"  ✅ Indexed {len(all_chunks)} chunks into ChromaDB in {index_time:.2f}s")
print(f"  💾 Persisted to: {CHROMA_DB_DIR}")

# ══════════════════════════════════════════════════════════════
# STEP 6: RETRIEVER SETUP (Dense + Sparse Hybrid)
# ══════════════════════════════════════════════════════════════
print("\n🔍 [STEP 6] Setting up Hybrid Retriever (Dense + BM25 Sparse)...")

# 6a: Dense Vector Retriever (Cosine Similarity via ChromaDB)
dense_retriever = vectorstore.as_retriever(
    search_type="mmr",             # Maximum Marginal Relevance for diversity
    search_kwargs={
        "k": TOP_K,
        "fetch_k": 20,             # Fetch 20 candidates, re-rank to top-K
        "lambda_mult": 0.7         # 0.7 = slightly favor relevance over diversity
    }
)
print("  ✅ Dense Vector Retriever (MMR) ready")

# 6b: Sparse BM25 Keyword Retriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(all_chunks)
bm25_retriever.k = TOP_K
print("  ✅ Sparse BM25 Retriever ready")

# 6c: Hybrid Ensemble Retriever (Reciprocal Rank Fusion)
from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]   # 40% BM25 (keywords) + 60% Dense (semantics)
)
print("  ✅ Hybrid Ensemble Retriever (RRF) ready")
print("     Weights: BM25=0.4, Dense=0.6")

# ══════════════════════════════════════════════════════════════
# STEP 7: LLM SETUP (Ollama - Local)
# ══════════════════════════════════════════════════════════════
print(f"\n🧠 [STEP 7] Connecting to Ollama LLM ({OLLAMA_MODEL})...")

from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0.1,       # Low temperature for factual, grounded answers
    num_predict=1024,      # Max tokens to generate
)

# Quick connectivity test
print("  ⏳ Testing Ollama connectivity...")
try:
    test_response = llm.invoke("Say 'Hello' in one word.")
    print(f"  ✅ Ollama ({OLLAMA_MODEL}) connected successfully!")
except Exception as e:
    print(f"  ❌ Ollama connection failed: {e}")
    print("  Make sure Ollama is running: `ollama serve`")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# STEP 8: RAG CHAIN CONSTRUCTION
# ══════════════════════════════════════════════════════════════
print("\n⛓️  [STEP 8] Building RAG Chain...")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Prompt template with strict grounding guardrails
RAG_PROMPT_TEMPLATE = """You are a helpful and precise research assistant. 
Answer the question based ONLY on the following context retrieved from research documents.

RULES:
1. Only use information present in the Context below.
2. If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this."
3. Be specific and cite which document/page the information comes from when possible.
4. Structure your answer clearly with bullet points or numbered lists when appropriate.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

QUESTION: {question}

ANSWER:"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def format_retrieved_docs(docs):
    """Format retrieved documents with source attribution."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        # Extract just the filename from the full path
        filename = os.path.basename(source)
        formatted.append(
            f"[Document {i} | Source: {filename} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)

# Build the RAG chain using LCEL (LangChain Expression Language)
rag_chain = (
    {
        "context": hybrid_retriever | format_retrieved_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("  ✅ RAG Chain built successfully!")
print("     Pipeline: Query → Hybrid Retriever → Format Context → Prompt → Ollama LLM → Answer")

# ══════════════════════════════════════════════════════════════
# STEP 9: INTERACTIVE Q&A SESSION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  🎉 RAG PIPELINE READY! Enter your questions below.")
print("  Type 'quit' or 'exit' to stop.")
print("  Type 'sources' to see raw retrieved chunks for the last query.")
print("  Type 'stats' to see pipeline statistics.")
print("=" * 70)

query_count = 0
total_retrieval_time = 0
total_generation_time = 0
last_retrieved_docs = []

while True:
    print()
    user_query = input("  📝 Your Question: ").strip()
    
    if not user_query:
        continue
    
    if user_query.lower() in ["quit", "exit", "q"]:
        print("\n  👋 Goodbye! RAG Pipeline session ended.")
        break
    
    if user_query.lower() == "stats":
        print(f"\n  📊 Pipeline Statistics:")
        print(f"     Queries processed     : {query_count}")
        print(f"     Total chunks indexed  : {len(all_chunks)}")
        print(f"     Total PDF pages       : {len(all_documents)}")
        print(f"     Embedding dimensions  : {len(test_vector)}")
        if query_count > 0:
            print(f"     Avg retrieval time    : {total_retrieval_time/query_count:.2f}s")
            print(f"     Avg generation time   : {total_generation_time/query_count:.2f}s")
        continue
    
    if user_query.lower() == "sources":
        if last_retrieved_docs:
            print(f"\n  📚 Retrieved Chunks for Last Query ({len(last_retrieved_docs)} chunks):")
            for i, doc in enumerate(last_retrieved_docs, 1):
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "N/A")
                preview = doc.page_content[:200].replace("\n", " ")
                print(f"\n  [{i}] Source: {source} | Page: {page}")
                print(f"      {preview}...")
        else:
            print("  No previous query results available.")
        continue
    
    # Execute the RAG pipeline
    query_count += 1
    print(f"\n  ⏳ Processing query #{query_count}...")
    
    # Step A: Retrieve
    retrieval_start = time.time()
    last_retrieved_docs = hybrid_retriever.invoke(user_query)
    retrieval_time = time.time() - retrieval_start
    total_retrieval_time += retrieval_time
    print(f"  🔍 Retrieved {len(last_retrieved_docs)} chunks in {retrieval_time:.2f}s")
    
    # Step B: Generate
    generation_start = time.time()
    try:
        answer = rag_chain.invoke(user_query)
        generation_time = time.time() - generation_start
        total_generation_time += generation_time
        
        print(f"  🧠 Generated answer in {generation_time:.2f}s")
        print(f"\n  {'─' * 60}")
        print(f"  📖 ANSWER:")
        print(f"  {'─' * 60}")
        # Word-wrap the answer for clean display
        for line in answer.split("\n"):
            wrapped = textwrap.fill(line, width=65, initial_indent="  ", subsequent_indent="  ")
            print(wrapped)
        print(f"  {'─' * 60}")
        
        # Show source summary
        sources = set()
        for doc in last_retrieved_docs:
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "?")
            sources.add(f"{src} (p.{page})")
        print(f"  📚 Sources: {', '.join(sorted(sources))}")
        
    except Exception as e:
        print(f"  ❌ Generation error: {e}")
        print("  Make sure Ollama is running: `ollama serve`")
