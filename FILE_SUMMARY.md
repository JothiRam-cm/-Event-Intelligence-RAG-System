# File-by-File Summary

## Python Files Overview

### 1. [`demo.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/demo.py) (53 lines)
**Role:** Streamlit web application interface  
**What it does:** Provides interactive UI for querying the RAG system  
**Key features:**
- Text input for user queries
- "Ask" button to trigger RAG pipeline
- Optional context viewer for debugging
- Expandable event details with metadata

**Dependencies:** `streamlit`, `src.rag_pipeline`, `src.retriever`

---

### 2. [`src/load_data.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/load_data.py) (29 lines)
**Role:** CSV data loader  
**What it does:** Loads and validates the event dataset  
**Key function:** `load_event_data()` → Returns DataFrame (3921 rows × 104 columns)

**Features:**
- File existence check
- Data shape inspection
- Column listing
- Sample row display

---

### 3. [`src/feature_engineering.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/feature_engineering.py) (103 lines)
**Role:** Transform structured data into narratives  
**What it does:** Converts CSV rows into human-readable event descriptions

**Key functions:**
- `safe_str()` - Handle missing values
- `build_event_text()` - Create narrative from row
- `build_metadata()` - Extract searchable metadata
- `normalize_component_id()` - Standardize IDs
- `create_event_documents()` - Generate LangChain Documents

**Output:** LangChain Document objects with page_content (text) and metadata (dict)

---

### 4. [`src/embedding.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/embedding.py) (11 lines)
**Role:** Embedding model configuration  
**What it does:** Provides the model for text-to-vector conversion

**Key function:** `get_embedding_model()` → Returns HuggingFaceEmbeddings

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- 384 dimensions
- Optimized for semantic similarity
- Fast inference

---

### 5. [`src/vectordb.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/vectordb.py) (29 lines)
**Role:** Vector database management  
**What it does:** Creates and loads ChromaDB instances

**Key functions:**
- `create_vector_db()` - Build new database from documents
- `load_vector_db()` - Load existing database from disk

**Storage:** `db/chroma/` (persistent)

---

### 6. [`src/build_vector_store.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/build_vector_store.py) (17 lines)
**Role:** One-time setup script  
**What it does:** Orchestrates vector database creation

**Workflow:**
1. Load CSV data
2. Create event documents
3. Get embedding model
4. Build and persist vector database

**Usage:** `python src/build_vector_store.py` (run once during setup)

---

### 7. [`src/retriever.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/retriever.py) (37 lines)
**Role:** Hybrid retrieval engine  
**What it does:** Finds relevant events using metadata filters + semantic search

**Key functions:**
- `extract_component_id()` - Parse component ID from query
- `normalize_id()` - Standardize ID format
- `get_relevant_events()` - Main retrieval function

**Logic:**
- Detects numeric IDs in query (e.g., "component 103")
- Applies metadata filter if ID found
- Performs semantic similarity search
- Returns top-k results

---

### 8. [`src/prompt.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/prompt.py) (22 lines)
**Role:** RAG prompt template  
**What it does:** Defines instructions for the LLM

**Key function:** `get_rag_prompt()` → Returns ChatPromptTemplate

**Prompt instructions:**
- Use ONLY provided context
- Summarize patterns
- Distinguish facts from missing info
- State when root cause unavailable
- Provide structured answers

---

### 9. [`src/llm.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/llm.py) (14 lines)
**Role:** LLM configuration (standalone version)  
**What it does:** Configures Google Gemini model

**Key function:** `get_llm()` → Returns ChatGoogleGenerativeAI

**Configuration:**
- Model: `gemini-1.5-flash`
- Temperature: 0.2 (low for consistency)
- API key from environment variable

---

### 10. [`src/rag_pipeline.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/rag_pipeline.py) (42 lines)
**Role:** Main RAG orchestrator  
**What it does:** Coordinates the complete RAG workflow

**Key function:** `run_rag(query)` → Returns answer string

**Pipeline:**
1. Retrieve relevant events
2. Build context from documents
3. Apply RAG prompt template
4. Invoke LLM with context + question
5. Return generated answer

**Note:** Uses `gemini-2.5-flash` (newer version)

---

### 11. [`src/test_retreiver.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/test_retreiver.py) (20 lines)
**Role:** Retrieval testing script  
**What it does:** Tests hybrid search without LLM

**Test query:** "Why are there so many critical alarms from component 103?"

**Output:**
- Number of retrieved documents
- First 600 chars of each document
- Metadata for each document

**Usage:** `python src/test_retreiver.py`

---

### 12. [`src/test_rag.py`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/src/test_rag.py) (14 lines)
**Role:** End-to-end RAG testing  
**What it does:** Tests complete pipeline with LLM

**Test query:** "Why are there so many critical alarms from component 103?"

**Output:** Final answer from RAG system

**Usage:** `python src/test_rag.py`

---

## Module Dependency Graph

```
demo.py
  └─ rag_pipeline.py
       ├─ retriever.py
       │    ├─ embedding.py
       │    └─ vectordb.py
       ├─ prompt.py
       └─ llm.py (inline)

build_vector_store.py
  ├─ load_data.py
  ├─ feature_engineering.py
  ├─ embedding.py
  └─ vectordb.py
```

---

## Total Code Statistics

| Metric | Value |
|--------|-------|
| **Total Python Files** | 12 |
| **Total Lines of Code** | 391 |
| **Average File Size** | 33 lines |
| **Largest File** | feature_engineering.py (103 lines) |
| **Smallest File** | embedding.py (11 lines) |

---

## File Categories

### Setup & Configuration (3 files)
- `embedding.py` - Embedding model
- `vectordb.py` - Database operations
- `llm.py` - LLM configuration

### Data Processing (2 files)
- `load_data.py` - CSV loading
- `feature_engineering.py` - Narrative generation

### Core Pipeline (3 files)
- `retriever.py` - Hybrid search
- `prompt.py` - Prompt template
- `rag_pipeline.py` - RAG orchestration

### User Interface (1 file)
- `demo.py` - Streamlit app

### Scripts (3 files)
- `build_vector_store.py` - Setup script
- `test_retreiver.py` - Retrieval test
- `test_rag.py` - RAG test

---

**For detailed function-level documentation, see [`DOCUMENTATION.md`](file:///c:/Users/cmjot/OneDrive/Desktop/Rag/rag_assignment_JR/DOCUMENTATION.md)**
