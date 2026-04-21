# GraphReason — Multi-Document AI Analysis

Upload up to 10 PDFs and ask questions across all of them simultaneously. GraphReason routes each question through a hybrid RAG pipeline — graph traversal, deep document reasoning, or both.


## Stack

- **Frontend:** Next.js (TypeScript)
- **Backend:** FastAPI (Python)
- **Auth:** Clerk (JWT)
- **Database:** PostgreSQL via Neon (asyncpg + SQLAlchemy)
- **Knowledge graph:** Neo4j
- **LLM:** GPT-4o / GPT-4o-mini (user-supplied API key)
- **PDF parsing:** pdfplumber + PyMuPDF


## How It Works

**4-stage processing pipeline:**

1. PDF extraction → structured JSON (text, tables, images via GPT-4o Vision)
2. Per-document RLM analysis (entity + concept extraction)
3. Cross-document analysis (contradictions, shared entities, thematic links)
4. Neo4j knowledge graph construction

**Two retrieval paths, auto-routed per question:**

- **Graph** — entity lookups, relationships, cross-document connections, contradictions
- **RLM** — deep reasoning, methodology analysis, nuanced passage interpretation
- **Hybrid** — both paths run in parallel, merged into a single cited answer


## Getting Started

### Backend

```bash
cd app/backend
pip install -r requirements.txt
uvicorn api:app --reload
```

Create `app/.env`:

```env
DATABASE_URL=postgresql+asyncpg://...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...
ENCRYPTION_KEY=...
FRONTEND_URL=http://localhost:3000
```

### Frontend

```bash
cd app/frontend
npm install
npm run dev
```

Create `app/frontend/.env.local`:

```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Requirements

- Python 3.11+
- Node.js 18+
- Neo4j instance (local or Aura)
- Neon (or any Postgres) database
- Clerk account
- OpenAI API key (entered per-user in the app)
