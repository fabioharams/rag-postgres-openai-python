# RAG on PostgreSQL — Knowledge Base

> A comprehensive reference for the `rag-postgres-openai-python` repository.
> Last updated: 2026-03-27

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Architecture](#2-architecture)
- [3. Repository Structure](#3-repository-structure)
- [4. Backend (FastAPI)](#4-backend-fastapi)
  - [4.1 Application Entry Point](#41-application-entry-point)
  - [4.2 API Endpoints](#42-api-endpoints)
  - [4.3 RAG Implementation](#43-rag-implementation)
  - [4.4 Database Models](#44-database-models)
  - [4.5 Search Engine (Hybrid Search)](#45-search-engine-hybrid-search)
  - [4.6 Embeddings](#46-embeddings)
  - [4.7 OpenAI Client Factory](#47-openai-client-factory)
  - [4.8 Dependency Injection](#48-dependency-injection)
  - [4.9 Prompt Templates](#49-prompt-templates)
- [5. Frontend (React + FluentUI)](#5-frontend-react--fluentui)
  - [5.1 Pages and Routing](#51-pages-and-routing)
  - [5.2 Components](#52-components)
  - [5.3 API Communication](#53-api-communication)
  - [5.4 Build Configuration](#54-build-configuration)
- [6. Infrastructure (Azure Bicep)](#6-infrastructure-azure-bicep)
  - [6.1 Azure Resources](#61-azure-resources)
  - [6.2 Bicep Module Map](#62-bicep-module-map)
  - [6.3 Key Parameters](#63-key-parameters)
  - [6.4 Environment Variables in Production](#64-environment-variables-in-production)
- [7. Database & Seed Data](#7-database--seed-data)
- [8. Testing](#8-testing)
- [9. Evaluations](#9-evaluations)
- [10. CI/CD Pipelines](#10-cicd-pipelines)
- [11. Scripts](#11-scripts)
- [12. Configuration Reference](#12-configuration-reference)
  - [12.1 Environment Variables](#121-environment-variables)
  - [12.2 LLM Provider Options](#122-llm-provider-options)
  - [12.3 pyproject.toml Tooling Config](#123-pyprojecttoml-tooling-config)
- [13. Development Workflow](#13-development-workflow)
- [14. Key Design Decisions](#14-key-design-decisions)
- [15. Related Documentation](#15-related-documentation)

---

## 1. Project Overview

**RAG on PostgreSQL** is a web-based chat application that uses OpenAI models to answer questions about data stored in a PostgreSQL database with the pgvector extension. It demonstrates a production-ready Retrieval-Augmented Generation (RAG) pattern on Azure.

| Attribute | Value |
|-----------|-------|
| **Backend** | Python 3.12, FastAPI, SQLAlchemy (async), pgvector |
| **Frontend** | React 18, TypeScript, Fluent UI, Vite |
| **Database** | PostgreSQL 15 with pgvector extension |
| **AI Models** | OpenAI GPT-4o-mini (chat), text-embedding-3-large (embeddings) |
| **Deployment Target** | Azure Container Apps |
| **IaC** | Bicep (Azure Resource Manager templates) |
| **Deployment Tool** | Azure Developer CLI (`azd`) |
| **Authentication** | User-assigned Managed Identity (Azure Entra ID) |

### Key Features

- **Hybrid search**: Combines pgvector vector similarity search with PostgreSQL full-text search, merged via Reciprocal Rank Fusion (RRF).
- **OpenAI function calling**: Converts natural language queries into SQL WHERE clauses (e.g., "Climbing gear cheaper than $30?" → `WHERE price < 30`).
- **Multiple LLM providers**: Supports Azure OpenAI, OpenAI.com, Ollama (local), and GitHub Models.
- **Streaming responses**: Real-time token delivery via NDJSON Server-Sent Events.
- **Advanced RAG flow**: Optional two-agent pipeline with query rewriting and filter extraction.
- **Observability**: Azure Application Insights + OpenTelemetry instrumentation.

---

## 2. Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND (React / Vite)                   │
│  React 18 + Fluent UI + TypeScript                         │
│  Chat interface, settings panel, analysis panel             │
│  Built to: src/backend/static/                             │
└────────────────────────┬────────────────────────────────────┘
                         │  HTTP POST /chat  (JSON or NDJSON stream)
                         │  Uses @microsoft/ai-chat-protocol
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 BACKEND (FastAPI - Port 8000)                │
│  Python 3.12 + SQLAlchemy async + openai-agents            │
│  Serves API endpoints + built frontend static files         │
└──────────┬──────────────────┬──────────────────┬────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
┌──────────────────┐ ┌───────────────┐ ┌────────────────────┐
│   PostgreSQL 15  │ │  Azure OpenAI │ │ Application        │
│   + pgvector     │ │               │ │ Insights           │
│                  │ │ • Chat model  │ │ + Log Analytics     │
│ • Items table    │ │   (GPT-4o-   │ │                    │
│ • HNSW indexes   │ │    mini)     │ │ OpenTelemetry      │
│ • Full-text      │ │ • Embed model│ │ instrumentation    │
│   search         │ │   (3-large)  │ │                    │
│ • Vector search  │ │ • Eval model │ │                    │
│                  │ │   (GPT-4)    │ │                    │
└──────────────────┘ └───────────────┘ └────────────────────┘
```

### RAG Data Flow

```
1. User sends question
       │
       ▼
2. [Simple Flow]  ──or──  [Advanced Flow]
       │                        │
       │                   3a. LLM rewrites query
       │                   3b. Extracts filters (brand, price)
       │                   3c. Calls search_database() tool
       │                        │
       ▼                        ▼
4. Embed query → vector (text-embedding-3-large, 1024 dims)
       │
       ▼
5. Hybrid search on PostgreSQL
   ├─ Vector: HNSW cosine distance on embedding column
   ├─ Full-text: ts_rank_cd with plainto_tsquery
   └─ Fusion: RRF score = 1/(k+vector_rank) + 1/(k+fts_rank)
       │
       ▼
6. Top N items returned
       │
       ▼
7. "Answerer" agent generates response using sources + prompt
       │
       ▼
8. Response returned (JSON or NDJSON stream)
   ├─ message: AI answer with citations
   ├─ data_points: retrieved database rows
   └─ thoughts: step-by-step reasoning trace
```

---

## 3. Repository Structure

```
rag-postgres-openai-python/
├── src/
│   ├── backend/                     # FastAPI application
│   │   ├── fastapi_app/             # Main Python package
│   │   │   ├── __init__.py          # App factory (create_app)
│   │   │   ├── api_models.py        # Pydantic request/response models
│   │   │   ├── dependencies.py      # FastAPI dependency injection
│   │   │   ├── embeddings.py        # Embedding generation logic
│   │   │   ├── openai_clients.py    # OpenAI/Azure client factories
│   │   │   ├── postgres_engine.py   # SQLAlchemy async engine setup
│   │   │   ├── postgres_models.py   # ORM models (Item table)
│   │   │   ├── postgres_searcher.py # Hybrid search implementation
│   │   │   ├── rag_base.py          # Abstract RAG base class
│   │   │   ├── rag_simple.py        # Simple RAG flow
│   │   │   ├── rag_advanced.py      # Advanced RAG with query rewriting
│   │   │   ├── query_rewriter.py    # Query → filters via function calling
│   │   │   ├── setup_postgres_database.py  # Schema creation script
│   │   │   ├── setup_postgres_seeddata.py  # Data seeding script
│   │   │   ├── setup_postgres_azurerole.py # Azure role setup
│   │   │   ├── update_embeddings.py # Batch embedding generator
│   │   │   ├── seed_data.json       # Sample product data with embeddings
│   │   │   ├── routes/
│   │   │   │   ├── api_routes.py    # REST API endpoints
│   │   │   │   └── frontend_routes.py # Static file serving
│   │   │   ├── prompts/
│   │   │   │   ├── answer.txt       # Answerer system prompt
│   │   │   │   ├── query.txt        # Query rewriter prompt
│   │   │   │   └── query_fewshots.json # Few-shot examples
│   │   │   └── static/              # Built frontend assets
│   │   ├── Dockerfile               # Container image definition
│   │   ├── entrypoint.sh            # Docker entrypoint
│   │   ├── pyproject.toml           # Package metadata & dependencies
│   │   └── requirements.txt         # Pinned dependencies (364 packages)
│   │
│   └── frontend/                    # React TypeScript application
│       ├── src/
│       │   ├── index.tsx            # Entry point + routing
│       │   ├── api/                 # Backend API client
│       │   ├── components/          # 10 component directories
│       │   ├── pages/               # Chat, Layout, NoPage
│       │   └── assets/              # SVGs and images
│       ├── package.json             # npm dependencies & scripts
│       ├── vite.config.ts           # Vite bundler config
│       ├── tsconfig.json            # TypeScript config
│       └── index.html               # HTML template
│
├── infra/                           # Azure Bicep infrastructure
│   ├── main.bicep                   # Top-level orchestration
│   ├── main.parameters.json         # Parameter definitions
│   ├── web.bicep                    # Web app container config
│   ├── backend-dashboard.bicep      # Monitoring dashboard
│   └── core/                        # Resource modules
│       ├── ai/                      # OpenAI + AI Foundry
│       ├── database/postgresql/     # PostgreSQL Flexible Server
│       ├── host/                    # Container Apps + Registry
│       ├── monitor/                 # App Insights + Log Analytics
│       └── security/                # RBAC role assignments
│
├── tests/                           # Test suite
│   ├── conftest.py                  # Fixtures and test client
│   ├── data.py                      # Test data constants
│   ├── mocks.py                     # Mock objects
│   ├── test_api_routes.py           # API endpoint tests
│   ├── test_postgres_searcher.py    # Search functionality tests
│   ├── test_embeddings.py           # Embedding tests
│   ├── test_openai_clients.py       # Client factory tests
│   ├── test_postgres_engine.py      # DB engine tests
│   ├── test_dependencies.py         # DI tests
│   ├── test_frontend_routes.py      # Frontend serving tests
│   ├── e2e.py                       # Playwright E2E tests
│   └── snapshots/                   # Response snapshot files
│
├── evals/                           # RAG quality evaluation
│   ├── evaluate.py                  # Main evaluation runner
│   ├── generate_ground_truth.py     # Ground truth generator
│   ├── safety_evaluation.py         # Content safety evaluation
│   ├── eval_config.json             # Evaluation metrics config
│   ├── ground_truth.jsonl           # Ground truth Q&A dataset
│   └── results/                     # Evaluation results
│
├── scripts/                         # Deployment & setup scripts
│   ├── setup_postgres_database.sh/.ps1
│   ├── setup_postgres_seeddata.sh/.ps1
│   ├── setup_postgres_azurerole.sh/.ps1
│   └── load_python_env.sh
│
├── docs/                            # Documentation
│   ├── rag_flow.md                  # RAG architecture explanation
│   ├── customize_data.md            # Guide for custom data
│   ├── deploy_existing.md           # Deploying with existing resources
│   ├── evaluation.md                # RAG evaluation methodology
│   ├── safety_evaluation.md         # Safety/harm evaluation
│   ├── loadtesting.md               # Locust load testing
│   ├── monitoring.md                # Application Insights setup
│   ├── using_entra_auth.md          # Entra ID authentication
│   └── images/                      # Architecture diagrams & screenshots
│
├── .github/workflows/               # CI/CD pipelines
│   ├── azure-dev.yaml               # Deploy to Azure
│   ├── app-tests.yaml               # Python tests (3 versions × 3 OS)
│   ├── python-code-quality.yaml     # Ruff + mypy
│   ├── bicep-security-scan.yaml     # IaC security scan
│   └── evaluate.yaml                # RAG evaluation
│
├── .devcontainer/                   # Dev container config
│   ├── devcontainer.json            # VS Code settings + features
│   ├── docker-compose.yaml          # App + PostgreSQL services
│   └── Dockerfile                   # Dev container image
│
├── azure.yaml                       # Azure Developer CLI config
├── pyproject.toml                   # Root project config (ruff, mypy, pytest)
├── requirements-dev.txt             # Development dependencies
├── locustfile.py                    # Load testing script
├── .env.sample                      # Environment variable template
├── .pre-commit-config.yaml          # Pre-commit hooks
├── AGENTS.md                        # Instructions for coding agents
├── CONTRIBUTING.md                  # Contribution guidelines
└── LICENSE.md                       # MIT License
```

---

## 4. Backend (FastAPI)

### 4.1 Application Entry Point

**File**: `src/backend/fastapi_app/__init__.py`

The `create_app()` factory function initializes the FastAPI application:

```
create_app(testing=False)
  → lifespan() async context manager
    → common_parameters()           # Read all env vars
    → create_postgres_engine_from_env()  # Async SQLAlchemy engine
    → create_openai_chat_client()   # Chat LLM client
    → create_openai_embed_client()  # Embedding client
  → FastAPI(docs_url="/docs", lifespan=lifespan)
  → include_router(api_routes)      # /api/* endpoints
  → mount(frontend_routes)          # Static file serving at /
```

All shared state (DB engine, OpenAI clients, config context) is attached to `request.state` during the lifespan and accessed via dependency injection.

### 4.2 API Endpoints

**File**: `src/backend/fastapi_app/routes/api_routes.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/items/{id}` | `GET` | Fetch a single item by ID |
| `/similar` | `GET` | Find N most similar items by vector distance |
| `/search` | `GET` | Search items by text, vector, or hybrid query |
| `/chat` | `POST` | RAG chat (non-streaming) — returns `RetrievalResponse` |
| `/chat/stream` | `POST` | RAG chat (streaming) — returns NDJSON stream of `RetrievalResponseDelta` |

**Chat endpoint flow:**
1. Instantiate `SimpleRAGChat` or `AdvancedRAGChat` based on `use_advanced_flow` override
2. Call `prepare_context()` to search database and build thought steps
3. Call `answer()` or `answer_stream()` to invoke OpenAI agent
4. Return response with message, data points, and thought process

### 4.3 RAG Implementation

The RAG system uses a three-tier architecture:

#### Base Class (`rag_base.py`)

- Abstract class `RAGChatBase` defining the RAG interface
- Loads prompt templates from the `prompts/` directory
- Methods: `get_chat_params()`, `prepare_rag_request()`, `prepare_context()`, `answer()`, `answer_stream()`

#### Simple RAG (`rag_simple.py`)

```
User Question → Embed query → Hybrid search → Format sources → LLM answers
```

- `SimpleRAGChat` — single "Answerer" agent
- Direct search using user query (no rewriting)
- Uses `openai-agents` library with `OpenAIChatCompletionsModel`

#### Advanced RAG (`rag_advanced.py`)

```
User Question → "Searcher" agent rewrites query + extracts filters
             → search_database() tool call → Hybrid search
             → "Answerer" agent generates response
```

- `AdvancedRAGChat` — dual-agent pipeline:
  - **Searcher Agent**: Interprets user query, extracts `PriceFilter`/`BrandFilter`, calls `search_database()` function tool
  - **Answerer Agent**: Generates final answer with citations from search results
- Uses few-shot examples (`query_fewshots.json`) for better filter extraction

### 4.4 Database Models

**File**: `src/backend/fastapi_app/postgres_models.py`

```python
class Item(Base):
    __tablename__ = "items"

    id: int            # Primary key, autoincrement
    type: str          # Product category (e.g., "shoes")
    brand: str         # Brand name (e.g., "AirStrider")
    name: str          # Product name
    description: str   # Product description
    price: float       # Product price
    embedding_3l: Vector(1024)   # text-embedding-3-large embeddings
    embedding_nomic: Vector(768) # nomic-embed-text embeddings (for Ollama)
```

**Indexes:**
- `hnsw_index_for_cosine_items_embedding_3l` — HNSW index, cosine distance, m=16, ef_construction=64
- `hnsw_index_for_cosine_items_embedding_nomic` — HNSW index for Ollama embeddings

**Key methods:**
- `to_dict(include_embedding)` — Serializes to dictionary
- `to_str_for_rag()` — Formats as "Name: ... Description: ... Price: ... Brand: ... Type: ..."
- `to_str_for_embedding()` — Formats for embedding generation

### 4.5 Search Engine (Hybrid Search)

**File**: `src/backend/fastapi_app/postgres_searcher.py`

Class `PostgresSearcher` implements three search modes:

| Mode | Technique | SQL Pattern |
|------|-----------|-------------|
| **Vector** | HNSW cosine distance | `ORDER BY embedding <=> :query_vector` |
| **Text** | PostgreSQL full-text search | `to_tsvector('english', description) @@ plainto_tsquery(...)` with `ts_rank_cd` |
| **Hybrid** | RRF (Reciprocal Rank Fusion) | Vector + FTS CTEs → FULL OUTER JOIN → `1/(k+vector_rank) + 1/(k+fts_rank)` |

**Filtering support:**
- `PriceFilter` — comparisons: `>`, `<`, `>=`, `<=`, `=`
- `BrandFilter` — comparisons: `=`, `!=`
- Filters are generated by the Advanced RAG's Searcher agent via function calling

### 4.6 Embeddings

**File**: `src/backend/fastapi_app/embeddings.py`

`compute_text_embedding()` calls the configured embeddings API:

| Model | Dimensions | Provider |
|-------|-----------|----------|
| `text-embedding-3-large` | 1024 (configurable) | Azure OpenAI / OpenAI.com |
| `text-embedding-3-small` | Configurable | Azure OpenAI / OpenAI.com |
| `text-embedding-ada-002` | 1536 (fixed) | Azure OpenAI / OpenAI.com |
| `nomic-embed-text` | 768 | Ollama (local) |

### 4.7 OpenAI Client Factory

**File**: `src/backend/fastapi_app/openai_clients.py`

Creates async OpenAI clients based on environment configuration:

| Provider | `OPENAI_CHAT_HOST` | Client Class | Auth |
|----------|-------------------|--------------|------|
| Azure OpenAI | `azure` | `AsyncAzureOpenAI` | API key or Managed Identity token |
| OpenAI.com | `openai` | `AsyncOpenAI` | API key |
| Ollama | `ollama` | `AsyncOpenAI` | None (local endpoint) |
| GitHub Models | `github` | `AsyncOpenAI` | GitHub token |

Azure authentication supports:
- Direct API key (`AZURE_OPENAI_KEY`)
- Managed Identity (`APP_IDENTITY_ID`)
- Azure Developer CLI credential (fallback)

### 4.8 Dependency Injection

**File**: `src/backend/fastapi_app/dependencies.py`

FastAPI `Depends()` system provides:

| Type Alias | Provides |
|------------|----------|
| `CommonDeps` | `FastAPIAppContext` — model names, deployment config, embedding settings |
| `DBSession` | `AsyncSession` — SQLAlchemy async database session |
| `ChatClient` | `AsyncOpenAI` — OpenAI chat client |
| `EmbeddingsClient` | `AsyncOpenAI` — OpenAI embeddings client |

### 4.9 Prompt Templates

**Directory**: `src/backend/fastapi_app/prompts/`

| File | Purpose |
|------|---------|
| `answer.txt` | System prompt for the Answerer agent — instructs LLM to act as a salesperson, cite sources, and only use provided data |
| `query.txt` | System prompt for the Searcher agent — instructs LLM to rewrite queries and extract filters |
| `query_fewshots.json` | Few-shot examples showing how to call `search_database()` with filters |

---

## 5. Frontend (React + FluentUI)

### 5.1 Pages and Routing

**Entry**: `src/frontend/src/index.tsx` — Uses React Router with hash-based routing.

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | `Chat` | Main chat interface with input, responses, and analysis panel |
| `*` | `NoPage` | 404 page |
| (wrapper) | `Layout` | App shell with header/navigation |

### 5.2 Components

| Component | Purpose |
|-----------|---------|
| **Answer** | Renders AI responses (markdown, loading skeleton, error states, citations) |
| **AnalysisPanel** | Side panel with tabs: Thought Process, Supporting Content, Citation |
| **ClearChatButton** | Clears conversation history |
| **Example** | Displays example prompts for new users |
| **MarkdownViewer** | Markdown rendering with syntax highlighting |
| **QuestionInput** | Text input field for user questions |
| **SettingsButton** | Opens configuration panel |
| **SupportingContent** | Displays retrieved database rows as citations |
| **UserChatMessage** | Renders user messages in the chat thread |
| **VectorSettings** | Toggle controls for retrieval mode (text/vector/hybrid) |

### 5.3 API Communication

Uses the `@microsoft/ai-chat-protocol` library:

```typescript
const chatClient = new AIChatProtocolClient("/chat");

// Streaming
const result = await chatClient.getStreamedCompletion(messages, {
  context: {
    overrides: {
      use_advanced_flow: boolean,
      retrieval_mode: "hybrid" | "vectors" | "text",
      top: number,
      temperature: number,
      prompt_template: string
    }
  }
});

// Non-streaming
const result = await chatClient.getCompletion(messages, options);
```

**Response structure:**
- `message` — AI answer with citations
- `context.data_points` — Retrieved database rows
- `context.thoughts` — Array of `ThoughtStep` objects (title, description, props)
- `context.followup_questions` — Suggested follow-ups

### 5.4 Build Configuration

**Vite** (`vite.config.ts`):
- Build output: `src/backend/static/` (served by FastAPI)
- Code splitting: Separate chunks for `@fluentui/react-icons`, `@fluentui/react`, and vendor libs
- Dev proxy: `/chat` → `http://localhost:8000`
- Source maps enabled

**Key dependencies:**
- `@fluentui/react` + `@fluentui/react-components` + `@fluentui/react-icons`
- `@azure/msal-react` + `@azure/msal-browser` (Azure authentication)
- `marked` (markdown parsing) + `dompurify` (HTML sanitization)
- `react-syntax-highlighter` (code blocks)
- `@react-spring/web` (animations)

---

## 6. Infrastructure (Azure Bicep)

### 6.1 Azure Resources

| Resource | Service | Configuration |
|----------|---------|---------------|
| **PostgreSQL** | Azure Database for PostgreSQL Flexible Server | v15, Standard_B1ms (Burstable), 32GB, Entra-only auth, pgvector |
| **Container App** | Azure Container Apps | Hosts FastAPI backend on port 8000 |
| **Container Registry** | Azure Container Registry | Stores Docker images |
| **Container Apps Environment** | Azure Container Apps | Managed Kubernetes environment |
| **Azure OpenAI** | Azure Cognitive Services | Chat (GPT-4o-mini) + Embedding (text-embedding-3-large) deployments |
| **Application Insights** | Azure Monitor | Performance monitoring + custom traces |
| **Log Analytics** | Azure Monitor | Centralized log storage |
| **Managed Identity** | Azure Entra ID | User-assigned identity for service auth |
| **AI Foundry** (optional) | Azure AI | Hub + Project for model management |

### 6.2 Bicep Module Map

```
infra/
├── main.bicep                    ← Subscription-scoped orchestrator
├── main.parameters.json          ← Parameter definitions
├── web.bicep                     ← Web container app config
├── backend-dashboard.bicep       ← App Insights dashboard
└── core/
    ├── ai/
    │   ├── cognitiveservices.bicep    ← Azure OpenAI (S0 SKU, Entra auth)
    │   └── ai-foundry.bicep          ← AI Foundry Hub + Project
    ├── database/postgresql/
    │   └── flexibleserver.bicep      ← PostgreSQL Flexible Server
    ├── host/
    │   ├── container-apps.bicep          ← Environment + Registry orchestration
    │   ├── container-apps-environment.bicep ← Container Apps Environment
    │   ├── container-registry.bicep      ← ACR
    │   ├── container-app.bicep           ← Container App definition
    │   └── container-app-upsert.bicep    ← Container App create/update
    ├── monitor/
    │   ├── monitoring.bicep          ← Log Analytics + App Insights
    │   ├── applicationinsights.bicep ← App Insights resource
    │   └── loganalytics.bicep        ← Log Analytics workspace
    └── security/
        ├── role.bicep                ← RBAC role assignments
        └── registry-access.bicep     ← ACR access control
```

### 6.3 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chatModelName` | `gpt-4o-mini` | Chat model |
| `chatDeploymentVersion` | `2024-07-18` | Chat model version |
| `chatDeploymentCapacity` | `30` | Chat deployment TPM |
| `embedModelName` | `text-embedding-3-large` | Embedding model |
| `embedDimensions` | `1024` | Embedding vector dimensions |
| `embedDeploymentCapacity` | `30` | Embedding deployment TPM |
| `evalModelName` | `gpt-4` | Evaluation model (optional) |
| `deployAzureOpenAI` | `true` | Whether to create OpenAI resource |
| `useAiProject` | `false` | Whether to create AI Foundry project |

### 6.4 Environment Variables in Production

The Container App receives these from `main.bicep`:

```
POSTGRES_HOST, POSTGRES_USERNAME, POSTGRES_DATABASE, POSTGRES_SSL
AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY (secret)
AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_CHAT_MODEL
AZURE_OPENAI_EMBED_DEPLOYMENT, AZURE_OPENAI_EMBED_DIMENSIONS
OPENAI_CHAT_HOST, OPENAI_EMBED_HOST
OPENAICOM_KEY (secret), OPENAICOM_CHAT_MODEL
APPLICATIONINSIGHTS_CONNECTION_STRING
RUNNING_IN_PRODUCTION=true
```

---

## 7. Database & Seed Data

### Schema

The application uses a single `items` table with product data and pre-computed embeddings:

```sql
CREATE TABLE items (
    id          SERIAL PRIMARY KEY,
    type        TEXT,          -- e.g., "shoes", "jackets"
    brand       TEXT,          -- e.g., "AirStrider", "SummitKing"
    name        TEXT,          -- Product name
    description TEXT,          -- Product description
    price       FLOAT,         -- Product price
    embedding_3l  VECTOR(1024), -- text-embedding-3-large embeddings
    embedding_nomic VECTOR(768) -- nomic-embed-text embeddings (Ollama)
);

-- HNSW indexes for fast approximate nearest neighbor search
CREATE INDEX hnsw_index_for_cosine_items_embedding_3l
    ON items USING hnsw (embedding_3l vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX hnsw_index_for_cosine_items_embedding_nomic
    ON items USING hnsw (embedding_nomic vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### Setup Scripts

| Script | Purpose |
|--------|---------|
| `setup_postgres_database.py` | Enables pgvector extension, creates tables and indexes via `Base.metadata.create_all` |
| `setup_postgres_seeddata.py` | Reads `seed_data.json`, inserts items idempotently (skips if ID exists) |
| `setup_postgres_azurerole.py` | Creates Azure AD role for managed identity auth on Azure PostgreSQL |
| `update_embeddings.py` | Batch generates embeddings for all items (in seed data or in database) |

### Seed Data Format

`seed_data.json` contains an array of products:

```json
[
  {
    "id": 1,
    "type": "shoes",
    "brand": "AirStrider",
    "name": "TrailMaster Pro Hiking Boots",
    "description": "Durable waterproof hiking boots...",
    "price": 89.99,
    "embedding_3l": [0.012, -0.034, ...],
    "embedding_nomic": [0.005, 0.015, ...]
  }
]
```

---

## 8. Testing

### Test Suite

**Framework**: pytest + pytest-asyncio + pytest-snapshot  
**Coverage requirement**: 85% minimum (`--cov-fail-under=85`)

| Test File | Tests |
|-----------|-------|
| `test_api_routes.py` | Chat endpoints (simple/advanced, streaming/non-streaming), item retrieval, search |
| `test_postgres_searcher.py` | Hybrid search, vector search, text search, filter application |
| `test_embeddings.py` | Embedding generation and dimension handling |
| `test_openai_clients.py` | Client factory for all 4 providers |
| `test_postgres_engine.py` | Database engine initialization |
| `test_dependencies.py` | Dependency injection resolution |
| `test_frontend_routes.py` | Static file serving |
| `e2e.py` | Playwright browser E2E tests |

### Snapshot Files

Response snapshots in `tests/snapshots/test_api_routes/`:

- `simple_chat_flow_response.json`
- `advanced_chat_flow_response.json`
- `simple_chat_streaming_flow_response.jsonlines`
- `advanced_chat_streaming_flow_response.jsonlines`
- `simple_chat_flow_message_history_response.json`

### Running Tests

```bash
# Unit tests (requires PostgreSQL)
pytest -s -vv --cov --cov-fail-under=85

# E2E tests (requires running app)
playwright install chromium --with-deps
pytest tests/e2e.py --tracing=retain-on-failure
```

---

## 9. Evaluations

**Directory**: `evals/`

| File | Purpose |
|------|---------|
| `evaluate.py` | Runs RAG quality metrics (groundedness, relevance, coherence, citations_matched) |
| `generate_ground_truth.py` | Generates Q&A pairs from seed data for evaluation |
| `safety_evaluation.py` | Tests for harmful content, jailbreak, and bias using Azure AI |
| `eval_config.json` | Metric configuration and thresholds |
| `ground_truth.jsonl` | Ground truth dataset |
| `results/baseline/` | Baseline evaluation results (summary.json, eval_results.jsonl) |

### Custom Metric

`citations_matched` — Checks that cited sources in the answer actually exist in the retrieved data points.

---

## 10. CI/CD Pipelines

**Directory**: `.github/workflows/`

| Workflow | Trigger | What It Does |
|----------|---------|--------------|
| `azure-dev.yaml` | Push to `main`, manual | Full Azure deployment via `azd up` with OIDC federation |
| `app-tests.yaml` | Push/PR to `main` | Runs pytest on Python 3.10/3.11/3.12 × macOS/Windows/Ubuntu |
| `python-code-quality.yaml` | Code changes | Ruff linting + mypy type checking |
| `bicep-security-scan.yaml` | Infra changes | Azure Bicep template security validation |
| `evaluate.yaml` | Manual | RAG quality evaluation suite |

### Adding New `azd` Environment Variables

When adding new environment variables, update all three:
1. `infra/main.parameters.json` — Parameter with Bicep-friendly name mapped to env var
2. `infra/main.bicep` — New Bicep parameter at top + add to `webAppEnv` object
3. `.github/workflows/azure-dev.yml` — Under `env` (from `vars` or `secrets` for @secure)

---

## 11. Scripts

**Directory**: `scripts/`

| Script | Variants | Purpose |
|--------|----------|---------|
| `setup_postgres_database` | `.sh`, `.ps1` | Initialize PostgreSQL schema (calls Python setup script) |
| `setup_postgres_seeddata` | `.sh`, `.ps1` | Seed database with sample data |
| `setup_postgres_azurerole` | `.sh`, `.ps1` | Configure Azure AD role for app identity |
| `setup_openai_role` | `.sh` | Configure OpenAI service principal access |
| `load_python_env` | `.sh` | Load Python environment variables for scripts |

These scripts are invoked by `azure.yaml` as `postprovision` hooks during `azd up`.

---

## 12. Configuration Reference

### 12.1 Environment Variables

#### PostgreSQL

| Variable | Example | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | Database hostname |
| `POSTGRES_USERNAME` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `POSTGRES_DATABASE` | `postgres` | Database name |
| `POSTGRES_SSL` | `disable` / `require` | SSL mode |

#### Azure OpenAI

| Variable | Example | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | `https://myservice.openai.azure.com` | Service endpoint |
| `AZURE_OPENAI_VERSION` | `2024-03-01-preview` | API version |
| `AZURE_OPENAI_KEY` | (optional) | API key; uses Managed Identity if unset |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4o-mini` | Chat deployment name |
| `AZURE_OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model name |
| `AZURE_OPENAI_EMBED_DEPLOYMENT` | `text-embedding-3-large` | Embedding deployment |
| `AZURE_OPENAI_EMBED_MODEL` | `text-embedding-3-large` | Embedding model |
| `AZURE_OPENAI_EMBED_DIMENSIONS` | `1024` | Embedding vector size |
| `AZURE_OPENAI_EMBEDDING_COLUMN` | `embedding_3l` | DB column for embeddings |

#### OpenAI.com

| Variable | Description |
|----------|-------------|
| `OPENAICOM_KEY` | OpenAI API key |
| `OPENAICOM_CHAT_MODEL` | e.g., `gpt-3.5-turbo` |
| `OPENAICOM_EMBED_MODEL` | e.g., `text-embedding-3-large` |
| `OPENAICOM_EMBED_DIMENSIONS` | e.g., `1024` |

#### Ollama (Local)

| Variable | Description |
|----------|-------------|
| `OLLAMA_ENDPOINT` | e.g., `http://localhost:11434/v1` |
| `OLLAMA_CHAT_MODEL` | e.g., `llama3.1` |
| `OLLAMA_EMBED_MODEL` | e.g., `nomic-embed-text` |
| `OLLAMA_EMBEDDING_COLUMN` | `embedding_nomic` |

#### GitHub Models

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub personal access token |
| `GITHUB_MODEL` | e.g., `openai/gpt-4o` |
| `GITHUB_EMBED_MODEL` | e.g., `openai/text-embedding-3-large` |
| `GITHUB_EMBED_DIMENSIONS` | e.g., `1024` |

### 12.2 LLM Provider Options

| Provider | `OPENAI_CHAT_HOST` | `OPENAI_EMBED_HOST` | Notes |
|----------|-------------------|---------------------|-------|
| Azure OpenAI | `azure` | `azure` | Recommended for production. Uses Managed Identity. |
| OpenAI.com | `openai` | `openai` | Requires API key. |
| Ollama | `ollama` | `ollama` | Local LLM. Use `llama3.1` (supports function calling). |
| GitHub Models | `github` | `github` | Requires GitHub token. |

> **Note:** Chat and embedding providers can be mixed (e.g., Azure for chat, Ollama for embeddings).

### 12.3 pyproject.toml Tooling Config

```toml
[tool.ruff]
line-length = 120
target-version = "py39"
# Rules: E (pycodestyle errors), F (pyflakes), I (isort), UP (pyupgrade)

[tool.mypy]
check_untyped_defs = true
# Ignores: pgvector.*, evaltools.*

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
pythonpath = ["src/backend"]
# async mode, snapshot testing, coverage
```

---

## 13. Development Workflow

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements-dev.txt
pip install -e src/backend

# 2. Configure environment
cp .env.sample .env
# Edit .env with your credentials

# 3. Set up database
python ./src/backend/fastapi_app/setup_postgres_database.py
python ./src/backend/fastapi_app/setup_postgres_seeddata.py

# 4. Build frontend
cd src/frontend && npm install && npm run build && cd ../..

# 5. Run backend
python -m uvicorn fastapi_app:create_app --factory --reload

# 6. (Optional) Run frontend dev server with hot reload
cd src/frontend && npm run dev
```

### Code Quality Checks

```bash
ruff check .                       # Lint
ruff format .                      # Format
mypy . --python-version 3.12       # Type check
pytest -s -vv --cov --cov-fail-under=85  # Tests + coverage
```

### Deploy to Azure

```bash
azd auth login
azd env new
azd up
```

---

## 14. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hybrid search (vector + FTS + RRF)** | Combines semantic understanding (vectors) with keyword precision (full-text), merged via RRF for best-of-both accuracy |
| **pgvector with HNSW indexes** | Fast approximate nearest neighbor search within PostgreSQL — no separate vector database needed |
| **Two RAG flows (Simple + Advanced)** | Simple for speed; Advanced for accuracy with query rewriting and filter extraction via function calling |
| **openai-agents library** | Provides structured agent framework with tool calling, streaming, and multi-agent orchestration |
| **4 LLM providers** | Flexibility for local dev (Ollama), prototyping (GitHub Models/OpenAI.com), and production (Azure OpenAI) |
| **Frontend built into backend static** | Single deployment artifact — Container App serves both API and UI |
| **Managed Identity (no API keys)** | Azure security best practice — eliminates secret rotation |
| **Bicep over Terraform** | Native Azure IaC with tight Azure integration and smaller footprint |
| **Azure Developer CLI (azd)** | Unified developer experience for provision + deploy + monitor |
| **Async throughout** | `asyncpg` + `AsyncSession` + `AsyncOpenAI` — high throughput with non-blocking I/O |

---

## 15. Related Documentation

| Document | Path | Description |
|----------|------|-------------|
| RAG Flow | `docs/rag_flow.md` | Detailed explanation of Simple and Advanced RAG flows |
| Custom Data | `docs/customize_data.md` | Guide for replacing sample data with your own |
| Deploy Existing | `docs/deploy_existing.md` | Using pre-existing Azure resources |
| Monitoring | `docs/monitoring.md` | Application Insights setup and traces |
| Evaluation | `docs/evaluation.md` | RAG quality evaluation methodology |
| Safety Eval | `docs/safety_evaluation.md` | Content safety and harm evaluation |
| Load Testing | `docs/loadtesting.md` | Locust-based performance testing |
| Entra Auth | `docs/using_entra_auth.md` | Azure Entra ID authentication for PostgreSQL |
| Contributing | `CONTRIBUTING.md` | Contribution guidelines |
| Agent Instructions | `AGENTS.md` | Instructions for coding agents working on this repo |
