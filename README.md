# Conditional Synthetic Tabular Data Generator

A production-ready API that generates realistic conditional synthetic rows
from the [UCI Adult Income](https://archive.ics.uci.edu/dataset/2/adult) dataset
using a Gaussian Copula or Conditional Histogram engine, validates fidelity with
TSTR / KS / JS metrics, and provides a Groq-powered conversational agent.

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <https://github.com/ShivamShrivastava18/Kronos_assignment> && cd Kronos_assignment

# 2. Set your Groq key (only needed for /agent/* endpoints)
cp .env.example .env
echo "GROQ_API_KEY=your_key_here" >> .env

# 3. Build & run (no manual setup beyond this)
docker-compose up --build
```

The API is now live at **http://localhost:8000**.  
Interactive docs: **http://localhost:8000/docs**
Interactable Frontend at: **http://localhost:8080**

---
## Endpoints — curl examples

### Health & Meta

```bash
# Liveness + model status
curl http://localhost:8000/health

# Aggregate generation stats
curl http://localhost:8000/metrics/summary
```

### Dataset

```bash
# Schema, value ranges, class distribution
curl http://localhost:8000/dataset/info

# 5 real rows (default)
curl "http://localhost:8000/dataset/sample?n=5"

# Seeded sample
curl "http://localhost:8000/dataset/sample?n=3&random_seed=42"
```

### Generate — structured constraints

```bash
# Copula: 500 rows, income >50K, age 40-60, specific occupations
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "constraints": {
      "income":     ">50K",
      "age":        {"gte": 40, "lte": 60},
      "occupation": ["Exec-managerial", "Prof-specialty"]
    },
    "n_rows":      500,
    "strategy":    "copula",
    "random_seed": 42
  }' | jq '{generation_id, n_rows_returned, constraints_applied}'

# Histogram fallback, female workers, long hours
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "constraints": {
      "sex":           "Female",
      "hours_per_week": {"gte": 45}
    },
    "n_rows":   200,
    "strategy": "histogram"
  }' | jq '.generation_id'
```

### Generate — natural language

```bash
curl -s -X POST http://localhost:8000/generate/from-text \
  -H "Content-Type: application/json" \
  -d '{
    "text":      "high income men over 40 working as Exec-managerial",
    "n_rows":    300,
    "strategy":  "copula",
    "random_seed": 7
  }' | jq '{generation_id, n_rows_returned, parsed_from_text, parser_warnings}'
```

### Validate

```bash
# Store the generation_id from a /generate call first:
GEN_ID="<paste-generation-id-here>"

# Full validation suite
curl -s -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d "{\"generation_id\": \"$GEN_ID\"}" \
  | jq '{verdict, tstr_accuracy, baseline_accuracy, fidelity_score}'

# Retrieve cached result
curl "http://localhost:8000/validate/$GEN_ID" | jq '{verdict, fidelity_score}'

# Compare two runs side-by-side
curl -s -X POST http://localhost:8000/validate/compare \
  -H "Content-Type: application/json" \
  -d '{
    "generation_id_a": "<GEN_ID_A>",
    "generation_id_b": "<GEN_ID_B>"
  }' | jq '{winner, diff}'
```

### Agent (requires GROQ_API_KEY)

```bash
# Start a conversation
curl -s -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session-1",
    "message":    "Generate 500 rows of high-income women over 40 and check the fidelity."
  }' | jq '{reply, generation_id, tool_calls_made}'

# Follow-up in the same session
curl -s -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session-1",
    "message":    "Try the histogram strategy instead and compare."
  }' | jq .

# Inspect conversation history
curl http://localhost:8000/agent/history/demo-session-1 | jq '.messages | length'

# Clear a session
curl -s -X DELETE http://localhost:8000/agent/session/demo-session-1 -w "%{http_code}"
```

---

## Project structure

```
app/
  main.py               # FastAPI app factory + lifespan
  config.py             # Settings via pydantic-settings
  schemas.py            # All Pydantic request/response models
  services.py           # Generation & validation service layer
  storage.py            # In-memory stores (generations, validations, sessions)
  data/
    loader.py           # UCI Adult loader, schema builder, caching
  generators/
    constraints.py      # Constraint normalization + row filtering
    base.py             # Abstract generator interface
    copula.py           # Gaussian Copula (full conditional sampling)
    histogram.py        # Conditional histogram (robust fallback)
  validators/
    metrics.py          # KS-test, Jensen-Shannon divergence
    validator.py        # TSTR pipeline (XGBoost / LightGBM)
  parsers/
    text_parser.py      # Deterministic rule-based NL constraint parser
  agent/
    tools.py            # Tool specs + dispatcher for the LLM
    groq_agent.py       # Stateful Groq tool-calling loop
  routers/
    dataset.py          # GET /dataset/*
    generate.py         # POST /generate, POST /generate/from-text
    validate.py         # POST /validate, GET /validate/{id}, POST /validate/compare
    agent.py            # POST /agent/chat, GET /agent/history, DELETE /agent/session
    health.py           # GET /health, GET /metrics/summary
```

---

## Environment variables

| Variable        | Required | Default                      | Description                  |
|-----------------|----------|------------------------------|------------------------------|
| `GROQ_API_KEY`  | for agent| —                            | Free key from console.groq.com |
| `GROQ_MODEL`    | no       | `llama-3.3-70b-versatile`    | Groq model slug              |
| `LOG_LEVEL`     | no       | `info`                       | Python logging level         |

---

## Validation verdict rule

`PASS` when `|TSTR_accuracy − baseline_accuracy| ≤ 0.05`.

The baseline is computed once (train-on-real, test-on-real with a 75/25 split)
and cached for the process lifetime to make comparison cheap and stable.
