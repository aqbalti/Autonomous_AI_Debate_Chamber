# ⚖️ Autonomous AI Debate Chamber
**Zenvyro Labs · Autonomous Agent Simulation**

Two AI agents debate any topic in real time. A SciKit-Learn RandomForestRegressor judges them and crowns a winner.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (separate terminal)
ollama serve
ollama pull llama3.2

# 3. Run
python app.py
```

Open **http://127.0.0.1:5000**

---

## 🏗️ Architecture

```
app.py                      ← Flask + SocketIO + REST API (Zenvyro routes)
services/
  llm.py                   ← Ollama HTTP wrapper
  ai_service.py            ← DebateConductor (Agent A Advocate + Agent B Challenger)
  memory_manager.py        ← Chronological debate history → LLM prompt injection
  feature_extractor.py     ← 10 NLP features for ML scoring
  ml_judge.py              ← DebateRegressionJudge (RandomForestRegressor)
templates/index.html       ← Zenvyro Labs UI (blue vs red, judge overlay)
static/css/style.css       ← Exact Zenvyro design system
static/js/script.js        ← SocketIO live updates, judge modal
historical_debates.csv     ← Training data (auto-used on first run)
models/judge_model.pkl     ← Cached trained model
```

---

## 🌐 REST API

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serve UI |
| `/start` | POST | Start autonomous SocketIO debate |
| `/api/debate/start` | POST | Initialize debate (REST) |
| `/api/debate/next-turn` | POST | Manual turn trigger |
| `/api/machine-learning/train` | POST | Retrain RF model |
| `/api/machine-learning/evaluate` | POST | Score two text inputs |
| `/status` | GET | Current debate state |
| `/winner` | GET | Final ML verdict |

---

## 🤖 Agents

| | Agent A — Advocate | Agent B — Challenger |
|---|---|---|
| Color | Blue `#3b82f6` | Red `#ef4444` |
| Style | Scientific, logical | Philosophical, critical |
| Approach | Facts & evidence | Challenges assumptions |

---

## 🧠 ML Features

| Feature | Why chosen |
|---|---|
| word_count | Longer arguments = more substance |
| sentence_count | Structured thinking |
| avg_word_length | Vocabulary sophistication |
| lexical_diversity | Richer language = stronger framing |
| sentiment_score | Positive framing = more persuasive |
| complexity_score | Analytical depth |
| question_count | Rhetorical engagement |
| avg_sentence_length | Argument structure |

**Why RandomForestRegressor:** Non-linear, scale-invariant, ensemble reduces overfitting.

---

## 🛠️ Future Improvements
- Real VADER/TextBlob sentiment scoring
- NLTK-based complexity metrics  
- Debate transcript export (PDF)
- Multiple LLM provider support
- Leaderboard of past debates
