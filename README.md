# ⚖️ Fake News Debater

**Multi-Agent AI System for Automated Misinformation Detection**

A research-grade NLP + Agentic AI project that uses adversarial debate between autonomous agents to fact-check news articles — with built-in explainability.

---

## 🏗️ Architecture

```
User submits news article
         |
         v
   [Claim Extractor]          ← spaCy NER + Groq LLM
         |
    _____|_____
    |         |
    v         v
[Verifier]  [Falsifier]       ← Both search web independently
 Agent       Agent
    |         |
 Evidence   Evidence
    |_________|
         |
         v
   [Judge Agent]              ← Weighs both sides, gives verdict
         |
         v
  Final Verdict + Report
```

### How It Works

1. **Claim Extractor** — Uses spaCy Named Entity Recognition to identify key entities (people, organizations, dates, locations), then sends the article + entities to Groq LLM to extract 3-5 verifiable factual claims.

2. **Verifier Agent** (Pro-Truth) — Searches the web for evidence that *supports* each claim, scores evidence using an NLI stance detector, and builds a verification argument.

3. **Falsifier Agent** (Counter-Truth) — Mirrors the verifier but searches for *contradicting* evidence and builds a counter-argument.

4. **Judge Agent** — Weighs evidence from both sides, delivers per-claim verdicts (SUPPORTED/REFUTED/UNVERIFIABLE), and a final article verdict (REAL/FAKE/MISLEADING) with confidence scores.

---

## 🛠️ Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| LLM | Groq API (Llama 3.3 70B) | Free (14,400 req/day) |
| Web Search | DuckDuckGo + Serper.dev | Free |
| NER | spaCy (local) | Free (offline) |
| Stance Detection | HuggingFace Inference API | Free |
| Frontend | Streamlit | Free |
| Language | Python 3.13 | Free |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
cd fake-news-debater
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set API Keys

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_key        # Optional (fallback search)
HF_API_TOKEN=your_huggingface_token
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
fake-news-debater/
├── agents/
│   ├── claim_extractor.py     # spaCy NER + Groq LLM claim extraction
│   ├── verifier_agent.py      # Pro-truth agent (searches for support)
│   ├── falsifier_agent.py     # Anti-truth agent (searches for contradictions)
│   └── judge_agent.py         # Impartial judge (weighs both sides)
├── tools/
│   ├── web_search.py          # DuckDuckGo + Serper.dev fallback
│   ├── stance_detector.py     # HuggingFace NLI stance scoring
│   └── article_scraper.py     # URL → article text extraction
├── app.py                     # Streamlit frontend
├── config.py                  # Central configuration
├── requirements.txt
├── .env                       # API keys (gitignored)
└── README.md
```

---

## 🔬 Research Contributions

1. **Adversarial Debate Framing** — Two agents with opposing objectives produce richer signal than single-model classifiers
2. **Claim-Level Granularity** — Per-claim verdicts, not just per-article
3. **Stance Detection as Evidence Scoring** — NLI model repurposed for ranking web evidence quality
4. **Built-in Explainability** — The debate transcript IS the explanation

**Suggested Paper Title:**
*"Adversarial Multi-Agent Debate for Automated Misinformation Detection"*

---

## ⚠️ Free Tier Limitations

| Issue | Workaround |
|-------|-----------|
| Groq rate limits | 1-second delay between API calls |
| DuckDuckGo throttling | Auto-fallback to Serper.dev |
| HuggingFace cold start | Retry with 30s wait |
| Streamlit timeout | Results cached with `@st.cache_data` |

---

## 📄 License

MIT License — Built for research and education.
