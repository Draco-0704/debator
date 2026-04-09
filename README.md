# ⚖️ Fake News Debater

**Production-Ready Multi-Agent AI System for Automated Misinformation Detection**

A high-fidelity, research-grade NLP + Agentic AI project that uses adversarial debate between autonomous agents to fact-check news articles — with built-in explainability, resilience, and speed optimizations.

---

## 🏗️ Architecture & Pipeline

```text
User submits news article (or URL)
         |
         v
   [Claim Extractor]          ← spaCy NER + Groq LLM
         |
    _____|_____
    |         |
    v         v
[Verifier]  [Falsifier]       ← Both search the web independently
 Agent       Agent              (Parallel Execution)
    |         |
 Evidence   Evidence          ← HuggingFace NLI Stance Detection
    |_________|                 (Parallelized, router API)
         |
         v
   [Judge Agent]              ← Impartial judge evaluates arguments
         |                      and delivers verdict
         v
 Final Verdict + Report       ← JSON Export available
```

### Detailed Agent Workflow

1. **Claim Extractor** — Uses spaCy Named Entity Recognition to anchor key entities (People, Orgs, Dates, Locations). The article text and entities are sent to Groq LLM (with forced JSON parsing logic) to systematically extract 3-5 verifiable factual claims.
2. **Verifier Agent (Pro-Truth)** — Generates independent search queries aiming to validate the claim. It retrieves search results, scrapes the web pages (capped at 1,500 chars for speed), runs NLI-based stance detection to find supporting evidence, and constructs a persuasive verification argument.
3. **Falsifier Agent (Counter-Truth)** — Works in parallel with the Verifier, but runs adversarial search queries aiming to debunk the claim. It utilizes the same stance-detection framework to isolate contradictions and constructs a counter-argument.
4. **Judge Agent** — Acts as the impartial arbitrator. It weighs the raw evidence surface area against the rhetoric of both agents, delivering targeted claim-level verdicts (SUPPORTED / REFUTED / UNVERIFIABLE), confidence metrics, and an overall article judgment (REAL / FAKE / MISLEADING).

---

## 🚀 Key Features & Upgrades

This system has been upgraded for production readiness with several robust engineering mechanisms:

- **Parallel Processing:** Verifier and Falsifier agent pipelines run concurrently using Python's `ThreadPoolExecutor`. Furthermore, stance detection on fetched evidence runs in parallel for rapid inference.
- **Resilient LLM Client (`groq_client.py`):** Features a thread-safe singleton pattern and exponential backoff retry logic for `429 Rate Limit` and `5xx Server Error` statuses.
- **Robust Inference API:** Uses the latest HuggingFace Router infrastructure (`router.huggingface.co`) for stable deployment of the `DeBERTa-v3` NLI stance detection model.
- **Result Caching:** Inputs are hashed (`sha256`); if an identical article is submitted, the expensive pipeline is bypassed and cached results are rendered instantly.
- **Input Validation:** Strict parsing ensures articles meet length requirements (50 chars min, 50,000 chars max).
- **JSON Exportability:** Built-in capability to download the full debate logic, claims, evidence, and verdicts as an interoperable JSON file.
- **Premium UI:** Deployed on Streamlit with custom CSS injecting a modern, glassmorphic dark-theme design.

---

## 🛠️ Tech Stack & Economics

| Component | Tool / Library | Infrastructure / Cost |
|-----------|----------------|-----------------------|
| LLM Reasoning | Groq API (Llama 3.3 70B) | Free Tier (14.4k req/day) |
| Web Search | DuckDuckGo + Serper.dev | Free Tier |
| NLP & NER | spaCy (`en_core_web_sm`) | Free (Offline/Local) |
| Stance Detection | HuggingFace Inference API (NLI) | Free Serverless Router |
| Frontend | Streamlit | UI Rendering |
| Execution | Python 3.13, ThreadPoolExecutor| Concurrency |

---

## 💻 Quick Start

### 1. Clone & Environment Setup

```bash
git clone https://github.com/Draco-0704/debator.git
cd debator

# Install dependencies
pip install -r requirements.txt

# Download local NLP model for entity extraction
python -m spacy download en_core_web_sm
```

### 2. Configure API Keys

Create a `.env` file in the root directory (this is `.gitignore`'d for security):

```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_key
HF_API_TOKEN=your_huggingface_token
```

### 3. Run the App

```bash
streamlit run app.py
```
View the dashboard at `http://localhost:8501`.

---

## 📁 Project Structure

```text
debater/
├── agents/                    # Core Agent Logic
│   ├── claim_extractor.py     # spaCy + Groq parsing
│   ├── verifier_agent.py      # Pro-truth retrieval pipeline
│   ├── falsifier_agent.py     # Debunking retrieval pipeline
│   └── judge_agent.py         # Verdict generation
├── tools/                     # Utilities & Integrations
│   ├── article_scraper.py     # BeautifulSoup HTML text extractor
│   ├── groq_client.py         # Resilient LLM wrapper with backoff
│   ├── stance_detector.py     # HF NLI stance classification
│   └── web_search.py          # DDG + Serper search engine operations
├── tests/                     # Unit test suites
│   ├── test_judge_agent.py
│   └── test_stance_detector.py
├── app.py                     # Streamlit frontend & UI definitions
├── config.py                  # Global parameters & limits
├── requirements.txt           # Python dependencies
└── .gitignore
```

---

## 🧪 Testing

The project includes unit testing for core logic and label mapping.
Run tests using:

```bash
python -m unittest discover tests/ -v
```

---

## 🔬 Research Significance

This project demonstrates a major shift in automated fact-checking:
1. **Adversarial Framing:** Two agents with explicitly opposing objectives eliminate "yes-man" confirmation bias that plagues single-model checkers.
2. **Deterministic Evidence Scoring:** Instead of relying entirely on LLM hallucination-prone text understanding, the pipeline anchors truth using a dedicated NLI (Natural Language Inference) transformer.
3. **Traceable Explainability:** The system does not just output a percentage confidence; the entire debate log IS the explanation.

---

## 📄 License

MIT License — Built for research, OSINT, and educational purposes.
