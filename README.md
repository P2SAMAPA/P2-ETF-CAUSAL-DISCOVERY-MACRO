# Causal Discovery with Macro-Augmented Graph (PC Algorithm)

Implements the PC algorithm (Spirtes, Glymour, Scheines) to discover a causal graph where nodes include ETFs and macro variables (VIX, DXY, yields). Conditional independence tests (partial correlation) are used. The per‑ETF score is the number of directed causal edges from macro variables to that ETF – a measure of macro‑driven structure.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Augmented graph: ETF nodes + all macro variable nodes
- PC algorithm with partial correlation and significance threshold α
- Score = number of incoming causal edges from macro nodes
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-causal-discovery-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slow for large graphs; reduce `MAX_COND_SIZE` for speed)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF is strongly causally driven by macro factors → likely to respond to macro news.
- Low score → ETF is driven by other ETFs or idiosyncratic factors.

## Requirements

See `requirements.txt`.
