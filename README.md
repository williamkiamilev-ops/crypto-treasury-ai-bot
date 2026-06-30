# Crypto mNAV Treasury Analysis Trading Bot

An AI-assisted (LangChain + OpenAI) autonomous paper-trading agent focused on crypto treasury equities. The bot builds a watchlist of companies holding Bitcoin and other crypto assets, calculates mNAV (market value to net asset value), identifies premium/discount opportunities versus peers, and can execute paper trades via Alpaca while sending Discord alerts and reports.

> **For research and paper trading only.** This is not financial advice. Do not use with real capital without thorough independent review, backtesting, and risk controls.

## Live Performance (March–May 2026)

The system has been running live paper trades (intermittently due to token costs and school workload) since March 2026.

**Notable Results:**
- **HUT**: +133.8% total P/L
- **CLSK**: +45.6% total P/L
- Multiple executed trades across CLSK, HUT, MARA, COIN, and TSLA
- Generated and acted on mNAV-based signals with automated risk controls and reporting

## What It Does

- Builds and maintains a crypto treasury company universe
- Tracks on-balance-sheet crypto holdings (Bitcoin, Ethereum, etc.)
- Computes mNAV using treasury holdings, underlying crypto prices, equity prices, and capital structure estimates
- Detects relative value opportunities (premiums/discounts vs peers and historical levels)
- Generates directional, pairs, or hedged trade ideas with position sizing and risk notes
- Supports one-shot or continuous autonomous sessions
- Executes paper trades via Alpaca (with local JSON fallback)
- Sends real-time Discord trade alerts and periodic performance reports
- Maintains persistent memory (`agent_memory.json`) so future runs can reference prior decisions

## Project Structure

```text
.
├── main.py                     # LangChain/OpenAI agent entry point
├── tools.py                    # Market data, mNAV calculation, broker, optimizer, reporting
├── refresh_holdings.py         # Standalone holdings refresh
├── requirements.txt
├── .env.example
├── paper_account.json          # Local paper trading state
├── treasury_holdings.json
├── treasury_universe.json
├── agent_memory.json
├── run_continuous_agent.ps1
├── run_refresh_holdings.ps1
└── install_scheduled_tasks.ps1
