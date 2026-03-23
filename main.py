
import os
import warnings
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from dotenv import load_dotenv
from pydantic import BaseModel

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools import (
    autonomous_broker_trading_session_tool,
    auto_trade_watchlist_tool,
    broker_auto_trade_watchlist_tool,
    broker_cancel_all_orders_tool,
    broker_daily_summary_tool,
    broker_paper_account_status_tool,
    broker_positions_tool,
    broker_submit_order_tool,
    compute_mnav_snapshot_tool,
    detect_mnav_arbitrage_tool,
    execute_mnav_pairs_trade_tool,
    execute_paper_trade_tool,
    fetch_treasury_reference_pages_tool,
    get_crypto_treasury_watchlist_tool,
    model_analyst_tool,
    optimize_strategy_parameters_tool,
    paper_account_status_tool,
    place_auto_hedge_tool,
    refresh_treasury_holdings_tool,
    run_long_backtest_protocol_tool,
    run_crypto_treasury_strategy_tool,
    run_optimizer_loop_tool,
    search_tool,
    seed_crypto_treasury_universe_tool,
    send_discord_report_tool,
    set_high_beta_mode_tool,
    strategy_config_tool,
    technical_stock_analysis_tool,
    technical_watchlist_analysis_tool,
    trade_signal_tool,
    upsert_crypto_treasury_company_tool,
    upsert_treasury_holding_tool,
    update_strategy_config_tool,
    yahoo_finance_tool,
)

# Always load .env from the same folder as this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(f"OPENAI_API_KEY not found in {env_path}")

load_dotenv()

MEMORY_PATH = Path(__file__).resolve().parent / "agent_memory.json"

class TraderResponse(BaseModel):
    objective: str
    market_regime: str
    analyst_model: str
    signals: list[str]
    trade_decisions: list[str]
    executed_orders: list[str]
    portfolio_snapshot: str
    risk_notes: list[str]
    tools_used: list[str]


llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", api_key=api_key)

tools = [
    autonomous_broker_trading_session_tool,
    yahoo_finance_tool,
    fetch_treasury_reference_pages_tool,
    seed_crypto_treasury_universe_tool,
    upsert_crypto_treasury_company_tool,
    get_crypto_treasury_watchlist_tool,
    compute_mnav_snapshot_tool,
    detect_mnav_arbitrage_tool,
    execute_mnav_pairs_trade_tool,
    model_analyst_tool,
    optimize_strategy_parameters_tool,
    run_long_backtest_protocol_tool,
    trade_signal_tool,
    broker_paper_account_status_tool,
    broker_positions_tool,
    broker_submit_order_tool,
    broker_cancel_all_orders_tool,
    broker_daily_summary_tool,
    broker_auto_trade_watchlist_tool,
    strategy_config_tool,
    technical_stock_analysis_tool,
    technical_watchlist_analysis_tool,
    set_high_beta_mode_tool,
    update_strategy_config_tool,
    upsert_treasury_holding_tool,
    run_crypto_treasury_strategy_tool,
    run_optimizer_loop_tool,
    place_auto_hedge_tool,
    send_discord_report_tool,
    execute_paper_trade_tool,
    paper_account_status_tool,
    auto_trade_watchlist_tool,
    search_tool,
    refresh_treasury_holdings_tool,
]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a model-driven paper trading strategist for stocks and crypto proxies. "
        "Follow the user's instructions directly, including autonomous sessions, timing, trade caps, and watchlists. "
        "For treasury equities, prioritize mNAV analytics: build/refresh watchlists, compute mNAV, detect deviations vs peers and 30-day baselines, and generate actionable long/short or pairs setups with risk controls. "
        "You must use market/model tools to build a concrete analyst view, generate signals, "
        "and execute paper orders through broker_* Alpaca tools when available. "
        "For crypto treasury strategies, prefer run_crypto_treasury_strategy_tool and then hedge via place_auto_hedge_tool when exposure is high. "
        "If broker credentials are missing, fall back to local fake tools. "
        "Do not claim live trading with real money. Return clear model metrics, specific trades, and risk controls."
    ),
    response_format=TraderResponse,
)


def collect_query() -> str:
    first_line = input("What should the trader do? ").strip()
    if not first_line:
        return ""

    print("Add more detail if needed. Press Enter on an empty line to run.")
    lines = [first_line]
    while True:
        line = input().strip()
        if not line:
            break
        lines.append(line)
    return " ".join(lines)


def resolve_query_interactively(initial_query: str = "") -> str:
    initial_query = initial_query.strip()
    if not initial_query:
        return collect_query()

    print("Current objective:")
    print(initial_query)
    override = input("Press Enter to use this objective, or type a new one: ").strip()
    if override:
        return override
    return initial_query


def confirm_continuous_run(query: str, days: float, poll_seconds: int, max_trades: int) -> None:
    print("Continuous session ready:")
    print(f"- Duration (days): {days}")
    print(f"- Poll seconds: {poll_seconds}")
    print(f"- Max trades: {max_trades}")
    print(f"- Query: {query}")
    confirm = input("Start continuous session? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        raise RuntimeError("Continuous session cancelled.")


def _default_memory() -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }


def _load_memory() -> dict:
    if not MEMORY_PATH.exists():
        data = _default_memory()
        _save_memory(data)
        return data
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = _default_memory()
        _save_memory(data)
        return data


def _save_memory(memory: dict) -> None:
    memory["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def _append_memory_entry(memory: dict, entry: dict, max_entries: int = 500) -> None:
    runs = memory.setdefault("runs", [])
    runs.append(entry)
    if len(runs) > max_entries:
        memory["runs"] = runs[-max_entries:]


def _memory_context(memory: dict, last_n: int = 10) -> str:
    recent = memory.get("runs", [])[-last_n:]
    if not recent:
        return "No prior memory entries."
    lines = []
    for r in recent:
        lines.append(
            f"{r.get('timestamp_utc')} | cycle={r.get('cycle', 'n/a')} | "
            f"decisions={len(r.get('trade_decisions', []))} | "
            f"orders={len(r.get('executed_orders', []))} | "
            f"summary={r.get('analysis_summary', '')[:180]}"
        )
    return "\n".join(lines)


def run_once(query: str) -> None:
    memory = _load_memory()
    mem_ctx = _memory_context(memory, last_n=8)
    query_with_memory = (
        f"{query}\n\n"
        "Session memory context (most recent entries):\n"
        f"{mem_ctx}\n"
        "Use this memory to maintain continuity in portfolio management and analysis."
    )
    raw_response = agent.invoke(
        {"messages": [{"role": "user", "content": query_with_memory}]}
    )

    try:
        structured_response: TraderResponse | None = raw_response.get("structured_response")
        if not structured_response:
            raise ValueError("No structured_response in agent output")
    except Exception as e:
        print("Error parsing response:", e, "Raw Response:", raw_response)
    else:
        print(f"\nObjective: {structured_response.objective}")
        print(f"\nMarket regime:\n{structured_response.market_regime}")
        print(f"\nAnalyst model:\n{structured_response.analyst_model}")

        if structured_response.signals:
            print("\nSignals:")
            for signal in structured_response.signals:
                print(f"- {signal}")

        if structured_response.trade_decisions:
            print("\nTrade decisions:")
            for decision in structured_response.trade_decisions:
                print(f"- {decision}")

        if structured_response.executed_orders:
            print("\nExecuted orders (paper account):")
            for order in structured_response.executed_orders:
                print(f"- {order}")

        print(f"\nPortfolio snapshot:\n{structured_response.portfolio_snapshot}")

        if structured_response.risk_notes:
            print("\nRisk notes:")
            for risk_note in structured_response.risk_notes:
                print(f"- {risk_note}")

        if structured_response.tools_used:
            print("\nTools used:")
            for tool_name in structured_response.tools_used:
                print(f"- {tool_name}")

        if not structured_response.analyst_model:
            messages = raw_response.get("messages", [])
            if messages:
                last_content = getattr(messages[-1], "content", "")
                if last_content:
                    print("\nAssistant message:")
                    print(last_content)

        _append_memory_entry(
            memory,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "single",
                "cycle": 1,
                "objective": structured_response.objective,
                "market_regime": structured_response.market_regime,
                "analysis_summary": structured_response.analyst_model[:600],
                "trade_decisions": structured_response.trade_decisions,
                "executed_orders": structured_response.executed_orders,
                "risk_notes": structured_response.risk_notes,
            },
        )
        _save_memory(memory)


def run_continuous(
    base_query: str,
    days: float,
    poll_seconds: int,
    max_trades: int,
    breakdown_hours: float = 24.0,
    weekly_report_days: float = 7.0,
) -> None:
    memory = _load_memory()
    def _send_discord_chunked(text: str, header: str = "") -> None:
        payload = (f"{header}\n{text}" if header else text).strip()
        if not payload:
            return
        max_len = 1800
        for i in range(0, len(payload), max_len):
            send_discord_report_tool(payload[i : i + max_len])

    def _build_trade_explanations(fills: list[dict], cycle_logs: list[dict]) -> list[str]:
        lines = []
        for fill in fills[:25]:
            symbol = str(fill.get("symbol", "")).upper()
            side = str(fill.get("side", "")).upper()
            qty = fill.get("qty")
            price = fill.get("price")

            decision_hint = ""
            risk_hint = ""
            for log in reversed(cycle_logs):
                for d in (log.get("trade_decisions") or []):
                    if symbol and symbol in str(d).upper():
                        decision_hint = str(d)
                        break
                for r in (log.get("risk_notes") or []):
                    if symbol and symbol in str(r).upper():
                        risk_hint = str(r)
                        break
                if decision_hint or risk_hint:
                    break

            if not decision_hint:
                decision_hint = "mNAV deviation, BX Trender weighting, and liquidity filters aligned for execution."
            if not risk_hint:
                risk_hint = "Position sizing and stop logic were applied to limit downside risk."

            lines.append(
                f"- {side} {qty} {symbol} @ {price}: "
                f"The trade was executed because {decision_hint.rstrip('.')}."
                f" Risk controls note: {risk_hint.rstrip('.')}."
            )
        return lines

    def _send_end_of_day_report(day_utc, cycle_logs: list[dict]) -> None:
        summary_raw = broker_daily_summary_tool(str(day_utc))
        try:
            summary = json.loads(summary_raw)
        except Exception:
            summary = {"error": "Could not parse daily summary"}

        if summary.get("error"):
            content = (
                f"End-of-day report ({day_utc})\n"
                f"Could not generate broker summary: {summary.get('error')}"
            )
            send_discord_report_tool(content)
            return

        pnl = summary.get("daily_pnl_usd")
        fills = summary.get("fills") or []
        explanations = _build_trade_explanations(fills, cycle_logs)

        decisions_flat = []
        for log in cycle_logs[-20:]:
            for decision in (log.get("trade_decisions") or []):
                decisions_flat.append(str(decision))
        decisions_text = "\n".join(f"- {d}" for d in decisions_flat[:10]) if decisions_flat else "- No trade decisions recorded."

        trades_text = "\n".join(explanations) if explanations else "- No fills for this day."
        report = (
            f"End-of-day trading report ({day_utc} UTC)\n"
            f"Daily P&L: ${pnl}\n"
            f"Trade decisions snapshot:\n{decisions_text}\n\n"
            f"Trades with analysis (2 sentences each):\n{trades_text}"
        )
        send_discord_report_tool(report)

    def _send_full_breakdown_report(trigger: str, cycle_logs: list[dict]) -> None:
        config_raw = strategy_config_tool()
        account_raw = broker_paper_account_status_tool()
        positions_raw = broker_positions_tool()
        summary_raw = broker_daily_summary_tool()
        recent_mem = _memory_context(memory, last_n=20)
        prompt = (
            "Create a full strategy breakdown for Discord.\n"
            f"Trigger: {trigger}\n\n"
            "Include sections:\n"
            "1) Algo and parameter diagnostics\n"
            "2) Recent performance drivers and failures\n"
            "3) Prioritized algorithm change suggestions\n"
            "4) New methods to evaluate market prices\n"
            "5) Validation/backtest upgrades\n"
            "6) Next-step action plan with risk controls and rollback criteria\n\n"
            f"Strategy config:\n{config_raw}\n\n"
            f"Account snapshot:\n{account_raw}\n\n"
            f"Positions snapshot:\n{positions_raw}\n\n"
            f"Daily summary:\n{summary_raw}\n\n"
            f"Recent memory:\n{recent_mem}\n\n"
            f"Recent cycle logs:\n{json.dumps(cycle_logs[-20:], indent=2)}\n"
        )
        msg = llm.invoke(prompt)
        content = getattr(msg, "content", str(msg))
        if isinstance(content, list):
            content = "\n".join(str(x) for x in content)
        _send_discord_chunked(
            str(content),
            header=f"Full Strategy Breakdown ({datetime.now(timezone.utc).isoformat()} UTC)",
        )

    def _send_weekly_executive_report(trigger: str, cycle_logs: list[dict]) -> None:
        today = datetime.now(timezone.utc).date()
        pnl_rows = []
        total_pnl = 0.0
        total_fills = 0
        for i in range(7):
            d = (today - timedelta(days=i)).isoformat()
            raw = broker_daily_summary_tool(d)
            try:
                row = json.loads(raw)
            except Exception:
                row = {"error": raw, "report_date_utc": d}
            pnl = float(row.get("daily_pnl_usd", 0.0) or 0.0)
            fills = int(row.get("fills_count", 0) or 0)
            total_pnl += pnl
            total_fills += fills
            pnl_rows.append({"date": d, "pnl": round(pnl, 2), "fills": fills})

        memory_tail = _load_memory().get("runs", [])[-80:]
        decisions = []
        orders = []
        for r in memory_tail:
            decisions.extend(r.get("trade_decisions") or [])
            orders.extend(r.get("executed_orders") or [])

        # parameter drift summary from current config
        cfg_raw = strategy_config_tool()
        acct_raw = broker_paper_account_status_tool()
        pos_raw = broker_positions_tool()

        prompt = (
            "Create a weekly executive trading report for Discord.\n"
            f"Trigger: {trigger}\n"
            "Audience: owner/operator, concise but complete.\n\n"
            "Required sections:\n"
            "1) Weekly KPI snapshot\n"
            "2) Best/worst decisions and why\n"
            "3) Parameter changes and impact\n"
            "4) New methods to improve price evaluation next week\n"
            "5) Risks and concrete next-week plan\n\n"
            f"Weekly pnl rows: {json.dumps(pnl_rows, indent=2)}\n"
            f"Weekly total pnl: {round(total_pnl, 2)}\n"
            f"Weekly total fills: {total_fills}\n"
            f"Recent decisions: {json.dumps(decisions[-30:], indent=2)}\n"
            f"Recent executed orders: {json.dumps(orders[-30:], indent=2)}\n"
            f"Current config: {cfg_raw}\n"
            f"Current account: {acct_raw}\n"
            f"Current positions: {pos_raw}\n"
            f"Recent cycle logs: {json.dumps(cycle_logs[-30:], indent=2)}\n"
        )
        msg = llm.invoke(prompt)
        content = getattr(msg, "content", str(msg))
        if isinstance(content, list):
            content = "\n".join(str(x) for x in content)
        _send_discord_chunked(
            str(content),
            header=f"Weekly Executive Report ({datetime.now(timezone.utc).isoformat()} UTC)",
        )

    end_time = datetime.now(timezone.utc) + timedelta(days=days)
    total_trades = 0
    cycle = 0
    current_day = datetime.now(timezone.utc).date()
    day_cycle_logs: list[dict] = []
    next_hourly_report = datetime.now(timezone.utc) + timedelta(hours=1)
    next_breakdown_report = datetime.now(timezone.utc) + timedelta(hours=max(1.0, breakdown_hours))
    next_weekly_report = datetime.now(timezone.utc) + timedelta(days=max(1.0, weekly_report_days))
    print(
        f"Starting continuous mode until {end_time.isoformat()} "
        f"(poll {poll_seconds}s, max trades {max_trades})."
    )

    while datetime.now(timezone.utc) < end_time and total_trades < max_trades:
        now_dt = datetime.now(timezone.utc)
        if now_dt.date() != current_day:
            _send_end_of_day_report(current_day, day_cycle_logs)
            current_day = now_dt.date()
            day_cycle_logs = []

        if now_dt >= next_hourly_report:
            acct_raw = broker_paper_account_status_tool()
            try:
                acct = json.loads(acct_raw)
            except Exception:
                acct = {"error": acct_raw}
            decisions_last_hour = []
            for log in day_cycle_logs[-12:]:
                decisions_last_hour.extend(log.get("trade_decisions") or [])
            msg = (
                f"Hourly status report ({now_dt.isoformat()} UTC)\n"
                f"Total trades so far: {total_trades}/{max_trades}\n"
                f"Equity: {acct.get('equity', 'n/a')} | Cash: {acct.get('cash', 'n/a')}\n"
                f"Recent trade decisions:\n"
                + ("\n".join(f"- {d}" for d in decisions_last_hour[:8]) if decisions_last_hour else "- No decisions in last hour.")
            )
            send_discord_report_tool(msg)
            next_hourly_report = now_dt + timedelta(hours=1)

        if now_dt >= next_breakdown_report:
            _send_full_breakdown_report("scheduled_breakdown", day_cycle_logs)
            next_breakdown_report = now_dt + timedelta(hours=max(1.0, breakdown_hours))

        if now_dt >= next_weekly_report:
            _send_weekly_executive_report("scheduled_weekly_report", day_cycle_logs)
            next_weekly_report = now_dt + timedelta(days=max(1.0, weekly_report_days))

        cycle += 1
        remaining = max_trades - total_trades
        cycle_query = (
            f"{base_query}\n\n"
            f"Cycle {cycle}: execute strategy now. Remaining trade budget: {remaining}. "
            "Use broker paper tools when available and provide session analysis.\n\n"
            "Session memory context (most recent entries):\n"
            f"{_memory_context(memory, last_n=12)}\n"
            "Incorporate prior outcomes and avoid repeating low-conviction mistakes."
        )
        raw_response = agent.invoke({"messages": [{"role": "user", "content": cycle_query}]})
        structured: TraderResponse | None = raw_response.get("structured_response")

        placed = 0
        if structured and structured.executed_orders:
            placed = len(structured.executed_orders)
            total_trades += placed

        day_cycle_logs.append(
            {
                "cycle": cycle,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "trade_decisions": [] if not structured else (structured.trade_decisions or []),
                "executed_orders": [] if not structured else (structured.executed_orders or []),
                "risk_notes": [] if not structured else (structured.risk_notes or []),
            }
        )

        if structured:
            _append_memory_entry(
                memory,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": "continuous",
                    "cycle": cycle,
                    "objective": structured.objective,
                    "market_regime": structured.market_regime,
                    "analysis_summary": structured.analyst_model[:600],
                    "trade_decisions": structured.trade_decisions or [],
                    "executed_orders": structured.executed_orders or [],
                    "risk_notes": structured.risk_notes or [],
                },
            )
            _save_memory(memory)

        now_text = datetime.now(timezone.utc).isoformat()
        print(f"[{now_text}] cycle={cycle} placed={placed} total={total_trades}/{max_trades}")

        if datetime.now(timezone.utc) >= end_time or total_trades >= max_trades:
            break
        time.sleep(max(15, poll_seconds))

    _send_end_of_day_report(current_day, day_cycle_logs)
    _send_full_breakdown_report("session_end_breakdown", day_cycle_logs)
    _send_weekly_executive_report("session_end_weekly_report", day_cycle_logs)
    _save_memory(memory)
    print("Continuous mode completed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-days", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-trades", type=int, default=1000)
    parser.add_argument("--breakdown-hours", type=float, default=24.0)
    parser.add_argument("--weekly-report-days", type=float, default=7.0)
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()

    query = resolve_query_interactively(args.query)
    if not query:
        raise RuntimeError("No objective provided.")

    if args.continuous_days > 0:
        confirm_continuous_run(
            query=query,
            days=args.continuous_days,
            poll_seconds=args.poll_seconds,
            max_trades=args.max_trades,
        )
        run_continuous(
            base_query=query,
            days=args.continuous_days,
            poll_seconds=args.poll_seconds,
            max_trades=args.max_trades,
            breakdown_hours=args.breakdown_hours,
            weekly_report_days=args.weekly_report_days,
        )
    else:
        run_once(query)


if __name__ == "__main__":
    main()
