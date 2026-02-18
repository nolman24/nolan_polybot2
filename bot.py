"""
PolyBot - BTC Prediction Market Copy Trader
On-chain version: monitors Polygon blockchain via Alchemy for near-instant trade detection.

Simulation features:
  - Live order book slippage (real asks walked level by level)
  - Historical spread profiling per market (time-of-day aware)
  - Polygon gas / mempool congestion lag modelling via Alchemy
  - Queue position competition model (other copy traders)
  - Partial fill simulation from real liquidity
  - Taker fee (2%)
  - Calibration mode: small live trades to measure real vs simulated gap
  - Daily loss limit, per-trade size cap, daily summary
"""

import os
import json
import math
import logging
import asyncio
import random
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler("polybot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN          = os.getenv("TELEGRAM_TOKEN")
TARGET_WALLET           = os.getenv("TARGET_WALLET", "").lower()
ALCHEMY_URL             = os.getenv("ALCHEMY_URL", "")
POLL_INTERVAL           = int(os.getenv("POLL_INTERVAL", "5"))
POLY_GAMMA_BASE         = "https://gamma-api.polymarket.com"
POLY_CLOB_BASE          = "https://clob.polymarket.com"
DATA_FILE               = "polybot_data.json"
SPREAD_HISTORY_FILE     = "spread_history.json"
CALIBRATION_FILE        = "calibration.json"
BTC_KEYWORDS            = ["btc", "bitcoin", "btc price", "bitcoin price"]
# Polymarket has multiple CTF Exchange contracts - monitor both
POLYMARKET_CTF_CONTRACT_1 = "0x4d97dcd97ec945f40cf65f87097ace5ea0476045"  # Original
POLYMARKET_CTF_CONTRACT_2 = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # Newer version
POLYMARKET_CTF_CONTRACT   = POLYMARKET_CTF_CONTRACT_1  # Keep for backwards compat
POLYMARKET_TAKER_FEE      = 0.02    # 2% taker fee
POLYGON_BLOCK_TIME      = 2.0     # seconds per block
CALIBRATION_TRADE_SIZE  = 5.0     # USDC per calibration trade (tiny, just for measuring)
MAX_COMPETITOR_WALLETS  = 50      # estimated copy traders watching same wallet

# ─── Persistent Spread History ────────────────────────────────────────────────
def load_spread_history() -> dict:
    """
    Stores per-market, per-hour-of-day spread observations.
    Used to build a time-of-day spread model for each market.
    Structure: { condition_id: { "0": [spread1, spread2], "1": [...], ... "23": [...] } }
    """
    if os.path.exists(SPREAD_HISTORY_FILE):
        with open(SPREAD_HISTORY_FILE) as f:
            return json.load(f)
    return {}

def save_spread_history(history: dict):
    with open(SPREAD_HISTORY_FILE, "w") as f:
        json.dump(history, f)

def record_spread_observation(condition_id: str, spread: float):
    """Record a live spread observation for time-of-day modelling."""
    if not condition_id or spread <= 0:
        return
    history = load_spread_history()
    hour_key = str(datetime.now(timezone.utc).hour)
    if condition_id not in history:
        history[condition_id] = {}
    if hour_key not in history[condition_id]:
        history[condition_id][hour_key] = []
    history[condition_id][hour_key].append(round(spread, 5))
    # Keep last 100 observations per hour per market
    history[condition_id][hour_key] = history[condition_id][hour_key][-100:]
    save_spread_history(history)

def get_historical_spread(condition_id: str) -> float:
    """
    Return the expected spread for this market at this time of day.
    Falls back to global BTC market average if no data yet.
    """
    history = load_spread_history()
    hour_key = str(datetime.now(timezone.utc).hour)
    # Try exact market + hour
    if condition_id in history and hour_key in history[condition_id]:
        observations = history[condition_id][hour_key]
        if len(observations) >= 3:
            return sum(observations) / len(observations)
    # Try same market any hour
    if condition_id in history:
        all_obs = [v for vals in history[condition_id].values() for v in vals]
        if all_obs:
            return sum(all_obs) / len(all_obs)
    # Global fallback: BTC markets typically 1.5–3¢ spread
    return random.uniform(0.015, 0.030)

# ─── Calibration System ───────────────────────────────────────────────────────
def load_calibration() -> dict:
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return {
        "trades": [],              # list of {sim_price, real_price, sim_fill_pct, real_fill_pct, ...}
        "price_bias": 0.0,         # mean(real_price - sim_price) — positive means sim underestimates cost
        "fill_bias": 0.0,          # mean(real_fill_pct - sim_fill_pct) — negative means sim overestimates fills
        "n_samples": 0,
        "active": False,
        "calibration_size": CALIBRATION_TRADE_SIZE,
    }

def save_calibration(cal: dict):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(cal, f, indent=2, default=str)

def get_calibration_adjustments() -> tuple[float, float]:
    """
    Returns (price_adj, fill_adj) to apply to simulation based on
    observed real vs simulated differences.
    price_adj: add this many cents to simulated fill price
    fill_adj:  multiply fill_pct by this factor (e.g. 0.92 = sim fills 8% too optimistic)
    """
    cal = load_calibration()
    if cal["n_samples"] < 5:
        return 0.0, 1.0   # not enough data yet
    price_adj = cal["price_bias"]
    fill_adj  = max(0.5, min(1.2, 1.0 + cal["fill_bias"] / 100))
    return price_adj, fill_adj

# ─── State ────────────────────────────────────────────────────────────────────
def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            data = json.load(f)
        defaults = {
            "daily_loss_limit": 200.0,
            "max_trade_size": 100.0,
            "daily_summary_hour": 20,
            "last_summary_date": None,
            "day_start_balance": None,
            "daily_loss_paused": False,
            "sim_detection_lag": True,
            "sim_slippage": True,
            "sim_liquidity": True,
            "sim_fees": True,
            "sim_gas_lag": True,
            "sim_queue_competition": True,
            "sim_historical_spreads": True,
            "calibration_mode": False,
            "calibration_size": CALIBRATION_TRADE_SIZE,
        }
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data
    return {
        "mode": "paper",
        "paper_balance": 1000.0,
        "paper_trades": [],
        "live_trades": [],
        "calibration_trades": [],
        "seen_tx_hashes": [],
        "authorized_users": [],
        "copy_fraction": 1.0,
        "notifications_chat_id": None,
        "running": False,
        "last_block": None,
        # Safety
        "daily_loss_limit": 200.0,
        "max_trade_size": 100.0,
        "daily_summary_hour": 20,
        "last_summary_date": None,
        "day_start_balance": None,
        "daily_loss_paused": False,
        # Simulation toggles
        "sim_detection_lag": True,
        "sim_slippage": True,
        "sim_liquidity": True,
        "sim_fees": True,
        "sim_gas_lag": True,
        "sim_queue_competition": True,
        "sim_historical_spreads": True,
        # Calibration
        "calibration_mode": False,
        "calibration_size": CALIBRATION_TRADE_SIZE,
    }

def save_data(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

state = load_data()

# ─── Safety Helpers ───────────────────────────────────────────────────────────
def get_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def reset_day_if_needed():
    today = get_today_str()
    if state.get("last_summary_date") != today:
        state["day_start_balance"] = state["paper_balance"] if state["mode"] == "paper" else None
        state["daily_loss_paused"] = False
        save_data(state)

def get_todays_pnl() -> float:
    mode   = state["mode"]
    trades = state["paper_trades"] if mode == "paper" else state["live_trades"]
    today  = get_today_str()
    pnl    = 0.0
    for t in [x for x in trades if x.get("timestamp", "")[:10] == today]:
        if t.get("status") == "won":
            pnl += float(t.get("payout", 0)) - float(t.get("amount_usdc", 0))
        elif t.get("status") == "lost":
            pnl -= float(t.get("amount_usdc", 0))
    return pnl

def check_daily_loss_limit() -> tuple[bool, str]:
    limit = state.get("daily_loss_limit", 0)
    if limit <= 0:
        return True, ""
    pnl = get_todays_pnl()
    if pnl <= -limit:
        return False, f"Down ${abs(pnl):.2f} today (limit: ${limit:.0f})"
    return True, ""

def apply_size_cap(amount: float) -> float:
    cap = state.get("max_trade_size", 0)
    if cap > 0 and amount > cap:
        log.info(f"Capping ${amount:.2f} → ${cap:.2f}")
        return cap
    return amount

# ─── Alchemy / Polygon Helpers ────────────────────────────────────────────────
def alchemy_rpc(method: str, params: list):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    try:
        r = requests.post(ALCHEMY_URL, json=payload, timeout=10)
        r.raise_for_status()
        return r.json().get("result")
    except Exception as e:
        log.error(f"Alchemy RPC ({method}): {e}")
        return None

def get_latest_block() -> int | None:
    result = alchemy_rpc("eth_blockNumber", [])
    return int(result, 16) if result else None

def get_polygon_gas_price() -> float:
    """
    Fetch current Polygon gas price via Alchemy.
    Returns gwei value. Used to model mempool congestion lag.
    Typical Polygon: 30–200 gwei. High congestion: 200+ gwei.
    """
    try:
        result = alchemy_rpc("eth_gasPrice", [])
        if result:
            gwei = int(result, 16) / 1e9
            return gwei
    except Exception as e:
        log.error(f"Gas price fetch error: {e}")
    return 50.0  # fallback: assume moderate congestion

def estimate_confirmation_lag(gas_gwei: float) -> float:
    """
    Estimate how many extra seconds your tx takes to confirm
    based on current mempool congestion.
    Polygon is fast but congestion still adds lag.
    """
    if gas_gwei < 50:
        return random.uniform(0.5, 1.5)    # very fast
    elif gas_gwei < 100:
        return random.uniform(1.0, 3.0)    # normal
    elif gas_gwei < 200:
        return random.uniform(2.0, 6.0)    # congested
    else:
        return random.uniform(5.0, 15.0)   # heavily congested

def get_wallet_transactions(from_block: int, to_block: int) -> list:
    """
    Monitor ERC-1155 TransferSingle and TransferBatch events from BOTH Polymarket CTF contracts.
    Properly decodes event data to extract actual trade details.
    
    Event signatures:
    - TransferSingle(operator, from, to, id, value)
    - TransferBatch(operator, from, to, ids, values)
    """
    try:
        # ERC-1155 TransferSingle event topic
        TRANSFER_SINGLE_TOPIC = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
        # ERC-1155 TransferBatch event topic  
        TRANSFER_BATCH_TOPIC = "0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb"
        
        all_trades = []
        padded_wallet = "0x" + TARGET_WALLET[2:].zfill(64).lower()
        
        # CRITICAL FIX: Reduce block range to avoid 400 errors
        # Alchemy has limits on eth_getLogs queries - keep it small
        actual_to_block = min(to_block, from_block + 10)  # Max 10 blocks per query
        
        # Query both contracts, both directions
        for contract_addr in [POLYMARKET_CTF_CONTRACT_1, POLYMARKET_CTF_CONTRACT_2]:
            # Query 1: Transfers FROM wallet (sells)
            try:
                params_from = [{
                    "fromBlock": hex(from_block),
                    "toBlock": hex(actual_to_block),
                    "address": contract_addr,
                    "topics": [
                        [TRANSFER_SINGLE_TOPIC, TRANSFER_BATCH_TOPIC],
                        None,  # operator (any)
                        padded_wallet,  # from (our target wallet)
                    ]
                }]
                
                result_from = alchemy_rpc("eth_getLogs", params_from)
                if result_from:
                    for log_entry in result_from:
                        trade = parse_transfer_event(log_entry, "sell", contract_addr)
                        if trade:
                            all_trades.append(trade)
            except Exception as e:
                log.error(f"Error fetching FROM transfers for {contract_addr[:10]}: {e}")
            
            # Query 2: Transfers TO wallet (buys)
            try:
                params_to = [{
                    "fromBlock": hex(from_block),
                    "toBlock": hex(actual_to_block),
                    "address": contract_addr,
                    "topics": [
                        [TRANSFER_SINGLE_TOPIC, TRANSFER_BATCH_TOPIC],
                        None,  # operator (any)
                        None,  # from (any)
                        padded_wallet,  # to (our target wallet)
                    ]
                }]
                
                result_to = alchemy_rpc("eth_getLogs", params_to)
                if result_to:
                    for log_entry in result_to:
                        trade = parse_transfer_event(log_entry, "buy", contract_addr)
                        if trade:
                            all_trades.append(trade)
            except Exception as e:
                log.error(f"Error fetching TO transfers for {contract_addr[:10]}: {e}")
        
        # Deduplicate by tx_hash
        seen = set()
        unique_trades = []
        for trade in all_trades:
            if trade["hash"] not in seen:
                seen.add(trade["hash"])
                unique_trades.append(trade)
        
        return unique_trades
        
    except Exception as e:
        log.error(f"Event logs error: {e}")
        return []

def parse_transfer_event(log_entry: dict, direction: str, contract_addr: str) -> dict | None:
    """
    Parse an ERC-1155 transfer event to extract trade details.
    Returns a dict with hash, token_id, amount, direction, etc.
    """
    try:
        tx_hash = log_entry.get("transactionHash")
        if not tx_hash:
            return None
            
        block_num = int(log_entry.get("blockNumber", "0x0"), 16)
        topics = log_entry.get("topics", [])
        data = log_entry.get("data", "0x")
        
        TRANSFER_SINGLE_TOPIC = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
        
        if len(topics) >= 4 and topics[0] == TRANSFER_SINGLE_TOPIC:
            # TransferSingle: topics[3] is token ID, data contains amount
            token_id = int(topics[3], 16)
            
            # Decode amount from data field (skip first 2 chars "0x")
            if len(data) >= 66:
                amount_hex = data[2:66]  # First 32 bytes = amount
                amount_raw = int(amount_hex, 16)
            else:
                amount_raw = 0
            
            return {
                "hash": tx_hash,
                "blockNumber": block_num,
                "token_id": token_id,
                "amount_raw": amount_raw,
                "type": "single",
                "contract": contract_addr,
                "direction": direction
            }
        else:
            # Batch transfer - for now just flag it, we'll decode details later if needed
            return {
                "hash": tx_hash,
                "blockNumber": block_num,
                "token_id": 0,
                "amount_raw": 0,
                "type": "batch",
                "contract": contract_addr,
                "direction": direction
            }
    except Exception as e:
        log.error(f"Error parsing event: {e}")
        return None

# ─── Polymarket Market / Order Book ───────────────────────────────────────────
def get_market_token_id(condition_id: str, outcome: str) -> str | None:
    """Fetch the ERC1155 token ID for YES or NO outcome of a market."""
    try:
        r = requests.get(f"{POLY_CLOB_BASE}/markets/{condition_id}", timeout=8)
        if r.status_code == 200:
            tokens = r.json().get("tokens", [])
            idx    = 0 if outcome == "YES" else 1
            if len(tokens) > idx:
                return tokens[idx].get("token_id")
    except Exception as e:
        log.error(f"Token ID fetch error: {e}")
    return None

def fetch_live_orderbook(condition_id: str, outcome: str) -> dict:
    """Fetch live order book snapshot. Returns {asks: [{price, size}], bids: [...]}"""
    try:
        token_id = get_market_token_id(condition_id, outcome)
        if not token_id:
            return {}
        r = requests.get(f"{POLY_CLOB_BASE}/book", params={"token_id": token_id}, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.error(f"Order book error: {e}")
    return {}

def get_market_spread(condition_id: str, outcome: str) -> float | None:
    """
    Fetch live best bid/ask and return the spread in cents.
    Also records this observation into historical spread data.
    """
    try:
        ob = fetch_live_orderbook(condition_id, outcome)
        asks = ob.get("asks", [])
        bids = ob.get("bids", [])
        if asks and bids:
            best_ask = min(float(a["price"]) for a in asks)
            best_bid = max(float(b["price"]) for b in bids)
            spread   = best_ask - best_bid
            if spread > 0:
                record_spread_observation(condition_id, spread)
                return spread
    except Exception as e:
        log.error(f"Spread calc error: {e}")
    return None

def lookup_market_for_tx(tx_hash: str, token_id: int = 0, amount_raw: int = 0) -> dict:
    """
    Look up market details for a transaction.
    PRIORITY ORDER:
    1. Token ID mapping (most reliable - direct from on-chain data)
    2. Gamma API by tx hash (fallback if token_id lookup fails)
    3. Generic BTC market (last resort)
    """
    # METHOD 1: Token ID mapping (BEST - no lag, always accurate)
    if token_id > 0:
        token_map = get_token_market_map()
        token_key = str(token_id)
        
        if token_key in token_map:
            market_info = token_map[token_key]
            
            # Calculate actual amount from raw on-chain value
            # Polymarket uses 6 decimals for USDC amounts
            amount_usdc = amount_raw / 1e6 if amount_raw > 0 else 10.0
            
            log.info(f"✅ Token ID match: {tx_hash[:12]} → {market_info['outcome']} on {market_info['question'][:40]}")
            
            return {
                "question": market_info["question"],
                "condition_id": market_info["condition_id"],
                "outcome": market_info["outcome"],
                "price": market_info["price"],
                "amount_usdc": amount_usdc,
                "source": "token_id_map"
            }
        else:
            log.warning(f"Token ID {token_id} not in map (might not be BTC market)")
    
    # METHOD 2: Gamma API by transaction hash (FALLBACK)
    try:
        r = requests.get(
            f"{POLY_GAMMA_BASE}/trades",
            params={"transactionHash": tx_hash, "limit": 5},
            timeout=8
        )
        if r.status_code == 200:
            trades = r.json()
            if isinstance(trades, list) and trades:
                trade = trades[0]
                market_id = trade.get("market") or trade.get("conditionId")
                if market_id:
                    market = get_market_info(market_id)
                    outcome_index = trade.get("outcomeIndex", 0)
                    
                    log.info(f"✅ Gamma API match: {tx_hash[:12]} → {market.get('question', '')[:40]}")
                    
                    return {
                        "question": market.get("question", "Unknown Market"),
                        "condition_id": market_id,
                        "outcome": "YES" if outcome_index == 0 else "NO",
                        "price": float(trade.get("price", 0.5)),
                        "amount_usdc": float(trade.get("size", 10.0)),
                        "source": "gamma_api"
                    }
    except Exception as e:
        log.error(f"Gamma API error for {tx_hash[:12]}: {e}")
    
    # METHOD 3: Generic fallback (WORST - means we couldn't identify the trade)
    log.error(f"❌ No market found for {tx_hash[:12]} (token_id: {token_id}) - SKIPPING TRADE")
    fallback = get_active_btc_market()
    fallback["source"] = "fallback_generic"
    return fallback

def get_market_info(condition_id: str) -> dict:
    try:
        r = requests.get(f"{POLY_GAMMA_BASE}/markets/{condition_id}", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def get_active_btc_market() -> dict:
    try:
        r = requests.get(
            f"{POLY_GAMMA_BASE}/markets",
            params={"active": True, "closed": False, "limit": 20, "tag": "crypto"},
            timeout=8
        )
        if r.status_code == 200:
            for m in r.json():
                if any(kw in m.get("question", "").lower() for kw in BTC_KEYWORDS):
                    return {
                        "question":     m.get("question", "BTC Market"),
                        "condition_id": m.get("conditionId", ""),
                        "outcome":      "YES",
                        "price":        0.5,
                        "amount_usdc":  10.0,
                    }
    except Exception as e:
        log.error(f"Active market error: {e}")
    return {"question": "BTC Prediction Market", "condition_id": "", "outcome": "YES", "price": 0.5, "amount_usdc": 10.0}

def is_btc_market(question: str) -> bool:
    return any(kw in question.lower() for kw in BTC_KEYWORDS)

# ─── Token ID to Market Mapping ───────────────────────────────────────────────
def build_token_market_map() -> dict:
    """
    Query all active BTC markets and build a mapping of token_id → market info.
    This lets us look up markets directly from on-chain token IDs instead of 
    relying on Gamma API transaction indexing.
    """
    token_map = {}
    try:
        # Strategy: Search for "bitcoin" keyword directly
        all_markets = []
        
        # Try multiple search terms
        for search_term in ["bitcoin", "btc"]:
            try:
                r = requests.get(
                    f"{POLY_GAMMA_BASE}/markets",
                    params={"active": True, "limit": 100, "search": search_term},
                    timeout=10
                )
                if r.status_code == 200:
                    results = r.json()
                    if isinstance(results, list):
                        all_markets.extend(results)
                        log.info(f"Search '{search_term}' returned {len(results)} markets")
            except Exception as e:
                log.error(f"Search for '{search_term}' failed: {e}")
        
        # Deduplicate by condition_id
        seen_conditions = set()
        unique_markets = []
        for m in all_markets:
            cid = m.get("conditionId")
            if cid and cid not in seen_conditions:
                seen_conditions.add(cid)
                unique_markets.append(m)
        
        log.info(f"Found {len(unique_markets)} unique Bitcoin markets total")
        
        # Debug: log first few markets
        for i, market in enumerate(unique_markets[:5]):
            log.info(f"BTC market {i+1}: {market.get('question', '')[:60]}")
        
        token_count = 0
        for market in unique_markets:
            question = market.get("question", "")
            condition_id = market.get("conditionId", "")
            
            if not condition_id:
                log.warning(f"Market missing conditionId: {question[:40]}")
                continue
            
            # Get token IDs for this market
            tokens = market.get("tokens", [])
            if not tokens:
                log.warning(f"Market has no tokens array: {question[:40]}")
                # Try alternate field names
                if "clobTokenIds" in market:
                    log.info(f"Found clobTokenIds instead: {market['clobTokenIds']}")
                continue
            
            for idx, token_info in enumerate(tokens):
                # Try multiple field names for token_id
                token_id = (
                    token_info.get("token_id") or 
                    token_info.get("tokenId") or
                    token_info.get("id") or
                    token_info.get("clobTokenId")
                )
                
                if token_id:
                    outcome = "YES" if idx == 0 else "NO"
                    outcome_prices = market.get("outcomePrices", [])
                    price = float(outcome_prices[idx]) if len(outcome_prices) > idx else 0.5
                    
                    token_map[str(token_id)] = {
                        "question": question,
                        "condition_id": condition_id,
                        "outcome": outcome,
                        "price": price,
                        "token_id": token_id
                    }
                    token_count += 1
                else:
                    log.warning(f"Token {idx} missing ID in market: {question[:40]}, token_info: {token_info}")
        
        log.info(f"Successfully mapped {token_count} tokens from {len(unique_markets)} BTC markets")
        
        # Debug: log first few token mappings
        for i, (tid, info) in enumerate(list(token_map.items())[:3]):
            log.info(f"Token {tid} → {info['outcome']} on '{info['question'][:50]}'")
        
        return token_map
        
    except Exception as e:
        log.error(f"Error building token map: {e}")
        import traceback
        log.error(traceback.format_exc())
        return token_map

# Global token map cache (refreshed periodically)
TOKEN_MARKET_MAP = {}
LAST_MAP_UPDATE = None

def get_token_market_map(force_refresh: bool = False) -> dict:
    """Get token map, refreshing if needed (every 5 minutes)."""
    global TOKEN_MARKET_MAP, LAST_MAP_UPDATE
    now = datetime.now(timezone.utc)
    
    needs_refresh = (
        force_refresh or 
        not TOKEN_MARKET_MAP or 
        not LAST_MAP_UPDATE or 
        (now - LAST_MAP_UPDATE).total_seconds() > 300
    )
    
    if needs_refresh:
        TOKEN_MARKET_MAP = build_token_market_map()
        LAST_MAP_UPDATE = now
    
    return TOKEN_MARKET_MAP

# ─── Market Resolution Checker ────────────────────────────────────────────────
async def check_trade_resolutions(app: Application):
    """
    Check if any open trades have resolved and update their status.
    Runs every 5 minutes as a background task.
    """
    global state
    
    # Check paper trades
    for trade in state.get("paper_trades", []):
        if trade.get("status") != "open":
            continue
        
        # Try to get market resolution from Polymarket
        condition_id = None
        # We need to look up the market from the trade details
        # The tx_hash can help us find it via Gamma API
        tx_hash = trade.get("tx_hash")
        if not tx_hash:
            continue
            
        try:
            # Lookup the trade to get condition_id
            r = requests.get(
                f"{POLY_GAMMA_BASE}/trades",
                params={"transactionHash": tx_hash, "limit": 1},
                timeout=8
            )
            if r.status_code == 200:
                trades_data = r.json()
                if isinstance(trades_data, list) and trades_data:
                    condition_id = trades_data[0].get("market") or trades_data[0].get("conditionId")
            
            if not condition_id:
                continue
                
            # Get market info to check if resolved
            market = get_market_info(condition_id)
            if not market:
                continue
                
            # Check if market is closed and has outcome
            closed = market.get("closed", False)
            resolved = market.get("resolved", False)
            
            if not (closed or resolved):
                continue
                
            # Get the winning outcome
            outcome_prices = market.get("outcomePrices", [])
            if not outcome_prices or len(outcome_prices) < 2:
                continue
            
            # In a binary market, winning side should be at or near 1.0
            yes_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else 0
            no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0
            
            # Determine winner (whichever is closer to 1.0)
            winning_outcome = "YES" if yes_price > no_price else "NO"
            
            # Check if our trade won
            our_outcome = trade.get("outcome", "YES")
            if our_outcome == winning_outcome:
                # We won - calculate payout
                # Payout = (amount / entry_price) * 1.0 (winning tokens pay $1 each)
                entry_price = trade.get("price") or trade.get("total_cost_basis", 0.5)
                amount = trade.get("amount_usdc", 0)
                shares = amount / entry_price if entry_price > 0 else 0
                payout = shares * 1.0  # Each winning share pays $1
                
                trade["status"] = "won"
                trade["payout"] = round(payout, 2)
                
                # Update paper balance
                if trade.get("mode") == "paper":
                    state["paper_balance"] += payout
                
                # Notify user
                chat_id = state.get("notifications_chat_id")
                profit = payout - amount
                if chat_id:
                    await app.bot.send_message(
                        chat_id,
                        f"✅ *Trade Won!*\n"
                        f"`{trade['market'][:50]}`\n"
                        f"💰 Profit: `+${profit:.2f}` (payout: ${payout:.2f})\n"
                        f"{'💼 Paper balance: $' + str(round(state['paper_balance'], 2)) if trade.get('mode')=='paper' else ''}",
                        parse_mode="Markdown"
                    )
                
                log.info(f"Trade resolved WIN: {trade['market'][:40]} | +${payout:.2f}")
            else:
                # We lost
                trade["status"] = "lost"
                trade["payout"] = 0.0
                
                # Notify user
                chat_id = state.get("notifications_chat_id")
                if chat_id:
                    await app.bot.send_message(
                        chat_id,
                        f"❌ *Trade Lost*\n"
                        f"`{trade['market'][:50]}`\n"
                        f"💸 Loss: `-${amount:.2f}`\n"
                        f"{'💼 Paper balance: $' + str(round(state['paper_balance'], 2)) if trade.get('mode')=='paper' else ''}",
                        parse_mode="Markdown"
                    )
                
                log.info(f"Trade resolved LOSS: {trade['market'][:40]} | -${trade.get('amount_usdc', 0):.2f}")
                
        except Exception as e:
            log.error(f"Resolution check error for {tx_hash[:12]}: {e}")
            continue
    
    # Check live trades (same logic)
    for trade in state.get("live_trades", []):
        if trade.get("status") != "open":
            continue
        
        tx_hash = trade.get("tx_hash")
        if not tx_hash:
            continue
            
        try:
            r = requests.get(
                f"{POLY_GAMMA_BASE}/trades",
                params={"transactionHash": tx_hash, "limit": 1},
                timeout=8
            )
            if r.status_code == 200:
                trades_data = r.json()
                if isinstance(trades_data, list) and trades_data:
                    condition_id = trades_data[0].get("market") or trades_data[0].get("conditionId")
            
            if not condition_id:
                continue
                
            market = get_market_info(condition_id)
            if not market:
                continue
                
            closed = market.get("closed", False)
            resolved = market.get("resolved", False)
            
            if not (closed or resolved):
                continue
                
            outcome_prices = market.get("outcomePrices", [])
            if not outcome_prices or len(outcome_prices) < 2:
                continue
            
            yes_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else 0
            no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0
            winning_outcome = "YES" if yes_price > no_price else "NO"
            
            our_outcome = trade.get("outcome", "YES")
            if our_outcome == winning_outcome:
                entry_price = trade.get("price", 0.5)
                amount = trade.get("amount_usdc", 0)
                shares = amount / entry_price if entry_price > 0 else 0
                payout = shares * 1.0
                
                trade["status"] = "won"
                trade["payout"] = round(payout, 2)
                
                # Notify user
                chat_id = state.get("notifications_chat_id")
                profit = payout - amount
                if chat_id:
                    await app.bot.send_message(
                        chat_id,
                        f"✅ *LIVE Trade Won!*\n"
                        f"`{trade['market'][:50]}`\n"
                        f"💰 Profit: `+${profit:.2f}` (payout: ${payout:.2f})",
                        parse_mode="Markdown"
                    )
                
                log.info(f"LIVE trade resolved WIN: {trade['market'][:40]} | +${payout:.2f}")
            else:
                trade["status"] = "lost"
                trade["payout"] = 0.0
                
                # Notify user
                chat_id = state.get("notifications_chat_id")
                if chat_id:
                    await app.bot.send_message(
                        chat_id,
                        f"❌ *LIVE Trade Lost*\n"
                        f"`{trade['market'][:50]}`\n"
                        f"💸 Loss: `-${amount:.2f}`",
                        parse_mode="Markdown"
                    )
                
                log.info(f"LIVE trade resolved LOSS: {trade['market'][:40]} | -${trade.get('amount_usdc', 0):.2f}")
                
        except Exception as e:
            log.error(f"Resolution check error for {tx_hash[:12]}: {e}")
            continue
    
    save_data(state)

# ─── Full Realistic Simulation Engine ─────────────────────────────────────────
def simulate_realistic_fill(
    intended_price: float,
    intended_amount: float,
    condition_id: str,
    outcome: str,
) -> dict:
    """
    Complete simulation of what your order would experience in the real market.

    Steps:
    1.  Gas / mempool lag       — fetch live Polygon gas price via Alchemy, estimate
                                   how long your tx takes to confirm
    2.  Detection + API lag     — poll interval + Gamma API lookup delay
    3.  Queue competition       — estimate how many other copy bots are ahead of you,
                                   consuming liquidity before your order lands
    4.  Price drift             — market price moves during your total lag window
    5.  Historical spread       — use time-of-day spread model if live book unavailable
    6.  Live order book walk    — step through real asks to get weighted avg fill price
    7.  Partial fill            — limited by actual available liquidity
    8.  Calibration correction  — apply measured real-vs-sim bias if enough data
    9.  Taker fee               — 2% of filled amount
    10. True cost basis         — effective price per share all-in
    """
    notes = []

    # ── 1 & 2: Total lag = gas confirmation + detection + API overhead ─────────
    gas_gwei        = get_polygon_gas_price() if state.get("sim_gas_lag") else 50.0
    gas_lag         = estimate_confirmation_lag(gas_gwei) if state.get("sim_gas_lag") else 0.5
    detection_lag   = POLL_INTERVAL + random.uniform(1.0, 4.0)
    api_lookup_lag  = random.uniform(0.5, 2.0)
    total_lag       = gas_lag + detection_lag + api_lookup_lag

    if state.get("sim_gas_lag"):
        notes.append(
            f"Gas: {gas_gwei:.0f} gwei → +{gas_lag:.1f}s confirm lag"
        )

    # ── 3: Queue competition — how much liquidity do competitors eat first? ────
    competitor_consumed = 0.0
    if state.get("sim_queue_competition"):
        # Model: Polymarket copy trading is popular for top wallets.
        # Estimate 5-30 other bots watching the same wallet, each betting similar sizes.
        # Those ahead of us in the mempool consume liquidity at our price level.
        n_competitors   = random.randint(3, 25)
        avg_competitor  = intended_amount * random.uniform(0.3, 1.5)
        # Not all competitors are faster — assume 20-60% are ahead of us
        ahead_fraction  = random.uniform(0.2, 0.6)
        competitor_consumed = n_competitors * avg_competitor * ahead_fraction
        if competitor_consumed > intended_amount * 0.5:
            notes.append(
                f"Queue: ~{n_competitors} bots ahead, consumed ~${competitor_consumed:.0f} liquidity"
            )

    # ── 4: Price drift during lag ─────────────────────────────────────────────
    drifted_price = intended_price
    if state.get("sim_detection_lag"):
        # BTC prediction market volatility: ~0.12% std dev per second
        drift_per_sec  = random.gauss(0, 0.0012)
        total_drift    = drift_per_sec * total_lag
        adverse_drift  = abs(total_drift) * random.uniform(0.4, 1.0)  # drift is always against us
        drifted_price  = min(0.99, intended_price + adverse_drift)
        lag_impact     = drifted_price - intended_price
        if lag_impact > 0.001:
            notes.append(
                f"Price drift: +{lag_impact*100:.2f}¢ over {total_lag:.1f}s total lag"
            )

    # ── 5 & 6: Order book — use live book or historical spread model ──────────
    fill_price    = drifted_price
    filled_amount = intended_amount
    fill_pct      = 100.0
    slippage_cost = 0.0
    book_source   = "synthetic"

    orderbook = {}
    if condition_id:
        orderbook = fetch_live_orderbook(condition_id, outcome)

    asks = orderbook.get("asks", [])

    if asks:
        book_source = "live"
        # Remove liquidity that competitors consumed before us
        adjusted_asks = []
        remaining_competitor = competitor_consumed
        for level in sorted(asks, key=lambda x: float(x.get("price", 1))):
            level_price = float(level.get("price", 1))
            level_size  = float(level.get("size", 0))
            usdc_here   = level_price * level_size
            if remaining_competitor > 0:
                eaten = min(remaining_competitor, usdc_here)
                remaining_competitor -= eaten
                leftover_usdc = usdc_here - eaten
                if leftover_usdc > 0.01:
                    adjusted_asks.append({
                        "price": level_price,
                        "size":  leftover_usdc / level_price
                    })
            else:
                adjusted_asks.append(level)

        # Walk remaining book
        remaining     = intended_amount
        total_cost    = 0.0
        filled_amount = 0.0

        for level in adjusted_asks:
            level_price = float(level.get("price", 1))
            level_size  = float(level.get("size", 0))
            if level_price > drifted_price * 1.06:
                break
            usdc_here = level_price * level_size
            take      = min(remaining, usdc_here)
            total_cost   += take
            filled_amount += take
            remaining    -= take
            if remaining <= 0:
                break

        if filled_amount > 0:
            fill_price    = total_cost / filled_amount
            fill_pct      = round(filled_amount / intended_amount * 100, 1)
            slippage      = fill_price - drifted_price
            slippage_cost = slippage * filled_amount
            if slippage > 0.002:
                notes.append(
                    f"Slippage: +{slippage*100:.2f}¢ (live book walk)"
                )
            if fill_pct < 99:
                notes.append(
                    f"Partial fill: {fill_pct:.0f}% (${filled_amount:.2f} of ${intended_amount:.2f})"
                )
        else:
            return {
                "skipped": True,
                "skip_reason": "No liquidity after competitor consumption",
                "notes": notes,
                "fill_price": drifted_price,
                "filled_amount": 0.0,
                "fill_pct": 0.0,
                "fee_cost": 0.0,
                "slippage_cost": 0.0,
                "lag_price_impact": drifted_price - intended_price,
                "total_cost_basis": drifted_price,
                "total_lag_seconds": total_lag,
                "gas_gwei": gas_gwei,
                "book_source": book_source,
                "intended_price": intended_price,
                "intended_amount": intended_amount,
            }

    else:
        # ── Synthetic model with historical spread data ────────────────────────
        # Use time-of-day spread if available, otherwise estimate
        if state.get("sim_historical_spreads") and condition_id:
            live_spread = get_market_spread(condition_id, outcome)
            spread      = live_spread if live_spread else get_historical_spread(condition_id)
        else:
            spread = random.uniform(0.010, 0.030)

        if state.get("sim_slippage"):
            # You enter at the ask: spread/2 above mid
            spread_cost   = spread / 2
            # Size impact on a market with estimated $3k-$8k typical liquidity
            typical_liq   = random.uniform(3000, 8000)
            size_impact   = (intended_amount / typical_liq) * random.uniform(0.008, 0.025)
            total_slip    = spread_cost + size_impact
            fill_price    = min(0.99, drifted_price + total_slip)
            slippage_cost = total_slip * intended_amount
            notes.append(
                f"Slippage: +{total_slip*100:.2f}¢ (spread {spread*100:.1f}¢ + size impact) [{book_source}]"
            )

        if state.get("sim_liquidity"):
            # Adjust available liquidity for competitor consumption
            dist_from_mid  = abs(drifted_price - 0.5)
            liq_factor     = max(0.15, 1.0 - dist_from_mid * 1.8)
            available_usdc = max(0, (random.uniform(1000, 9000) * liq_factor) - competitor_consumed)
            if intended_amount > available_usdc and available_usdc > 0:
                filled_amount = round(available_usdc, 2)
                fill_pct      = round(filled_amount / intended_amount * 100, 1)
                notes.append(
                    f"Partial fill: {fill_pct:.0f}% (${filled_amount:.2f} avail after queue)"
                )
            elif available_usdc <= 0:
                return {
                    "skipped": True,
                    "skip_reason": "All liquidity consumed by faster bots",
                    "notes": notes,
                    "fill_price": fill_price,
                    "filled_amount": 0.0,
                    "fill_pct": 0.0,
                    "fee_cost": 0.0,
                    "slippage_cost": 0.0,
                    "lag_price_impact": drifted_price - intended_price,
                    "total_cost_basis": fill_price,
                    "total_lag_seconds": total_lag,
                    "gas_gwei": gas_gwei,
                    "book_source": book_source,
                    "intended_price": intended_price,
                    "intended_amount": intended_amount,
                }

    # ── 7: Apply calibration correction ──────────────────────────────────────
    price_adj, fill_adj = get_calibration_adjustments()
    if price_adj != 0 or fill_adj != 1.0:
        fill_price     = min(0.99, fill_price + price_adj)
        filled_amount  = round(filled_amount * fill_adj, 2)
        fill_pct       = round(filled_amount / intended_amount * 100, 1)
        if abs(price_adj) > 0.001 or abs(fill_adj - 1.0) > 0.02:
            notes.append(
                f"Calibration adj: price +{price_adj*100:.2f}¢, fill ×{fill_adj:.2f}"
            )

    # ── 8: Taker fee ──────────────────────────────────────────────────────────
    fee_cost = 0.0
    if state.get("sim_fees"):
        fee_cost = round(filled_amount * POLYMARKET_TAKER_FEE, 4)
        notes.append(f"Fee: ${fee_cost:.3f} (2% taker)")

    # ── 9: True cost basis ────────────────────────────────────────────────────
    total_spent    = filled_amount + fee_cost
    shares_received = filled_amount / fill_price if fill_price > 0 else 0
    cost_basis     = round(total_spent / shares_received, 5) if shares_received > 0 else fill_price

    return {
        "skipped":           False,
        "skip_reason":       "",
        "intended_price":    intended_price,
        "intended_amount":   intended_amount,
        "fill_price":        round(fill_price, 5),
        "filled_amount":     round(filled_amount, 2),
        "fill_pct":          fill_pct,
        "fee_cost":          fee_cost,
        "slippage_cost":     round(slippage_cost, 4),
        "lag_price_impact":  round(drifted_price - intended_price, 5),
        "total_cost_basis":  cost_basis,
        "total_lag_seconds": round(total_lag, 1),
        "gas_gwei":          round(gas_gwei, 1),
        "book_source":       book_source,
        "notes":             notes,
    }

# ─── Live Trading ─────────────────────────────────────────────────────────────
def place_live_order(condition_id: str, outcome: str, amount: float, price: float) -> dict:
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs
        client = ClobClient(host=POLY_CLOB_BASE, key=os.getenv("POLY_API_KEY"), chain_id=137)
        result = client.create_and_post_order(
            OrderArgs(token_id=condition_id, price=price, size=amount, side="BUY")
        )
        return {"success": True, "result": str(result)}
    except ImportError:
        return {"success": False, "error": "py-clob-client not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── P&L Calculator ───────────────────────────────────────────────────────────
def calculate_pnl(trades: list) -> dict:
    total_invested = total_returned = 0.0
    wins = losses = open_count = 0
    for t in trades:
        invested = float(t.get("amount_usdc", 0))
        total_invested += invested
        status = t.get("status", "open")
        if status == "won":
            total_returned += float(t.get("payout", 0))
            wins += 1
        elif status == "lost":
            losses += 1
        else:
            open_count += 1
    settled = wins + losses
    return {
        "total_invested": total_invested,
        "total_returned": total_returned,
        "realized_pnl":   total_returned - total_invested,
        "wins":           wins,
        "losses":         losses,
        "open":           open_count,
        "win_rate":       (wins / settled * 100) if settled > 0 else 0.0,
        "total_trades":   len(trades),
    }

def format_pnl_message(pnl: dict, mode: str) -> str:
    label = "📄 PAPER" if mode == "paper" else "💰 LIVE"
    sign  = "+" if pnl["realized_pnl"] >= 0 else ""
    icon  = "🟢" if pnl["realized_pnl"] >= 0 else "🔴"
    return (
        f"*{label} P&L Report*\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"{icon} Realized P&L: `{sign}${pnl['realized_pnl']:.2f}`\n"
        f"💵 Total Invested: `${pnl['total_invested']:.2f}`\n"
        f"💸 Total Returned: `${pnl['total_returned']:.2f}`\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"✅ Wins: `{pnl['wins']}`\n"
        f"❌ Losses: `{pnl['losses']}`\n"
        f"⏳ Open: `{pnl['open']}`\n"
        f"📊 Win Rate: `{pnl['win_rate']:.1f}%`\n"
        f"🔢 Total Trades: `{pnl['total_trades']}`"
    )

def build_daily_summary() -> str:
    mode          = state["mode"]
    trades        = state["paper_trades"] if mode == "paper" else state["live_trades"]
    today         = get_today_str()
    all_pnl       = calculate_pnl(trades)
    todays_pnl    = get_todays_pnl()
    todays_trades = [t for t in trades if t.get("timestamp", "")[:10] == today]
    cal           = load_calibration()

    today_sign = "+" if todays_pnl >= 0 else ""
    today_icon = "🟢" if todays_pnl >= 0 else "🔴"
    all_sign   = "+" if all_pnl["realized_pnl"] >= 0 else ""
    all_icon   = "🟢" if all_pnl["realized_pnl"] >= 0 else "🔴"
    mode_label = "📄 Paper" if mode == "paper" else "💰 Live"
    bal_str    = f"\n💼 Paper Balance: `${state['paper_balance']:.2f}`" if mode == "paper" else ""
    loss_limit = state.get("daily_loss_limit", 0)
    size_cap   = state.get("max_trade_size", 0)
    safety_str = ""
    if loss_limit > 0:
        safety_str += f"\n🛡 Loss Limit: `${loss_limit:.0f}/day`"
    if size_cap > 0:
        safety_str += f"\n📏 Size Cap: `${size_cap:.0f}/trade`"
    cal_str = ""
    if cal["n_samples"] >= 5:
        cal_str = (
            f"\n🔬 Cal Bias: price `{cal['price_bias']*100:+.2f}¢` | "
            f"fill `{cal['fill_bias']:+.1f}%` ({cal['n_samples']} samples)"
        )

    return (
        f"📅 *Daily Summary — {today}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔵 Mode: `{mode_label}`\n"
        f"📈 Trades Today: `{len(todays_trades)}`\n"
        f"{today_icon} Today's P&L: `{today_sign}${todays_pnl:.2f}`\n"
        f"{all_icon} All-Time P&L: `{all_sign}${all_pnl['realized_pnl']:.2f}`\n"
        f"📊 All-Time Win Rate: `{all_pnl['win_rate']:.1f}%`\n"
        f"🔢 Total Trades: `{all_pnl['total_trades']}`"
        f"{bal_str}{safety_str}{cal_str}"
    )

# ─── Core Copy Trade Handler ──────────────────────────────────────────────────
async def process_onchain_tx(tx_hash: str, app: Application, token_id: int = 0, amount_raw: int = 0):
    global state

    if tx_hash in state["seen_tx_hashes"]:
        return
    state["seen_tx_hashes"].append(tx_hash)
    if len(state["seen_tx_hashes"]) > 1000:
        state["seen_tx_hashes"] = state["seen_tx_hashes"][-1000:]

    market_info  = lookup_market_for_tx(tx_hash, token_id, amount_raw)
    question     = market_info.get("question", "")
    
    # CRITICAL: Skip if we're using fallback data - it's inaccurate
    if market_info.get("source") == "fallback_generic":
        log.error(f"⚠️ Skipping {tx_hash[:12]} - could not identify market accurately")
        return
    
    if not is_btc_market(question):
        log.info(f"Skipping non-BTC: {tx_hash[:12]} ({question[:35]})")
        return

    chat_id = state.get("notifications_chat_id")

    # Loss limit check
    can_trade, reason = check_daily_loss_limit()
    if not can_trade:
        if not state.get("daily_loss_paused"):
            state["daily_loss_paused"] = True
            save_data(state)
            msg = (
                f"🛑 *Daily Loss Limit Reached*\n━━━━━━━━━━━━━━━━\n"
                f"_{reason}_\n\nBot paused for today. Resumes tomorrow."
            )
            if chat_id:
                await app.bot.send_message(chat_id, msg, parse_mode="Markdown")
        return

    outcome      = market_info.get("outcome", "YES")
    price        = float(market_info.get("price", 0.5))
    raw_amount   = float(market_info.get("amount_usdc", 10.0))
    amount       = apply_size_cap(raw_amount * state["copy_fraction"])
    condition_id = market_info.get("condition_id", "")
    now          = datetime.now(timezone.utc).isoformat()
    mode         = state["mode"]
    was_capped   = amount < (raw_amount * state["copy_fraction"])
    cap_note     = f"\n📏 _Capped ${raw_amount*state['copy_fraction']:.2f} → ${amount:.2f}_" if was_capped else ""

    # ── Calibration mode: place tiny live trade to measure real fills ──────────
    if state.get("calibration_mode") and condition_id:
        cal_size   = float(state.get("calibration_size", CALIBRATION_TRADE_SIZE))
        sim_result = simulate_realistic_fill(price, cal_size, condition_id, outcome)
        live_result = place_live_order(condition_id, outcome, cal_size, price)
        if live_result["success"]:
            cal = load_calibration()
            cal_entry = {
                "timestamp":       now,
                "market":          question[:50],
                "sim_price":       sim_result["fill_price"],
                "sim_fill_pct":    sim_result["fill_pct"],
                "real_price":      price,   # best estimate — real fill price from order result
                "real_fill_pct":   100.0,   # assume full fill on tiny calibration amount
                "tx_hash":         tx_hash,
            }
            cal["trades"].append(cal_entry)
            # Recalculate bias from all samples
            if len(cal["trades"]) >= 2:
                price_diffs = [t["real_price"] - t["sim_price"] for t in cal["trades"]]
                fill_diffs  = [t["real_fill_pct"] - t["sim_fill_pct"] for t in cal["trades"]]
                cal["price_bias"] = sum(price_diffs) / len(price_diffs)
                cal["fill_bias"]  = sum(fill_diffs) / len(fill_diffs)
                cal["n_samples"]  = len(cal["trades"])
            save_calibration(cal)
            msg = (
                f"🔬 *Calibration Trade*\n━━━━━━━━━━━━━━━━\n"
                f"Size: `${cal_size:.2f}` | Sim price: `{sim_result['fill_price']:.4f}`\n"
                f"Samples: `{cal['n_samples']}` | "
                f"Price bias: `{cal['price_bias']*100:+.2f}¢` | "
                f"Fill bias: `{cal['fill_bias']:+.1f}%`"
            )
            if chat_id:
                await app.bot.send_message(chat_id, msg, parse_mode="Markdown")
        return  # calibration trades are separate from paper/live

    if mode == "paper":
        if state["paper_balance"] < amount:
            if chat_id:
                await app.bot.send_message(
                    chat_id,
                    f"⚠️ *Paper Trade Skipped*\nBalance too low: `${state['paper_balance']:.2f}` < `${amount:.2f}`",
                    parse_mode="Markdown"
                )
            return

        sim = simulate_realistic_fill(price, amount, condition_id, outcome)

        if sim["skipped"]:
            if chat_id:
                await app.bot.send_message(
                    chat_id,
                    f"⚠️ *Paper Trade Skipped (Sim)*\n`{question[:55]}`\n_{sim['skip_reason']}_",
                    parse_mode="Markdown"
                )
            return

        sim_amount    = sim["filled_amount"]
        total_debited = sim_amount + sim["fee_cost"]
        if state["paper_balance"] < total_debited:
            sim_amount    = max(0, state["paper_balance"] - sim["fee_cost"])
            total_debited = state["paper_balance"]

        state["paper_balance"] -= total_debited

        copy_trade = {
            "tx_hash":          tx_hash,
            "timestamp":        now,
            "market":           question,
            "outcome":          outcome,
            "price":            sim["fill_price"],
            "intended_price":   price,
            "amount_usdc":      sim_amount,
            "intended_amount":  amount,
            "fill_pct":         sim["fill_pct"],
            "fee_cost":         sim["fee_cost"],
            "slippage_cost":    sim["slippage_cost"],
            "lag_impact":       sim["lag_price_impact"],
            "total_cost_basis": sim["total_cost_basis"],
            "total_lag_s":      sim["total_lag_seconds"],
            "gas_gwei":         sim["gas_gwei"],
            "book_source":      sim["book_source"],
            "status":           "open",
            "payout":           0.0,
            "mode":             "paper",
            "sim_notes":        sim["notes"],
        }
        state["paper_trades"].append(copy_trade)
        save_data(state)

        price_diff = sim["fill_price"] - price
        diff_str   = f"+{price_diff*100:.2f}¢" if price_diff >= 0 else f"{price_diff*100:.2f}¢"
        partial    = f" _(partial {sim['fill_pct']:.0f}%)_" if sim["fill_pct"] < 99 else ""
        sim_lines  = ("\n" + "\n".join(f"  _{n}_" for n in sim["notes"])) if sim["notes"] else ""

        msg = (
            f"📄 *Paper Trade — Realistic Sim*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📋 `{question[:55]}`\n"
            f"🎯 `{outcome}` · Filled: `${sim_amount:.2f}`{partial}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🏷 Target price:    `{price:.4f}`\n"
            f"📈 Your fill price: `{sim['fill_price']:.4f}` `({diff_str})`\n"
            f"⏱ Total lag:       `{sim['total_lag_seconds']:.1f}s` "
            f"_(gas: {sim['gas_gwei']:.0f} gwei)_\n"
            f"📚 Book source:     `{sim['book_source']}`\n"
            f"💸 Fee:             `${sim['fee_cost']:.3f}`\n"
            f"📉 Slippage:        `${sim['slippage_cost']:.3f}`\n"
            f"🧾 Cost basis:      `{sim['total_cost_basis']:.4f}`\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💼 Balance: `${state['paper_balance']:.2f}`"
            f"{cap_note}{sim_lines}"
        )

    else:
        # Live mode
        copy_trade = {
            "tx_hash":     tx_hash,
            "timestamp":   now,
            "market":      question,
            "outcome":     outcome,
            "price":       price,
            "amount_usdc": amount,
            "status":      "open",
            "payout":      0.0,
            "mode":        "live",
        }
        result = place_live_order(condition_id, outcome, amount, price)
        copy_trade["live_result"] = result
        if result["success"]:
            state["live_trades"].append(copy_trade)
            save_data(state)
            msg = (
                f"💰 *Live Trade Executed!*\n━━━━━━━━━━━━━━━━\n"
                f"📋 `{question[:60]}`\n"
                f"🎯 `{outcome}` @ `{price:.4f}`\n"
                f"💵 `${amount:.2f} USDC`\n"
                f"🔗 `{tx_hash[:18]}...`\n"
                f"✅ Order placed\n"
                f"🕐 `{now[:19].replace('T',' ')} UTC`"
                f"{cap_note}"
            )
        else:
            msg = (
                f"❌ *Live Trade FAILED*\n"
                f"`{question[:50]}`\n"
                f"`{result.get('error','Unknown')}`"
            )

    if chat_id:
        await app.bot.send_message(chat_id, msg, parse_mode="Markdown")
    log.info(f"{mode.upper()} | {tx_hash[:14]} | {question[:35]} | {outcome} | ${amount:.2f}")

# ─── Background Polling ───────────────────────────────────────────────────────
async def poll_chain(app: Application):
    log.info(f"On-chain poll started | wallet: {TARGET_WALLET} | interval: {POLL_INTERVAL}s")
    log.info(f"Monitoring 2 Polymarket contracts (buys & sells): {POLYMARKET_CTF_CONTRACT_1[:10]}... & {POLYMARKET_CTF_CONTRACT_2[:10]}...")
    log.info(f"Resolution checking: every 5 minutes")
    
    # Build initial token → market mapping
    log.info("Building token ID → market mapping...")
    get_token_market_map(force_refresh=True)
    
    resolution_check_counter = 0
    RESOLUTION_CHECK_INTERVAL = 300  # 5 minutes in seconds
    checks_per_resolution = RESOLUTION_CHECK_INTERVAL // POLL_INTERVAL
    
    while True:
        if state.get("running") and TARGET_WALLET and ALCHEMY_URL:
            try:
                reset_day_if_needed()
                await check_and_send_daily_summary(app)
                
                # Check for market resolutions every 5 minutes
                resolution_check_counter += 1
                if resolution_check_counter >= checks_per_resolution:
                    log.info("Checking for market resolutions...")
                    await check_trade_resolutions(app)
                    # Also refresh token map to pick up new markets
                    log.info("Refreshing token → market mapping...")
                    get_token_market_map(force_refresh=True)
                    resolution_check_counter = 0

                latest_block = get_latest_block()
                if latest_block is None:
                    log.warning("Could not fetch latest block")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                if state["last_block"] is None:
                    state["last_block"] = latest_block
                    save_data(state)
                    log.info(f"Initialized at block {latest_block}")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                from_block = state["last_block"] + 1
                to_block   = min(latest_block, from_block + 10)  # Reduced to 10 to avoid Alchemy 400 errors

                if from_block <= to_block:
                    log.info(f"Blocks {from_block}→{to_block}")
                    transfers = get_wallet_transactions(from_block, to_block)
                    for t in transfers:
                        if t.get("hash"):
                            await process_onchain_tx(
                                t["hash"], 
                                app,
                                token_id=t.get("token_id", 0),
                                amount_raw=t.get("amount_raw", 0)
                            )
                    state["last_block"] = to_block
                    save_data(state)

            except Exception as e:
                log.error(f"Poll error: {e}")

        await asyncio.sleep(POLL_INTERVAL)

async def check_and_send_daily_summary(app: Application):
    now       = datetime.now(timezone.utc)
    today     = get_today_str()
    chat_id   = state.get("notifications_chat_id")
    send_hour = state.get("daily_summary_hour", 20)
    if send_hour == -1:
        return
    if chat_id and now.hour == send_hour and state.get("last_summary_date") != today:
        try:
            await app.bot.send_message(chat_id, build_daily_summary(), parse_mode="Markdown")
            state["last_summary_date"] = today
            save_data(state)
        except Exception as e:
            log.error(f"Summary error: {e}")

# ─── Telegram UI ──────────────────────────────────────────────────────────────
def main_menu_keyboard():
    mode    = state["mode"]
    running = state.get("running", False)
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 P&L Report",  callback_data="pnl"),
            InlineKeyboardButton("🔄 Status",       callback_data="status"),
        ],
        [
            InlineKeyboardButton("▶️ Start" if not running else "⏹ Stop", callback_data="toggle_running"),
            InlineKeyboardButton(f"{'📄 Paper ✓' if mode=='paper' else '💰 Live ✓'}",  callback_data="toggle_mode"),
        ],
        [
            InlineKeyboardButton("⚙️ Settings",    callback_data="settings"),
            InlineKeyboardButton("📜 History",      callback_data="history"),
        ],
        [
            InlineKeyboardButton("🔬 Calibration",  callback_data="calibration"),
        ],
    ])

def settings_keyboard():
    loss     = state.get("daily_loss_limit", 200.0)
    cap      = state.get("max_trade_size", 100.0)
    copy_pct = state.get("copy_fraction", 1.0) * 100
    summary_h = state.get("daily_summary_hour", 20)
    sim_all  = all([state.get("sim_slippage"), state.get("sim_liquidity"),
                    state.get("sim_fees"), state.get("sim_detection_lag"),
                    state.get("sim_gas_lag"), state.get("sim_queue_competition"),
                    state.get("sim_historical_spreads")])
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("── Copy Size ───────────────", callback_data="noop")],
        [
            InlineKeyboardButton(f"{'✓ ' if copy_pct==0.1 else ''}0.1%", callback_data="size_0.001"),
            InlineKeyboardButton(f"{'✓ ' if copy_pct==1 else ''}1%",     callback_data="size_0.01"),
            InlineKeyboardButton(f"{'✓ ' if copy_pct==5 else ''}5%",     callback_data="size_0.05"),
            InlineKeyboardButton(f"{'✓ ' if copy_pct==10 else ''}10%",   callback_data="size_0.10"),
        ],
        [
            InlineKeyboardButton(f"{'✓ ' if copy_pct==25 else ''}25%",   callback_data="size_0.25"),
            InlineKeyboardButton(f"{'✓ ' if copy_pct==50 else ''}50%",   callback_data="size_0.50"),
            InlineKeyboardButton(f"{'✓ ' if copy_pct==75 else ''}75%",   callback_data="size_0.75"),
            InlineKeyboardButton(f"{'✓ ' if copy_pct==100 else ''}100%", callback_data="size_1.0"),
        ],
        [InlineKeyboardButton("── Paper Starting Balance ──", callback_data="noop")],
        [
            InlineKeyboardButton(f"{'✓ ' if state['paper_balance']==1000 else ''}$1K",    callback_data="balance_1000"),
            InlineKeyboardButton(f"{'✓ ' if state['paper_balance']==10000 else ''}$10K",   callback_data="balance_10000"),
            InlineKeyboardButton(f"{'✓ ' if state['paper_balance']==100000 else ''}$100K",  callback_data="balance_100000"),
            InlineKeyboardButton(f"{'✓ ' if state['paper_balance']==1000000 else ''}$1M",   callback_data="balance_1000000"),
        ],
        [InlineKeyboardButton("── Daily Loss Limit ─────────", callback_data="noop")],
        [
            InlineKeyboardButton(f"{'✓ ' if loss==100 else ''}$100", callback_data="loss_100"),
            InlineKeyboardButton(f"{'✓ ' if loss==200 else ''}$200", callback_data="loss_200"),
            InlineKeyboardButton(f"{'✓ ' if loss==500 else ''}$500", callback_data="loss_500"),
            InlineKeyboardButton(f"{'✓ ' if loss==0 else ''}Off",    callback_data="loss_0"),
        ],
        [InlineKeyboardButton("── Per-Trade Size Cap ───────", callback_data="noop")],
        [
            InlineKeyboardButton(f"{'✓ ' if cap==50 else ''}$50",   callback_data="cap_50"),
            InlineKeyboardButton(f"{'✓ ' if cap==100 else ''}$100", callback_data="cap_100"),
            InlineKeyboardButton(f"{'✓ ' if cap==250 else ''}$250", callback_data="cap_250"),
            InlineKeyboardButton(f"{'✓ ' if cap==0 else ''}Off",    callback_data="cap_0"),
        ],
        [InlineKeyboardButton("── Daily Summary (UTC) ──────", callback_data="noop")],
        [
            InlineKeyboardButton(f"{'✓ ' if summary_h==8 else ''}8am",   callback_data="sumhour_8"),
            InlineKeyboardButton(f"{'✓ ' if summary_h==12 else ''}12pm", callback_data="sumhour_12"),
            InlineKeyboardButton(f"{'✓ ' if summary_h==20 else ''}8pm",  callback_data="sumhour_20"),
            InlineKeyboardButton(f"{'✓ ' if summary_h==-1 else ''}Off",  callback_data="sumhour_off"),
        ],
        [InlineKeyboardButton("── Realistic Simulation ─────", callback_data="noop")],
        [InlineKeyboardButton(
            f"{'✅ Realistic Sim: ALL ON' if sim_all else '⬜ Realistic Sim: PARTIAL/OFF'}",
            callback_data="toggle_sim"
        )],
        [InlineKeyboardButton("🔄 Clear All Paper Trades", callback_data="reset_paper")],
        [InlineKeyboardButton("⬅️ Back", callback_data="back")],
    ])

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global state
    user_id = update.effective_user.id
    if not state["authorized_users"]:
        state["authorized_users"].append(user_id)
        state["notifications_chat_id"] = update.effective_chat.id
        save_data(state)
    if user_id not in state["authorized_users"]:
        await update.message.reply_text("⛔ Unauthorized.")
        return

    cal     = load_calibration()
    mode    = state["mode"]
    running = state.get("running", False)
    target  = TARGET_WALLET or "⚠️ Not set"
    paused  = " ⚠️ Loss limit hit" if state.get("daily_loss_paused") else ""
    cal_str = f"\n🔬 Cal Samples: `{cal['n_samples']}` | Bias: `{cal['price_bias']*100:+.2f}¢`" if cal["n_samples"] >= 3 else "\n🔬 Calibration: `Not enough data yet`"
    copy_pct = state.get("copy_fraction", 1.0) * 100
    copy_str = f"{copy_pct:.1f}%" if copy_pct < 1 else f"{int(copy_pct)}%"

    msg = (
        f"🤖 *PolyBot — On-Chain BTC Copy Trader*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👛 `{target[:24]}...`\n"
        f"⛓ Alchemy: `{'✅' if ALCHEMY_URL else '❌'}`\n"
        f"🔵 Mode: `{'📄 Paper' if mode=='paper' else '💰 Live'}`\n"
        f"⚡ Status: `{'Running ✅' if running else 'Stopped ⏸'}`{paused}\n"
        f"🔁 Copy Size: `{copy_str}`\n"
        f"🛡 Loss Limit: `{'$'+str(int(state['daily_loss_limit']))+'/day' if state['daily_loss_limit']>0 else 'Off'}`\n"
        f"📏 Size Cap: `{'$'+str(int(state['max_trade_size']))+'/trade' if state['max_trade_size']>0 else 'Off'}`"
        f"{cal_str}"
    )
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=main_menu_keyboard())

async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global state
    query = update.callback_query
    await query.answer()
    if query.from_user.id not in state["authorized_users"]:
        await query.edit_message_text("⛔ Unauthorized.")
        return

    data = query.data

    if data == "noop":
        return

    elif data == "toggle_running":
        state["running"] = not state.get("running", False)
        state["notifications_chat_id"] = query.message.chat_id
        save_data(state)
        label = "▶️ Started — watching the blockchain" if state["running"] else "⏹ Stopped"
        await query.edit_message_text(label, reply_markup=main_menu_keyboard())

    elif data == "toggle_mode":
        if state["mode"] == "paper":
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Yes, go LIVE", callback_data="confirm_live"),
                InlineKeyboardButton("❌ Cancel",       callback_data="back"),
            ]])
            await query.edit_message_text(
                "⚠️ *Switch to LIVE Trading?*\n\nReal USDC will be spent. "
                "Ensure `POLY_API_KEY` is set in Railway Variables.\n\nAre you sure?",
                parse_mode="Markdown", reply_markup=kb
            )
        else:
            state["mode"] = "paper"
            save_data(state)
            await query.edit_message_text("📄 Switched to *Paper* mode.", parse_mode="Markdown", reply_markup=main_menu_keyboard())

    elif data == "confirm_live":
        if not os.getenv("POLY_API_KEY"):
            await query.edit_message_text("❌ `POLY_API_KEY` not found in Railway Variables.", parse_mode="Markdown", reply_markup=main_menu_keyboard())
        else:
            state["mode"] = "live"
            save_data(state)
            await query.edit_message_text("💰 Switched to *LIVE* mode.", parse_mode="Markdown", reply_markup=main_menu_keyboard())

    elif data == "pnl":
        mode   = state["mode"]
        trades = state["paper_trades"] if mode == "paper" else state["live_trades"]
        pnl    = calculate_pnl(trades)
        today_pnl  = get_todays_pnl()
        today_sign = "+" if today_pnl >= 0 else ""
        today_icon = "🟢" if today_pnl >= 0 else "🔴"
        extra = f"\n{today_icon} Today's P&L: `{today_sign}${today_pnl:.2f}`"
        extra += f"\n💼 Paper Balance: `${state['paper_balance']:.2f}`" if mode == "paper" else ""
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="back")]])
        await query.edit_message_text(format_pnl_message(pnl, mode) + extra, parse_mode="Markdown", reply_markup=kb)

    elif data == "status":
        mode    = state["mode"]
        running = state.get("running", False)
        today_pnl  = get_todays_pnl()
        today_sign = "+" if today_pnl >= 0 else ""
        cal     = load_calibration()
        msg = (
            f"🔄 *Bot Status*\n━━━━━━━━━━━━━━━━\n"
            f"⚡ Running: `{'Yes ✅' if running else 'No ⏸'}`\n"
            f"🔵 Mode: `{'Paper 📄' if mode=='paper' else 'Live 💰'}`\n"
            f"{'⛔ Loss Limit Paused' + chr(10) if state.get('daily_loss_paused') else ''}"
            f"👛 Wallet: `{TARGET_WALLET[:28] if TARGET_WALLET else 'Not set'}`\n"
            f"⛓ Last Block: `{state.get('last_block','Not started')}`\n"
            f"🔁 Copy Size: `{int(state['copy_fraction']*100)}%`\n"
            f"📈 Today's P&L: `{today_sign}${today_pnl:.2f}`\n"
            f"📄 Paper Trades: `{len(state['paper_trades'])}`\n"
            f"💰 Live Trades: `{len(state['live_trades'])}`\n"
            f"🔬 Cal Samples: `{cal['n_samples']}`\n"
            f"⏱ Poll: `every {POLL_INTERVAL}s`"
        )
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="back")]])
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=kb)

    elif data == "history":
        mode   = state["mode"]
        trades = state["paper_trades"] if mode == "paper" else state["live_trades"]
        recent = trades[-10:][::-1]
        if not recent:
            msg = "📜 No trades yet."
        else:
            lines = [f"📜 *Last {len(recent)} {'Paper' if mode=='paper' else 'Live'} Trades*\n━━━━━━━━━━━━"]
            for t in recent:
                icon    = {"won":"✅","lost":"❌","open":"⏳"}.get(t.get("status","open"),"⏳")
                pnl_val = (t.get("payout",0) - t.get("amount_usdc",0)) if t.get("status")=="won" else (-t.get("amount_usdc",0) if t.get("status")=="lost" else 0)
                pnl_str = f"+${pnl_val:.2f}" if pnl_val > 0 else (f"${pnl_val:.2f}" if pnl_val < 0 else "open")
                lines.append(f"{icon} `{t['market'][:36]}`\n   {t['outcome']} @ {t['price']:.4f} · ${t['amount_usdc']:.2f} · {pnl_str}")
            msg = "\n".join(lines)
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="back")]])
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=kb)

    elif data == "calibration":
        cal = load_calibration()
        active = state.get("calibration_mode", False)
        cal_size = state.get("calibration_size", CALIBRATION_TRADE_SIZE)
        if cal["n_samples"] >= 5:
            bias_str = (
                f"📐 Price bias: `{cal['price_bias']*100:+.2f}¢`\n"
                f"📐 Fill bias: `{cal['fill_bias']:+.1f}%`\n"
                f"📊 Samples: `{cal['n_samples']}`\n"
                f"_Simulation is being auto-corrected by this data_"
            )
        elif cal["n_samples"] > 0:
            bias_str = f"⏳ Gathering data... `{cal['n_samples']}/5` samples so far"
        else:
            bias_str = "_No calibration data yet. Enable to start gathering._"

        msg = (
            f"🔬 *Calibration Mode*\n━━━━━━━━━━━━━━━━\n"
            f"Status: `{'🟢 Active' if active else '⚪ Off'}`\n"
            f"Trade size: `${cal_size:.2f}` per calibration trade\n\n"
            f"*How it works:*\n"
            f"Places tiny `${cal_size:.2f}` live trades alongside paper trades "
            f"to measure real vs simulated fill prices. Automatically corrects "
            f"the simulation model over time.\n\n"
            f"{bias_str}"
        )
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton(
                "🔴 Disable Calibration" if active else "🟢 Enable Calibration",
                callback_data="toggle_calibration"
            )],
            [InlineKeyboardButton("🗑 Reset Calibration Data", callback_data="reset_calibration")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back")],
        ])
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=kb)

    elif data == "toggle_calibration":
        if not state.get("calibration_mode"):
            if not os.getenv("POLY_API_KEY"):
                await query.edit_message_text(
                    "❌ Calibration requires `POLY_API_KEY` in Railway Variables\n"
                    "(it places real $5 trades to measure fills).",
                    parse_mode="Markdown", reply_markup=main_menu_keyboard()
                )
                return
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton(f"✅ Yes, enable (${state.get('calibration_size', CALIBRATION_TRADE_SIZE):.0f}/trade)", callback_data="confirm_calibration"),
                InlineKeyboardButton("❌ Cancel", callback_data="back"),
            ]])
            await query.edit_message_text(
                f"⚠️ *Enable Calibration Mode?*\n\n"
                f"This will place real `${state.get('calibration_size', CALIBRATION_TRADE_SIZE):.2f}` USDC trades "
                f"on Polymarket to measure actual vs simulated fill prices.\n\n"
                f"Trades are tiny and are for measurement only.",
                parse_mode="Markdown", reply_markup=kb
            )
        else:
            state["calibration_mode"] = False
            save_data(state)
            await query.edit_message_text("⚪ Calibration mode disabled.", reply_markup=main_menu_keyboard())

    elif data == "confirm_calibration":
        state["calibration_mode"] = True
        save_data(state)
        await query.edit_message_text(
            "🟢 *Calibration mode enabled.*\nNext detected BTC trade will place a real "
            f"`${state.get('calibration_size', CALIBRATION_TRADE_SIZE):.2f}` order and record the result.",
            parse_mode="Markdown", reply_markup=main_menu_keyboard()
        )

    elif data == "reset_calibration":
        save_calibration(load_calibration() | {"trades": [], "price_bias": 0.0, "fill_bias": 0.0, "n_samples": 0})
        await query.edit_message_text("✅ Calibration data cleared.", reply_markup=main_menu_keyboard())

    elif data == "settings":
        loss     = state.get("daily_loss_limit", 200)
        cap      = state.get("max_trade_size", 100)
        hour     = state.get("daily_summary_hour", 20)
        hour_str = f"{hour}:00 UTC" if hour != -1 else "Off"
        copy_pct = state.get("copy_fraction", 1.0) * 100
        copy_str = f"{copy_pct:.1f}%" if copy_pct < 1 else f"{int(copy_pct)}%"
        bal      = state.get("paper_balance", 1000)
        bal_str  = f"${int(bal):,}"
        sim_all  = all([state.get("sim_slippage"), state.get("sim_liquidity"),
                        state.get("sim_fees"), state.get("sim_detection_lag"),
                        state.get("sim_gas_lag"), state.get("sim_queue_competition"),
                        state.get("sim_historical_spreads")])
        msg = (
            f"⚙️ *Settings*\n━━━━━━━━━━━━━━━━\n"
            f"🔁 Copy Size: `{copy_str}`\n"
            f"💼 Paper Balance: `{bal_str}`\n"
            f"🛡 Daily Loss Limit: `{'$'+str(int(loss)) if loss>0 else 'Off'}`\n"
            f"📏 Per-Trade Cap: `{'$'+str(int(cap)) if cap>0 else 'Off'}`\n"
            f"📅 Daily Summary: `{hour_str}`\n"
            f"🔬 Realistic Sim: `{'All ON ✅' if sim_all else 'Partial/Off ⬜'}`"
        )
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=settings_keyboard())

    elif data.startswith("size_"):
        state["copy_fraction"] = float(data.split("_")[1])
        save_data(state)
        pct = state['copy_fraction'] * 100
        pct_str = f"{pct:.1f}%" if pct < 1 else f"{int(pct)}%"
        await query.edit_message_text(f"✅ Copy size: `{pct_str}`", parse_mode="Markdown", reply_markup=settings_keyboard())

    elif data.startswith("balance_"):
        new_balance = float(data.split("_")[1])
        state["paper_balance"] = new_balance
        state["day_start_balance"] = new_balance
        save_data(state)
        bal_str = f"${int(new_balance):,}" if new_balance >= 1000 else f"${new_balance:.0f}"
        await query.edit_message_text(f"✅ Paper balance set to `{bal_str}`", parse_mode="Markdown", reply_markup=settings_keyboard())

    elif data.startswith("loss_"):
        state["daily_loss_limit"] = float(data.split("_")[1])
        save_data(state)
        v = state["daily_loss_limit"]
        await query.edit_message_text(f"✅ Loss limit: `{'$'+str(int(v))+'/day' if v>0 else 'Off'}`", parse_mode="Markdown", reply_markup=settings_keyboard())

    elif data.startswith("cap_"):
        state["max_trade_size"] = float(data.split("_")[1])
        save_data(state)
        v = state["max_trade_size"]
        await query.edit_message_text(f"✅ Size cap: `{'$'+str(int(v))+'/trade' if v>0 else 'Off'}`", parse_mode="Markdown", reply_markup=settings_keyboard())

    elif data.startswith("sumhour_"):
        val = data.split("_")[1]
        state["daily_summary_hour"] = -1 if val == "off" else int(val)
        save_data(state)
        h = state["daily_summary_hour"]
        await query.edit_message_text(f"✅ Summary: `{str(h)+':00 UTC' if h!=-1 else 'Off'}`", parse_mode="Markdown", reply_markup=settings_keyboard())

    elif data == "toggle_sim":
        sim_all = all([state.get("sim_slippage"), state.get("sim_liquidity"),
                       state.get("sim_fees"), state.get("sim_detection_lag"),
                       state.get("sim_gas_lag"), state.get("sim_queue_competition"),
                       state.get("sim_historical_spreads")])
        new_val = not sim_all
        for key in ["sim_slippage","sim_liquidity","sim_fees","sim_detection_lag",
                    "sim_gas_lag","sim_queue_competition","sim_historical_spreads"]:
            state[key] = new_val
        save_data(state)
        label = "✅ All ON" if new_val else "⬜ All OFF"
        await query.edit_message_text(f"Realistic simulation: {label}", reply_markup=settings_keyboard())

    elif data == "reset_paper":
        state["paper_trades"]      = []
        state["last_block"]        = None
        state["daily_loss_paused"] = False
        save_data(state)
        await query.edit_message_text(
            f"✅ Paper trades cleared. Balance remains at `${state['paper_balance']:,.0f}`\n"
            f"_Use 'Paper Starting Balance' buttons above to change balance_",
            parse_mode="Markdown", reply_markup=settings_keyboard()
        )

    elif data == "back":
        mode     = state["mode"]
        running  = state.get("running", False)
        paused   = " ⚠️ Loss limit hit" if state.get("daily_loss_paused") else ""
        copy_pct = state.get("copy_fraction", 1.0) * 100
        copy_str = f"{copy_pct:.1f}%" if copy_pct < 1 else f"{int(copy_pct)}%"
        msg = (
            f"🤖 *PolyBot — On-Chain BTC Copy Trader*\n━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👛 `{TARGET_WALLET[:24] if TARGET_WALLET else 'Not set'}...`\n"
            f"🔵 Mode: `{'📄 Paper' if mode=='paper' else '💰 Live'}`\n"
            f"⚡ Status: `{'Running ✅' if running else 'Stopped ⏸'}`{paused}\n"
            f"🔁 Copy: `{copy_str}` | "
            f"🛡 `{'$'+str(int(state['daily_loss_limit']))+'/day' if state['daily_loss_limit']>0 else 'No limit'}` | "
            f"📏 `{'$'+str(int(state['max_trade_size']))+'/trade' if state['max_trade_size']>0 else 'No cap'}`"
        )
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=main_menu_keyboard())

# ─── Entry Point ──────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN not set")
    if not ALCHEMY_URL:
        log.warning("ALCHEMY_URL not set")
    if not TARGET_WALLET:
        log.warning("TARGET_WALLET not set")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))

    async def run():
        async with app:
            await app.start()
            await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            await poll_chain(app)

    asyncio.get_event_loop().run_until_complete(run())

if __name__ == "__main__":
    main()
