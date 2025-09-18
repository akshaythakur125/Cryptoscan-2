#!/usr/bin/env python3
# scanner_github.py
"""
Crypto intraday scanner (CoinGecko + optional CCXT funding check)
- Filters coins by market cap rank (default 40-100)
- Looks for 1-hour moves, 24h confirmation, and higher-than-normal volume
- Sends an email if candidates are found
"""

import os
import time
import math
import json
import smtplib
import logging
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
import pandas as pd
import numpy as np

# optional CCXT import; script will continue if CCXT not available
try:
    import ccxt
    CCXT_AVAILABLE = True
except Exception:
    CCXT_AVAILABLE = False

# -----------------------
# Config (via env / defaults)
# -----------------------
COINGECKO_API = "https://api.coingecko.com/api/v3"
RANK_MIN = int(os.getenv("RANK_MIN", "40"))
RANK_MAX = int(os.getenv("RANK_MAX", "100"))
TOP_N = int(os.getenv("TOP_N", "250"))  # fetch this many from coinGecko (>= RANK_MAX)
API_RATE_LIMIT_SECONDS = float(os.getenv("API_RATE_LIMIT_SECONDS", "0.6"))

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")
SENDER_NAME = os.getenv("SENDER_NAME", SMTP_USER)

# threshold tuning
MIN_1H_PCT = float(os.getenv("MIN_1H_PCT", "2.0"))     # e.g. at least +2% in 1h
MIN_24H_PCT = float(os.getenv("MIN_24H_PCT", "3.0"))   # e.g. at least +3% in 24h
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.4"))  # volume > median*factor

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------
# Helpers
# -----------------------
def rate_limit_sleep():
    if API_RATE_LIMIT_SECONDS > 0:
        time.sleep(API_RATE_LIMIT_SECONDS)

def fetch_coingecko_markets(vs_currency="usd", per_page=250, page=1, price_change_pct="1h,24h"):
    """Fetch /coins/markets from CoinGecko."""
    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "price_change_percentage": price_change_pct,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    rate_limit_sleep()
    return resp.json()

def try_fetch_funding_rate(ccxt_exchange, symbol):
    """Try to fetch funding rate via ccxt if available; return None on fail."""
    if not CCXT_AVAILABLE:
        return None
    try:
        ex = getattr(ccxt, ccxt_exchange)()
        # many exchanges require symbol as e.g. "BTC/USDT:USDT" or "BTC/USDT"
        # ccxt has no unified method across all exchanges for funding rates; try safe call
        if hasattr(ex, "fetch_funding_rate"):
            return ex.fetch_funding_rate(symbol)
        # fallback: some exchanges have fetchFundingRate or fetchFundingRates
        if hasattr(ex, "fetchFundingRate"):
            return ex.fetchFundingRate(symbol)
        if hasattr(ex, "fetchFundingRates"):
            rates = ex.fetchFundingRates([symbol])
            # find rate for symbol
            for r in rates:
                if r.get("symbol") == symbol:
                    return r
            return None
    except Exception as e:
        logging.debug("Funding fetch error: %s", e)
        return None

def send_email(subject, html_body):
    """Send an email using SMTP. Requires SMTP_USER and SMTP_PASSWORD to be set."""
    if not SMTP_USER or not SMTP_PASSWORD or not EMAIL_RECIPIENT:
        logging.warning("SMTP or recipient not configured. Skipping email send.")
        return False, "SMTP/recipient not configured"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{SENDER_NAME} <{SMTP_USER}>"
    msg["To"] = EMAIL_RECIPIENT

    part_html = MIMEText(html_body, "html")
    msg.attach(part_html)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
        server.ehlo()
        if SMTP_PORT == 587:
            server.starttls()
            server.ehlo()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [EMAIL_RECIPIENT], msg.as_string())
        server.quit()
        logging.info("Email sent to %s", EMAIL_RECIPIENT)
        return True, "ok"
    except Exception as e:
        logging.exception("Failed to send email")
        return False, str(e)

# -----------------------
# Main scanning logic
# -----------------------
def run_scan():
    try:
        logging.info("Starting scan: rank range %d-%d", RANK_MIN, RANK_MAX)
        markets = fetch_coingecko_markets(per_page=TOP_N, price_change_pct="1h,24h")
        if not isinstance(markets, list) or len(markets) == 0:
            logging.error("No market data from CoinGecko")
            return

        # Convert to DataFrame for easier computations
        df = pd.DataFrame(markets)
        # Ensure rank and required columns exist
        if "market_cap_rank" not in df.columns:
            logging.error("CoinGecko data missing market_cap_rank")
            return

        # Filter by rank
        mask = df["market_cap_rank"].between(RANK_MIN, RANK_MAX)
        sel = df[mask].copy()
        if sel.empty:
            logging.info("No coins in the requested rank range (%d - %d).", RANK_MIN, RANK_MAX)
            return

        # Clean up some columns and compute signals
        # CoinGecko may provide price_change_percentage_1h_in_currency and price_change_percentage_24h_in_currency
        sel["price_change_1h"] = sel.get("price_change_percentage_1h_in_currency", sel.get("price_change_percentage_24h", np.nan))
        # fallback: coin might not have 1h; try to pull from price_change_percentage_24h_in_currency if 1h absent
        if "price_change_percentage_24h_in_currency" in sel.columns and "price_change_percentage_1h_in_currency" not in sel.columns:
            sel["price_change_1h"] = np.nan

        sel["price_change_24h"] = sel.get("price_change_percentage_24h_in_currency", sel.get("price_change_24h", np.nan))

        # volume handling
        sel["total_volume"] = sel.get("total_volume", 0)
        median_vol = sel["total_volume"].median() if not sel["total_volume"].isna().all() else 0

        # build candidate list
        candidates = []
        for _, row in sel.iterrows():
            try:
                coin_id = row.get("id")
                symbol = row.get("symbol", "").upper()
                name = row.get("name", coin_id)
                rank = int(row.get("market_cap_rank", -1))
                price = row.get("current_price", float("nan"))
                vol = float(row.get("total_volume") or 0)
                change_1h = float(row.get("price_change_1h") or 0)
                change_24h = float(row.get("price_change_24h") or 0)

                # Basic signal: 1h move above threshold AND 24h confirms (same direction) AND volume above median*factor
                signal = False
                reasons = []
                if change_1h >= MIN_1H_PCT:
                    reasons.append(f"1h {change_1h:.2f}% ≥ {MIN_1H_PCT}%")
                if change_24h >= MIN_24H_PCT:
                    reasons.append(f"24h {change_24h:.2f}% ≥ {MIN_24H_PCT}%")
                if median_vol > 0 and vol >= median_vol * VOLUME_MULTIPLIER:
                    reasons.append(f"vol {vol:.0f} ≥ median*{VOLUME_MULTIPLIER:.2f}")
                # require at least two of the three conditions to avoid too many false positives
                if len(reasons) >= 2:
                    signal = True

                funding_info = None
                # try optional funding rate: map coin to a futures symbol (best-effort)
                if CCXT_AVAILABLE:
                    # try commonly used perpetual symbol for Binance: e.g. "BTC/USDT:USDT" or "BTC/USDT"
                    try_symbols = []
                    if symbol:
                        try_symbols = [f"{symbol}/USDT", f"{symbol}/USD", f"{symbol}/USDT:USDT"]
                    funding_info = None
                    for s in try_symbols:
                        fr = try_fetch_funding_rate("binance", s)
                        if fr:
                            funding_info = fr
                            break

                if signal:
                    candidates.append({
                        "id": coin_id, "name": name, "symbol": symbol, "rank": rank,
                        "price": price, "vol": vol,
                        "1h": change_1h, "24h": change_24h,
                        "reasons": reasons, "funding": funding_info
                    })
            except Exception:
                logging.exception("Error processing coin row")
                continue

        # Compose result
        if not candidates:
            logging.info("No candidates found this run.")
            return

        # Create simple HTML report
        html = "<h2>Crypto Scanner — Candidates</h2>"
        html += f"<p>Rank range: {RANK_MIN}-{RANK_MAX}. Found {len(candidates)} candidates.</p>"
        html += "<table border='1' cellpadding='6' cellspacing='0'>"
        html += "<tr><th>Rank</th><th>Name (symbol)</th><th>Price</th><th>1h%</th><th>24h%</th><th>Volume</th><th>Reasons</th><th>Funding (if any)</th></tr>"
        for c in sorted(candidates, key=lambda x: x["rank"]):
            fr = c["funding"]
            fr_str = ""
            if isinstance(fr, dict):
                # format commonly available funding info
                fr_rate = fr.get("fundingRate") or fr.get("rate") or fr.get("funding_rate")
                fr_time = fr.get("timestamp")
                if fr_rate is not None:
                    fr_str = f"{fr_rate}"
                else:
                    fr_str = json.dumps(fr)[:200]
            elif fr:
                fr_str = str(fr)
            html += ("<tr>"
                     f"<td>{c['rank']}</td>"
                     f"<td>{c['name']} ({c['symbol']})</td>"
                     f"<td>{c['price']}</td>"
                     f"<td>{c['1h']:.2f}%</td>"
                     f"<td>{c['24h']:.2f}%</td>"
                     f"<td>{int(c['vol']):,}</td>"
                     f"<td>{'; '.join(c['reasons'])}</td>"
                     f"<td>{fr_str}</td>"
                     "</tr>")
        html += "</table>"

        subject = f"[Scanner] {len(candidates)} candidate(s) found"
        ok, info = send_email(subject, html)
        if not ok:
            logging.error("Email failed: %s", info)
        else:
            logging.info("Report emailed successfully.")
    except Exception as e:
        logging.exception("Scan aborted due to exception")
        # send failure email (best effort)
        try:
            body = f"<p>Scanner crashed with exception:</p><pre>{traceback.format_exc()}</pre>"
            send_email("[Scanner] ERROR", body)
        except Exception:
            logging.exception("Also failed to send crash email")

if __name__ == "__main__":
    run_scan()
