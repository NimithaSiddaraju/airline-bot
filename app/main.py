# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import httpx
import math, re
from typing import Tuple, Optional, List

# ============================================================
# Remote reference data (no local files)
# ============================================================
AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
COLS = ["id","name","city","country","IATA","ICAO","lat","lon","alt_ft","tz_offset","dst","tzdb","type","source"]
airports = pd.read_csv(AIRPORTS_URL, header=None, names=COLS)

# Lightweight preprocess for matching
airports["city_l"] = airports["city"].fillna("").str.lower()
airports["name_l"] = airports["name"].fillna("").str.lower()
iata_set = set(airports["IATA"].dropna().astype(str))

# ============================================================
# Helpers
# ============================================================

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def detect_iata_tokens(user_msg: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z]{3}\b", user_msg)
    hits = [t.upper() if t.isupper() else None for t in tokens]
    hits = [h for h in hits if h and h in iata_set]

    if not hits and len(user_msg.strip()) == 3:
        t = user_msg.strip().upper()
        if t in iata_set:
            hits = [t]
    return hits

# --- FAQs ---
def get_tsa_liquids_summary():
    url = "https://www.tsa.gov/travel/security-screening/whatcanibring/items/travel-size-toiletries"
    summary = ("TSA liquids rule (3-1-1): containers ≤ 3.4 oz / 100 mL; "
               "all containers fit in one quart-size transparent bag; one bag per passenger; "
               "place in bin for screening. Larger volumes → checked bag.")
    return summary, url

def get_faa_powerbank_summary():
    url = "https://www.faa.gov/hazmat/packsafe/lithium-batteries"
    summary = ("Power banks (lithium batteries): carry-on only (no checked). "
               "≤100 Wh allowed without airline approval; 100–160 Wh requires airline approval; "
               "protect terminals from short circuit.")
    return summary, url

# --- Airline baggage links ---
AIRLINE_LINKS = {
    "american": "https://www.aa.com/i18n/travel-info/baggage/baggage.jsp",
    "aa":       "https://www.aa.com/i18n/travel-info/baggage/baggage.jsp",
    "delta":    "https://www.delta.com/traveling-with-us/baggage",
    "dl":       "https://www.delta.com/traveling-with-us/baggage",
    "united":   "https://www.united.com/en/us/fly/travel/baggage.html",
    "ua":       "https://www.united.com/en/us/fly/travel/baggage.html",
    "southwest":"https://www.southwest.com/help/baggage",
    "wn":       "https://www.southwest.com/help/baggage",
    "alaska":   "https://www.alaskaair.com/travel-info/baggage/overview",
    "as":       "https://www.alaskaair.com/travel-info/baggage/overview",
    "jetblue":  "https://www.jetblue.com/help/baggage",
    "b6":       "https://www.jetblue.com/help/baggage",
    # Intl
    "air canada": "https://www.aircanada.com/ca/en/aco/home/plan/baggage.html",
    "ac":         "https://www.aircanada.com/ca/en/aco/home/plan/baggage.html",
    "british airways": "https://www.britishairways.com/en-us/information/baggage-essentials",
    "ba":              "https://www.britishairways.com/en-us/information/baggage-essentials",
    "lufthansa": "https://www.lufthansa.com/us/en/baggage-overview",
    "lh":        "https://www.lufthansa.com/us/en/baggage-overview",
    "emirates":  "https://www.emirates.com/us/english/before-you-fly/baggage/",
    "emirates airlines": "https://www.emirates.com/us/english/before-you-fly/baggage/",
    "ek":        "https://www.emirates.com/us/english/before-you-fly/baggage/",
    "qatar":     "https://www.qatarairways.com/en-us/baggage/allowance.html",
    "qatar airways": "https://www.qatarairways.com/en-us/baggage/allowance.html",
    "qr":        "https://www.qatarairways.com/en-us/baggage/allowance.html",
    "singapore": "https://www.singaporeair.com/en_UK/us/travel-info/baggage/",
    "sq":        "https://www.singaporeair.com/en_UK/us/travel-info/baggage/",
}

ALIAS_TO_NAME = {
    "aa":"American Airlines","dl":"Delta Air Lines","ua":"United Airlines",
    "wn":"Southwest Airlines","as":"Alaska Airlines","b6":"JetBlue",
    "ac":"Air Canada","ba":"British Airways","lh":"Lufthansa",
    "ek":"Emirates","qr":"Qatar Airways","sq":"Singapore Airlines"
}

BAGGAGE_KEYWORDS = [
    "baggage","bags","luggage","checked bag","checked bags","carry-on",
    "carry on","carryon","baggage allowance","bag fee","bag fees","allowance",
    "how many bags","how much luggage","how much baggage"
]

def get_airline_baggage_link(text: str):
    t = normalize(text)

    # Multi-word checks first
    for key in ["air canada", "british airways", "emirates airlines", "qatar airways"]:
        if key in t:
            return key.title(), AIRLINE_LINKS[key]

    # Single-word / aliases
    for key, url in AIRLINE_LINKS.items():
        if re.search(rf"\b{re.escape(key)}\b", t):
            return ALIAS_TO_NAME.get(key, key.title()), url

    return None, None

# --- Power bank calculator ---
def powerbank_wh_from_text(text: str) -> Optional[Tuple[float, float]]:
    t = text.lower().replace(",", " ")
    m_wh = re.search(r"(\d+(\.\d+)?)\s*wh\b", t)
    if m_wh:
        return float(m_wh.group(1)), None
    m_mah = re.search(r"(\d+(\.\d+)?)\s*mah\b", t)
    if m_mah:
        mah = float(m_mah.group(1))
        m_v = re.search(r"(\d+(\.\d+)?)\s*v\b", t)
        v = float(m_v) if m_v else 3.7
        wh = (mah/1000.0) * v
        return wh, v
    return None

def classify_wh(wh: float):
    if wh <= 100:
        return "Allowed in carry-on without airline approval (no checked baggage)."
    elif wh <= 160:
        return "Carry-on allowed with airline approval (no checked baggage)."
    else:
        return "Not allowed for passenger aircraft (exceeds 160 Wh)."

# ============================================================
# AviationStack Query
# ============================================================
def query_aviationstack(iata_code: str, direction: str = "from"):
    url = "http://api.aviationstack.com/v1/flights"
    params = {
        "access_key": "5c405fe0aa56286d8e7698ff945dff76",  # your key
        "limit": 5,
    }
    if direction == "from":
        params["dep_iata"] = iata_code
    else:
        params["arr_iata"] = iata_code

    try:
        resp = httpx.get(url, params=params, timeout=30.0)
        if resp.status_code != 200:
            return None, f"AviationStack status {resp.status_code}: {resp.text}"

        data = resp.json().get("data", [])
        flights = []
        for f in data:
            airline = f.get("airline", {}).get("name", "Unknown")
            flight_no = f.get("flight", {}).get("iata", "N/A")
            dep = f.get("departure", {}).get("iata", "???")
            arr = f.get("arrival", {}).get("iata", "???")
            status = f.get("flight_status", "unknown")
            flights.append(f"{airline} {flight_no} {dep}→{arr} — {status}")
        return flights, None

    except Exception as e:
        return None, f"AviationStack error: {e}"

# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="Airline Chatbot", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat(q: Query):
    user_msg = q.message.strip()
    user_l   = normalize(user_msg)

    # 1) FAQs
    if any(k in user_l for k in ["liquid","toiletries","3-1-1","3 1 1","100ml","100 ml"]):
        info, src = get_tsa_liquids_summary()
        return {"answer": info, "source": src}

    if any(k in user_l for k in ["power bank","powerbank","battery","lithium","mah","wh"]):
        calc = powerbank_wh_from_text(user_l)
        if calc:
            wh, v = calc
            verdict = classify_wh(wh)
            v_txt = "" if v is None else f" using {v} V,"
            return {"answer": f"Estimated capacity ≈ {wh:.1f} Wh{v_txt} which falls under: {verdict}",
                    "source": "https://www.faa.gov/hazmat/packsafe/lithium-batteries"}
        info, src = get_faa_powerbank_summary()
        return {"answer": info, "source": src}

    if any(k in user_l for k in BAGGAGE_KEYWORDS):
        name, link = get_airline_baggage_link(user_l)
        if link:
            return {"answer": f"Here’s the official baggage policy for {name}:", "source": link}
        return {"answer": "Tell me the airline (e.g., 'United baggage', 'AA baggage allowance', 'Emirates baggage')."}

    # 2) Flights
    if "flight" in user_l or "flights" in user_l:
        codes = detect_iata_tokens(user_msg)
        if codes:
            code = codes[0]
            if "to" in user_l:
                flights, err = query_aviationstack(code, "to")
                direction = f"to {code}"
            else:
                flights, err = query_aviationstack(code, "from")
                direction = f"from {code}"
            if err:
                return {"answer": f"Could not fetch flight data ({err})."}
            if flights:
                return {"answer": f"Found {len(flights)} flights {direction}. Examples:\n" + "\n".join(flights)}
            return {"answer": f"No flights found {direction} right now."}

    # 3) Airport lookup
    codes = detect_iata_tokens(user_msg)
    for code in codes:
        r = airports.loc[airports["IATA"] == code, ["name","city","country","IATA","ICAO"]]
        if not r.empty:
            row = r.iloc[0]
            return {"answer": f"{row['IATA']} = {row['name']} in {row['city']}, {row['country']} (ICAO {row['ICAO']})."}

    by_city = airports.loc[airports["city_l"].str.contains(user_l, na=False), ["name","city","country","IATA","ICAO"]]
    if not by_city.empty:
        row = by_city.iloc[0]
        return {"answer": f"Airport in {row['city']}: {row['name']} (IATA {row['IATA']}, ICAO {row['ICAO']})."}

    by_name = airports.loc[airports["name_l"].str.contains(user_l, na=False), ["name","city","country","IATA","ICAO"]]
    if not by_name.empty:
        row = by_name.iloc[0]
        return {"answer": f"{row['name']} is in {row['city']}, {row['country']} (IATA {row['IATA']}, ICAO {row['ICAO']})."}

    # Fallback
    return {
        "answer": ("I can help with:\n"
                   "• Airline baggage links (e.g., 'United baggage', 'AA carry on size', 'Emirates baggage')\n"
                   "• TSA liquids & FAA power banks (e.g., 'what's the liquids rule', 'is 20000 mAh allowed')\n"
                   "• Live flights by IATA code (e.g., 'Flights from LAX', 'Flights to DFW')\n"
                   "• Airport info by code/name/city (e.g., 'DFW', 'airport in Tokyo').")
    }
