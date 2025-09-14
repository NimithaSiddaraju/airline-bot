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
    """Return 3-letter IATA codes typed by user"""
    tokens = re.findall(r"\b[a-zA-Z]{3}\b", user_msg)
    hits = [t.upper() if t.isupper() else None for t in tokens]
    hits = [h for h in hits if h and h in iata_set]
    if not hits and len(user_msg.strip()) == 3:
        t = user_msg.strip().upper()
        if t in iata_set:
            hits = [t]
    return hits

# ============================================================
# AviationStack API wrapper
# ============================================================
AVIATIONSTACK_KEY = "5c405fe0aa56286d8e7698ff945dff76"

def query_aviationstack(dep: str = None, arr: str = None):
    url = "http://api.aviationstack.com/v1/flights"
    params = {"access_key": AVIATIONSTACK_KEY}
    if dep:
        params["dep_iata"] = dep
    if arr:
        params["arr_iata"] = arr
    try:
        resp = httpx.get(url, params=params, timeout=30.0)
        if resp.status_code != 200:
            return None, f"AviationStack status {resp.status_code}"
        data = resp.json()
        if "error" in data:
            return None, f"AviationStack error {data['error']}"
        return data.get("data", []), None
    except Exception as e:
        return None, f"AviationStack error: {e}"

# ============================================================
# FAQs
# ============================================================
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

# Airline baggage links
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
    "air canada": "https://www.aircanada.com/ca/en/aco/home/plan/baggage.html",
    "ac":         "https://www.aircanada.com/ca/en/aco/home/plan/baggage.html",
    "british airways": "https://www.britishairways.com/en-us/information/baggage-essentials",
    "ba":              "https://www.britishairways.com/en-us/information/baggage-essentials",
    "lufthansa": "https://www.lufthansa.com/us/en/baggage-overview",
    "lh":        "https://www.lufthansa.com/us/en/baggage-overview",
    "emirates":  "https://www.emirates.com/us/english/before-you-fly/baggage/",
    "ek":        "https://www.emirates.com/us/english/before-you-fly/baggage/",
    "qatar":     "https://www.qatarairways.com/en-us/baggage/allowance.html",
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

def get_airline_baggage_link(text: str):
    t = normalize(text)
    for key in ["air canada","british airways"]:
        if key in t:
            return key.title(), AIRLINE_LINKS[key]
    for key, url in AIRLINE_LINKS.items():
        if key in ["air canada","british airways"]:
            continue
        if key in t:
            return ALIAS_TO_NAME.get(key, key.title()), url
    return None, None

# ============================================================
# Intent detection
# ============================================================
BAGGAGE_KEYWORDS = ["baggage","bags","luggage","carry-on","carry on","carryon"]

def is_liquids_intent(user_l: str) -> bool:
    return any(k in user_l for k in ["liquid","toiletries","3-1-1","100ml"])

def is_powerbank_intent(user_l: str) -> bool:
    return any(k in user_l for k in ["power bank","battery","lithium","mah","wh"])

def is_baggage_intent(user_l: str) -> bool:
    return any(k in user_l for k in BAGGAGE_KEYWORDS)

def is_live_flights_intent(user_l: str) -> bool:
    return any(k in user_l for k in ["flight","flights","planes","from","to"])

# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="Airline Chatbot", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

class Query(BaseModel):
    message: str

@app.get("/")
def root():
    return {"msg": "Airline Chatbot API. Use POST /chat."}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(q: Query):
    user_msg = q.message.strip()
    user_l   = normalize(user_msg)

    # FAQs
    if is_liquids_intent(user_l):
        info, src = get_tsa_liquids_summary()
        return {"answer": info, "source": src}

    if is_powerbank_intent(user_l):
        info, src = get_faa_powerbank_summary()
        return {"answer": info, "source": src}

    if is_baggage_intent(user_l):
        name, link = get_airline_baggage_link(user_l)
        if link:
            return {"answer": f"Here’s the official baggage policy for {name}:", "source": link}
        return {"answer": "Tell me the airline (e.g., 'United baggage', 'AA baggage allowance')."}

    # Live flights
    if is_live_flights_intent(user_l):
        codes = detect_iata_tokens(user_msg)
        if codes:
            dep_code = codes[0]
            flights, err = query_aviationstack(dep=dep_code)
            if err:
                return {"answer": err}
            if flights:
                examples = []
                for f in flights[:5]:
                    airline = f["airline"]["name"] if f.get("airline") else "Unknown"
                    flight_no = f["flight"]["iata"] if f.get("flight") else "N/A"
                    dep = f["departure"]["iata"] or "?"
                    arr = f["arrival"]["iata"] or "?"
                    status = f.get("flight_status", "unknown")
                    examples.append(f"{airline} {flight_no} {dep}→{arr} — {status}")
                return {"answer": f"Found {len(flights)} flights from {dep_code}. Examples: {', '.join(examples)}."}
            else:
                return {"answer": f"No flights found from {dep_code} right now."}

    # Airport info
    iata_hits = detect_iata_tokens(user_msg)
    for code in iata_hits:
        r = airports.loc[airports["IATA"] == code, ["name","city","country","IATA","ICAO"]]
        if not r.empty:
            row = r.iloc[0]
            return {"answer": f"{row['IATA']} = {row['name']} in {row['city']}, {row['country']} (ICAO {row['ICAO']})."}

    # Fallback
    return {
        "answer": (
            "I can help with:\n"
            "• Live flights by IATA code (e.g., 'Flights from LAX')\n"
            "• Airline baggage links (e.g., 'United baggage')\n"
            "• TSA liquids rule\n"
            "• FAA power bank rules\n"
            "• Airport info by code or city"
        )
    }
