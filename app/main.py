# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import httpx
import re
from typing import Tuple, Optional, List

# ============================================================
# Remote reference data
# ============================================================
AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
COLS = ["id","name","city","country","IATA","ICAO","lat","lon","alt_ft","tz_offset","dst","tzdb","type","source"]
airports = pd.read_csv(AIRPORTS_URL, header=None, names=COLS)

airports["city_l"] = airports["city"].fillna("").str.lower()
airports["name_l"] = airports["name"].fillna("").str.lower()
iata_set = set(airports["IATA"].dropna().astype(str))

# ============================================================
# AviationStack API (your API key)
# ============================================================
AVIATIONSTACK_KEY = "5c405fe0aa56286d8e7698ff945dff76"

def query_aviationstack(dep_iata: str):
    """Fetch live flights departing from a given IATA airport code"""
    url = "http://api.aviationstack.com/v1/flights"
    params = {
        "access_key": AVIATIONSTACK_KEY,
        "dep_iata": dep_iata,
        "limit": 5
    }
    try:
        resp = httpx.get(url, params=params, timeout=15.0)
        if resp.status_code != 200:
            return None, f"AviationStack status {resp.status_code}"
        data = resp.json()
        return data.get("data", []), None
    except Exception as e:
        return None, f"AviationStack error: {e}"

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
    "aa": "https://www.aa.com/i18n/travel-info/baggage/baggage.jsp",
    "delta": "https://www.delta.com/traveling-with-us/baggage",
    "dl": "https://www.delta.com/traveling-with-us/baggage",
    "united": "https://www.united.com/en/us/fly/travel/baggage.html",
    "ua": "https://www.united.com/en/us/fly/travel/baggage.html",
    "southwest": "https://www.southwest.com/help/baggage",
    "wn": "https://www.southwest.com/help/baggage",
    "alaska": "https://www.alaskaair.com/travel-info/baggage/overview",
    "as": "https://www.alaskaair.com/travel-info/baggage/overview",
    "jetblue": "https://www.jetblue.com/help/baggage",
    "b6": "https://www.jetblue.com/help/baggage",
    "air canada": "https://www.aircanada.com/ca/en/aco/home/plan/baggage.html",
    "ac": "https://www.aircanada.com/ca/en/aco/home/plan/baggage.html",
    "british airways": "https://www.britishairways.com/en-us/information/baggage-essentials",
    "ba": "https://www.britishairways.com/en-us/information/baggage-essentials",
    "lufthansa": "https://www.lufthansa.com/us/en/baggage-overview",
    "lh": "https://www.lufthansa.com/us/en/baggage-overview",
    "emirates": "https://www.emirates.com/us/english/before-you-fly/baggage/",
    "ek": "https://www.emirates.com/us/english/before-you-fly/baggage/",
    "qatar": "https://www.qatarairways.com/en-us/baggage/allowance.html",
    "qr": "https://www.qatarairways.com/en-us/baggage/allowance.html",
    "singapore": "https://www.singaporeair.com/en_UK/us/travel-info/baggage/",
    "sq": "https://www.singaporeair.com/en_UK/us/travel-info/baggage/",
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
        if re.search(rf"\b{re.escape(key)}\b", t):
            return ALIAS_TO_NAME.get(key, key.title()), url
    return None, None

# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="Airline Chatbot", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nimithasiddaraju.github.io"],  # ✅ your frontend
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    message: str

@app.get("/")
def root():
    return {"msg": "Airline Chatbot API (AviationStack). Use POST /chat or open /docs."}

@app.post("/chat")
async def chat(q: Query):
    user_msg = q.message.strip()
    user_l   = normalize(user_msg)

    # FAQ intents
    if any(k in user_l for k in ["liquid","toiletries","3-1-1","100ml","100 ml"]):
        info, src = get_tsa_liquids_summary()
        return {"answer": info, "source": src}

    if any(k in user_l for k in ["power bank","powerbank","battery","lithium","mah","wh"]):
        info, src = get_faa_powerbank_summary()
        return {"answer": info, "source": src}

    if any(k in user_l for k in ["baggage","bags","luggage","checked bag","carry-on","carry on"]):
        name, link = get_airline_baggage_link(user_l)
        if link:
            return {"answer": f"Here’s the official baggage policy for {name}:", "source": link}
        return {"answer": "Tell me the airline (e.g., 'United baggage', 'AA baggage allowance')."}

    # Live flights intent
    if any(k in user_l for k in ["flight","flights","aircraft","planes","plane","traffic"]):
        codes = detect_iata_tokens(user_msg)
        if codes:
            dep_code = codes[0]
            flights, err = query_aviationstack(dep_code)
            if err:
                return {"answer": f"Could not fetch live data ({err}). Try again shortly."}
            if flights:
                examples = []
                for f in flights[:5]:
                    flight = f.get("flight", {})
                    dep = f.get("departure", {})
                    arr = f.get("arrival", {})
                    examples.append(f"{flight.get('iata', '??')} {dep.get('iata', '?')}→{arr.get('iata', '?')}")
                return {"answer": f"Found {len(flights)} flights from {dep_code}. Examples: {', '.join(examples)}."}
            else:
                return {"answer": f"No live flights found for {dep_code} right now."}

    return {
        "answer": ("I can help with:\n"
                   "• Airline baggage links\n"
                   "• TSA liquids & FAA power banks\n"
                   "• Live flights by IATA code (e.g., 'Flights from LAX')\n"
                   "• Airport info by code/name/city")
    }
