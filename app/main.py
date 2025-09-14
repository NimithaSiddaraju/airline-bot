# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import httpx
import re
import math
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

# ============================================================
# AviationStack API (replaces OpenSky)
# ============================================================
AVIATIONSTACK_KEY = "5c405fe0aa56286d8e7698ff945dff76"
AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"

def query_aviationstack(dep_code: Optional[str] = None, arr_code: Optional[str] = None):
    try:
        params = {"access_key": AVIATIONSTACK_KEY}
        if dep_code:
            params["dep_iata"] = dep_code
        if arr_code:
            params["arr_iata"] = arr_code

        resp = httpx.get(AVIATIONSTACK_URL, params=params, timeout=30.0)
        if resp.status_code != 200:
            return None, f"AviationStack status {resp.status_code}"
        data = resp.json()
        return data.get("data", []), None
    except Exception as e:
        return None, f"AviationStack error: {e}"

# ============================================================
# FAQs (summaries & links)
# ============================================================
def get_tsa_liquids_summary():
    url = "https://www.tsa.gov/travel/security-screening/whatcanibring/items/travel-size-toiletries"
    summary = ("TSA liquids rule (3-1-1): containers ≤ 3.4 oz / 100 mL; "
               "all containers fit in one quart-size bag; one bag per passenger; "
               "larger volumes → checked bag.")
    return summary, url

def get_faa_powerbank_summary():
    url = "https://www.faa.gov/hazmat/packsafe/lithium-batteries"
    summary = ("Power banks (lithium batteries): carry-on only. "
               "≤100 Wh allowed without airline approval; 100–160 Wh requires approval; "
               "no checked baggage.")
    return summary, url

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
}
ALIAS_TO_NAME = {
    "aa":"American Airlines","dl":"Delta Air Lines","ua":"United Airlines",
    "wn":"Southwest Airlines","as":"Alaska Airlines","b6":"JetBlue"
}

BAGGAGE_KEYWORDS = ["baggage","bags","luggage","checked bag","carry-on","carry on"]

def get_airline_baggage_link(text: str):
    t = normalize(text)
    for key, url in AIRLINE_LINKS.items():
        if re.search(rf"\b{re.escape(key)}\b", t):
            return ALIAS_TO_NAME.get(key, key.title()), url
    return None, None

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
    return {"msg": "Airline Chatbot API. Use POST /chat or open /docs for Swagger UI."}

@app.post("/chat")
async def chat(q: Query):
    user_msg = q.message.strip()
    user_l   = normalize(user_msg)

    # FAQs
    if any(k in user_l for k in ["liquid","toiletries","3-1-1","100ml","100 ml"]):
        info, src = get_tsa_liquids_summary()
        return {"answer": info, "source": src}

    if any(k in user_l for k in ["power bank","powerbank","battery","lithium","mah","wh"]):
        info, src = get_faa_powerbank_summary()
        return {"answer": info, "source": src}

    if any(k in user_l for k in BAGGAGE_KEYWORDS):
        name, link = get_airline_baggage_link(user_l)
        if link:
            return {"answer": f"Here’s the official baggage policy for {name}:", "source": link}
        return {"answer": "Tell me the airline (e.g., 'United baggage', 'AA baggage allowance')."}

    # Flights intent
    codes = detect_iata_tokens(user_msg)
    if codes:
        code = codes[0]
        if "to " in user_l:
            flights, err = query_aviationstack(arr_code=code)
            if err:
                return {"answer": f"Could not fetch live data ({err})."}
            if flights:
                examples = []
                for f in flights[:5]:
                    fl = f.get("flight", {})
                    dep = f.get("departure", {})
                    arr = f.get("arrival", {})
                    examples.append(f"{fl.get('iata', '??')} {dep.get('iata','?')}→{arr.get('iata','?')}")
                return {"answer": f"Found {len(flights)} flights to {code}. Examples: {', '.join(examples)}."}
            return {"answer": f"No live flights arriving at {code} right now."}
        else:
            flights, err = query_aviationstack(dep_code=code)
            if err:
                return {"answer": f"Could not fetch live data ({err})."}
            if flights:
                examples = []
                for f in flights[:5]:
                    fl = f.get("flight", {})
                    dep = f.get("departure", {})
                    arr = f.get("arrival", {})
                    examples.append(f"{fl.get('iata', '??')} {dep.get('iata','?')}→{arr.get('iata','?')}")
                return {"answer": f"Found {len(flights)} flights from {code}. Examples: {', '.join(examples)}."}
            return {"answer": f"No live flights departing {code} right now."}

    # Airport lookup
    by_city = airports.loc[airports["city_l"].str.contains(user_l, na=False), ["name","city","country","IATA"]]
    if not by_city.empty:
        row = by_city.iloc[0]
        return {"answer": f"Airport in {row['city']}: {row['name']} (IATA {row['IATA']})."}

    by_name = airports.loc[airports["name_l"].str.contains(user_l, na=False), ["name","city","country","IATA"]]
    if not by_name.empty:
        row = by_name.iloc[0]
        return {"answer": f"{row['name']} is in {row['city']}, {row['country']} (IATA {row['IATA']})."}

    return {"answer": (
        "I can help with:\n"
        "• Airline baggage links\n"
        "• TSA liquids & FAA power banks\n"
        "• Live flights by IATA code (e.g., 'Flights from LAX' or 'Flights to DFW')\n"
        "• Airport info by code/name/city"
    )}
