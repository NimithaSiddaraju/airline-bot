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

def find_location_from_message(msg_lower: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    m = re.search(r"(near|around|over|in|at)\s+([a-zA-Z .'\-]+)", msg_lower)
    candidate = (m.group(2).strip() if m else msg_lower).lower()
    city_hits = airports.loc[
        airports["city_l"].str.contains(candidate, na=False),
        ["city","name","lat","lon","country","IATA"]
    ].dropna(subset=["lat","lon"])
    if not city_hits.empty:
        lat = float(city_hits["lat"].astype(float).mean())
        lon = float(city_hits["lon"].astype(float).mean())
        label = f"{city_hits.iloc[0]['city']} ({city_hits.iloc[0]['country']})"
        return lat, lon, label
    name_hits = airports.loc[
        airports["name_l"].str.contains(candidate, na=False),
        ["city","name","lat","lon","country","IATA"]
    ].dropna(subset=["lat","lon"])
    if not name_hits.empty:
        lat = float(name_hits["lat"].astype(float).mean())
        lon = float(name_hits["lon"].astype(float).mean())
        label = f"{name_hits.iloc[0]['name']} ({name_hits.iloc[0]['city']})"
        return lat, lon, label
    return None, None, None

# --- FAQs (sync summaries; link to official sources) ---
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

# --- Airline baggage links router ---
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

BAGGAGE_KEYWORDS = [
    "baggage","bags","luggage","checked bag","checked bags","carry-on",
    "carry on","carryon","baggage allowance","bag fee","bag fees","allowance",
    "how many bags","how much luggage","how much baggage"
]

def get_airline_baggage_link(text: str):
    t = normalize(text)

    # Multi-word airline names
    for key in ["air canada","british airways"]:
        if key in t:
            return key.title(), AIRLINE_LINKS[key]

    # Looser matching for all other airlines
    for key, url in AIRLINE_LINKS.items():
        if key in ["air canada","british airways"]:
            continue
        if key in t:  # substring match
            return ALIAS_TO_NAME.get(key, key.title()), url
    return None, None

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

@app.get("/")
def root():
    return {"msg": "Airline Chatbot API. Use POST /chat or open /docs for Swagger UI."}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(q: Query):
    user_msg = q.message.strip()
    user_l   = normalize(user_msg)

    # ---------- TSA liquids ----------
    if any(k in user_l for k in ["liquid","toiletries","3-1-1","3 1 1","100ml","100 ml"]):
        info, src = get_tsa_liquids_summary()
        return {"answer": info, "source": src}

    # ---------- Power banks ----------
    if any(k in user_l for k in ["power bank","powerbank","battery","lithium","mah","wh"]):
        info, src = get_faa_powerbank_summary()
        return {"answer": info, "source": src}

    # ---------- Airline baggage ----------
    if any(k in user_l for k in BAGGAGE_KEYWORDS):
        name, link = get_airline_baggage_link(user_l)
        if link:
            return {"answer": f"Here’s the official baggage policy for {name}:", "source": link}
        return {"answer": "Tell me the airline (e.g., 'United baggage', 'AA baggage allowance')."}

    # ---------- Airport info ----------
    iata_hits = detect_iata_tokens(user_msg)
    for code in iata_hits:
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

    return {
        "answer": ("I can help with:\n"
                   "• Airline baggage links (e.g., 'United baggage', 'AA carry on size')\n"
                   "• TSA liquids & FAA power banks (e.g., 'what's the liquids rule', 'is 20000 mAh allowed')\n"
                   "• Live flights by IATA code (future: 'Flights from LAX')\n"
                   "• Airport info by code/name/city (e.g., 'DFW', 'airport in Tokyo').")
    }
