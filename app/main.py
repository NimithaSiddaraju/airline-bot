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
    """
    Return 3-letter IATA codes that the user actually typed as codes.
    Accept whole-word 3-letter tokens.
    """
    tokens = re.findall(r"\b[a-zA-Z]{3}\b", user_msg)
    hits = [t.upper() for t in tokens if t.upper() in iata_set]

    if not hits and len(user_msg.strip()) == 3:
        t = user_msg.strip().upper()
        if t in iata_set:
            hits = [t]
    return hits

# ============================================================
# FAQs (sync summaries; link to official sources)
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

# ============================================================
# Airline baggage links
# ============================================================
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
    # Check multi-word airlines
    for key in ["air canada","british airways"]:
        if key in t:
            return key.title(), AIRLINE_LINKS[key]
    # Check others loosely
    for key, url in AIRLINE_LINKS.items():
        if key in ["air canada","british airways"]:
            continue
        if key in t:  # looser match
            return ALIAS_TO_NAME.get(key, key.title()), url
    return None, None

# ============================================================
# Power bank calculator
# ============================================================
def powerbank_wh_from_text(text: str) -> Optional[Tuple[float, float]]:
    t = text.lower().replace(",", " ")
    m_wh = re.search(r"(\d+(\.\d+)?)\s*wh\b", t)
    if m_wh:
        return float(m_wh.group(1)), None
    m_mah = re.search(r"(\d+(\.\d+)?)\s*mah\b", t)
    if m_mah:
        mah = float(m_mah.group(1))
        m_v = re.search(r"(\d+(\.\d+)?)\s*v\b", t)
        v = float(m_v.group(1)) if m_v else 3.7
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

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(q: Query):
    user_msg = q.message.strip()
    user_l   = normalize(user_msg)

    # ---------- TSA Liquids ----------
    if any(k in user_l for k in ["liquid","toiletries","3-1-1","3 1 1","100ml","100 ml"]):
        info, src = get_tsa_liquids_summary()
        return {"answer": info, "source": src}

    # ---------- Power banks ----------
    if any(k in user_l for k in ["power bank","powerbank","battery","lithium","mah","wh"]):
        calc = powerbank_wh_from_text(user_l)
        if calc:
            wh, v = calc
            verdict = classify_wh(wh)
            v_txt = "" if v is None else f" using {v} V,"
            return {
                "answer": f"Estimated capacity ≈ {wh:.1f} Wh{v_txt} which falls under: {verdict}",
                "source": "https://www.faa.gov/hazmat/packsafe/lithium-batteries"
            }
        info, src = get_faa_powerbank_summary()
        return {"answer": info, "source": src}

    # ---------- Airline baggage ----------
    if any(k in user_l for k in ["baggage","bags","luggage","checked bag","carry-on","carry on","carryon","bag fee","allowance"]):
        name, link = get_airline_baggage_link(user_l)
        if link:
            return {"answer": f"Here’s the official baggage policy for {name}:", "source": link}
        return {"answer": "Tell me the airline (e.g., 'United baggage', 'AA baggage allowance')."}

    # ---------- Live flights (AviationStack) ----------
    if user_l.startswith("flights from") or user_l.startswith("flights to"):
        direction = "departures" if "from" in user_l else "arrivals"
        codes = detect_iata_tokens(user_msg)
        if codes:
            code = codes[0]
            url = f"http://api.aviationstack.com/v1/{direction}"
            try:
                resp = httpx.get(url, params={
                    "access_key": "5c405fe0aa56286d8e7698ff945dff76",  # your API key
                    "airport_iata": code
                }, timeout=20.0)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if data:
                        examples = []
                        for f in data[:5]:
                            airline = f.get("airline", {}).get("name", "Unknown Airline")
                            flight_no = f.get("flight", {}).get("iata", "N/A")
                            dep = f.get("departure", {}).get("iata", "???")
                            arr = f.get("arrival", {}).get("iata", "???")
                            status = f.get("flight_status", "scheduled")
                            examples.append(f"{flight_no} ({airline}) {dep}→{arr} — {status}")
                        return {
                            "answer": f"Found {len(data)} {direction} for {code}. Examples: " + ", ".join(examples)
                        }
                    else:
                        return {"answer": f"No {direction} found for {code} at this time."}
                else:
                    return {"answer": f"AviationStack error {resp.status_code}: {resp.text}"}
            except Exception as e:
                return {"answer": f"AviationStack error: {str(e)}"}

    # ---------- Airport lookup ----------
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

    # ---------- Fallback ----------
    return {
        "answer": ("I can help with:\n"
                   "• Live flights (departures & arrivals by IATA code)\n"
                   "• Airline baggage links\n"
                   "• TSA liquids & FAA power banks\n"
                   "• Airport info by code/name/city")
    }
