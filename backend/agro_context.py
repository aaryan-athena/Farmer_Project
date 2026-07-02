"""
agro_context.py  —  KisaanMitra location/soil/weather enrichment layer.

Given a diagnosed disease (class_index 0-16) and a coordinate (lat, lon),
this builds a context-aware advisory by combining:

    1. Soil      -> offline nearest-reference-point lookup over Indian
                    agricultural regions (no external API; ships with the app).
    2. Weather   -> Open-Meteo current + short forecast (free, no API key).
                    Uses stdlib urllib so it adds NO new requirements.
    3. Disease   -> per-class plant-pathology metadata + a weather-driven
                    "disease pressure" heuristic.

The disease keys (0-16) follow the verified alphabetical CLASS_MAP, so the
output joins directly onto whatever /predict already returns.
"""

from __future__ import annotations
import json
import math
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------- #
# 1. SOIL  —  offline lookup
# --------------------------------------------------------------------------- #
# Each reference point: (lat, lon, place, state, soil_key).
# Query coordinate is matched to its NEAREST reference point (haversine).
# Dense in the wheat/rice/sugarcane belt and especially Haryana (target region);
# degrades gracefully elsewhere. Extend by simply adding rows.

SOIL_REFERENCE_POINTS = [
    # --- Haryana (target region) ---
    (29.69, 76.99, "Karnal",     "Haryana",       "alluvial"),
    (30.38, 76.78, "Ambala",     "Haryana",       "alluvial"),
    (28.89, 76.61, "Rohtak",     "Haryana",       "alluvial"),
    (28.46, 77.03, "Gurugram",   "Haryana",       "alluvial"),
    (29.15, 75.72, "Hisar",      "Haryana",       "arid_sandy"),
    (29.53, 75.03, "Sirsa",      "Haryana",       "arid_sandy"),
    # --- Punjab ---
    (30.90, 75.86, "Ludhiana",   "Punjab",        "alluvial"),
    (31.63, 74.87, "Amritsar",   "Punjab",        "alluvial"),
    # --- Uttar Pradesh ---
    (28.98, 77.71, "Meerut",     "Uttar Pradesh", "alluvial"),
    (26.85, 80.95, "Lucknow",    "Uttar Pradesh", "alluvial"),
    (25.32, 82.97, "Varanasi",   "Uttar Pradesh", "alluvial"),
    # --- Bihar / Bengal / Assam (Gangetic + Brahmaputra) ---
    (25.59, 85.14, "Patna",      "Bihar",         "alluvial"),
    (22.57, 88.36, "Kolkata",    "West Bengal",   "alluvial"),
    (26.14, 91.74, "Guwahati",   "Assam",         "alluvial"),
    # --- Rajasthan (arid) ---
    (26.91, 75.79, "Jaipur",     "Rajasthan",     "arid_sandy"),
    (26.24, 73.02, "Jodhpur",    "Rajasthan",     "arid_sandy"),
    (28.02, 73.31, "Bikaner",    "Rajasthan",     "arid_sandy"),
    # --- Gujarat ---
    (23.02, 72.57, "Ahmedabad",  "Gujarat",       "alluvial"),
    (22.30, 70.80, "Rajkot",     "Gujarat",       "black"),
    # --- Madhya Pradesh / Maharashtra (Deccan black soils) ---
    (23.25, 77.41, "Bhopal",     "Madhya Pradesh","black"),
    (22.72, 75.86, "Indore",     "Madhya Pradesh","black"),
    (21.15, 79.09, "Nagpur",     "Maharashtra",   "black"),
    (18.52, 73.86, "Pune",       "Maharashtra",   "black"),
    (19.99, 73.79, "Nashik",     "Maharashtra",   "black"),
    # --- South (red / laterite) ---
    (12.97, 77.59, "Bengaluru",  "Karnataka",     "red"),
    (15.36, 75.12, "Hubli",      "Karnataka",     "black"),
    (17.39, 78.49, "Hyderabad",  "Telangana",     "red"),
    (16.51, 80.65, "Vijayawada", "Andhra Pradesh","alluvial"),
    (13.08, 80.27, "Chennai",    "Tamil Nadu",    "red"),
    (10.79, 79.14, "Thanjavur",  "Tamil Nadu",    "alluvial"),
    (11.02, 76.96, "Coimbatore", "Tamil Nadu",    "red"),
    (9.93,  76.27, "Kochi",      "Kerala",        "laterite"),
    (20.30, 85.82, "Bhubaneswar","Odisha",        "laterite"),
    # --- Hill / forest ---
    (30.32, 78.03, "Dehradun",   "Uttarakhand",   "alluvial"),
    (31.10, 77.17, "Shimla",     "Himachal Pradesh","forest_hill"),
]

SOIL_PROFILES = {
    "alluvial": {
        "name": "Alluvial soil",
        "texture": "sandy loam to clay loam",
        "ph_tendency": "neutral to slightly alkaline (6.5-8.0)",
        "drainage": "moderate to good",
        "notes": "Fertile and responsive to fertiliser; suits most field crops.",
    },
    "arid_sandy": {
        "name": "Arid / sandy soil",
        "texture": "sandy, low organic matter",
        "ph_tendency": "alkaline, can turn saline/sodic (7.5-9.0)",
        "drainage": "rapid (poor water retention)",
        "notes": "Holds little water; needs frequent light irrigation. Watch for salinity build-up.",
    },
    "black": {
        "name": "Black (regur / vertisol) soil",
        "texture": "clayey, high montmorillonite",
        "ph_tendency": "neutral to alkaline (7.0-8.5)",
        "drainage": "poor when wet, cracks when dry",
        "notes": "High water retention; prone to waterlogging in heavy rain, which raises root-rot risk.",
    },
    "red": {
        "name": "Red soil",
        "texture": "sandy loam to loam",
        "ph_tendency": "slightly acidic to neutral (5.5-7.0)",
        "drainage": "good",
        "notes": "Lower fertility; benefits from organic matter, nitrogen and zinc correction.",
    },
    "laterite": {
        "name": "Laterite soil",
        "texture": "gravelly loam, leached",
        "ph_tendency": "acidic (4.5-6.0)",
        "drainage": "good but nutrient-poor",
        "notes": "Heavily leached; liming and organic amendments improve productivity.",
    },
    "forest_hill": {
        "name": "Forest / hill soil",
        "texture": "loam, variable, often stony",
        "ph_tendency": "acidic on slopes",
        "drainage": "good (slope runoff)",
        "notes": "Thin on slopes and erosion-prone; terracing and mulching help retain moisture.",
    },
}


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def get_soil(lat: float, lon: float) -> dict:
    """Return the soil profile of the nearest known agricultural reference point."""
    nearest = min(
        SOIL_REFERENCE_POINTS,
        key=lambda p: _haversine_km(lat, lon, p[0], p[1]),
    )
    rlat, rlon, place, state, key = nearest
    profile = dict(SOIL_PROFILES[key])
    profile.update({
        "soil_key": key,
        "nearest_reference": place,
        "state": state,
        "distance_km": round(_haversine_km(lat, lon, rlat, rlon), 1),
    })
    return profile


# --------------------------------------------------------------------------- #
# 2. WEATHER  —  Open-Meteo (free, keyless, stdlib only)
# --------------------------------------------------------------------------- #

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
# In-process cache: {(lat_rounded, lon_rounded): (fetched_at_epoch, weather_dict)}
# Coordinates are rounded to ~1 km so nearby diagnoses share one fetch.
_WEATHER_CACHE: dict = {}
_WEATHER_CACHE_TTL = 900  # 15 minutes


def get_weather(lat: float, lon: float, timeout: float = 8.0, retries: int = 2) -> dict | None:
    """Fetch current conditions + short forecast from Open-Meteo (no API key, no signup).

    Open-Meteo's weather_code follows the WMO scheme the frontend's icon table
    already expects, and its uptime/latency are far more consistent than
    wttr.in (which this used to call and would intermittently time out,
    causing the weather chip to silently disappear from results).

    Cached in-process for 15 minutes per ~1 km grid cell so repeat requests
    from the same field don't hammer the service."""
    import time
    key = (round(lat, 2), round(lon, 2))
    now = time.time()
    cached = _WEATHER_CACHE.get(key)
    if cached and (now - cached[0]) < _WEATHER_CACHE_TTL:
        return cached[1]

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
        "hourly": "precipitation",
        "forecast_days": 3,
        "timezone": "auto",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "KisaanMitra/1.0"})

    raw = None
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue

    if raw is None:
        # Serve a stale cached value rather than nothing, if we have one.
        if cached:
            return cached[1]
        return {"_debug_error": last_error}

    try:
        cur = raw["current"]
        temp_c = cur.get("temperature_2m")
        humidity = cur.get("relative_humidity_2m")
        precip_now = float(cur.get("precipitation") or 0)
        weather_code = cur.get("weather_code")

        # Sum hourly precipitation for the 48 hours starting at the current hour.
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        precips = hourly.get("precipitation", [])
        current_time = cur.get("time", "")
        start = next((i for i, t in enumerate(times) if t >= current_time), 0)
        rain_next_48h = round(sum(precips[start:start + 48]), 1)
    except (KeyError, ValueError, TypeError, IndexError) as e:
        return {"_debug_error": f"ParseError: {type(e).__name__}: {e}"}

    result = {
        "temp_c": temp_c,
        "humidity_pct": humidity,
        "precip_now_mm": precip_now,
        "rain_next_48h_mm": rain_next_48h,
        "weather_code": weather_code,
    }
    _WEATHER_CACHE[key] = (now, result)
    return result


# --------------------------------------------------------------------------- #
# 3. DISEASE metadata  (keyed by verified alphabetical class_index)
# --------------------------------------------------------------------------- #
# type: fungal | oomycete | bacterial | viral | healthy
# temp_opt: (min, max) °C window in which the pathogen is most active
# management: concise, disease-specific baseline action

DISEASE_INFO = {
    0:  {"crop": "Corn", "name": "Common rust", "type": "fungal", "temp_opt": (16, 25),
         "management": "Apply a protectant fungicide (e.g. mancozeb) at first pustules; favour resistant hybrids next season."},
    1:  {"crop": "Corn", "name": "Healthy", "type": "healthy"},
    2:  {"crop": "Corn", "name": "Northern leaf blight", "type": "fungal", "temp_opt": (18, 27),
         "management": "Remove lower infected leaves, rotate crops, and use a triazole/strobilurin fungicide if lesions spread up the plant."},
    3:  {"crop": "Potato", "name": "Early blight", "type": "fungal", "temp_opt": (24, 29),
         "management": "Maintain plant vigour (avoid N stress); spray mancozeb/chlorothalonil on a 7-10 day schedule once lesions appear."},
    4:  {"crop": "Potato", "name": "Healthy", "type": "healthy"},
    5:  {"crop": "Potato", "name": "Late blight", "type": "oomycete", "temp_opt": (10, 24),
         "management": "Act fast — this spreads explosively. Use a systemic fungicide (cymoxanil/metalaxyl + mancozeb); destroy infected foliage."},
    6:  {"crop": "Rice", "name": "Bacterial blight", "type": "bacterial", "temp_opt": (25, 34),
         "management": "No effective foliar cure — reduce nitrogen, drain the field, remove infected debris, and plant resistant varieties next season."},
    7:  {"crop": "Rice", "name": "Rice blast", "type": "fungal", "temp_opt": (20, 30),
         "management": "Cut back nitrogen, keep consistent water, and apply tricyclazole at early lesion/neck stage."},
    8:  {"crop": "Rice", "name": "Healthy", "type": "healthy"},
    9:  {"crop": "Sugarcane", "name": "Healthy", "type": "healthy"},
    10: {"crop": "Sugarcane", "name": "Mosaic", "type": "viral", "temp_opt": None,
         "management": "Virus is aphid-spread and seed-borne — rogue out infected clumps, control aphids, and plant only certified disease-free setts. Fungicides do not work."},
    11: {"crop": "Sugarcane", "name": "Red rot", "type": "fungal", "temp_opt": (25, 32),
         "management": "Uproot and burn affected clumps, improve drainage, and use resistant varieties with hot-water-treated setts."},
    12: {"crop": "Wheat", "name": "Black (stem) rust", "type": "fungal", "temp_opt": (20, 30),
         "management": "Spray propiconazole/tebuconazole at first sign; grow resistant varieties — this rust can collapse a crop late in the season."},
    13: {"crop": "Wheat", "name": "Brown (leaf) rust", "type": "fungal", "temp_opt": (15, 25),
         "management": "Apply a triazole fungicide at early pustule stage; monitor closely as it can shift toward stem rust look-alikes."},
    14: {"crop": "Wheat", "name": "Healthy", "type": "healthy"},
    15: {"crop": "Wheat", "name": "Leaf blight", "type": "fungal", "temp_opt": (20, 30),
         "management": "Use balanced nutrition, seed treatment, and mancozeb/propiconazole sprays; rotate away from cereals."},
    16: {"crop": "Wheat", "name": "Yellow (stripe) rust", "type": "fungal", "temp_opt": (10, 18),
         "management": "Spray propiconazole promptly — yellow rust thrives in cool humid Haryana winters and spreads fast across fields."},
}


# --------------------------------------------------------------------------- #
# 4. DISEASE PRESSURE  —  weather-driven heuristic
# --------------------------------------------------------------------------- #

def assess_disease_pressure(info: dict, weather: dict | None) -> str:
    """Return 'high' | 'moderate' | 'low' | 'n/a' from weather + pathogen window."""
    if info["type"] in ("healthy", "viral"):
        return "n/a"
    if not weather or weather.get("temp_c") is None:
        return "unknown"

    temp = weather["temp_c"]
    humidity = weather.get("humidity_pct") or 0
    wet = (weather.get("precip_now_mm") or 0) > 0.2 or (weather.get("rain_next_48h_mm") or 0) > 5

    score = 0.0
    lo, hi = info["temp_opt"]
    if lo <= temp <= hi:
        score += 1.0
    elif lo - 3 <= temp <= hi + 3:
        score += 0.5

    if humidity >= 80:
        score += 1.0
    elif humidity >= 65:
        score += 0.5

    if wet:
        score += 1.0

    if score >= 2.5:
        return "high"
    if score >= 1.5:
        return "moderate"
    return "low"


# --------------------------------------------------------------------------- #
# 5. ADVISORY  —  combine disease + soil + weather into actionable lines
# --------------------------------------------------------------------------- #

DISCLAIMER = ("Guidance is indicative. Confirm dose and product with your local "
              "Krishi Vigyan Kendra (KVK) or agriculture extension officer before spraying.")


def build_advisory(class_index: int, lat: float, lon: float,
                   weather: dict | None = None) -> dict:
    """
    Main entry point. Pass the predicted class_index and the user's coordinate.
    `weather` can be injected (for testing); otherwise it is fetched live.
    """
    info = DISEASE_INFO.get(int(class_index))
    if info is None:
        return {"error": f"Unknown class_index {class_index}"}

    soil = get_soil(lat, lon)
    if weather is None:
        weather = get_weather(lat, lon)
    pressure = assess_disease_pressure(info, weather)

    advisory: list[str] = []

    # -- Healthy ----------------------------------------------------------- #
    if info["type"] == "healthy":
        advisory.append(f"No disease detected on this {info['crop'].lower()} sample — keep monitoring.")
        if weather and pressure != "unknown":
            advisory.append("Continue routine scouting after rain or high-humidity spells, when most diseases establish.")
        advisory.append(soil["notes"])
        return _assemble(info, soil, weather, pressure, advisory)

    # -- Disease present --------------------------------------------------- #
    # 5a. pressure framing
    if pressure == "high":
        advisory.append("Current weather strongly favours this disease — treat promptly (within ~48 hours).")
    elif pressure == "moderate":
        advisory.append("Weather is moderately favourable for spread — scout daily and be ready to treat.")
    elif pressure == "low":
        advisory.append("Weather is currently unfavourable for rapid spread, but treat visible infection to be safe.")

    # 5b. baseline disease management
    advisory.append(info["management"])

    # 5c. spray-timing from weather
    if weather and weather.get("temp_c") is not None and info["type"] != "viral":
        rain = weather.get("rain_next_48h_mm") or 0
        temp = weather["temp_c"]
        humidity = weather.get("humidity_pct") or 0
        if rain > 5:
            advisory.append(f"Rain (~{rain} mm) is expected in the next 48 h — use a rain-fast/systemic product or "
                            "spray right after the rain so it isn't washed off.")
        elif temp >= 32 and humidity < 50:
            advisory.append("Conditions are hot and dry — spray in the early morning or evening to avoid evaporation and leaf scorch.")

    # 5d. soil modifier
    if info["type"] != "viral":
        wet_weather = bool(weather and ((weather.get("rain_next_48h_mm") or 0) > 5 or (weather.get("precip_now_mm") or 0) > 0.2))
        if soil["soil_key"] == "black" and wet_weather:
            advisory.append("Soil here is heavy black/clay with poor drainage — in this wet spell, improve field drainage "
                            "and avoid over-irrigation to limit waterlogging and root rot.")
        elif soil["soil_key"] == "arid_sandy":
            advisory.append("Sandy soil drains fast and holds little water — keep moisture even and avoid water stress, "
                            "which weakens the plant's defences.")
        elif soil["soil_key"] == "laterite":
            advisory.append("Laterite soil is acidic and leached — correct pH with lime and add organics to keep plants vigorous against infection.")

    advisory.append(soil["notes"])
    return _assemble(info, soil, weather, pressure, advisory)


def _assemble(info, soil, weather, pressure, advisory):
    return {
        "crop": info["crop"],
        "disease": info["name"],
        "pathogen_type": info["type"],
        "disease_pressure": pressure,
        "soil": {
            "type": soil["name"],
            "texture": soil["texture"],
            "ph_tendency": soil["ph_tendency"],
            "drainage": soil["drainage"],
            "nearest_reference": soil["nearest_reference"],
            "state": soil["state"],
        },
        "weather": weather,
        "advisory": advisory,
        "disclaimer": DISCLAIMER,
    }


# --------------------------------------------------------------------------- #
# Offline demo (weather injected so it runs without network)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Karnal, Haryana — cool, humid winter morning with rain coming
    winter_wet = {"temp_c": 14.0, "humidity_pct": 88, "precip_now_mm": 0.0, "rain_next_48h_mm": 9.0, "weather_code": 61}
    print(json.dumps(build_advisory(16, 29.69, 76.99, weather=winter_wet), indent=2, ensure_ascii=False))
    print("\n" + "=" * 70 + "\n")
    # Nagpur (black soil) — warm humid, Potato late blight, rain coming
    monsoon = {"temp_c": 22.0, "humidity_pct": 90, "precip_now_mm": 1.2, "rain_next_48h_mm": 18.0, "weather_code": 63}
    print(json.dumps(build_advisory(5, 21.15, 79.09, weather=monsoon), indent=2, ensure_ascii=False))
    print("\n" + "=" * 70 + "\n")
    # Hisar (arid sandy) — healthy wheat, dry
    dry = {"temp_c": 33.0, "humidity_pct": 30, "precip_now_mm": 0.0, "rain_next_48h_mm": 0.0, "weather_code": 0}
    print(json.dumps(build_advisory(14, 29.15, 75.72, weather=dry), indent=2, ensure_ascii=False))
