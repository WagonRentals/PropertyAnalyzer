"""
Property Scorer — Streamlit web app
===================================
Scores a single property address for golf cart / LSV rental viability across
three dimensions:

  1. Competitor activity   — Google Places (via lsv_competitor_scorer.py)
  2. STR pricing           — AirROI licensed API (polygon search)
  3. LSV route safety      — OpenStreetMap (via lsv_route_analyzer.py)

The fleet is PARKED AT the property, so the LSV section specifically checks the
property's exit road, the immediate 0.25-mi vicinity, and connected reachability
— a cart that can't legally leave the lot makes the site a non-starter.

Run
---
    pip install streamlit streamlit-folium geopy osmnx networkx pandas shapely requests
    export GOOGLE_PLACES_API_KEY="..."     # competitor section
    export AIRROI_API_KEY="..."            # STR section
    streamlit run property_scorer.py

Everything is cached per-address under ./.property_scorer_cache/, so re-submitting
an address is instant and (for STR) free.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import folium
import networkx as nx
import osmnx as ox
import requests
import streamlit as st
from folium.plugins import MeasureControl
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from shapely.geometry import MultiPoint, Point

from lsv_competitor_scorer import score_market, MarketResult
from lsv_route_analyzer import (
    fetch_network, extract_lsv_subgraph, edge_is_lsv_legal, edge_is_forbidden,
    edge_speed_mph, nearest_connected_node, isochrone_node_set, isochrone_area_km2,
    LSV_CRUISE_MPH,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_ROOT = Path("./.property_scorer_cache")
GEOCODE_DIR = CACHE_ROOT / "geocode"
COMPETITOR_DIR = CACHE_ROOT / "competitors"
STR_DIR = CACHE_ROOT / "str"
LSV_DIR = CACHE_ROOT / "lsv"
for _d in (GEOCODE_DIR, COMPETITOR_DIR, STR_DIR, LSV_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Brand logo — pulled directly from the live Wagon site (square app icon).
LOGO_URL = "https://wagon.rent/assets/wagon-app-icon.png"

STR_CACHE_MAX_AGE_DAYS = 30
METERS_PER_MILE = 1609.34

AIRROI_API_KEY = os.environ.get("AIRROI_API_KEY", "")
AIRROI_POLYGON_URL = "https://api.airroi.com/listings/search/polygon"
AIRROI_PAGE_SIZE = 10         # AirROI caps pageSize at 10
AIRROI_MAX_PAGES = 20         # up to ~200 listings/address (cost/time guard)
AIRROI_EST_COST = "$0.25–1.50"

GOOGLE_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "")

# STR nightly price bands (label, lower, upper)
PRICE_BANDS = [
    ("Budget", 0, 100), ("Mid-range", 100, 200), ("Upscale", 200, 400),
    ("Luxury", 400, 800), ("Ultra-luxury", 800, 1600),
]
PRICE_BAR_MAX = 1600

# LSV scoring thresholds
VICINITY_RADIUS_M = 400               # ~0.25 mi
VICINITY_MAX_FORBIDDEN_PCT = 20.0
CONNECTED_MIN_KM = 5.0
AREA_30MIN_MIN_KM2 = 5.0
AMENITIES_MIN = 10

AMENITY_TAGS = {
    "amenity": ["restaurant", "cafe", "bar", "fast_food"],
    "shop": True,
    "natural": ["beach"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeoCandidate:
    address: str
    lat: float
    lon: float


@dataclass
class STRResult:
    available: bool
    median_adr: float = 0.0
    avg_adr: float = 0.0
    listing_count: int = 0
    pct_entire: float = 0.0
    pct_private: float = 0.0
    pct_shared: float = 0.0
    median_rating: float = 0.0
    message: str = ""


@dataclass
class LSVResult:
    score: float
    exit_road_legal: bool
    exit_road_name: str
    exit_road_speed: float
    vicinity_clean: bool
    pct_vicinity_forbidden: float
    connected_km: float
    area_15min_km2: float
    area_30min_km2: float
    area_45min_km2: float
    amenities: dict
    warnings: list[str]
    subscores: dict
    legal_geojson: str = ""
    forbidden_geojson: str = ""
    all_geojson: str = ""
    isochrone_coords: list = field(default_factory=list)
    sparse: bool = False


# ---------------------------------------------------------------------------
# Geometry / utility helpers
# ---------------------------------------------------------------------------

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.7613  # earth radius in miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def circle_polygon(lat: float, lon: float, radius_miles: float,
                   n_vertices: int = 32) -> list[dict]:
    """A closed ring of {latitude, longitude} points approximating a circle —
    the shape AirROI's polygon search expects."""
    pts = []
    lat_per_mile = 1.0 / 69.0
    lon_per_mile = 1.0 / (69.0 * max(0.01, math.cos(math.radians(lat))))
    for i in range(n_vertices):
        theta = 2 * math.pi * i / n_vertices
        pts.append({
            "latitude": round(lat + radius_miles * lat_per_mile * math.sin(theta), 6),
            "longitude": round(lon + radius_miles * lon_per_mile * math.cos(theta), 6),
        })
    return pts


def price_band(adr: float) -> tuple[str, float]:
    """Return (band label, 0..1 position on the price bar) for an ADR."""
    label = PRICE_BANDS[-1][0]
    for name, lo, hi in PRICE_BANDS:
        if adr < hi:
            label = name
            break
    return label, min(1.0, max(0.0, adr / PRICE_BAR_MAX))


def _median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


# ---------------------------------------------------------------------------
# Geocoding (Nominatim)
# ---------------------------------------------------------------------------

GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def _geocode_nominatim(address: str) -> list[GeoCandidate]:
    """Primary geocoder (free OSM). Returns [] on any failure so we can fall
    back. Timeout is generous because the public endpoint is slow from servers."""
    try:
        geolocator = Nominatim(user_agent="wagon-property-analyzer", timeout=8)
        locs = geolocator.geocode(address, exactly_one=False, limit=5,
                                  addressdetails=True)
    except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError):
        return []
    return [GeoCandidate(address=l.address, lat=l.latitude, lon=l.longitude)
            for l in (locs or [])]


def _geocode_google(address: str) -> list[GeoCandidate]:
    """Fallback geocoder using the Google Geocoding API (reliable from cloud
    hosts). Requires the Geocoding API to be enabled on the GOOGLE_PLACES key."""
    if not GOOGLE_API_KEY:
        return []
    try:
        data = requests.get(GOOGLE_GEOCODE_URL,
                            params={"address": address, "key": GOOGLE_API_KEY},
                            timeout=15).json()
    except requests.RequestException:
        return []
    if data.get("status") != "OK":
        return []
    out = []
    for res in data.get("results", [])[:5]:
        loc = res.get("geometry", {}).get("location", {})
        if "lat" in loc and "lng" in loc:
            out.append(GeoCandidate(address=res.get("formatted_address", address),
                                    lat=loc["lat"], lon=loc["lng"]))
    return out


def geocode(address: str) -> list[GeoCandidate]:
    """Geocode an address: Nominatim first, Google as a fallback. Cached per
    address so repeats are instant and free."""
    key = "".join(c if c.isalnum() else "_" for c in address.lower())[:80]
    cache = GEOCODE_DIR / f"{key}.json"
    if cache.exists():
        return [GeoCandidate(**c) for c in json.loads(cache.read_text())]

    candidates = _geocode_nominatim(address)
    if not candidates:
        candidates = _geocode_google(address)   # licensed, server-reliable fallback
    if candidates:
        cache.write_text(json.dumps([asdict(c) for c in candidates]))
    return candidates


# ---------------------------------------------------------------------------
# Score 1 — Competitor activity
# ---------------------------------------------------------------------------

def _comp_cache(lat: float, lon: float, radius: float) -> Path:
    return COMPETITOR_DIR / f"{lat:.4f}_{lon:.4f}_r{radius:g}.json"


def score_competitors(lat: float, lon: float, radius_miles: float) -> dict:
    """Wraps lsv_competitor_scorer.score_market and reshapes the top 5 (by
    review count) with distance + Maps link. Cached per address."""
    cache = _comp_cache(lat, lon, radius_miles)
    if cache.exists():
        return json.loads(cache.read_text())

    result: MarketResult = score_market(f"property@{lat:.4f},{lon:.4f}", lat, lon,
                                        radius_miles=radius_miles, verbose=False)
    operational = [c for c in result.competitors if c.business_status == "OPERATIONAL"]
    top5 = sorted(operational, key=lambda c: c.review_count, reverse=True)[:5]
    data = {
        "score": result.score,
        "count": result.operational_count,
        "top5": [{
            "name": c.name, "rating": c.rating, "reviews": c.review_count,
            "distance_mi": round(haversine_miles(lat, lon, c.lat, c.lon), 1),
            "maps_url": c.google_maps_url, "lat": c.lat, "lon": c.lon,
        } for c in top5],
    }
    cache.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------------------
# Score 2 — STR pricing (AirROI)
# ---------------------------------------------------------------------------

def _extract_listings(payload) -> list:
    """Pull the listings array out of AirROI's response, tolerant of the exact
    envelope key (listings/data/results/items, a bare list, or first list-of-dicts)."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("listings", "data", "results", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        for v in payload.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v
    return []


def _str_cache(lat: float, lon: float, radius: float) -> Path:
    return STR_DIR / f"{lat:.4f}_{lon:.4f}_r{radius:g}.json"


def str_cache_fresh(lat: float, lon: float, radius: float) -> bool:
    p = _str_cache(lat, lon, radius)
    if not p.exists():
        return False
    age_days = (time.time() - p.stat().st_mtime) / 86400
    return age_days < STR_CACHE_MAX_AGE_DAYS


def load_str_cache(lat: float, lon: float, radius: float) -> Optional[STRResult]:
    if str_cache_fresh(lat, lon, radius):
        return STRResult(**json.loads(_str_cache(lat, lon, radius).read_text()))
    return None


def fetch_str_airroi(lat: float, lon: float, radius_miles: float) -> STRResult:
    """Query AirROI polygon search, aggregate listing stats, cache 30 days."""
    if not AIRROI_API_KEY:
        return STRResult(available=False, message="AIRROI_API_KEY not set.")

    polygon = circle_polygon(lat, lon, radius_miles, n_vertices=32)
    headers = {"x-api-key": AIRROI_API_KEY, "Content-Type": "application/json"}
    listings: list[dict] = []
    try:
        for page in range(1, AIRROI_MAX_PAGES + 1):
            body = {
                "polygon": polygon,
                "pagination": {"page": page, "pageSize": AIRROI_PAGE_SIZE},
                "currency": "USD",
            }
            resp = requests.post(AIRROI_POLYGON_URL, json=body, headers=headers, timeout=30)
            if resp.status_code == 402:
                return STRResult(available=False,
                                 message="AirROI credit exhausted — add credits at airroi.com.")
            if resp.status_code in (401, 403):
                return STRResult(available=False, message="AirROI auth failed — check AIRROI_API_KEY.")
            if not resp.ok:
                return STRResult(available=False,
                                 message=f"AirROI HTTP {resp.status_code}: {resp.text[:200]}")
            payload = resp.json()
            batch = _extract_listings(payload)
            if not batch:
                break
            listings.extend(batch)
            if len(batch) < AIRROI_PAGE_SIZE:
                break
    except requests.RequestException as e:
        return STRResult(available=False, message=f"AirROI request failed: {e}")

    # Geofence precisely to the radius (the 32-gon slightly under/over-shoots).
    kept = []
    for L in listings:
        la, lo = L.get("latitude"), L.get("longitude")
        if la is None or lo is None:
            continue
        if haversine_miles(lat, lon, la, lo) <= radius_miles:
            kept.append(L)

    if not kept:
        res = STRResult(available=False, message="STR data unavailable in this market.")
        _str_cache(lat, lon, radius_miles).write_text(json.dumps(asdict(res)))
        return res

    rates = [L.get("ttm_avg_rate") for L in kept if L.get("ttm_avg_rate")]
    ratings = [L.get("rating_overall") for L in kept if L.get("rating_overall")]
    n = len(kept)
    rt = [str(L.get("room_type", "")).lower() for L in kept]
    pct = lambda kind: round(100 * sum(1 for r in rt if kind in r) / n, 1) if n else 0.0
    res = STRResult(
        available=True,
        median_adr=round(_median(rates), 0),
        avg_adr=round(sum(rates) / len(rates), 0) if rates else 0.0,
        listing_count=n,
        pct_entire=pct("entire"),
        pct_private=pct("private"),
        pct_shared=pct("shared"),
        median_rating=round(_median(ratings), 2),
    )
    _str_cache(lat, lon, radius_miles).write_text(json.dumps(asdict(res)))
    return res


# ---------------------------------------------------------------------------
# Score 3 — LSV safety
# ---------------------------------------------------------------------------

def _lsv_cache(lat: float, lon: float, radius: float) -> Path:
    return LSV_DIR / f"{lat:.4f}_{lon:.4f}_r{radius:g}.json"


def _lsv_json_default(o):
    """Coerce numpy scalars (np.float64/np.int64/np.bool_) to native types."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def analyze_lsv_safety(lat: float, lon: float, radius_miles: float) -> LSVResult:
    cache = _lsv_cache(lat, lon, radius_miles)
    if cache.exists():
        try:
            return LSVResult(**json.loads(cache.read_text()))
        except Exception:
            pass  # stale/corrupt cache — fall through and recompute

    radius_m = min(int(radius_miles * METERS_PER_MILE), 8000)
    warnings: list[str] = []

    G = fetch_network(lat, lon, radius_m)
    if G.number_of_edges() == 0:
        return LSVResult(score=0, exit_road_legal=False, exit_road_name="", exit_road_speed=0,
                         vicinity_clean=False, pct_vicinity_forbidden=100.0, connected_km=0,
                         area_15min_km2=0, area_30min_km2=0, area_45min_km2=0,
                         amenities={"restaurants": 0, "shops": 0, "beaches": 0, "total": 0},
                         warnings=["OSM road network is too sparse here to analyze."],
                         subscores={}, sparse=True)

    # 1. Exit road — nearest edge to the property point.
    u, v, k = ox.nearest_edges(G, lon, lat)
    edge_data = G.edges[u, v, k]
    exit_legal = edge_is_lsv_legal(edge_data)
    name = edge_data.get("name", "")
    if isinstance(name, list):
        name = name[0] if name else ""
    exit_speed = edge_speed_mph(edge_data)
    if not exit_legal:
        warnings.append("Exit road is not LSV-legal. Carts cannot safely operate from this property.")

    # 2. Immediate 0.25-mi vicinity.
    try:
        Gv = fetch_network(lat, lon, VICINITY_RADIUS_M)
        vt = sum(d.get("length", 0) for _, _, d in Gv.edges(data=True))
        vl = sum(d.get("length", 0) for _, _, d in Gv.edges(data=True) if edge_is_lsv_legal(d))
        pct_forbidden = round((1 - vl / vt) * 100, 1) if vt > 0 else 100.0
    except Exception:
        pct_forbidden = 100.0
    vicinity_clean = pct_forbidden <= VICINITY_MAX_FORBIDDEN_PCT
    if not vicinity_clean:
        warnings.append(f"{pct_forbidden:.0f}% of roads within 0.25 mi are not LSV-legal.")

    # 3. Connected LSV-legal component reachable from the property.
    H = extract_lsv_subgraph(G)
    start = nearest_connected_node(H, lat, lon)
    reach = (nx.descendants(H, start) | {start}) if start is not None else set()
    connected_km = round(sum(
        d.get("length", 0) for a, b, d in H.edges(data=True)
        if a in reach and b in reach) / 1000, 1)

    # 4. Isochrones.
    n15 = isochrone_node_set(H, start, 15 * 60)
    n30 = isochrone_node_set(H, start, 30 * 60)
    n45 = isochrone_node_set(H, start, 45 * 60)
    area15, area30, area45 = (round(isochrone_area_km2(H, s), 2) for s in (n15, n30, n45))

    iso_coords: list = []
    hull_poly = None
    if len(n30) >= 3:
        hull = MultiPoint([(H.nodes[n]["x"], H.nodes[n]["y"]) for n in n30]).convex_hull
        if hull.geom_type == "Polygon":
            hull_poly = hull
            iso_coords = [[y, x] for x, y in hull.exterior.coords]

    # 5. Amenities reachable (within the 30-min isochrone polygon).
    amen = {"restaurants": 0, "shops": 0, "beaches": 0, "total": 0}
    if hull_poly is not None:
        try:
            feats = ox.features_from_polygon(hull_poly, AMENITY_TAGS)
            if "amenity" in feats.columns:
                amen["restaurants"] = int(feats["amenity"].isin(
                    AMENITY_TAGS["amenity"]).sum())
            if "shop" in feats.columns:
                amen["shops"] = int(feats["shop"].notna().sum())
            if "natural" in feats.columns:
                amen["beaches"] = int((feats["natural"] == "beach").sum())
            amen["total"] = amen["restaurants"] + amen["shops"] + amen["beaches"]
        except Exception:
            pass

    # --- compose score ---
    sub = {
        "exit_road": 30 if exit_legal else 0,
        "vicinity_clean": 20 if vicinity_clean else 0,
        "connected": 20 if connected_km > CONNECTED_MIN_KM else 0,
        "reach_30min": 20 if area30 > AREA_30MIN_MIN_KM2 else 0,
        "amenities": 10 if amen["total"] >= AMENITIES_MIN else 0,
    }
    score = sum(sub.values())
    if not exit_legal:
        score = min(score, 30)
    elif not vicinity_clean:
        score = min(score, 50)

    # Road layers for the map.
    def _gj(graph, predicate):
        edges = [(a, b, kk) for a, b, kk, d in graph.edges(keys=True, data=True) if predicate(d)]
        if not edges:
            return ""
        sub_g = graph.edge_subgraph(edges)
        try:
            return ox.graph_to_gdfs(sub_g, nodes=False).to_json()
        except Exception:
            return ""

    result = LSVResult(
        score=score, exit_road_legal=exit_legal, exit_road_name=name or "(unnamed road)",
        exit_road_speed=round(exit_speed, 0), vicinity_clean=vicinity_clean,
        pct_vicinity_forbidden=pct_forbidden, connected_km=connected_km,
        area_15min_km2=area15, area_30min_km2=area30, area_45min_km2=area45,
        amenities=amen, warnings=warnings, subscores=sub,
        legal_geojson=_gj(G, edge_is_lsv_legal),
        forbidden_geojson=_gj(G, edge_is_forbidden),
        all_geojson=_gj(G, lambda d: True),
        isochrone_coords=iso_coords,
    )
    # Caching is best-effort — a serialization/disk hiccup must never crash the app.
    try:
        cache.write_text(json.dumps(asdict(result), default=_lsv_json_default))
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def score_color(score: float) -> str:
    if score >= 70:
        return "#2e7d32"
    if score >= 40:
        return "#f9a825"
    return "#c62828"


def build_map(lat: float, lon: float, lsv: LSVResult, competitors: dict) -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron",
                   control_scale=True)

    if lsv.all_geojson:
        folium.GeoJson(lsv.all_geojson, name="All roads",
                       style_function=lambda _: {"color": "#bdbdbd", "weight": 1, "opacity": 0.5}
                       ).add_to(m)
    if lsv.isochrone_coords:
        folium.Polygon(lsv.isochrone_coords, color="#2e7d32", weight=1,
                       fill=True, fillColor="#2e7d32", fillOpacity=0.12,
                       name="30-min cart reach", tooltip="30-min cart reach").add_to(m)
    if lsv.legal_geojson:
        folium.GeoJson(lsv.legal_geojson, name="LSV-legal roads",
                       style_function=lambda _: {"color": "#2e7d32", "weight": 3, "opacity": 0.9}
                       ).add_to(m)
    if lsv.forbidden_geojson:
        folium.GeoJson(lsv.forbidden_geojson, name="Forbidden roads",
                       style_function=lambda _: {"color": "#c62828", "weight": 4, "opacity": 0.9}
                       ).add_to(m)

    for c in competitors.get("top5", []):
        folium.CircleMarker(
            [c["lat"], c["lon"]], radius=4 + min(12, c["reviews"] / 50),
            color="#1565c0", weight=1, fillColor="#1e88e5", fillOpacity=0.7,
            tooltip=f"{c['name']} — {c['reviews']} reviews",
        ).add_to(m)

    folium.Marker([lat, lon], tooltip="Property",
                  icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.add_child(MeasureControl(primary_length_unit="miles"))
    return m


# ---------------------------------------------------------------------------
# Pricing bar (custom HTML)
# ---------------------------------------------------------------------------

def pricing_bar_html(adr: float) -> str:
    label, pos = price_band(adr)
    pct = pos * 100
    return f"""
    <div style="margin: 8px 0 4px;">
      <div style="position: relative; height: 22px; border-radius: 4px;
           background: linear-gradient(to right,#66bb6a 0%,#9ccc65 20%,#ffee58 40%,
           #ffa726 65%,#ef5350 100%);">
        <div style="position:absolute; left:{pct:.1f}%; top:-4px; transform:translateX(-50%);
             font-weight:bold; color:#222;">▼</div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:10px; color:#666;">
        <span>$0</span><span>$100</span><span>$200</span><span>$400</span>
        <span>$800</span><span>$1600+</span>
      </div>
      <div style="text-align:center; margin-top:4px; font-weight:600;">{label}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render_results(lat: float, lon: float, resolved: str, radius: float):
    # Competitor + LSV are cheap/cached; STR is gated separately.
    comp_ok = bool(GOOGLE_API_KEY) or _comp_cache(lat, lon, radius).exists()
    with st.spinner("Analyzing roads (first run downloads the OSM network)…"):
        lsv = analyze_lsv_safety(lat, lon, radius)

    competitors = {"score": 0, "count": 0, "top5": []}
    if comp_ok:
        try:
            competitors = score_competitors(lat, lon, radius)
        except Exception as e:
            st.warning(f"Competitor lookup failed: {e}")

    # ---- Map ----
    st.subheader("Map")
    try:
        from streamlit_folium import st_folium
        st_folium(build_map(lat, lon, lsv, competitors), height=480, use_container_width=True)
    except Exception as e:
        st.warning(f"Map could not be rendered: {e}")

    col1, col2, col3 = st.columns(3)

    # ---- Column 1: Competitors ----
    with col1:
        st.markdown("### 🏪 Competitor Activity")
        if not comp_ok:
            st.info("Set `GOOGLE_PLACES_API_KEY` to enable competitor scoring.")
        else:
            st.metric("Score", f"{competitors['score']:.0f} / 100")
            st.caption(f"{competitors['count']} operational rentals within {radius:g} miles")
            for c in competitors["top5"]:
                link = f"[Maps]({c['maps_url']})" if c["maps_url"] else ""
                st.markdown(
                    f"**{c['name']}** — ⭐{c['rating']} ({c['reviews']:,} reviews) · "
                    f"{c['distance_mi']} mi {link}")

    # ---- Column 2: STR pricing ----
    with col2:
        st.markdown("### 🏠 STR Pricing")
        # Cache key survives reruns via session_state, so a fetched result —
        # success OR error — is shown instead of being discarded by a rerun.
        sk = f"str_{lat:.4f}_{lon:.4f}_r{radius:g}"
        cached = load_str_cache(lat, lon, radius) or st.session_state.get(sk)
        if cached is None:
            if not AIRROI_API_KEY:
                st.info("Set `AIRROI_API_KEY` to enable STR pricing. "
                        "[Get a key](https://www.airroi.com/api).")
            else:
                st.caption(f"Not cached. AirROI query ≈ {AIRROI_EST_COST} in credits.")
                if st.button("Fetch STR data from AirROI", key="fetch_str"):
                    with st.spinner("Querying AirROI…"):
                        cached = fetch_str_airroi(lat, lon, radius)
                    st.session_state[sk] = cached
        if cached is not None:
            if not cached.available:
                st.warning(cached.message or "STR data unavailable.")
            else:
                st.metric("Median ADR (TTM)", f"${cached.median_adr:,.0f}")
                st.markdown(pricing_bar_html(cached.median_adr), unsafe_allow_html=True)
                st.caption(f"{cached.listing_count} active listings within {radius:g} mi")
                st.markdown(
                    f"- **{cached.pct_entire:.0f}%** entire homes · "
                    f"{cached.pct_private:.0f}% private · {cached.pct_shared:.0f}% shared\n"
                    f"- Median rating: **⭐{cached.median_rating}**\n"
                    f"- Avg ADR: ${cached.avg_adr:,.0f}")

    # ---- Column 3: LSV safety ----
    with col3:
        st.markdown("### 🛺 LSV Safety")
        st.metric("Score", f"{lsv.score:.0f} / 100")
        if not lsv.exit_road_legal:
            st.error(f"**Exit road: NOT LSV-legal** ⚠️\n\n{lsv.exit_road_name} "
                     f"(~{lsv.exit_road_speed:.0f} mph). Carts cannot safely operate here.")
        else:
            st.success(f"Exit road: LSV-legal ✓ ({lsv.exit_road_name}, ~{lsv.exit_road_speed:.0f} mph)")
        st.markdown(
            f"- First 0.25 mi: {'✅ pass' if lsv.vicinity_clean else '❌ fail'} "
            f"({lsv.pct_vicinity_forbidden:.0f}% not legal)\n"
            f"- Connected network: **{lsv.connected_km} km** of LSV-legal roads\n"
            f"- 30-min reach: **{lsv.area_30min_km2} km²** "
            f"(15/45: {lsv.area_15min_km2}/{lsv.area_45min_km2})\n"
            f"- Amenities in range: 🍽 {lsv.amenities['restaurants']} · "
            f"🛍 {lsv.amenities['shops']} · 🏖 {lsv.amenities['beaches']}")
        if lsv.sparse:
            st.warning("OSM network too sparse for full road analysis.")


def main():
    st.set_page_config(page_title="Property Analyzer", page_icon=LOGO_URL, layout="wide")

    logo_col, title_col = st.columns([1, 4], vertical_alignment="center")
    logo_col.image(LOGO_URL, width=110)
    title_col.markdown("# Property Analyzer")
    st.caption("Score a property for golf cart / LSV rental viability — "
               "competitor activity, STR pricing, and cart route safety.")

    if not GOOGLE_API_KEY:
        st.warning("`GOOGLE_PLACES_API_KEY` not set — competitor scoring will be unavailable.")
    if not AIRROI_API_KEY:
        st.warning("`AIRROI_API_KEY` not set — STR pricing will be unavailable.")

    address = st.text_input("Property address", placeholder="123 Main St, Tybee Island, GA")
    radius = st.slider("Search radius (miles)", 1.0, 10.0, 5.0, 0.5)
    submitted = st.button("Submit", type="primary")

    if submitted and address.strip():
        with st.spinner("Geocoding…"):
            try:
                candidates = geocode(address.strip())
            except Exception:
                st.error("Address lookup (OpenStreetMap Nominatim) is temporarily "
                         "unavailable or rate-limited from the server. Please wait a "
                         "few seconds and try again.")
                st.stop()
        if not candidates:
            st.error("Couldn't find that address (tried OpenStreetMap, then Google). "
                     "Check the spelling or add city/state/ZIP. If this keeps happening, "
                     "make sure the **Geocoding API** is enabled on your Google key.")
            return
        st.session_state["candidates"] = [asdict(c) for c in candidates]
        st.session_state["radius"] = radius

    cands = st.session_state.get("candidates")
    if cands:
        labels = [c["address"] for c in cands]
        idx = 0
        if len(cands) > 1:
            st.info("Multiple matches — confirm which one you meant:")
            choice = st.selectbox("Resolved address", labels, index=0)
            idx = labels.index(choice)
        chosen = cands[idx]
        st.success(f"📍 Resolved: {chosen['address']}")
        render_results(chosen["lat"], chosen["lon"], chosen["address"],
                       st.session_state.get("radius", 5.0))


if __name__ == "__main__":
    main()
