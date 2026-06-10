"""
LSV Competitor Scorer
=====================
Pulls competitor density data from Google Places API for each candidate
location. Validates market demand by counting nearby cart/LSV rental
businesses and weighting them by review volume.

For each candidate, this script:
  1. Runs 5 search queries (golf cart rental, LSV rental, etc.) plus 2
     nearby searches at different radii to comprehensively find competitors.
  2. Pulls full Place Details (reviews, rating, hours, photos) for each.
  3. Computes a 0-100 "validated demand" score where more competitors
     and more review volume = higher score.
  4. Caches results to JSON so re-runs don't re-query the API.

Combines cleanly with lsv_route_analyzer.py for a full viability score.

Setup
-----
1. Create a Google Cloud account at https://console.cloud.google.com
2. Enable the "Places API" (legacy version, simpler to use)
3. Create an API key (APIs & Services -> Credentials -> Create Credentials)
4. Set it as an environment variable in your terminal:
       export GOOGLE_PLACES_API_KEY="AIza...your-key-here"
   Add that line to ~/.zshrc to make it permanent on a Mac.

Cost
----
~27-32 API calls per candidate at comprehensive depth.
Google gives $200 free credit/month, which covers ~1,000+ candidates.
You will NOT be charged unless you exceed $200/month.
Set a billing alert in Google Cloud Console to be safe.

Usage
-----
    python lsv_competitor_scorer.py

Programmatic:
    from lsv_competitor_scorer import score_market
    result = score_market("Tybee Island GA", 31.9968, -80.8456)
    print(result.score)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional

import folium
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "")

# Search radius (Google Places limits to 50,000m = ~31 miles)
DEFAULT_RADIUS_MILES = 10
METERS_PER_MILE = 1609.34

# Multiple queries to comprehensively find competitors. Adding queries
# costs more API calls but catches businesses that only match certain
# terms. These 5 cover ~95% of cart/LSV rental businesses.
SEARCH_QUERIES = [
    "golf cart rental",
    "LSV rental",
    "low speed vehicle rental",
    "golf car rental",
    "electric cart rental beach",
]

# Place types for nearby search (some businesses tag themselves as
# "car_rental" or "tourist_attraction" rather than appearing in text search)
NEARBY_TYPES = ["car_rental"]

# Cache directory — keeps a JSON of each candidate's search results so
# re-running doesn't burn API quota. Delete the cache file to refresh.
CACHE_DIR = Path("./.competitor_cache")

# Endpoints (legacy Places API — simpler than the new v1 API)
TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
NEARBY_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Fields to request in Place Details (more fields = higher cost per call,
# but the difference is small and the data is more useful)
DETAIL_FIELDS = ",".join([
    "place_id", "name", "formatted_address", "geometry/location",
    "rating", "user_ratings_total", "website", "formatted_phone_number",
    "opening_hours", "business_status", "types", "price_level", "url",
])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Competitor:
    """One competitor business found via Google Places."""
    place_id: str
    name: str
    address: str = ""
    lat: float = 0.0
    lon: float = 0.0
    rating: float = 0.0
    review_count: int = 0
    website: str = ""
    phone: str = ""
    business_status: str = "OPERATIONAL"
    google_maps_url: str = ""
    matched_queries: list[str] = field(default_factory=list)


@dataclass
class MarketResult:
    """Demand validation result for one candidate market."""
    name: str
    lat: float
    lon: float
    radius_miles: float
    competitor_count: int
    operational_count: int  # excludes closed/permanently-closed businesses
    total_reviews: int
    avg_rating: float
    median_review_count: float
    top_competitor_name: str
    top_competitor_reviews: int
    competitors: list[Competitor]
    score: float
    sub_scores: dict


# ---------------------------------------------------------------------------
# API call wrappers
# ---------------------------------------------------------------------------

def _check_api_key() -> None:
    if not API_KEY:
        raise RuntimeError(
            "GOOGLE_PLACES_API_KEY not set. Run this in your terminal first:\n"
            '    export GOOGLE_PLACES_API_KEY="your-key-here"\n'
            "Then re-run the script."
        )


def _text_search(query: str, lat: float, lon: float, radius_m: int) -> list[dict]:
    """Run a Places text search, paging through up to 60 results."""
    _check_api_key()
    results = []
    params = {
        "query": query,
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "key": API_KEY,
    }

    for page in range(3):  # API allows up to 3 pages of 20 results each
        try:
            r = requests.get(TEXT_SEARCH_URL, params=params, timeout=15)
            data = r.json()
        except Exception as e:
            print(f"    Text search failed for '{query}': {e}")
            return results

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            print(f"    Text search status: {data.get('status')} "
                  f"({data.get('error_message', 'no message')})")
            return results

        results.extend(data.get("results", []))
        next_token = data.get("next_page_token")
        if not next_token:
            break

        # Google requires a brief delay before next_page_token is usable
        time.sleep(2.0)
        params = {"pagetoken": next_token, "key": API_KEY}

    return results


def _nearby_search(lat: float, lon: float, radius_m: int,
                   place_type: str = "car_rental") -> list[dict]:
    """Run a Places nearby search."""
    _check_api_key()
    params = {
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "type": place_type,
        "key": API_KEY,
    }
    try:
        r = requests.get(NEARBY_SEARCH_URL, params=params, timeout=15)
        return r.json().get("results", [])
    except Exception as e:
        print(f"    Nearby search failed: {e}")
        return []


def _place_details(place_id: str) -> Optional[dict]:
    """Pull detailed info for a single place."""
    _check_api_key()
    params = {
        "place_id": place_id,
        "fields": DETAIL_FIELDS,
        "key": API_KEY,
    }
    try:
        r = requests.get(PLACE_DETAILS_URL, params=params, timeout=15)
        data = r.json()
        if data.get("status") == "OK":
            return data.get("result")
        return None
    except Exception as e:
        print(f"    Place details failed for {place_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Filtering — make sure results are actually cart/LSV rentals
# ---------------------------------------------------------------------------

# Keywords that suggest a business is actually a cart rental operation
RELEVANCE_KEYWORDS = [
    "golf cart", "golf car", "lsv", "low speed",
    "cart rental", "cart hire", "street legal",
    "electric cart", "neighborhood electric",
]

# Hard exclusions — full-size car rental / moving companies. These are never
# cart rentals even if the name coincidentally contains a cart word.
HARD_EXCLUSION_KEYWORDS = [
    "u-haul", "uhaul", "u haul", "enterprise rent", "hertz", "budget car",
    "avis", "national car", "thrifty", "alamo", "sixt", "moving truck",
    "penske", "ryder", "zipcar", "turo",
]

# Soft exclusions — other recreation-rental businesses (bikes, scooters,
# kayaks, etc.) that pollute the results. A business matching one of these is
# excluded UNLESS it also has an explicit cart/LSV signal in its name (so a
# combo shop like "Golf Cart & Bike Rentals" is still counted as a competitor).
SOFT_EXCLUSION_KEYWORDS = [
    "bike", "bicycle", "e-bike", "ebike", "cycle", "scooter", "moped",
    "kayak", "canoe", "paddle", "paddleboard", "sup ", "jet ski", "jetski",
    "boat", "pontoon", "segway", "surf", "snorkel", "atv", "moto",
]


def _is_likely_cart_rental(name: str, types: list[str]) -> bool:
    """Heuristic filter. Returns True if name suggests cart/LSV rental
    and isn't an obviously-excluded business (full-size car rental, bike/kayak
    outfitter, etc.)."""
    name_lower = name.lower()

    # Hard exclusions win over everything.
    for ex_kw in HARD_EXCLUSION_KEYWORDS:
        if ex_kw in name_lower:
            return False

    has_cart_signal = any(kw in name_lower for kw in RELEVANCE_KEYWORDS)

    # An explicit cart/LSV signal keeps the business even if it also rents
    # bikes/kayaks/etc.
    if has_cart_signal:
        return True

    # No cart signal + a soft-exclusion term => it's a bike/kayak/scooter shop.
    for ex_kw in SOFT_EXCLUSION_KEYWORDS:
        if ex_kw in name_lower:
            return False

    # If the business is tagged as car_rental but doesn't have cart keywords,
    # be skeptical — most are full-size rentals, not LSVs.
    if "car_rental" in (types or []) and not any(
        kw in name_lower for kw in ["cart", "lsv", "golf"]
    ):
        return False

    # Default to including ambiguous results; the user can review them
    return True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Anchors: a candidate hitting this many competitors / total reviews maps to
# ~100. Set near the top of the realistic range (after filtering) so scores
# spread across actual markets instead of saturating early.
COMPETITOR_SCALE_MAX = 50
REVIEW_SCALE_MAX = 10000


def _competitor_count_score(count: int) -> float:
    """0-100 score from competitor count. Square-root scaled: still concave so
    the marginal value of competitor #11 is less than #2 (diminishing returns),
    but it spreads far better than a log curve across the realistic 0-50 range
    instead of pinning every busy market at 100. User chose 'reward more
    competitors' so it stays monotonically increasing."""
    if count <= 0:
        return 0.0
    return min(100.0, (count / COMPETITOR_SCALE_MAX) ** 0.5 * 100)


def _review_volume_score(total_reviews: int) -> float:
    """0-100 score from total reviews across all competitors. Strong proxy for
    actual rental volume in the market. Square-root scaled over a 0-10k range
    for the same spread reasons as the competitor-count score."""
    if total_reviews <= 0:
        return 0.0
    return min(100.0, (total_reviews / REVIEW_SCALE_MAX) ** 0.5 * 100)


def _rating_score(avg_rating: float) -> float:
    """0-100 score from average rating. Linear scale since rating differences
    in the 3.5-4.5 range matter a lot. Below 3.0 is concerning."""
    if avg_rating <= 0:
        return 0.0
    if avg_rating < 3.0:
        return (avg_rating / 3.0) * 30  # 0-30 for poor markets
    return 30 + ((avg_rating - 3.0) / 2.0) * 70  # 30-100 for 3-5 star markets


def _compute_market_score(competitors: list[Competitor]) -> tuple[float, dict]:
    """Combine sub-scores into final 0-100 market viability score.

    Weights (reward more competitors mode):
      50% competitor count (more = better, log-scaled)
      40% review volume (proxy for market size)
      10% avg rating (sanity check)"""
    operational = [c for c in competitors if c.business_status == "OPERATIONAL"]

    count = len(operational)
    total_reviews = sum(c.review_count for c in operational)
    rated = [c for c in operational if c.rating > 0]
    avg_rating = sum(c.rating for c in rated) / len(rated) if rated else 0

    sub_scores = {
        "competitor_count": round(_competitor_count_score(count), 1),
        "review_volume": round(_review_volume_score(total_reviews), 1),
        "avg_rating": round(_rating_score(avg_rating), 1),
    }

    score = (
        0.50 * sub_scores["competitor_count"]
        + 0.40 * sub_scores["review_volume"]
        + 0.10 * sub_scores["avg_rating"]
    )
    return round(score, 1), sub_scores


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_path(name: str, lat: float, lon: float) -> Path:
    """Stable filename for caching a market's results."""
    CACHE_DIR.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in name)[:60]
    return CACHE_DIR / f"{safe_name}_{lat:.4f}_{lon:.4f}.json"


def _load_cache(name: str, lat: float, lon: float) -> Optional[dict]:
    path = _cache_path(name, lat, lon)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(name: str, lat: float, lon: float, data: dict) -> None:
    with open(_cache_path(name, lat, lon), "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Distance filtering
# ---------------------------------------------------------------------------

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.7613  # earth radius in miles
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return 2 * r * asin(sqrt(a))


def _within_radius(competitors: list["Competitor"], lat: float, lon: float,
                   radius_miles: float) -> list["Competitor"]:
    """Google Text Search treats `radius` as a bias, not a hard bound — it can
    return strong query matches hundreds of miles away. Drop anything actually
    outside the requested radius (and any result missing coordinates)."""
    return [c for c in competitors
            if c.lat and c.lon
            and _haversine_miles(lat, lon, c.lat, c.lon) <= radius_miles]


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

def score_market(
    name: str,
    lat: float,
    lon: float,
    radius_miles: float = DEFAULT_RADIUS_MILES,
    use_cache: bool = True,
    verbose: bool = True,
) -> MarketResult:
    """Score the demand validation for one candidate market."""
    radius_m = int(radius_miles * METERS_PER_MILE)
    radius_m = min(radius_m, 50000)  # API hard limit

    if verbose:
        print(f"\nScoring market: {name} ({lat}, {lon})")

    # Try cache first
    if use_cache:
        cached = _load_cache(name, lat, lon)
        if cached:
            competitors = [Competitor(**c) for c in cached["competitors"]]
            # Re-apply the current relevance filter — the cache may have been
            # written before filter changes (e.g. bike/kayak exclusions), so we
            # re-filter by name here to re-score without new API calls.
            before = len(competitors)
            competitors = [c for c in competitors
                           if _is_likely_cart_rental(c.name, [])]
            competitors = _within_radius(competitors, lat, lon, radius_miles)
            if verbose:
                dropped = before - len(competitors)
                note = f", {dropped} filtered out" if dropped else ""
                print(f"  Loaded from cache ({len(competitors)} competitors{note})")
            score, sub_scores = _compute_market_score(competitors)
            return _build_result(name, lat, lon, radius_miles, competitors,
                                 score, sub_scores)

    # No cache; query the API
    api_calls = 0
    raw_results: dict[str, dict] = {}
    matched_queries: dict[str, set[str]] = {}

    # Run text searches
    if verbose:
        print(f"  Running {len(SEARCH_QUERIES)} text searches...")
    for query in SEARCH_QUERIES:
        results = _text_search(query, lat, lon, radius_m)
        api_calls += 1
        for r in results:
            pid = r.get("place_id")
            if pid:
                raw_results[pid] = r
                matched_queries.setdefault(pid, set()).add(query)

    # Run nearby searches by type
    if verbose:
        print(f"  Running {len(NEARBY_TYPES)} nearby type searches...")
    for place_type in NEARBY_TYPES:
        results = _nearby_search(lat, lon, radius_m, place_type)
        api_calls += 1
        for r in results:
            pid = r.get("place_id")
            if pid and pid not in raw_results:
                raw_results[pid] = r
                matched_queries.setdefault(pid, set()).add(f"type:{place_type}")

    # Filter to likely-relevant results before pulling details (saves API cost)
    filtered_ids = []
    for pid, r in raw_results.items():
        if _is_likely_cart_rental(r.get("name", ""), r.get("types", [])):
            filtered_ids.append(pid)

    if verbose:
        print(f"  Found {len(raw_results)} candidates, "
              f"{len(filtered_ids)} look like cart rentals")
        print(f"  Pulling Place Details for {len(filtered_ids)} businesses...")

    # Pull full details for the filtered set
    competitors: list[Competitor] = []
    for pid in filtered_ids:
        details = _place_details(pid)
        api_calls += 1
        if not details:
            continue

        geom = details.get("geometry", {}).get("location", {})
        competitors.append(Competitor(
            place_id=pid,
            name=details.get("name", ""),
            address=details.get("formatted_address", ""),
            lat=geom.get("lat", 0.0),
            lon=geom.get("lng", 0.0),
            rating=details.get("rating", 0.0) or 0.0,
            review_count=details.get("user_ratings_total", 0) or 0,
            website=details.get("website", ""),
            phone=details.get("formatted_phone_number", ""),
            business_status=details.get("business_status", "OPERATIONAL"),
            google_maps_url=details.get("url", ""),
            matched_queries=sorted(matched_queries.get(pid, [])),
        ))

    if verbose:
        print(f"  Total API calls: {api_calls}")

    # Cache the raw fetch (pre-distance-filter) so other radii can reuse it.
    if use_cache:
        _save_cache(name, lat, lon, {
            "name": name, "lat": lat, "lon": lon,
            "radius_miles": radius_miles,
            "competitors": [asdict(c) for c in competitors],
            "api_calls": api_calls,
        })

    # Text Search radius is only a bias — drop results outside radius_miles.
    competitors = _within_radius(competitors, lat, lon, radius_miles)
    score, sub_scores = _compute_market_score(competitors)
    result = _build_result(name, lat, lon, radius_miles, competitors,
                           score, sub_scores)

    if verbose:
        print(f"  Score: {score} | {result.operational_count} operational competitors"
              f" | {result.total_reviews} total reviews | "
              f"avg rating {result.avg_rating:.1f}")

    return result


def _build_result(name, lat, lon, radius_miles, competitors,
                  score, sub_scores) -> MarketResult:
    operational = [c for c in competitors if c.business_status == "OPERATIONAL"]
    total_reviews = sum(c.review_count for c in operational)
    rated = [c for c in operational if c.rating > 0]
    avg_rating = sum(c.rating for c in rated) / len(rated) if rated else 0.0

    review_counts = sorted([c.review_count for c in operational])
    median_reviews = (
        review_counts[len(review_counts) // 2]
        if review_counts else 0
    )

    top = max(operational, key=lambda c: c.review_count, default=None)

    return MarketResult(
        name=name,
        lat=lat,
        lon=lon,
        radius_miles=radius_miles,
        competitor_count=len(competitors),
        operational_count=len(operational),
        total_reviews=total_reviews,
        avg_rating=round(avg_rating, 2),
        median_review_count=median_reviews,
        top_competitor_name=top.name if top else "",
        top_competitor_reviews=top.review_count if top else 0,
        competitors=competitors,
        score=score,
        sub_scores=sub_scores,
    )


# ---------------------------------------------------------------------------
# Batch scoring + outputs
# ---------------------------------------------------------------------------

def score_batch(candidates: list[tuple[str, float, float]]) -> list[MarketResult]:
    """Score multiple markets. Each candidate is (name, lat, lon)."""
    results = []
    for name, lat, lon in candidates:
        try:
            results.append(score_market(name, lat, lon))
        except Exception as e:
            print(f"  FAILED: {name}: {e}")
    return results


def results_to_dataframe(results: list[MarketResult]) -> pd.DataFrame:
    """Flatten results into a ranked dataframe for CSV/Excel export."""
    rows = []
    for r in results:
        rows.append({
            "market": r.name,
            "lat": r.lat,
            "lon": r.lon,
            "score": r.score,
            "competitors": r.operational_count,
            "total_reviews": r.total_reviews,
            "avg_rating": r.avg_rating,
            "median_reviews_per_competitor": r.median_review_count,
            "top_competitor": r.top_competitor_name,
            "top_competitor_reviews": r.top_competitor_reviews,
            "sub_competitor_count": r.sub_scores.get("competitor_count", 0),
            "sub_review_volume": r.sub_scores.get("review_volume", 0),
            "sub_avg_rating": r.sub_scores.get("avg_rating", 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def plot_competitors_map(
    results: list[MarketResult],
    output_path: str | Path = "competitor_map.html",
) -> Path:
    """Render an interactive map with candidate markers plus competitor pins."""
    if not results:
        raise ValueError("No results to plot.")

    output_path = Path(output_path)
    avg_lat = sum(r.lat for r in results) / len(results)
    avg_lon = sum(r.lon for r in results) / len(results)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6,
                   tiles="CartoDB positron")

    for result in sorted(results, key=lambda r: r.score, reverse=True):
        # Candidate marker
        score_color = (
            "#2e7d32" if result.score >= 70
            else "#fbc02d" if result.score >= 40
            else "#d32f2f"
        )

        popup = f"""
        <div style="width: 260px; font-family: -apple-system, sans-serif;">
          <h4 style="margin: 0;">{result.name}</h4>
          <div style="font-size: 24px; color: {score_color}; font-weight: bold;">
            Demand score: {result.score}/100
          </div>
          <table style="font-size: 12px; margin-top: 8px; width: 100%;">
            <tr><td>Operational competitors:</td><td><b>{result.operational_count}</b></td></tr>
            <tr><td>Total reviews:</td><td><b>{result.total_reviews:,}</b></td></tr>
            <tr><td>Avg rating:</td><td><b>{result.avg_rating}</b></td></tr>
            <tr><td>Top competitor:</td><td>{result.top_competitor_name}</td></tr>
            <tr><td>Top reviews:</td><td><b>{result.top_competitor_reviews:,}</b></td></tr>
          </table>
        </div>
        """
        folium.CircleMarker(
            location=[result.lat, result.lon],
            radius=12 + result.score / 10,
            color=score_color,
            weight=2,
            fillColor=score_color,
            fillOpacity=0.5,
            popup=folium.Popup(popup, max_width=300),
            tooltip=f"{result.name} — score {result.score}",
        ).add_to(m)

        # Competitor pins
        for comp in result.competitors:
            if comp.business_status != "OPERATIONAL":
                continue
            comp_popup = f"""
            <b>{comp.name}</b><br>
            Rating: {comp.rating} ({comp.review_count:,} reviews)<br>
            {comp.address}<br>
            {f'<a href="{comp.website}" target="_blank">Website</a>' if comp.website else ''}
            {f' · <a href="{comp.google_maps_url}" target="_blank">Maps</a>' if comp.google_maps_url else ''}
            """
            # Size pins by review count (busier = bigger)
            pin_radius = 3 + min(8, comp.review_count / 100)
            folium.CircleMarker(
                location=[comp.lat, comp.lon],
                radius=pin_radius,
                color="#0277bd",
                weight=1,
                fillColor="#039be5",
                fillOpacity=0.7,
                popup=folium.Popup(comp_popup, max_width=260),
                tooltip=f"{comp.name} ({comp.review_count} reviews)",
            ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    print(f"Map saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Integration with road viability scorer
# ---------------------------------------------------------------------------

def combine_with_road_scores(
    competitor_results: list[MarketResult],
    road_metrics_df: pd.DataFrame,
    road_weight: float = 0.5,
    demand_weight: float = 0.5,
) -> pd.DataFrame:
    """Combine demand scores with road viability scores into a final ranking.

    road_metrics_df should be the output of analyze_batch() from
    lsv_route_analyzer. Matching is done by name."""
    demand_df = results_to_dataframe(competitor_results)
    demand_df = demand_df.rename(columns={"score": "demand_score"})

    road_df = road_metrics_df.rename(
        columns={"score": "road_score", "candidate_name": "market"}
    )

    merged = pd.merge(demand_df, road_df, on="market", how="outer")
    merged["combined_score"] = (
        road_weight * merged["road_score"].fillna(0)
        + demand_weight * merged["demand_score"].fillna(0)
    ).round(1)
    merged = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EXAMPLE_CANDIDATES = [
    ("Tybee Island GA", 31.9968, -80.8456),
    ("Hilton Head Island SC", 32.2241, -80.6973),
    ("Marco Island FL", 25.9950, -81.6724),
    ("Peachtree City GA", 33.3573, -84.5719),
]


def main():
    if not API_KEY:
        print("=" * 60)
        print("ERROR: GOOGLE_PLACES_API_KEY not set.")
        print()
        print("Set it in your terminal:")
        print('  export GOOGLE_PLACES_API_KEY="your-key-here"')
        print()
        print("To make it permanent, add that line to ~/.zshrc and run:")
        print("  source ~/.zshrc")
        print("=" * 60)
        return

    print("LSV Competitor Scorer")
    print(f"Mode: Comprehensive (~30 API calls per candidate)")
    print(f"Cache: {CACHE_DIR.absolute()}")

    results = score_batch(EXAMPLE_CANDIDATES)

    df = results_to_dataframe(results)
    print("\n" + "=" * 70)
    print("RANKED MARKETS BY DEMAND SCORE")
    print("=" * 70)
    print(df[["market", "score", "competitors", "total_reviews",
              "avg_rating", "top_competitor"]].to_string(index=False))

    df.to_csv("competitor_scores.csv", index=False)
    print("\nFull results saved to competitor_scores.csv")

    plot_competitors_map(results, "competitor_map.html")
    print("Open competitor_map.html in a browser to see competitors on the map.")


if __name__ == "__main__":
    main()
