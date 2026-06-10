"""
LSV Route Viability Analyzer
=============================
Scores candidate locations (airports, marinas) for golf cart / LSV rental
viability based on the surrounding road network.

For each candidate, this script answers:
  1. What % of nearby roads are LSV-legal (<= 35 mph)?
  2. Is the LSV-legal network connected, or fragmented?
  3. How far can a cart reach in 15 / 30 / 45 minutes?
  4. If you provide destinations (vacation rental clusters, beaches, etc.),
     are they actually reachable by an LSV-legal path?

Usage
-----
    pip install osmnx networkx geopandas folium pandas shapely

    python lsv_route_analyzer.py

Or import and use programmatically:
    from lsv_route_analyzer import analyze_location, Candidate

    candidate = Candidate(
        name="Tybee Island GA - Airport",
        lat=31.99, lon=-80.85,
        destinations=[(31.98, -80.84), (32.00, -80.86)],
    )
    result = analyze_location(candidate)
    print(result.score, result.metrics)

Notes
-----
- OSM speed limit data is incomplete in small markets. The script falls
  back on road-type heuristics (residential = 25 mph, etc.). Trust the
  output for relative ranking; verify top candidates with Street View
  and local knowledge before committing.
- Default LSV speed threshold is 35 mph (matches FL/GA/SC/NC law).
  Adjust LSV_MAX_SPEED_MPH for other jurisdictions.
- Runtime is ~30-90 seconds per candidate depending on radius and
  network density. For 100+ candidates, parallelize or cache the
  graph downloads.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import folium
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LSV_MAX_SPEED_MPH = 35
"""Speed threshold for LSV-legal roads. 35 mph matches FL/GA/SC/NC law."""

LSV_CRUISE_MPH = 25
"""Realistic average travel speed for a golf cart / LSV, used for
isochrone calculations. Carts top out at 25 mph by design."""

DEFAULT_RADIUS_M = 5000
"""Default analysis radius around each candidate, in meters (~3.1 miles)."""

# Highway types that LSVs are categorically banned from, regardless of
# the posted speed limit. Even if OSM tags it as 25 mph, you can't ride
# a cart on a motorway shoulder.
FORBIDDEN_HIGHWAY_TYPES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "cycleway", "footway", "pedestrian", "path", "steps", "bridleway",
}

# Fallback speed assumptions (mph) when OSM doesn't have a maxspeed tag.
# These are based on US norms; adjust if you're analyzing other countries.
HIGHWAY_DEFAULT_SPEED_MPH = {
    "motorway": 70,
    "motorway_link": 50,
    "trunk": 55,
    "trunk_link": 40,
    "primary": 45,
    "primary_link": 35,
    "secondary": 40,
    "secondary_link": 30,
    "tertiary": 30,
    "tertiary_link": 25,
    "unclassified": 25,
    "residential": 25,
    "living_street": 15,
    "service": 15,
    "track": 20,
    "road": 30,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A candidate location to score (airport, marina, etc.)."""
    name: str
    lat: float
    lon: float
    category: str = "unknown"  # "airport", "marina", etc.
    destinations: list[tuple[float, float]] = field(default_factory=list)
    """Optional list of (lat, lon) destinations the LSV should be able to
    reach — e.g. vacation rental clusters, downtown, beach access."""


@dataclass
class AnalysisResult:
    """Output of analyzing one candidate."""
    candidate: Candidate
    metrics: dict
    score: float
    graph: nx.MultiDiGraph
    lsv_subgraph: nx.MultiDiGraph


# ---------------------------------------------------------------------------
# Speed limit parsing
# ---------------------------------------------------------------------------

_MAXSPEED_RE = re.compile(r"(\d+)\s*(mph|kmh|kph|km/h)?", re.IGNORECASE)


def parse_maxspeed_mph(maxspeed) -> float | None:
    """Parse an OSM maxspeed tag into mph.

    OSM stores maxspeed as strings like '25 mph', '40', '50 kmh', or
    sometimes as a list when an edge has multiple values. Returns None
    if it can't be parsed."""
    if maxspeed is None:
        return None
    if isinstance(maxspeed, list):
        # Edges sometimes have multiple maxspeed values; take the lowest
        # (most conservative) one.
        parsed = [parse_maxspeed_mph(m) for m in maxspeed]
        parsed = [p for p in parsed if p is not None]
        return min(parsed) if parsed else None
    if isinstance(maxspeed, (int, float)):
        return float(maxspeed)  # Assume mph if numeric

    m = _MAXSPEED_RE.search(str(maxspeed))
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "mph").lower()
    if unit in ("kmh", "kph", "km/h"):
        value = value * 0.621371  # km/h -> mph
    return value


def edge_speed_mph(data: dict) -> float:
    """Estimate the speed limit of an edge in mph.

    Uses the explicit maxspeed tag if present; otherwise falls back to
    a default based on the highway type. Returns a conservative high
    speed (50) if we can't classify the road at all."""
    explicit = parse_maxspeed_mph(data.get("maxspeed"))
    if explicit is not None:
        return explicit

    hwy = data.get("highway")
    # When an edge spans multiple road segments OSMnx returns a list.
    if isinstance(hwy, list):
        # Use the slowest default — typically the most restrictive segment.
        defaults = [HIGHWAY_DEFAULT_SPEED_MPH.get(h, 50) for h in hwy]
        return min(defaults) if defaults else 50
    return HIGHWAY_DEFAULT_SPEED_MPH.get(hwy, 50)


def edge_is_forbidden(data: dict) -> bool:
    """Returns True if the edge is on a road type LSVs cannot use."""
    hwy = data.get("highway")
    if isinstance(hwy, list):
        return any(h in FORBIDDEN_HIGHWAY_TYPES for h in hwy)
    return hwy in FORBIDDEN_HIGHWAY_TYPES


def edge_is_lsv_legal(data: dict) -> bool:
    """LSV-legal = not on a forbidden road type AND speed <= threshold."""
    if edge_is_forbidden(data):
        return False
    return edge_speed_mph(data) <= LSV_MAX_SPEED_MPH


# ---------------------------------------------------------------------------
# Network analysis
# ---------------------------------------------------------------------------

def fetch_network(lat: float, lon: float, radius_m: int = DEFAULT_RADIUS_M) -> nx.MultiDiGraph:
    """Download the drive network around a point."""
    G = ox.graph_from_point(
        (lat, lon),
        dist=radius_m,
        network_type="drive",
        simplify=True,
    )
    # Add travel times for routing. We'll override speeds with our LSV
    # cruise speed when computing cart-reachability later.
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def extract_lsv_subgraph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return a copy of G with only LSV-legal edges."""
    edges_to_keep = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
        if edge_is_lsv_legal(data)
    ]
    H = G.edge_subgraph(edges_to_keep).copy()
    # Override speed for cart-realistic travel time
    cart_speed_kph = LSV_CRUISE_MPH * 1.60934
    for u, v, k, data in H.edges(keys=True, data=True):
        length_m = data.get("length", 0)
        data["lsv_travel_time_s"] = (length_m / 1000) / cart_speed_kph * 3600
    return H


def total_edge_length_m(G: nx.MultiDiGraph) -> float:
    return sum(data.get("length", 0) for _, _, data in G.edges(data=True))


def largest_strongly_connected_component(G: nx.MultiDiGraph) -> set:
    """Return the node set of the largest strongly connected component.

    Strongly connected = can drive both ways. For LSV analysis this
    matters because one-way roads can fragment the network."""
    if G.number_of_nodes() == 0:
        return set()
    components = nx.strongly_connected_components(G)
    return max(components, key=len, default=set())


def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float):
    """Find the OSMnx node closest to a lat/lon point. Returns None if
    no nodes exist or the point isn't near the network."""
    if G.number_of_nodes() == 0:
        return None
    try:
        return ox.nearest_nodes(G, lon, lat)  # note: lon first
    except Exception as e:
        # Don't silently swallow — a masked error here (e.g. OSMnx 2.x needs
        # scikit-learn to search an unprojected graph) zeroes out every
        # isochrone and quietly tanks the score. Surface it, then degrade.
        warnings.warn(f"nearest_node failed ({type(e).__name__}: {e})")
        return None


def nearest_connected_node(G: nx.MultiDiGraph, lat: float, lon: float):
    """Nearest node that lies in G's largest strongly connected component.

    Snapping to the raw nearest node is fragile: a candidate sitting on an
    isolated stub (e.g. an airport access road) lands in a singleton component,
    so a directed isochrone from it reaches nothing and the area collapses to 0.
    Restricting the search to the main connected network gives a meaningful
    reachable area. Falls back to the plain nearest node if there's no
    sizeable component."""
    main = largest_strongly_connected_component(G)
    if len(main) < 2:
        return nearest_node(G, lat, lon)
    return nearest_node(G.subgraph(main), lat, lon)


def isochrone_node_set(G: nx.MultiDiGraph, start_node, max_seconds: float) -> set:
    """Return the set of nodes reachable from start_node within max_seconds
    of travel along LSV-legal roads at cart cruise speed."""
    if start_node is None or start_node not in G:
        return set()
    distances = nx.single_source_dijkstra_path_length(
        G, start_node, cutoff=max_seconds, weight="lsv_travel_time_s"
    )
    return set(distances.keys())


def isochrone_area_km2(G: nx.MultiDiGraph, node_set: set) -> float:
    """Estimate the area covered by a set of reachable nodes (km²).

    Uses the convex hull of node coordinates. Rough but useful for
    relative comparison between candidates."""
    if len(node_set) < 3:
        return 0.0
    points = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_set]
    gdf = ox.projection.project_gdf(
        ox.utils_geo._build_points_gdf(points) if hasattr(ox.utils_geo, "_build_points_gdf")
        else __import__("geopandas").GeoDataFrame(geometry=points, crs="EPSG:4326")
    )
    hull = gdf.unary_union.convex_hull
    return hull.area / 1_000_000  # m² to km²


def destinations_reachable(
    G: nx.MultiDiGraph,
    start_node,
    destinations: Iterable[tuple[float, float]],
    max_seconds: float = 1800,  # 30 minutes
) -> tuple[int, int]:
    """Returns (reachable_count, total_count) for a list of (lat, lon)
    destinations, checking whether each is reachable from start_node
    within max_seconds via LSV-legal roads."""
    reachable_nodes = isochrone_node_set(G, start_node, max_seconds)
    if not reachable_nodes:
        return 0, sum(1 for _ in destinations)

    reached = 0
    total = 0
    for lat, lon in destinations:
        total += 1
        dest_node = nearest_node(G, lat, lon)
        if dest_node is not None and dest_node in reachable_nodes:
            reached += 1
    return reached, total


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(metrics: dict) -> float:
    """Combine raw metrics into a 0–100 viability score.

    Weights:
      - 30% road coverage (% of roads that are LSV-legal)
      - 25% connectivity (largest connected component as % of LSV-legal)
      - 20% reachable area at 30 minutes
      - 25% destination reachability (if destinations provided)
              fallback: extra weight on connectivity if no destinations

    Tune these for your business judgment — these are defaults."""
    pct_legal = metrics["pct_lsv_legal"]
    pct_connected = metrics["pct_largest_component"]
    area_30min = metrics["reachable_area_30min_km2"]
    dest_pct = metrics.get("destinations_pct_reachable")

    # Normalize 30-min reachable area against an aspirational benchmark
    # (a perfectly-connected 5 mile radius is ~80 km²; we cap at 50).
    area_score = min(area_30min / 50.0, 1.0) * 100

    if dest_pct is not None:
        score = (
            0.30 * pct_legal
            + 0.25 * pct_connected
            + 0.20 * area_score
            + 0.25 * dest_pct
        )
    else:
        score = (
            0.35 * pct_legal
            + 0.40 * pct_connected
            + 0.25 * area_score
        )
    return round(score, 1)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_location(
    candidate: Candidate,
    radius_m: int = DEFAULT_RADIUS_M,
) -> AnalysisResult:
    """Analyze one candidate location end-to-end."""
    print(f"\nAnalyzing: {candidate.name}")
    print(f"  Fetching road network ({radius_m}m radius)...")
    G = fetch_network(candidate.lat, candidate.lon, radius_m)

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Filtering to LSV-legal roads (<= {LSV_MAX_SPEED_MPH} mph)...")
    H = extract_lsv_subgraph(G)

    total_length = total_edge_length_m(G)
    lsv_length = total_edge_length_m(H)
    pct_legal = (lsv_length / total_length * 100) if total_length > 0 else 0

    largest_cc = largest_strongly_connected_component(H)
    pct_largest = (
        sum(
            data.get("length", 0)
            for u, v, data in H.edges(data=True)
            if u in largest_cc and v in largest_cc
        ) / lsv_length * 100
    ) if lsv_length > 0 else 0

    start_node = nearest_connected_node(H, candidate.lat, candidate.lon)
    # Only count area reachable from the start node, on connected legal roads
    nodes_15 = isochrone_node_set(H, start_node, 15 * 60)
    nodes_30 = isochrone_node_set(H, start_node, 30 * 60)
    nodes_45 = isochrone_node_set(H, start_node, 45 * 60)

    area_15 = isochrone_area_km2(H, nodes_15)
    area_30 = isochrone_area_km2(H, nodes_30)
    area_45 = isochrone_area_km2(H, nodes_45)

    dest_metrics = {}
    if candidate.destinations:
        reached, total = destinations_reachable(
            H, start_node, candidate.destinations, max_seconds=30 * 60
        )
        dest_metrics["destinations_reached"] = reached
        dest_metrics["destinations_total"] = total
        dest_metrics["destinations_pct_reachable"] = (
            (reached / total * 100) if total > 0 else 0
        )

    metrics = {
        "candidate_name": candidate.name,
        "total_road_length_km": round(total_length / 1000, 2),
        "lsv_legal_road_length_km": round(lsv_length / 1000, 2),
        "pct_lsv_legal": round(pct_legal, 1),
        "pct_largest_component": round(pct_largest, 1),
        "reachable_area_15min_km2": round(area_15, 2),
        "reachable_area_30min_km2": round(area_30, 2),
        "reachable_area_45min_km2": round(area_45, 2),
        "nodes_reachable_30min": len(nodes_30),
        **dest_metrics,
    }
    score = compute_score(metrics)
    metrics["score"] = score

    print(f"  Score: {score} | LSV-legal: {pct_legal:.1f}% | "
          f"30-min reach: {area_30:.1f} km²")

    return AnalysisResult(
        candidate=candidate,
        metrics=metrics,
        score=score,
        graph=G,
        lsv_subgraph=H,
    )


def analyze_batch(candidates: list[Candidate], radius_m: int = DEFAULT_RADIUS_M) -> pd.DataFrame:
    """Analyze a batch of candidates and return a ranked DataFrame."""
    results = []
    for c in candidates:
        try:
            result = analyze_location(c, radius_m=radius_m)
            results.append(result.metrics)
        except Exception as e:
            print(f"  FAILED: {c.name} ({e})")
            results.append({"candidate_name": c.name, "score": 0, "error": str(e)})

    df = pd.DataFrame(results)
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_result(result: AnalysisResult, output_path: str | Path = "lsv_map.html") -> Path:
    """Render an interactive map showing the road network, LSV-legal subset,
    and isochrones around the candidate location."""
    c = result.candidate
    H = result.lsv_subgraph
    G = result.graph

    m = folium.Map(location=[c.lat, c.lon], zoom_start=14, tiles="OpenStreetMap")

    # Draw all roads in gray
    edges_all = ox.graph_to_gdfs(G, nodes=False)
    folium.GeoJson(
        edges_all,
        style_function=lambda _: {"color": "#999999", "weight": 1.5, "opacity": 0.6},
        name="All roads",
    ).add_to(m)

    # Overlay LSV-legal roads in green
    if H.number_of_edges() > 0:
        edges_lsv = ox.graph_to_gdfs(H, nodes=False)
        folium.GeoJson(
            edges_lsv,
            style_function=lambda _: {"color": "#2e7d32", "weight": 3, "opacity": 0.9},
            name="LSV-legal roads",
        ).add_to(m)

    # Candidate marker
    folium.Marker(
        [c.lat, c.lon],
        popup=f"<b>{c.name}</b><br>Score: {result.score}<br>"
              f"LSV-legal: {result.metrics['pct_lsv_legal']}%",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    # Destination markers
    for i, (lat, lon) in enumerate(c.destinations):
        folium.Marker(
            [lat, lon],
            popup=f"Destination {i + 1}",
            icon=folium.Icon(color="red", icon="flag"),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    output_path = Path(output_path)
    m.save(str(output_path))
    print(f"  Map saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

EXAMPLE_CANDIDATES = [
    Candidate(
        name="Tybee Island GA",
        lat=31.9968, lon=-80.8456,
        category="beach_town",
        destinations=[
            (31.9897, -80.8434),  # Tybee pier / downtown
            (32.0083, -80.8390),  # North beach
        ],
    ),
    Candidate(
        name="Hilton Head Island Airport SC",
        lat=32.2241, lon=-80.6973,
        category="airport",
        destinations=[
            (32.1838, -80.7414),  # Sea Pines area
            (32.1654, -80.7826),  # Forest Beach
        ],
    ),
    Candidate(
        name="Marco Island Executive Airport FL",
        lat=25.9950, lon=-81.6724,
        category="airport",
        destinations=[
            (25.9344, -81.7187),  # South Beach
            (25.9744, -81.7194),  # Marco downtown
        ],
    ),
    Candidate(
        name="Peachtree City GA (Falcon Field Airport)",
        lat=33.3573, lon=-84.5719,
        category="airport",
        destinations=[
            (33.3967, -84.5957),  # Peachtree City downtown
            (33.3733, -84.5950),  # Aberdeen Village
        ],
    ),
]


def main():
    """Run the analysis on the example candidates and save results."""
    print(f"LSV Route Analyzer | threshold = {LSV_MAX_SPEED_MPH} mph | "
          f"radius = {DEFAULT_RADIUS_M}m")

    df = analyze_batch(EXAMPLE_CANDIDATES)
    print("\n" + "=" * 70)
    print("RANKED RESULTS")
    print("=" * 70)
    print(df[["candidate_name", "score", "pct_lsv_legal", "pct_largest_component",
              "reachable_area_30min_km2"]].to_string(index=False))

    df.to_csv("lsv_analysis_results.csv", index=False)
    print("\nFull results saved to lsv_analysis_results.csv")

    # Generate a map for the top-scoring candidate
    if len(EXAMPLE_CANDIDATES) > 0:
        top_name = df.iloc[0]["candidate_name"]
        top_candidate = next(c for c in EXAMPLE_CANDIDATES if c.name == top_name)
        print(f"\nGenerating map for top candidate: {top_name}")
        result = analyze_location(top_candidate)
        plot_result(result, f"map_{top_name.replace(' ', '_').replace('/', '_')}.html")


if __name__ == "__main__":
    main()
