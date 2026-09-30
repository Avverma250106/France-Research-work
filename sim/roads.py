"""Street routing over a locally-built road graph.

The public routing services (OSRM's demo server, Valhalla) are not reachable
from every environment, and depending on one during a live demo is a liability
anyway. So the road network is fetched once from Overpass, reduced to a graph,
and cached in artifacts/roads.json; routing then runs locally with networkx and
needs no network at all.

Build the cache:   python3 -m sim.roads
Runtime:           RoadNetwork.load() -> .route((lon,lat), (lon,lat))

If the cache is absent the simulation falls back to approximated paths, so this
is an enhancement rather than a dependency.
"""

import json
import math
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import networkx as nx

ART = Path("artifacts")
CACHE = ART / "roads.json"
UA = {"User-Agent": "france-research-sim/0.1 (research)"}
OVERPASS = "https://overpass-api.de/api/interpreter"

HIGHWAYS = ("motorway|motorway_link|trunk|trunk_link|primary|primary_link"
            "|secondary|secondary_link|tertiary|tertiary_link")

# Ile-de-France core, split small enough that Overpass does not time out.
TILES = [
    (48.80, 2.20, 48.86, 2.33), (48.80, 2.33, 48.86, 2.46),
    (48.86, 2.20, 48.89, 2.33), (48.89, 2.20, 48.92, 2.33),
    (48.86, 2.33, 48.89, 2.395), (48.86, 2.395, 48.89, 2.46),
    (48.89, 2.33, 48.92, 2.395), (48.89, 2.395, 48.92, 2.46),
]

GRID = 0.01  # degrees, bucket size for nearest-node lookup


def _haversine(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(h))


def fetch_ways(tiles=TILES, tries=4, pause=4.0, verbose=True):
    """Pull drivable ways from Overpass, one tile at a time."""
    out = []
    for i, (s, w, n, e) in enumerate(tiles, 1):
        body = (f'[out:json][timeout:90];way[highway~"^({HIGHWAYS})$"]'
                f'({s},{w},{n},{e});out geom;')
        got = []
        for attempt in range(tries):
            try:
                req = urllib.request.Request(OVERPASS, data=body.encode(), headers=UA)
                with urllib.request.urlopen(req, timeout=150) as resp:
                    got = json.load(resp).get("elements", [])
                break
            except Exception as exc:
                if verbose:
                    print(f"    tile {i} try {attempt + 1}: {str(exc)[:60]}", flush=True)
                time.sleep(pause * 3)
        if verbose:
            print(f"  tile {i}/{len(tiles)}: {len(got)} ways", flush=True)
        out += got
        time.sleep(pause)
    return out


def build_graph(ways):
    """Ways -> undirected graph keyed on rounded coordinates.

    OSM shares exact node coordinates at intersections, so rounding to 6
    decimals preserves connectivity while collapsing float noise.
    """
    G = nx.Graph()
    for w in ways:
        geom = w.get("geometry") or []
        prev = None
        for p in geom:
            node = (round(p["lon"], 6), round(p["lat"], 6))
            if prev is not None and node != prev:
                G.add_edge(prev, node, weight=_haversine(prev, node))
            prev = node
    if G.number_of_nodes() == 0:
        return G
    # Keep the largest connected component; islands make routing fail randomly.
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def contract(G):
    """Collapse degree-2 chains into single edges that carry their geometry.

    The raw graph is ~65k nodes, nearly all of them mid-street shape points, and
    Dijkstra over that takes ~150ms per route - enough to stutter the event loop
    at demo speed. Contracting to junctions only cuts it by an order of
    magnitude while the stored geometry keeps vehicles on the real street line.
    """
    H = nx.Graph()
    junctions = {n for n in G.nodes() if G.degree(n) != 2}
    if not junctions:
        return G

    seen = set()
    for j in junctions:
        for nbr in G.neighbors(j):
            if (j, nbr) in seen:
                continue
            geom = [j]
            weight = 0.0
            prev, cur = j, nbr
            while True:
                weight += G[prev][cur]["weight"]
                geom.append(cur)
                if cur in junctions or cur == j:
                    break
                nxts = [x for x in G.neighbors(cur) if x != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
            seen.add((j, nbr))
            seen.add((cur, geom[-2] if len(geom) > 1 else j))
            if cur == j or weight <= 0:
                continue
            if H.has_edge(j, cur) and H[j][cur]["weight"] <= weight:
                continue
            H.add_edge(j, cur, weight=weight, geom=[list(p) for p in geom])
    return H


def save(G, path=CACHE):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = []
    for a, b, d in G.edges(data=True):
        geom = d.get("geom")
        edges.append([idx[a], idx[b], round(d["weight"], 1),
                      [[round(x, 6), round(y, 6)] for x, y in geom] if geom else None])
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"nodes": [list(n) for n in nodes], "edges": edges}))
    return path


class RoadNetwork:
    def __init__(self, G):
        self.G = G
        self.buckets = defaultdict(list)
        for n in G.nodes():
            self.buckets[(int(n[0] / GRID), int(n[1] / GRID))].append(n)
        self._cache = {}

    @classmethod
    def load(cls, path=CACHE):
        path = Path(path)
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            G = nx.Graph()
            nodes = [tuple(n) for n in d["nodes"]]
            G.add_nodes_from(nodes)
            for rec in d["edges"]:
                a, b, w = rec[0], rec[1], rec[2]
                geom = rec[3] if len(rec) > 3 else None
                G.add_edge(nodes[a], nodes[b], weight=w, geom=geom)
            return cls(G)
        except Exception:
            return None

    def nearest(self, pt):
        """Nearest graph node, searching outward through grid buckets."""
        cx, cy = int(pt[0] / GRID), int(pt[1] / GRID)
        for ring in range(0, 8):
            cands = []
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    if ring and max(abs(dx), abs(dy)) != ring:
                        continue
                    cands += self.buckets.get((cx + dx, cy + dy), ())
            if cands:
                return min(cands, key=lambda n: (n[0] - pt[0]) ** 2 + (n[1] - pt[1]) ** 2)
        return None

    def route(self, origin, dest):
        """Street path between two lon/lat points, or None if unroutable."""
        a, b = self.nearest(origin), self.nearest(dest)
        if a is None or b is None or a == b:
            return None
        key = (a, b)
        if key in self._cache:
            coords = self._cache[key]
        else:
            try:
                nodes = nx.shortest_path(self.G, a, b, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
            coords = self._expand(nodes)
            if len(self._cache) < 3000:
                self._cache[key] = coords
        return [list(origin)] + [list(c) for c in coords] + [list(dest)]

    def _expand(self, nodes):
        """Walk the junction path, splicing in each edge's stored geometry."""
        out = []
        for u, v in zip(nodes, nodes[1:]):
            geom = self.G[u][v].get("geom")
            seg = [tuple(p) for p in geom] if geom else [u, v]
            if seg and tuple(seg[0]) != tuple(u):
                seg = seg[::-1]
            if out and seg and out[-1] == seg[0]:
                seg = seg[1:]
            out += seg
        return out or list(nodes)


def main():
    print("Fetching drivable road network from Overpass...")
    ways = fetch_ways()
    print(f"  {len(ways)} ways")
    G = build_graph(ways)
    print(f"  raw graph : {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    G = contract(G)
    print(f"  contracted: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    if G.number_of_nodes() < 1000:
        print("  ERROR: graph too small to route on - not saving")
        raise SystemExit(1)
    p = save(G)
    print(f"  saved {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
