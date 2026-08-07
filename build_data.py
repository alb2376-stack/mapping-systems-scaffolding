"""
Rebuild data.js for the sidewalk shed website, using the same pandas/geopandas
pipeline as the Final Presentation notebook.

Run with the notebook's own environment:
    C:\\Users\\aqila\\anaconda3\\envs\\cdp\\python.exe build_data.py
"""

import json
import os

import numpy as np
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------- paths
DATA = r"C:\Users\aqila\OneDrive - Syracuse University\Documents\GitHub\cdp-mapping-systems-aqila\content\Assignments\Final Project\Data"
CSV = os.path.join(DATA, "Active_Sheds2.csv")
DISTRICTS = os.path.join(DATA, "geo_export_e5c4913d-d23e-47c5-ad1c-2bbe098562d8.shp")
PLUTO = os.path.join(DATA, "MapPLUTO_UNCLIPPED.shp")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.js")

# ---------------------------------------------------------------- cost model
RATE = 150          # $ per linear foot (installation + first 3 months)
MONTHLY_PCT = 0.06  # monthly rental after month 3 = ~6% of installation cost
FREE_MONTHS = 3

SIMPLIFY_TOL = 0.0002  # degrees; keeps data.js small without visible distortion

board_names = {
    101: "Battery Park City, Greenwich Village, Financial District, Tribeca",
    102: "Greenwich Village, SoHo, NoHo, Little Italy, Chinatown",
    103: "Lower East Side, Chinatown, East Village",
    104: "Chelsea, Clinton/Hell's Kitchen",
    105: "Midtown, Flatiron, Union Square",
    106: "Murray Hill, Gramercy, Stuyvesant Town, Turtle Bay",
    107: "Upper West Side, Lincoln Square, Manhattan Valley",
    108: "Upper East Side, Yorkville, Carnegie Hill",
    109: "Morningside Heights, Manhattanville, Hamilton Heights, West Harlem",
    110: "Central Harlem",
    111: "East Harlem",
    112: "Washington Heights, Inwood",
    201: "Mott Haven, Melrose, Port Morris",
    202: "Hunts Point, Longwood",
    203: "Morrisania, Crotona Park East",
    204: "Highbridge, Concourse Village",
    205: "Fordham, University Heights, Morris Heights",
    206: "East Tremont, Belmont, West Farms",
    207: "Kingsbridge Heights, Bedford Park, Norwood",
    208: "Riverdale, Fieldston, Kingsbridge",
    209: "Soundview, Parkchester, Castle Hill",
    210: "Throgs Neck, Co-op City, Pelham Bay",
    211: "Morris Park, Pelham Parkway, Laconia",
    212: "Williamsbridge, Baychester, Wakefield",
    301: "Greenpoint, Williamsburg",
    302: "Fort Greene, Clinton Hill",
    303: "Bedford-Stuyvesant",
    304: "Bushwick",
    305: "East New York, Starrett City, Cypress Hills",
    306: "Park Slope, Carroll Gardens, Red Hook, Cobble Hill",
    307: "Sunset Park, Windsor Terrace",
    308: "Crown Heights North, Prospect Heights",
    309: "Crown Heights South, Wingate, Lefferts Gardens",
    310: "Bay Ridge, Dyker Heights, Fort Hamilton",
    311: "Bensonhurst, Bath Beach",
    312: "Borough Park, Kensington, Ocean Parkway",
    313: "Coney Island, Brighton Beach, Sea Gate",
    314: "Flatbush, Midwood",
    315: "Sheepshead Bay, Gerritsen Beach, Manhattan Beach",
    316: "Brownsville, Ocean Hill",
    317: "East Flatbush, Rugby, Farragut",
    318: "Canarsie, Flatlands",
    401: "Astoria, Long Island City",
    402: "Sunnyside, Woodside",
    403: "Jackson Heights, East Elmhurst",
    404: "Elmhurst, Corona",
    405: "Ridgewood, Glendale, Maspeth, Middle Village",
    406: "Rego Park, Forest Hills",
    407: "Flushing, Murray Hill, Whitestone",
    408: "Fresh Meadows, Briarwood, Hillcrest, Jamaica Hills",
    409: "Kew Gardens, Richmond Hill, Woodhaven",
    410: "South Ozone Park, Ozone Park, Howard Beach",
    411: "Bayside, Douglaston, Little Neck",
    412: "Jamaica, St. Albans, Hollis",
    413: "Queens Village, Cambria Heights, Bellerose",
    414: "Rockaway, Broad Channel, Far Rockaway",
    501: "St. George, Stapleton, Port Richmond",
    502: "South Beach, Willowbrook, New Dorp",
    503: "Tottenville, Great Kills, Annadale",
}

# ---------------------------------------------------------------- 1. sheds
print("Reading sheds CSV...")
df = pd.read_csv(CSV)

df["Community Board Name"] = df["Community Board"].map(
    lambda code: board_names.get(int(code), f"Unknown/Park ({int(code)})")
)

df["First Permit Date"] = pd.to_datetime(df["First Permit Date"], errors="coerce")
df["Permit Expiration Date"] = pd.to_datetime(df["Permit Expiration Date"], errors="coerce")
df["Permit Days"] = (df["Permit Expiration Date"] - df["First Permit Date"]).dt.days

sheds_data = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Longitude Point"], df["Latitude Point"]),
    crs="EPSG:4326",
)

# keep only real community districts (drops parks/airports like 164, 355, 481)
sheds_data = sheds_data[sheds_data["Community Board"].isin(board_names.keys())]
sheds_data = sheds_data.dropna(subset=["Permit Days", "Sidewalk Shed/Linear Feet"])
print(f"  {len(sheds_data):,} sheds after filtering")

# cost model
sheds_data["Months Up"] = sheds_data["Permit Days"] / 30.44
sheds_data["Billable Months"] = (sheds_data["Months Up"] - FREE_MONTHS).clip(lower=0)
install = sheds_data["Sidewalk Shed/Linear Feet"] * RATE
sheds_data["Cost"] = install + (install * MONTHLY_PCT) * sheds_data["Billable Months"]

# ---------------------------------------------------------------- 2. sidewalk length
print("Reading MapPLUTO (attributes only)...")
lots = gpd.read_file(PLUTO, columns=["CD", "LotFront"], ignore_geometry=True)
lots["CD"] = pd.to_numeric(lots["CD"], errors="coerce")
lots["LotFront"] = pd.to_numeric(lots["LotFront"], errors="coerce").fillna(0)
sidewalk_length_per_district = lots.groupby("CD")["LotFront"].sum()
print(f"  {len(lots):,} lots -> {len(sidewalk_length_per_district)} districts")

# ---------------------------------------------------------------- 3. per-district stats
print("Aggregating per district...")
districts_out = []
for cd, g in sheds_data.groupby("Community Board"):
    cd = int(cd)
    days = np.sort(g["Permit Days"].to_numpy().astype(float))
    n = len(days)
    feet = float(g["Sidewalk Shed/Linear Feet"].sum())
    front = float(sidewalk_length_per_district.get(cd, 0))

    # quartiles the same way matplotlib does them (linear interpolation)
    q1, med, q3 = (float(v) for v in np.percentile(days, [25, 50, 75]))
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    inside = days[(days >= lo_fence) & (days <= hi_fence)]
    whisk_lo = float(inside.min()) if len(inside) else q1
    whisk_hi = float(inside.max()) if len(inside) else q3
    outliers = sorted({int(v) for v in days[(days < lo_fence) | (days > hi_fence)]})

    # same five-number summary for per-shed cost
    costs = np.sort(g["Cost"].to_numpy().astype(float))
    cq1, cmed, cq3 = (float(v) for v in np.percentile(costs, [25, 50, 75]))
    ciqr = cq3 - cq1
    c_lo_fence, c_hi_fence = cq1 - 1.5 * ciqr, cq3 + 1.5 * ciqr
    c_inside = costs[(costs >= c_lo_fence) & (costs <= c_hi_fence)]
    c_whisk_lo = float(c_inside.min()) if len(c_inside) else cq1
    c_whisk_hi = float(c_inside.max()) if len(c_inside) else cq3
    c_outliers = sorted({int(round(v)) for v in costs[(costs < c_lo_fence) | (costs > c_hi_fence)]})

    districts_out.append({
        "cd": cd,
        "name": board_names[cd],
        "count": int(n),
        "feet": round(feet),
        "frontage": round(front),
        "pct": round(min(feet / front * 100, 100), 2) if front else None,
        "avg_days": round(float(g["Permit Days"].mean())),
        "med_days": round(med),
        "q1": round(q1),
        "q3": round(q3),
        "whisk_lo": round(whisk_lo),
        "whisk_hi": round(whisk_hi),
        "outliers": outliers,
        "min_days": int(days[0]),
        "max_days": int(days[-1]),
        "cost_total": round(float(g["Cost"].sum())),
        "cost_avg": round(float(g["Cost"].mean())),
        "cost_q1": round(cq1),
        "cost_med": round(cmed),
        "cost_q3": round(cq3),
        "cost_wlo": round(c_whisk_lo),
        "cost_whi": round(c_whisk_hi),
        "cost_out": c_outliers,
        "cost_max": int(round(costs[-1])),
    })
districts_out.sort(key=lambda d: d["cd"])

# ---------------------------------------------------------------- 4. per-shed points
print("Building point list...")
addr = (
    sheds_data["House Number"].fillna("").astype(str).str.strip()
    + " "
    + sheds_data["Street Name"].fillna("").astype(str).str.strip()
).str.replace(r"\s+", " ", regex=True).str.title().str.slice(0, 44)

points_out = [
    [
        round(float(geom.x), 5),
        round(float(geom.y), 5),
        int(cb),
        round(float(ft)),
        int(dy),
        round(float(cost)),
        a,
        start.strftime("%m/%Y") if pd.notna(start) else "",
    ]
    for geom, cb, ft, dy, cost, a, start in zip(
        sheds_data.geometry,
        sheds_data["Community Board"],
        sheds_data["Sidewalk Shed/Linear Feet"],
        sheds_data["Permit Days"],
        sheds_data["Cost"],
        addr,
        sheds_data["First Permit Date"],
    )
]

# ---------------------------------------------------------------- 5. district shapes
print("Reading district boundaries...")
districts = gpd.read_file(DISTRICTS).to_crs("EPSG:4326")
districts["boro_cd"] = pd.to_numeric(districts["boro_cd"], errors="coerce")
districts = districts[districts["boro_cd"].isin(board_names.keys())].copy()
districts["geometry"] = districts.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)

geo_out = []
for _, row in districts.iterrows():
    geom = row.geometry
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    rings = [
        [[round(x, 5), round(y, 5)] for x, y in p.exterior.coords]
        for p in polys
        if p.exterior is not None
    ]
    if rings:
        geo_out.append({"cd": int(row["boro_cd"]), "rings": rings})

# ---------------------------------------------------------------- 6. write
payload = {
    "points": points_out,
    "districts": districts_out,
    "geo": geo_out,
    "meta": {
        "rate": RATE,
        "monthly_pct": MONTHLY_PCT,
        "free_months": FREE_MONTHS,
        "total_sheds": len(points_out),
        "total_cost": round(sum(d["cost_total"] for d in districts_out)),
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    f.write("const DATA = " + json.dumps(payload, separators=(",", ":")) + ";")

print(f"\nWrote {OUT}  ({os.path.getsize(OUT) / 1_000_000:.2f} MB)")
print(f"  {len(points_out):,} shed points")
print(f"  {len(districts_out)} districts")
print(f"  {len(geo_out)} district shapes")
print(f"  total estimated cost: ${payload['meta']['total_cost'] / 1_000_000:.1f}M")
