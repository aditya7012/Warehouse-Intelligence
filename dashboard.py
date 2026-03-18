"""
AI Dark Store Demand & Inventory Dashboard
==========================================
Powered by Streamlit + scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ZeptoDark · Demand Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS  – dark industrial theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1e2230;
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* ── headings ── */
h1 { font-family: 'Space Mono', monospace; font-size: 1.6rem !important; letter-spacing: -0.5px; color: #facc15 !important; }
h2 { font-family: 'Space Mono', monospace; font-size: 1.1rem !important; color: #94a3b8 !important; border-bottom: 1px solid #1e2230; padding-bottom: 6px; margin-top: 2rem !important; }
h3 { font-family: 'Space Mono', monospace; font-size: 0.9rem !important; color: #64748b !important; text-transform: uppercase; letter-spacing: 1px; }

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: #facc15; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #facc15 !important; font-family: 'Space Mono', monospace; }
[data-testid="stMetricDelta"] { font-size: 0.8rem; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e2230; border-radius: 8px; overflow: hidden; }

/* ── selectbox / widgets ── */
.stSelectbox label, .stMultiSelect label { color: #94a3b8 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
.stSelectbox > div > div { background: #1a1d27 !important; border-color: #2a2f42 !important; color: #e8eaf0 !important; }

/* ── status boxes ── */
.stAlert { border-radius: 8px; border-left-width: 3px; }

/* ── divider ── */
hr { border-color: #1e2230; }

/* ── sidebar labels ── */
.sidebar-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #475569;
    margin-bottom: 4px;
    margin-top: 12px;
}

/* ── tag badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-yellow { background: #3d3200; color: #facc15; border: 1px solid #facc15; }
.badge-red    { background: #3d0f0f; color: #f87171; border: 1px solid #f87171; }
.badge-green  { background: #0f3d1e; color: #4ade80; border: 1px solid #4ade80; }

/* ── page title bar ── */
.title-bar {
    background: linear-gradient(90deg, #13161e 0%, #1a1d27 100%);
    border: 1px solid #1e2230;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

/* ── section card ── */
.section-card {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA GENERATION
# Converts the raw Kaggle store-item sales
# CSV into the dark-store schema expected
# by this dashboard.
# ─────────────────────────────────────────

CITY_STORE_MAP = {
    "Bengaluru": ["BLR-DS-01", "BLR-DS-02", "BLR-DS-03"],
    "Mumbai":    ["MUM-DS-01", "MUM-DS-02"],
    "Delhi":     ["DEL-DS-01", "DEL-DS-02"],
    "Hyderabad": ["HYD-DS-01"],
    "Pune":      ["PNQ-DS-01"],
}

CATEGORY_PRODUCT_MAP = {
    "Snacks":    ["Lays Classic", "Kurkure Masala", "Bingo Mad Angles", "Doritos Nacho", "Too Yumm Multigrain"],
    "Beverages": ["Coca-Cola 500ml", "Red Bull 250ml", "Minute Maid Pulpy", "Frooti Mango", "Sting Energy"],
    "Dairy":     ["Amul Butter 500g", "Mother Dairy Milk 1L", "Britannia Cheese Slice", "Epigamia Greek Yogurt"],
    "Personal Care": ["Dove Soap 3pk", "Head & Shoulders 340ml", "Gillette Mach3 Blades", "Dettol Hand Wash"],
    "Staples":   ["Tata Salt 1kg", "Fortune Basmati Rice 5kg", "Aashirvaad Atta 5kg", "Daawat Brown Rice"],
}

FESTIVAL_DATES = {
    "Diwali":    "2025-11-01",
    "Christmas": "2025-12-25",
    "Eid":       "2025-04-01",
    "Dussehra": "2025-10-02",
}

SAFETY_STOCK = 20


@st.cache_data(show_spinner="Loading data…")
def load_data():
    """Load and transform the Kaggle CSV into the dark-store schema."""
    try:
        raw = pd.read_csv("zepto_darkstore_orders.csv")
        # If the real dark-store CSV is present, use it directly.
        if "dark_store_id" in raw.columns:
            raw["date"] = pd.to_datetime(raw["date"])
            raw["day"] = raw["date"].dt.dayofyear
            raw["day_of_week"] = raw["date"].dt.day_name()
            raw["festival"] = raw["date"].astype(str).map(
                {v: k for k, v in FESTIVAL_DATES.items()}
            )
            return raw
    except FileNotFoundError:
        pass

    # ── Synthetic generation from train.csv ──
    try:
        raw = pd.read_csv("train.csv")
    except FileNotFoundError:
        st.error("Could not find train.csv or zepto_darkstore_orders.csv. Please upload a data file.")
        st.stop()

    raw["date"] = pd.to_datetime(raw["date"])

    # Map numeric store/item ids → dark-store / product names
    stores = sorted(raw["store"].unique())
    items  = sorted(raw["item"].unique())

    all_cities    = list(CITY_STORE_MAP.keys())
    all_stores    = [s for ss in CITY_STORE_MAP.values() for s in ss]
    all_cats      = list(CATEGORY_PRODUCT_MAP.keys())
    all_products  = [p for ps in CATEGORY_PRODUCT_MAP.values() for p in ps]

    store_map = {s: all_stores[i % len(all_stores)] for i, s in enumerate(stores)}
    # city comes from store id prefix
    store_city_map = {}
    for city, ss in CITY_STORE_MAP.items():
        for s in ss:
            store_city_map[s] = city

    item_product_map = {it: all_products[i % len(all_products)] for i, it in enumerate(items)}
    item_cat_map     = {}
    for cat, prods in CATEGORY_PRODUCT_MAP.items():
        for p in prods:
            item_cat_map[p] = cat

    raw["dark_store_id"] = raw["store"].map(store_map)
    raw["city"]          = raw["dark_store_id"].map(store_city_map)
    raw["product_name"]  = raw["item"].map(item_product_map)
    raw["category"]      = raw["product_name"].map(item_cat_map)
    raw["orders"]        = raw["sales"]

    # Simulate stock_remaining
    rng = np.random.default_rng(42)
    raw["stock_remaining"] = (raw["orders"] * rng.uniform(0.8, 2.5, len(raw))).astype(int)

    raw["day"]         = raw["date"].dt.dayofyear
    raw["day_of_week"] = raw["date"].dt.day_name()
    raw["festival"]    = raw["date"].astype(str).map(
        {v: k for k, v in FESTIVAL_DATES.items()}
    )

    return raw[["date", "day", "day_of_week", "city", "dark_store_id",
                "category", "product_name", "orders", "stock_remaining", "festival"]]


data = load_data()


# ─────────────────────────────────────────
# SIDEBAR  – Filters
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ZeptoDark")
    st.markdown("<p style='color:#475569;font-size:0.78rem;margin-top:-8px;'>Demand Intelligence Platform</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p class='sidebar-label'>City</p>", unsafe_allow_html=True)
    city = st.selectbox("City", data["city"].unique(), label_visibility="collapsed")

    city_data = data[data["city"] == city]

    st.markdown("<p class='sidebar-label'>Dark Store</p>", unsafe_allow_html=True)
    store = st.selectbox("Store", city_data["dark_store_id"].unique(), label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Category</p>", unsafe_allow_html=True)
    categories = ["All Categories"] + sorted(data["category"].unique().tolist())
    category = st.selectbox("Category", categories, label_visibility="collapsed")

    if category == "All Categories":
        sku_data = data
    else:
        sku_data = data[data["category"] == category]

    st.markdown("<p class='sidebar-label'>Product / SKU</p>", unsafe_allow_html=True)
    product_list = ["All Products"] + sorted(sku_data["product_name"].unique().tolist())
    product = st.selectbox("Product", product_list, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<p style='color:#2a2f42;font-size:0.68rem;'>Built for Zepto Product Internship · 2025</p>",
        unsafe_allow_html=True
    )

store_data = sku_data[
    (sku_data["city"] == city) &
    (sku_data["dark_store_id"] == store)
]


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="title-bar">
  <div>
    <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#facc15;font-weight:700;">
      ⚡ Dark Store Operations
    </div>
    <div style="color:#475569;font-size:0.82rem;margin-top:4px;">
      {city} · {store} · {category} · {product}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# CASE 1 — STORE-LEVEL DASHBOARD
# ═══════════════════════════════════════════
if product == "All Products":

    st.markdown("## 🏪 Store Operations Dashboard")

    # ── KPI Row ──
    store_daily = store_data.groupby("date")["orders"].sum().reset_index()
    store_daily["day"] = store_daily["date"].dt.dayofyear

    if len(store_daily) >= 2:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(store_daily[["day"]], store_daily["orders"])
        next_day_val = store_daily["day"].max() + 1
        store_prediction = int(rf.predict([[next_day_val]])[0])
    else:
        store_prediction = int(store_daily["orders"].mean()) if len(store_daily) else 0

    orders_per_rider = 40
    riders_needed    = int(np.ceil(store_prediction / orders_per_rider))
    total_orders_7d  = int(store_data[store_data["date"] >= store_data["date"].max() - pd.Timedelta(days=7)]["orders"].sum())
    avg_daily        = int(store_daily["orders"].mean()) if len(store_daily) else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Predicted Orders Tomorrow", f"{store_prediction:,}")
    col2.metric("🛵 Delivery Partners Needed",  f"{riders_needed}")
    col3.metric("📊 Avg Daily Orders",           f"{avg_daily:,}")
    col4.metric("🗓️ Orders (Last 7 Days)",       f"{total_orders_7d:,}")

    # ── Historical Demand ──
    st.markdown("## 📈 Historical Demand — Store Level")
    st.line_chart(store_daily.set_index("date")["orders"], height=250, use_container_width=True)

    # ── Restock Table ──
    st.markdown("## 🔄 Items Needing Restock")

    results = []
    for p_name in store_data["product_name"].unique():
        temp = store_data[store_data["product_name"] == p_name].copy()
        if len(temp) < 2:
            continue
        lr = LinearRegression()
        lr.fit(temp[["day"]], temp["orders"])
        pred   = int(lr.predict([[temp["day"].max() + 1]])[0])
        stock  = int(temp.sort_values("date").iloc[-1]["stock_remaining"])
        restock = max(0, pred + SAFETY_STOCK - stock)
        cover_days = round(stock / max(pred, 1), 1)
        risk = "🔴 HIGH" if cover_days <= 2 else ("🟡 MEDIUM" if cover_days <= 5 else "🟢 LOW")
        if restock > 0:
            results.append({
                "Product":        p_name,
                "Category":       store_data[store_data["product_name"] == p_name]["category"].iloc[0],
                "Forecast Demand": pred,
                "Current Stock":  stock,
                "Restock Qty":    restock,
                "Days of Cover":  cover_days,
                "Risk":           risk,
            })

    if results:
        df_restock = pd.DataFrame(results).sort_values("Restock Qty", ascending=False)
        st.dataframe(
            df_restock,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Restock Qty":     st.column_config.NumberColumn(format="%d 📦"),
                "Days of Cover":   st.column_config.NumberColumn(format="%.1f d"),
                "Forecast Demand": st.column_config.NumberColumn(format="%d"),
                "Current Stock":   st.column_config.NumberColumn(format="%d"),
            }
        )
        st.info(f"⚠️  {len(df_restock)} SKU(s) require restocking across this store.")
    else:
        st.success("✅ All SKUs are sufficiently stocked.")

    # ── Store Comparison ──
    st.markdown("## 🏬 Dark Store Demand Comparison — " + city)
    city_store_orders = data[data["city"] == city].groupby("dark_store_id")["orders"].sum()
    st.bar_chart(city_store_orders, height=220, use_container_width=True)

    # ── Expansion Predictor ──
    st.markdown("## 🗺️ Dark Store Expansion Predictor")
    store_demand   = data[data["city"] == city].groupby("dark_store_id")["orders"].sum().reset_index()
    avg_demand     = store_demand["orders"].mean()
    high_demand_df = store_demand[store_demand["orders"] > avg_demand * 1.3]

    if len(high_demand_df) > 0:
        st.warning("🚀 High-demand stores detected — expansion opportunity identified.")
        st.dataframe(high_demand_df.rename(columns={"dark_store_id": "Store", "orders": "Total Orders"}),
                     use_container_width=True, hide_index=True)
        st.info("Recommendation: Consider opening satellite dark stores near these high-demand locations to reduce delivery radius and SLA breach risk.")
    else:
        st.success("✅ Current store network capacity is sufficient for this city.")


# ═══════════════════════════════════════════
# CASE 2 — SKU-LEVEL DASHBOARD
# ═══════════════════════════════════════════
else:
    st.markdown(f"## 📦 SKU Demand Intelligence — *{product}*")

    product_data = store_data[store_data["product_name"] == product].copy()

    if product_data.empty:
        st.warning("No data found for this SKU at the selected store.")
        st.stop()

    # ── Forecast ──
    X = product_data[["day"]]
    y = product_data["orders"]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    max_day      = X["day"].max()
    predictions  = rf.predict([[max_day + 1], [max_day + 2], [max_day + 3]])
    prediction   = int(predictions[0])
    current_stock = int(product_data.sort_values("date").iloc[-1]["stock_remaining"])
    restock_qty   = max(0, prediction + SAFETY_STOCK - current_stock)
    avg_demand    = product_data["orders"].mean()
    days_to_stockout = int(current_stock / max(avg_demand, 0.01))

    # ── KPI Row ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Predicted Demand Tomorrow",      f"{prediction}")
    col2.metric("🏷️ Current Stock",                   f"{current_stock}")
    col3.metric("🛡️ Safety Stock Buffer",             f"{SAFETY_STOCK}")
    delta_color = "inverse" if restock_qty > 0 else "normal"
    col4.metric("📦 Restock Required", f"{restock_qty}", delta="Action needed" if restock_qty > 0 else "All good", delta_color=delta_color)

    # ── Inventory Status ──
    st.markdown("## 🔋 Inventory Status")
    if restock_qty > 0:
        st.error(f"⚠️  Restock **{restock_qty} units** of *{product}* immediately to avoid stockout.")
    else:
        st.success(f"✅  Stock level is safe. No restock required for *{product}*.")

    # ── Demand Chart + 3-day forecast ──
    st.markdown("## 📈 Historical Demand + 3-Day Forecast")

    hist = product_data.set_index("date")["orders"].rename("Actual")
    future_dates  = pd.date_range(product_data["date"].max() + pd.Timedelta(days=1), periods=3)
    forecast_s    = pd.Series(predictions.astype(int), index=future_dates, name="Forecast")
    chart_df      = pd.concat([hist, forecast_s], axis=1)
    st.line_chart(chart_df, height=260, use_container_width=True)

    # ── Two-column: Stockout Risk + Demand Surge ──
    colA, colB = st.columns(2)

    with colA:
        st.markdown("## ⏱️ Stockout Risk")
        if days_to_stockout <= 2:
            st.error(f"**HIGH RISK** — Estimated stockout in **{days_to_stockout} day(s)**")
        elif days_to_stockout <= 5:
            st.warning(f"**MEDIUM RISK** — Estimated stockout in **{days_to_stockout} day(s)**")
        else:
            st.success(f"**LOW RISK** — Stock will last ~**{days_to_stockout} days** at current demand rate")

    with colB:
        st.markdown("## 🚀 Demand Surge Detector")
        weekday_avg = product_data[product_data["day_of_week"].isin(
            ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        )]["orders"].mean()
        weekend_avg = product_data[product_data["day_of_week"].isin(
            ["Saturday","Sunday"]
        )]["orders"].mean()
        festival_avg = product_data[product_data["festival"].notna()]["orders"].mean()

        st.write(f"**Avg Weekday Orders:** {int(weekday_avg) if not np.isnan(weekday_avg) else '–'}")
        st.write(f"**Avg Weekend Orders:** {int(weekend_avg) if not np.isnan(weekend_avg) else '–'}")

        if not np.isnan(weekend_avg) and not np.isnan(weekday_avg) and weekend_avg > weekday_avg * 1.2:
            st.warning("🚀 Weekend Demand Surge Detected — consider pre-loading stock on Fridays.")
        if not np.isnan(festival_avg):
            st.error(f"🎉 Festival Demand Surge Active — avg {int(festival_avg)} orders on festival days.")
        if (np.isnan(festival_avg)) and (np.isnan(weekend_avg) or weekend_avg <= weekday_avg * 1.2):
            st.success("Demand is stable — no surge events detected.")


# ─────────────────────────────────────────
# CITY HEATMAP  (always shown at bottom)
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("## 🗺️ City Demand Heatmap — " + city)
heatmap_data = data[data["city"] == city].groupby("dark_store_id")["orders"].sum()
st.bar_chart(heatmap_data, height=200, use_container_width=True)

st.markdown(
    "<p style='color:#2a2f42;font-size:0.72rem;text-align:center;margin-top:2rem;'>"
    "ZeptoDark Demand Intelligence · Built with Streamlit · 2025</p>",
    unsafe_allow_html=True
)
