# ZeptoDark — Deployment Guide

## Local Setup

```bash
pip install -r requirements.txt

# Put train.csv in the same folder as dashboard.py, then:
streamlit run dashboard.py
```

## Deploy to Streamlit Community Cloud (Free — Recommended)

1. Push your files to a **public GitHub repo**:
   ```
   your-repo/
   ├── dashboard.py
   ├── requirements.txt
   └── train.csv          ← include the data file
   ```

2. Go to **https://share.streamlit.io** → "New app"

3. Connect your GitHub repo, set:
   - **Main file path:** `dashboard.py`
   - **Python version:** 3.11

4. Click **Deploy** — live URL in ~2 minutes.

---

## Data Note

The dashboard works with either:

| File | Columns |
|---|---|
| `zepto_darkstore_orders.csv` | `date, city, dark_store_id, category, product_name, orders, stock_remaining` |
| `train.csv` (Kaggle) | `date, store, item, sales` (auto-mapped by the app) |

---

## UI Improvements Made

| Area | Before | After |
|---|---|---|
| Theme | Streamlit default white | Dark industrial (#0d0f14) |
| Typography | System fonts | Space Mono + DM Sans |
| KPI cards | Basic `st.metric` | Styled cards with hover border |
| Store-level KPIs | 2 metrics | 4 metrics (+ 7-day orders, avg daily) |
| Forecast chart | Orders only | Actual + 3-day forecast overlay |
| Restock table | Basic dataframe | + Category, Days of Cover, Risk column |
| Layout | Full width single column | Responsive 2 & 4-column grids |
| Sidebar | Raw widgets | Labelled, grouped, branded |
| Page config | Missing | Title, icon, wide layout |
| Data loading | No caching | `@st.cache_data` for speed |
| Error handling | Crashes if file missing | Graceful fallback + st.stop() |
