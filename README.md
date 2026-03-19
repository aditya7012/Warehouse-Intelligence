## 📌 Overview

I built an AI-powered operations intelligence system designed for quick-commerce dark stores to solve demand unpredictability, frequent stockouts, and inefficient delivery planning.

The system predicts SKU-level demand, recommends inventory restocking, and estimates delivery partner requirements using historical order data across cities, stores, and products.

---

## 🚨 Problem

Quick commerce platforms like Zepto and Blinkit operate under extreme time constraints where:

- Stockouts directly lead to lost revenue
- Demand fluctuates heavily across weekends and festivals
- Delivery partner allocation is often reactive, not predictive

Operations teams lack a unified system to make **data-driven daily decisions**.

---

## 👤 Users

- Dark Store Managers → Inventory decisions
- City Operations Teams → Delivery planning
- Supply Chain Teams → Restocking strategy

---

## 💡 Key Insight

Demand is not random — it follows predictable patterns:

- 📈 Weekend demand spikes
- 🎉 Festival-driven surges
- 📍 Store-level demand variation

This means demand can be **predicted and operationalized**.

---

## 🛠️ Solution

I designed a 4-layer intelligence system:

### 1. Demand Forecasting

Predicts future SKU demand using historical order patterns.

### 2. Inventory Recommendation

Calculates restock quantity using:

- Predicted demand
- Current stock
- Safety buffer

### 3. Delivery Planning

Estimates required delivery partners based on expected order volume.

### 4. Demand Surge Detection

Identifies abnormal spikes (weekends/festivals) to prevent operational overload.

---

## ⚙️ Product Logic

### Restock Formula

```
Restock = Predicted Demand + Safety Stock - Current Inventory
```

### Delivery Planning

```
Delivery Partners Needed = Total Orders / Orders per Rider
```

### Surge Detection

```
If Weekend Demand > 1.2 × Weekday Demand → Surge Triggered
```

---

### 📊 Metrics

---

### **Model / Prediction Metrics:**

- Demand forecast accuracy (MAPE / % error)
- Stockout prediction accuracy
- Demand surge detection precision

### **Product Metrics:**

- % of SKUs with automated restock recommendations
- % of demand spikes detected before occurrence
- Delivery partner allocation accuracy

### **Business Metrics:**

- Stockout rate reduction (~35% simulated)
- Increase in in-stock availability (%)
- Reduction in lost orders due to stockouts

### **Operational Efficiency Metrics:**

- Reduction in inventory planning time
- Improvement in rider utilization (%)
- Reduction in emergency restocking events

---

## 🧪 Trade-offs & Decisions

| Decision | Reason |
| --- | --- |
| Random Forest for forecasting | Handles non-linear demand patterns |
| Safety stock buffer | Reduces risk of stockouts |
| SKU + Store views | Different decision layers |
| Rule-based surge detection | Simple + interpretable |

---

## 📉 Limitations

- No real-time data ingestion
- Does not include external signals (weather, pricing, promos)
- Forecasting can be improved using time-series models

---

## 🚀 Future Improvements

- Real-time inventory sync
- Dynamic pricing integration
- Supplier-side automation
- Advanced ML models (ARIMA, LSTM)
- Store expansion optimization model
