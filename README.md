# Aadhaar Analytics Command Center

### UIDAI Data Hackathon 2026 Submission  
**Team:** DataVisionaries  
**Members:** Sampurna Niyogi, Hrichik Khandait  

---

## 🧠 Overview

The Aadhaar Analytics Command Center transforms UIDAI enrolment and update datasets into actionable intelligence for:

- Inclusion monitoring  
- Fraud risk detection  
- Operational planning  
- Policy decision support  

The platform analyzes biometric, demographic, and enrolment streams to highlight districts and pincodes requiring intervention.

> **Important:** All insights are derived from the provided UIDAI datasets only.  
> No synthetic or randomly generated records are used.

---

## 🎯 Problem Addressed

1. Low child enrolment in specific districts  
2. Abnormal adult-only enrolment spikes  
3. Centers with high acquisition but near-zero maintenance  
4. Lack of workload forecasting for PECs

---

## 🚀 Key Features

### 1. Smart Insights
- District Z-Score Gap Index for child enrolment  
- Acquisition vs Maintenance imbalance detection  
- Churn Index as migration proxy  

### 2. Anomaly Detection
- Isolation Forest on pincode volumes and age ratios  
- Identification of:
  - Ghost enrolment centers  
  - Mass address update mills  
  - Zero-maintenance districts

### 3. Forecasting
- XGBoost 30-day prediction  
- Seasonality based on school admission cycles  
- RMSE evaluation

### 4. Multivariate Analytics
- Parallel coordinate analysis  
- 3D trivariate state maturity cube  
- Correlation dashboards

---

## 🛠 Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Plotly, Scikit-learn, XGBoost  
- **Framework:** Streamlit  
- **ML Models:** Isolation Forest, XGBoost Regressor  

---

## 📊 Methodology

1. **Data Ingestion**
   - Load biometric, demographic, enrolment CSV files  
   - Date normalization and validation  

2. **Pre-processing**
   - Aggregations (State/District/Pincode)  
   - Feature engineering:
     - child_ratio  
     - adult_ratio  
     - churn_index  

3. **Analytics**
   - Statistical Z-scores  
   - Multivariate comparisons  
   - Time-series modelling  

4. **Output**
   - Interactive dashboards  
   - Actionable recommendations  
