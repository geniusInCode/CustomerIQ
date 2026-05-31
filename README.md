# CustomerIQ – Customer Segmentation & RFM Analytics Platform

CustomerIQ is a full-stack customer analytics platform built with Python and Flask that performs ETL processing, RFM analysis, customer segmentation, churn prediction, cohort analysis, A/B testing, and business intelligence reporting through an interactive dashboard and REST APIs.

Demo Link :- https://customeriq-b9py.onrender.com/

## Features

### ETL Pipeline
- Data generation and ingestion
- Data cleaning and transformation
- SQLite database loading

### Customer Analytics
- RFM (Recency, Frequency, Monetary) Analysis
- Customer segmentation using K-Means Clustering
- Customer cohort analysis
- Customer lifetime insights

### Machine Learning
- Churn prediction using Logistic Regression
- Feature importance analysis
- Model accuracy reporting

### Statistical Analysis
- A/B Testing using SciPy
- Premium vs Basic customer comparison
- Statistical significance testing

### SQL Analytics
- Revenue analysis by city
- Monthly revenue trends
- Customer transaction insights
- SQL JOINs and CTEs

### Data Visualization
- Interactive dashboard
- Chart.js visualizations
- Matplotlib charts
- Seaborn heatmaps

### REST API
- KPI endpoints
- Customer segment endpoints
- Revenue analytics endpoints
- Churn model endpoints
- Cohort analysis endpoints

---

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- SQLite
- Scikit-Learn
- SciPy
- Matplotlib
- Seaborn
- Chart.js
- Gunicorn
- Render

---

## Project Architecture

```text
CustomerIQ
│
├── app.py
├── requirements.txt
├── customeriq.db
├── render.yaml
├── README.md
│
├── ETL Pipeline
├── SQLite Database
├── RFM Analytics
├── Customer Segmentation
├── Churn Prediction
├── Cohort Analysis
├── A/B Testing
├── REST APIs
└── Dashboard
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/geniusInCode/CustomerIQ.git
cd CustomerIQ
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://localhost:5001
```

---

## API Endpoints

| Endpoint | Description |
|-----------|------------|
| `/api/kpis` | Business KPIs |
| `/api/segments` | Customer Segments |
| `/api/rfm-top` | Top Customers |
| `/api/city-analysis` | Revenue by City |
| `/api/monthly-revenue` | Monthly Revenue |
| `/api/churn-model` | Churn Prediction |
| `/api/cohort` | Cohort Analysis |
| `/api/ab-test` | A/B Testing Results |

---

## Deployment

### Render

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
gunicorn app:app
```

---

## Dashboard Preview

- Executive KPI Dashboard
- RFM Analytics
- Customer Segmentation
- SQL Analytics
- Churn Prediction
- Cohort Analysis
- A/B Testing
- REST API Explorer

---

## Author

**GeniusInCode**

Customer Analytics • Machine Learning • Data Science • Business Intelligence
