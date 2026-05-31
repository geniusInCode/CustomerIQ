"""
CustomerIQ – Customer Segmentation & RFM Analytics Platform
Tools: Python · Flask · Pandas · NumPy · SQLite/SQL · Matplotlib · Seaborn
       Scikit-Learn (KMeans) · SciPy · Chart.js · REST API · ETL Pipeline
Deployable on Render
"""

from flask import Flask, jsonify, send_file
import pandas as pd
import numpy as np
from scipy import stats
import sqlite3, io, base64, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta

app = Flask(__name__)

BG, GRID = '#1e293b', '#334155'

# ═══════════════════════════════════════════════════════════════════════════════
#  ETL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_raw():
    np.random.seed(99)
    n_customers = 800
    n_txns      = 4000

    cust_ids    = [f"C{str(i).zfill(4)}" for i in range(1, n_customers+1)]
    genders     = np.random.choice(['Male','Female','Other'], n_customers, p=[0.48,0.48,0.04])
    ages        = np.random.randint(18, 72, n_customers)
    cities      = np.random.choice(['Mumbai','Delhi','Bangalore','Hyderabad','Chennai','Pune'], n_customers)
    plans       = np.random.choice(['Basic','Standard','Premium'], n_customers, p=[0.40,0.35,0.25])
    join_dates  = pd.date_range('2022-01-01', '2024-01-01', periods=n_customers)
    churned     = np.random.choice([0,1], n_customers, p=[0.75,0.25])

    customers = pd.DataFrame({
        'customer_id': cust_ids, 'gender': genders, 'age': ages,
        'city': cities, 'plan': plans, 'join_date': join_dates, 'churned': churned
    })

    txn_custs   = np.random.choice(cust_ids, n_txns)
    txn_dates   = pd.to_datetime(np.random.choice(pd.date_range('2023-01-01','2024-12-31'), n_txns))
    categories  = np.random.choice(['Electronics','Fashion','Groceries','Beauty','Sports'], n_txns)
    amounts     = np.random.exponential(scale=2500, size=n_txns).clip(100, 50000)

    # Inject dirty data
    txn_amounts = amounts.copy().astype(float)
    txn_amounts[np.random.choice(n_txns, 40)] = np.nan
    txn_amounts[np.random.choice(n_txns, 20)] = -999

    transactions = pd.DataFrame({
        'txn_id': range(1, n_txns+1), 'customer_id': txn_custs,
        'date': txn_dates, 'category': categories, 'amount': txn_amounts
    })
    return customers, transactions


def clean_data(customers, transactions):
    # Clean transactions
    t = transactions.copy()
    t['amount'] = t['amount'].fillna(t['amount'].median())
    t = t[t['amount'] > 0].copy()
    t['amount'] = t['amount'].abs()
    t['month']   = t['date'].dt.month
    t['quarter'] = t['date'].dt.quarter
    t['year']    = t['date'].dt.year
    t['weekday'] = t['date'].dt.day_name()

    # RFM Calculation (as of end of 2024)
    snapshot = pd.Timestamp('2024-12-31')
    rfm = t.groupby('customer_id').agg(
        recency   = ('date',   lambda x: (snapshot - x.max()).days),
        frequency = ('txn_id', 'count'),
        monetary  = ('amount', 'sum')
    ).reset_index()
    rfm['monetary'] = rfm['monetary'].round(2)

    # RFM Scores (1-5)
    for col, label in [('recency','R'),('frequency','F'),('monetary','M')]:
        rfm[f'{label}_score'] = pd.qcut(
            rfm[col], q=5,
            labels=[5,4,3,2,1] if col=='recency' else [1,2,3,4,5],
            duplicates='drop'
        ).astype(int)
    rfm['rfm_score'] = rfm['R_score']*100 + rfm['F_score']*10 + rfm['M_score']

    return customers, t, rfm


def load_db(customers, transactions, rfm):
    conn = sqlite3.connect('customeriq.db')
    customers.to_sql('customers',    conn, if_exists='replace', index=False)
    transactions.to_sql('transactions', conn, if_exists='replace', index=False)
    rfm.to_sql('rfm', conn, if_exists='replace', index=False)
    conn.commit(); conn.close()


def run_etl():
    print("▶ CustomerIQ ETL running …")
    raw_c, raw_t    = generate_raw()
    cust, txn, rfm  = clean_data(raw_c, raw_t)
    load_db(cust, txn, rfm)
    print(f"✔ ETL done — {len(cust)} customers, {len(txn)} transactions, {len(rfm)} RFM records")
    return cust, txn, rfm


CUST, TXN, RFM = run_etl()


# ═══════════════════════════════════════════════════════════════════════════════
#  SQL HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def sql(q):
    conn = sqlite3.connect('customeriq.db')
    df   = pd.read_sql_query(q, conn); conn.close()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  KMEANS SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def segment_customers():
    X   = RFM[['recency','frequency','monetary']].copy()
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    km  = KMeans(n_clusters=4, random_state=42, n_init=10)
    RFM['segment_id'] = km.fit_predict(Xs)
    seg_map = {
        RFM.groupby('segment_id')['monetary'].mean().idxmax():  'Champions',
        RFM.groupby('segment_id')['recency'].mean().idxmin():   'At Risk',
        RFM.groupby('segment_id')['frequency'].mean().idxmax(): 'Loyal',
    }
    def label(x):
        return seg_map.get(x, 'New / Occasional')
    RFM['segment'] = RFM['segment_id'].apply(label)
    return RFM

RFM = segment_customers()


# ═══════════════════════════════════════════════════════════════════════════════
#  CHURN PREDICTION (Logistic Regression)
# ═══════════════════════════════════════════════════════════════════════════════

def build_churn_model():
    merged = CUST.merge(RFM[['customer_id','recency','frequency','monetary']], on='customer_id', how='left').fillna(0)
    merged['age_scaled'] = merged['age'] / 100
    feats  = ['recency','frequency','monetary','age_scaled']
    X, y   = merged[feats], merged['churned']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc     = StandardScaler()
    model  = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(sc.fit_transform(Xtr), ytr)
    ypred  = model.predict(sc.transform(Xte))
    acc    = accuracy_score(yte, ypred)
    coef   = sorted(zip(feats, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    return {'accuracy': round(acc,4), 'features': [{'name':f,'coef':round(c,4)} for f,c in coef]}

CHURN = build_churn_model()


# ═══════════════════════════════════════════════════════════════════════════════
#  COHORT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def build_cohort():
    df = TXN.merge(CUST[['customer_id','join_date']], on='customer_id')
    df['cohort_month'] = pd.to_datetime(df['join_date']).dt.to_period('Q').astype(str)
    df['txn_quarter']  = df['date'].dt.to_period('Q').astype(str)
    cohort = df.groupby(['cohort_month','txn_quarter'])['customer_id'].nunique().reset_index()
    cohort.columns = ['cohort','period','active_customers']
    return cohort.to_dict(orient='records')

COHORT = build_cohort()


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor=BG)
    buf.seek(0); data = base64.b64encode(buf.read()).decode()
    plt.close(fig); return data

def make_charts():
    plt.rcParams.update({'text.color':'#e2e8f0','axes.labelcolor':'#94a3b8',
                         'xtick.color':'#94a3b8','ytick.color':'#94a3b8'})
    charts = {}

    # 1. RFM Scatter — Recency vs Monetary coloured by Segment
    fig, ax = plt.subplots(figsize=(9,5), facecolor=BG); ax.set_facecolor(BG)
    colors  = {'Champions':'#4ade80','Loyal':'#38bdf8','At Risk':'#f87171','New / Occasional':'#facc15'}
    for seg, grp in RFM.groupby('segment'):
        ax.scatter(grp['recency'], grp['monetary']/1000, label=seg,
                   alpha=0.6, s=30, color=colors.get(seg,'#94a3b8'))
    ax.set_title('RFM Scatter — Recency vs Monetary by Segment', color='white', fontsize=13)
    ax.set_xlabel('Recency (days)'); ax.set_ylabel('Monetary (₹K)')
    ax.legend(labelcolor='white', facecolor=BG, edgecolor=GRID, fontsize=9)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color(GRID)
    charts['rfm_scatter'] = to_b64(fig)

    # 2. Segment Distribution — Seaborn countplot
    seg_counts = RFM['segment'].value_counts()
    fig, ax = plt.subplots(figsize=(8,4), facecolor=BG); ax.set_facecolor(BG)
    bars = ax.bar(seg_counts.index, seg_counts.values,
                  color=[colors.get(s,'#94a3b8') for s in seg_counts.index])
    ax.set_title('Customer Segment Distribution (KMeans)', color='white', fontsize=13)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color(GRID)
    charts['segments'] = to_b64(fig)

    # 3. Spend by Category — Seaborn
    cat_spend = TXN.groupby('category')['amount'].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8,4), facecolor=BG); ax.set_facecolor(BG)
    pal = ['#f87171','#fb923c','#facc15','#4ade80','#38bdf8']
    ax.barh(cat_spend.index, cat_spend.values/1e6, color=pal)
    ax.set_title('Total Spend by Category (₹M)', color='white', fontsize=13)
    ax.set_xlabel('Revenue (₹M)')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color(GRID)
    charts['category'] = to_b64(fig)

    # 4. RFM Score Heatmap — Seaborn
    fig, ax = plt.subplots(figsize=(7,5), facecolor=BG); ax.set_facecolor(BG)
    rfm_heat = RFM.pivot_table(index='R_score', columns='F_score', values='monetary', aggfunc='mean')
    sns.heatmap(rfm_heat, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, annot_kws={'color':'white','size':9})
    ax.set_title('Avg Monetary by R-Score vs F-Score', color='white', fontsize=13)
    charts['rfm_heat'] = to_b64(fig)

    return charts

CHARTS = make_charts()


# ═══════════════════════════════════════════════════════════════════════════════
#  REST API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/kpis')
def api_kpis():
    return jsonify({
        'total_customers'  : int(len(CUST)),
        'total_transactions': int(len(TXN)),
        'total_revenue'    : round(TXN['amount'].sum(), 2),
        'avg_order_value'  : round(TXN['amount'].mean(), 2),
        'churn_rate_pct'   : round(CUST['churned'].mean()*100, 2),
        'avg_frequency'    : round(RFM['frequency'].mean(), 2),
    })

@app.route('/api/segments')
def api_segments():
    seg = RFM.groupby('segment').agg(
        count       = ('customer_id','count'),
        avg_recency = ('recency','mean'),
        avg_freq    = ('frequency','mean'),
        avg_monetary= ('monetary','mean')
    ).round(2).reset_index()
    return jsonify(seg.to_dict(orient='records'))

@app.route('/api/rfm-top')
def api_rfm():
    result = sql("""
        SELECT r.customer_id, r.recency, r.frequency,
               ROUND(r.monetary,2) AS monetary,
               r.R_score, r.F_score, r.M_score, r.segment
        FROM   rfm r
        ORDER  BY r.monetary DESC
        LIMIT  20
    """)
    return jsonify(result.to_dict(orient='records'))

@app.route('/api/city-analysis')
def api_city():
    result = sql("""
        WITH city_stats AS (
            SELECT c.city,
                   COUNT(DISTINCT c.customer_id)  AS customers,
                   COUNT(t.txn_id)                AS transactions,
                   ROUND(SUM(t.amount),2)         AS revenue,
                   ROUND(AVG(t.amount),2)         AS avg_order
            FROM   customers c
            LEFT JOIN transactions t ON c.customer_id = t.customer_id
            GROUP  BY c.city
        )
        SELECT *, ROUND(revenue*100.0/(SELECT SUM(amount) FROM transactions),2) AS rev_share
        FROM   city_stats ORDER BY revenue DESC
    """)
    return jsonify(result.to_dict(orient='records'))

@app.route('/api/monthly-revenue')
def api_monthly():
    result = sql("""
        SELECT year, month,
               ROUND(SUM(amount),2)  AS revenue,
               COUNT(*)              AS transactions,
               COUNT(DISTINCT customer_id) AS unique_customers
        FROM   transactions
        GROUP  BY year, month ORDER BY year, month
    """)
    return jsonify(result.to_dict(orient='records'))

@app.route('/api/churn-model')
def api_churn():
    return jsonify(CHURN)

@app.route('/api/cohort')
def api_cohort():
    return jsonify(COHORT)

@app.route('/api/ab-test')
def api_ab():
    premium = TXN.merge(CUST[['customer_id','plan']], on='customer_id')
    a = premium[premium['plan']=='Premium']['amount']
    b = premium[premium['plan']=='Basic']['amount']
    t, p = stats.ttest_ind(a, b)
    return jsonify({
        'test'       : 'Premium vs Basic Avg Order Value',
        'premium'    : {'mean': round(a.mean(),2), 'n': len(a)},
        'basic'      : {'mean': round(b.mean(),2), 'n': len(b)},
        't_stat'     : round(t,4), 'p_value': round(p,4),
        'significant': bool(p<0.05),
        'conclusion' : 'Significant difference' if p<0.05 else 'No significant difference'
    })

@app.route('/chart/<name>')
def chart(name):
    if name in CHARTS:
        return send_file(io.BytesIO(base64.b64decode(CHARTS[name])), mimetype='image/png')
    return 'Not found', 404

# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CustomerIQ Analytics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',sans-serif}
.hdr{background:linear-gradient(135deg,#1a2e4a,#0f172a);padding:22px 32px;border-bottom:1px solid #1e293b}
.hdr h1{font-size:24px;color:#a78bfa;font-weight:700}
.hdr p{color:#94a3b8;font-size:12px;margin-top:3px}
.nav{display:flex;gap:8px;padding:14px 32px;background:#0f172a;border-bottom:1px solid #1e293b;flex-wrap:wrap}
.pill{padding:6px 16px;border-radius:20px;cursor:pointer;font-size:13px;border:1px solid #334155;color:#94a3b8;background:transparent;transition:all .2s}
.pill.on,.pill:hover{background:#a78bfa;color:#0f172a;border-color:#a78bfa;font-weight:700}
.wrap{padding:24px 32px}
.kgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:24px}
.kcard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px;transition:transform .2s,border-color .2s}
.kcard:hover{transform:translateY(-3px);border-color:#a78bfa}
.klbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px}
.kval{font-size:22px;font-weight:700;color:#a78bfa;margin-top:6px}
.ksub{font-size:11px;color:#94a3b8;margin-top:3px}
.cgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px;margin-bottom:20px}
.ccard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px;margin-bottom:20px}
.ccard h3{font-size:13px;color:#cbd5e1;margin-bottom:14px;font-weight:600}
.ccard img{width:100%;border-radius:8px}
.tcard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px;margin-bottom:20px;overflow-x:auto}
.tcard h3{font-size:13px;color:#cbd5e1;margin-bottom:12px;font-weight:600}
.qbox{background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:12px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:9px 12px;color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #334155}
td{padding:9px 12px;border-bottom:1px solid #1e293b;color:#cbd5e1}
tr:hover td{background:#334155}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:11px;font-weight:600}
.gn{background:#14532d;color:#4ade80}.bl{background:#2e1065;color:#a78bfa}.or{background:#431407;color:#fb923c}
.sgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;margin-bottom:20px}
.scard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px}
.scard h3{font-size:13px;color:#cbd5e1;margin-bottom:12px;font-weight:600}
.srow{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1e293b;font-size:13px}
.srow:last-child{border-bottom:none}.sk{color:#94a3b8}.sv{color:#a78bfa;font-weight:600}
.sec{display:none}.sec.on{display:block}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;background:#2e1065;color:#a78bfa;font-size:11px;margin:2px}
.mth{display:inline-block;padding:2px 8px;border-radius:4px;background:#14532d;color:#4ade80;font-size:11px;font-weight:700;margin-right:8px}
.acard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px;margin-bottom:14px}
</style></head>
<body>
<div class="hdr">
  <h1>👥 CustomerIQ Analytics</h1>
  <p>Python · Flask · Pandas · NumPy · SQLite/SQL · KMeans · Logistic Regression · Matplotlib · Seaborn · SciPy · Chart.js · ETL · RFM · Cohort Analysis</p>
</div>
<div class="nav">
  <button class="pill on" onclick="show('ov',this)">📈 Overview</button>
  <button class="pill" onclick="show('rfm',this)">🎯 RFM & Segments</button>
  <button class="pill" onclick="show('sq',this)">🗄️ SQL Analytics</button>
  <button class="pill" onclick="show('ch',this)">📊 Charts</button>
  <button class="pill" onclick="show('ml',this)">🤖 Churn Model</button>
  <button class="pill" onclick="show('ap',this)">⚡ REST API</button>
</div>
<div class="wrap">

<div id="ov" class="sec on">
  <div class="kgrid" id="kgrid"><div class="kcard"><div class="klbl">Loading…</div></div></div>
  <div class="cgrid">
    <div class="ccard"><h3>🔵 RFM Scatter by Segment — Matplotlib</h3><img src="/chart/rfm_scatter"></div>
    <div class="ccard"><h3>📊 Segment Distribution — KMeans (k=4)</h3><img src="/chart/segments"></div>
  </div>
  <div class="ccard"><h3>📈 Monthly Revenue Trend — Chart.js</h3><canvas id="monChart" height="80"></canvas></div>
</div>

<div id="rfm" class="sec">
  <div class="tcard">
    <h3>🎯 Top 20 Customers by Monetary Value — SQL ORDER BY</h3>
    <div class="qbox">SELECT customer_id, recency, frequency, monetary, R_score, F_score, M_score, segment FROM rfm ORDER BY monetary DESC LIMIT 20</div>
    <table id="trfm"></table>
  </div>
  <div class="ccard"><h3>🔥 RFM Score Heatmap (R vs F → Avg Monetary) — Seaborn</h3><img src="/chart/rfm_heat"></div>
</div>

<div id="sq" class="sec">
  <div class="tcard">
    <h3>🗺️ City Revenue Analysis — SQL CTE + JOIN</h3>
    <div class="qbox">WITH city_stats AS (SELECT c.city, COUNT(*) customers, SUM(t.amount) revenue … FROM customers c LEFT JOIN transactions t ON c.customer_id = t.customer_id GROUP BY c.city) SELECT *, revenue*100/total AS rev_share …</div>
    <table id="tcity"></table>
  </div>
  <div class="ccard"><h3>🛍️ Spend by Category — Seaborn</h3><img src="/chart/category"></div>
</div>

<div id="ch" class="sec">
  <div class="ccard"><h3>📊 Category Revenue — Chart.js Doughnut</h3><canvas id="catChart" height="120"></canvas></div>
  <div class="ccard" style="margin-top:20px"><h3>📦 Segment Size — Chart.js Bar</h3><canvas id="segChart" height="80"></canvas></div>
</div>

<div id="ml" class="sec">
  <div class="sgrid">
    <div class="scard"><h3>🤖 Churn Prediction — Logistic Regression</h3><div id="churnmet"></div></div>
    <div class="scard"><h3>🧪 A/B Test — Premium vs Basic AOV</h3><div id="abtest"></div></div>
  </div>
  <div class="ccard"><h3>🎯 Feature Importance (Churn Coefficients)</h3><canvas id="featChart" height="100"></canvas></div>
</div>

<div id="ap" class="sec">
  <div class="acard"><h3>⚡ REST API Endpoints</h3>
    <div style="margin-top:12px">
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/kpis</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/segments — KMeans customer segments</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/rfm-top — Top 20 customers by RFM</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/city-analysis — SQL CTE city performance</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/monthly-revenue — Time-series data</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/churn-model — Logistic Regression results</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/cohort — Cohort retention data</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/ab-test — A/B test Premium vs Basic</div>
    </div>
  </div>
  <div class="acard"><h3>🛠️ Tech Stack</h3><div style="margin-top:10px">
    <span class="tag">Python</span><span class="tag">Flask</span><span class="tag">Pandas</span><span class="tag">NumPy</span>
    <span class="tag">SQLite</span><span class="tag">SQL CTEs</span><span class="tag">SQL JOINs</span>
    <span class="tag">KMeans Clustering</span><span class="tag">Logistic Regression</span>
    <span class="tag">RFM Analysis</span><span class="tag">Cohort Analysis</span>
    <span class="tag">Matplotlib</span><span class="tag">Seaborn</span><span class="tag">Chart.js</span>
    <span class="tag">A/B Testing</span><span class="tag">SciPy</span><span class="tag">REST API</span><span class="tag">ETL Pipeline</span>
  </div></div>
</div>
</div>

<script>
function show(id,btn){
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('on'));
  document.querySelectorAll('.pill').forEach(p=>p.classList.remove('on'));
  document.getElementById(id).classList.add('on'); btn.classList.add('on');
}
const fmt=v=>'₹'+(v/1e6).toFixed(2)+'M';

async function loadKPIs(){
  const d=await fetch('/api/kpis').then(r=>r.json());
  document.getElementById('kgrid').innerHTML=`
    <div class="kcard"><div class="klbl">Total Customers</div><div class="kval">${d.total_customers.toLocaleString()}</div></div>
    <div class="kcard"><div class="klbl">Transactions</div><div class="kval">${d.total_transactions.toLocaleString()}</div></div>
    <div class="kcard"><div class="klbl">Total Revenue</div><div class="kval">${fmt(d.total_revenue)}</div></div>
    <div class="kcard"><div class="klbl">Avg Order Value</div><div class="kval">₹${d.avg_order_value.toLocaleString()}</div></div>
    <div class="kcard"><div class="klbl">Churn Rate</div><div class="kval">${d.churn_rate_pct}%</div></div>
    <div class="kcard"><div class="klbl">Avg Frequency</div><div class="kval">${d.avg_frequency}</div></div>`;
}

async function loadMonthly(){
  const d=await fetch('/api/monthly-revenue').then(r=>r.json());
  new Chart(document.getElementById('monChart'),{
    type:'line',
    data:{labels:d.map(x=>`${x.year}-${String(x.month).padStart(2,'0')}`),datasets:[
      {label:'Revenue',data:d.map(x=>x.revenue),borderColor:'#a78bfa',backgroundColor:'#a78bfa15',fill:true,tension:0.4},
      {label:'Unique Customers',data:d.map(x=>x.unique_customers*1000),borderColor:'#38bdf8',backgroundColor:'#38bdf815',fill:true,tension:0.4}
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:12},grid:{color:'#1e293b'}},y:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}}}}
  });
}

async function loadRFM(){
  const d=await fetch('/api/rfm-top').then(r=>r.json());
  const segColor={'Champions':'gn','Loyal':'bl','At Risk':'or','New / Occasional':'bl'};
  document.getElementById('trfm').innerHTML=`
    <tr><th>Customer ID</th><th>Recency</th><th>Frequency</th><th>Monetary</th><th>R</th><th>F</th><th>M</th><th>Segment</th></tr>
    ${d.map(x=>`<tr><td>${x.customer_id}</td><td>${x.recency}d</td><td>${x.frequency}</td><td>₹${Number(x.monetary).toLocaleString()}</td><td>${x.R_score}</td><td>${x.F_score}</td><td>${x.M_score}</td><td><span class="badge ${segColor[x.segment]||'bl'}">${x.segment}</span></td></tr>`).join('')}`;
}

async function loadCity(){
  const d=await fetch('/api/city-analysis').then(r=>r.json());
  document.getElementById('tcity').innerHTML=`
    <tr><th>City</th><th>Customers</th><th>Transactions</th><th>Revenue</th><th>Avg Order</th><th>Rev Share</th></tr>
    ${d.map(x=>`<tr><td>${x.city}</td><td>${x.customers}</td><td>${x.transactions}</td><td>${fmt(x.revenue)}</td><td>₹${Number(x.avg_order).toLocaleString()}</td><td><span class="badge bl">${x.rev_share}%</span></td></tr>`).join('')}`;
}

async function loadCharts(){
  const segs=await fetch('/api/segments').then(r=>r.json());
  const colors=['#4ade80','#38bdf8','#f87171','#facc15'];
  new Chart(document.getElementById('segChart'),{
    type:'bar',
    data:{labels:segs.map(s=>s.segment),datasets:[{label:'Customers',data:segs.map(s=>s.count),backgroundColor:colors,borderWidth:0}]},
    options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#94a3b8'},grid:{color:'#1e293b'}},y:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}}}}
  });
  const mon=await fetch('/api/monthly-revenue').then(r=>r.json());
  const catData=await fetch('/api/city-analysis').then(r=>r.json());
  new Chart(document.getElementById('catChart'),{
    type:'doughnut',
    data:{labels:catData.map(c=>c.city),datasets:[{data:catData.map(c=>c.revenue),backgroundColor:['#a78bfa','#38bdf8','#4ade80','#facc15','#f87171','#fb923c'],borderWidth:0}]},
    options:{responsive:true,plugins:{legend:{position:'right',labels:{color:'#94a3b8'}}}}
  });
}

async function loadML(){
  const d=await fetch('/api/churn-model').then(r=>r.json());
  const ab=await fetch('/api/ab-test').then(r=>r.json());
  document.getElementById('churnmet').innerHTML=`
    <div class="srow"><span class="sk">Model</span><span class="sv">Logistic Regression</span></div>
    <div class="srow"><span class="sk">Accuracy</span><span class="sv">${(d.accuracy*100).toFixed(1)}%</span></div>
    <div class="srow"><span class="sk">Features</span><span class="sv">${d.features.length}</span></div>`;
  document.getElementById('abtest').innerHTML=`
    <div class="srow"><span class="sk">Premium AOV</span><span class="sv">₹${ab.premium.mean.toLocaleString()}</span></div>
    <div class="srow"><span class="sk">Basic AOV</span><span class="sv">₹${ab.basic.mean.toLocaleString()}</span></div>
    <div class="srow"><span class="sk">P-Value</span><span class="sv">${ab.p_value}</span></div>
    <div class="srow"><span class="sk">Significant?</span><span class="sv">${ab.significant?'✅ Yes':'❌ No'}</span></div>`;
  new Chart(document.getElementById('featChart'),{
    type:'bar',
    data:{labels:d.features.map(f=>f.name),datasets:[{label:'Coefficient',data:d.features.map(f=>f.coef),backgroundColor:d.features.map(f=>f.coef>0?'#f87171aa':'#4ade80aa'),borderColor:d.features.map(f=>f.coef>0?'#f87171':'#4ade80'),borderWidth:2}]},
    options:{responsive:true,indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}},y:{ticks:{color:'#94a3b8'},grid:{color:'#1e293b'}}}}
  });
}

loadKPIs();loadMonthly();loadRFM();loadCity();loadCharts();loadML();
</script>
</body></html>"""

@app.route('/')
def index(): return HTML

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
