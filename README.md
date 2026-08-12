# 🧾 Procurement Basket Optimizer

A Streamlit web dashboard for multi-supplier catalog ingestion, demand planning, and Mixed-Integer Linear Programming (MILP) basket cost minimization.

## Features
- **Universal Catalog Ingestion**: Upload Excel (`.xlsx`, `.xls`), CSV, or PDF buying manuals, or paste text tables.
- **MILP Optimization Engine**: Built on `scipy.optimize.milp` to optimize required demand allocation, buffer stock forward-buying, supplier Minimum Order Quantities (MOQ), and Free Delivery Thresholds ($T_s$).
- **Visual Analytics**: Interactive cost allocation charts, supplier PO progress gauges, and itemized purchase basket tables.
- **Multi-Format Export**: Download itemized purchase baskets and supplier purchase order summaries as CSV files.

## Local Setup & Execution

1. Clone or download the repository.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deploying on Streamlit Community Cloud

1. Push this repository (`app.py`, `requirements.txt`, `.streamlit/config.toml`, `README.md`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select your repository, branch, and `app.py` as the main entry file.
4. Click **Deploy**.
