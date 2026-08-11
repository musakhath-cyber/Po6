import re
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Page Configuration
st.set_page_config(
    page_title="Procurement Basket Optimizer",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# 1. CATALOG & BUYING MANUAL INGESTION MODULE
# ======================================================================

PERISHABLE_CATEGORIES = {
    "fresh veg", "fresh produce", "produce", "meat", "poultry", "dairy",
    "butchery", "seafood", "fish", "bakery", "fruit", "fresh fruit", "veg",
    "vegetables", "fresh"
}
NON_PERISHABLE_CATEGORIES = {
    "dry goods", "chemicals", "cleaning", "packaging", "beverages", "grocery",
    "canned", "canned goods", "spices", "oils", "frozen", "dry", "non-perishable"
}

MIN_MATRIX_COLS = 60


def _normalize_columns(df, min_cols=MIN_MATRIX_COLS):
    """
    Pad a DataFrame out to a fixed minimum column count and give it
    plain integer column labels (0..n-1), without relying on `names=`
    at read time (which breaks if the file has more real columns than
    the fixed width).
    """
    df = df.copy()
    n_cols = df.shape[1]
    if n_cols < min_cols:
        for i in range(n_cols, min_cols):
            df[i] = ""
    df.columns = list(range(df.shape[1]))
    return df


def _parse_price_val(raw_val):
    if pd.isna(raw_val):
        return None
    val_str = str(raw_val).strip()
    if not val_str or val_str.lower() in ("nan", "none", "null", "0", "0.0", "-"):
        return None
    cleaned = re.sub(r"[^\d.,\-]", "", val_str)
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(".") > cleaned.rfind(","):
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts) <= 2:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    try:
        val = float(cleaned)
        return val if val > 0 else None
    except ValueError:
        return None


def _infer_perishable(category):
    if not category or not isinstance(category, str):
        return True
    cat_lower = category.strip().lower()
    if any(kw in cat_lower for kw in ["dry", "packaging", "chemical", "cleaning", "beverage", "canned", "frozen", "non-perishable"]):
        return False
    return True


def parse_buying_manual_bytes(file_bytes, filename="uploaded_file.csv", default_supplier_name="Uploaded Supplier"):
    """
    Universal buying manual parser supporting single-supplier sheets, multi-supplier matrix tables,
    Excel workbooks, CSV files, and PDF documents.

    Returns a (suppliers_dict, error_message) tuple. error_message is None on
    success (even if suppliers_dict ends up empty because no headers matched);
    it is set when parsing itself failed outright (e.g. missing pdfplumber),
    so the caller can show the user the real reason instead of a generic
    "couldn't detect columns" message.
    """
    fname_lower = filename.lower()
    is_excel = fname_lower.endswith((".xlsx", ".xls"))
    is_pdf = fname_lower.endswith(".pdf")

    all_dfs = []
    pdf_parse_error = None

    if is_pdf:
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    pdf_rows = []
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        if tables:
                            for tbl in tables:
                                for r in tbl:
                                    if r and any(r):
                                        pdf_rows.append([str(c).strip() if c is not None else "" for c in r])
                        else:
                            text = page.extract_text()
                            if text:
                                for line in text.split("\n"):
                                    parts = line.split()
                                    if parts:
                                        pdf_rows.append(parts)
                    if pdf_rows:
                        max_cols = max(len(r) for r in pdf_rows)
                        max_cols = max(max_cols, MIN_MATRIX_COLS)
                        padded_rows = [r + [""] * (max_cols - len(r)) for r in pdf_rows]
                        all_dfs.append(pd.DataFrame(padded_rows))
            except Exception as e:
                pdf_parse_error = str(e)
        else:
            pdf_parse_error = "pdfplumber is not installed, so PDF buying manuals can't be read."
    elif is_excel:
        try:
            xl = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
            for sheet in xl.sheet_names:
                try:
                    raw_df = xl.parse(sheet, header=None)
                    all_dfs.append(_normalize_columns(raw_df))
                except Exception:
                    pass
        except Exception:
            try:
                raw_df = pd.read_csv(io.BytesIO(file_bytes), header=None, on_bad_lines='skip')
                all_dfs.append(_normalize_columns(raw_df))
            except Exception:
                pass
    else:
        parsed = False
        for sep in [",", ";", "\t", "|"]:
            try:
                raw_df = pd.read_csv(io.BytesIO(file_bytes), header=None, sep=sep, on_bad_lines='skip')
                if raw_df.shape[1] > 1:
                    all_dfs.append(_normalize_columns(raw_df))
                    parsed = True
                    break
            except Exception:
                continue
        if not parsed:
            try:
                text_str = file_bytes.decode('utf-8', errors='ignore') if isinstance(file_bytes, bytes) else file_bytes
                raw_df = pd.read_csv(io.StringIO(text_str), header=None, on_bad_lines='skip')
                all_dfs.append(_normalize_columns(raw_df))
            except Exception:
                pass

    suppliers = {}

    header_kws = [
        "stock item", "item description", "product description", "product name",
        "deal price", "best price", "price excl", "customer products", "item",
        "product", "description", "particulars", "article", "details"
    ]

    for df in all_dfs:
        rows = df.values.tolist()
        i = 0
        n_rows = len(rows)

        while i < n_rows:
            row_vals = [str(val).strip() if not pd.isna(val) else "" for val in rows[i]]
            row_str_lower = " ".join(row_vals).lower()

            if any(kw in row_str_lower for kw in header_kws):
                current_cols = row_vals

                item_col_idx = None
                cat_col_idx = None
                uom_col_idx = None

                for c_idx, c_name in enumerate(current_cols):
                    cn = c_name.lower().strip()
                    if cn in ["stock item", "item description", "product description", "product name", "stock_item", "description", "customer products", "item", "product", "article"]:
                        item_col_idx = c_idx
                        break
                if item_col_idx is None:
                    for c_idx, c_name in enumerate(current_cols):
                        cn = c_name.lower().strip()
                        if ("item" in cn or "product" in cn or "description" in cn or "particular" in cn) and ("group" not in cn and "code" not in cn and "category" not in cn):
                            item_col_idx = c_idx
                            break
                if item_col_idx is None:
                    item_col_idx = 0

                for c_idx, c_name in enumerate(current_cols):
                    cn = c_name.lower().strip()
                    if "category" in cn or "dept" in cn or "group" in cn:
                        cat_col_idx = c_idx
                        break

                for c_idx, c_name in enumerate(current_cols):
                    cn = c_name.lower().strip()
                    if "uom" in cn or "unit" in cn or "pack" in cn or "measure" in cn or "size" in cn:
                        uom_col_idx = c_idx
                        break

                ignore_kws = {
                    "category", "bm", "stock item", "item group", "uom", "code", "new code",
                    "best price", "preferred supplier", "item description", "product description",
                    "barcode", "product code", "product sub category", "product category", "measure",
                    "pack", "deal price excl", "april", "january", "february", "march", "price excl vat", "price",
                    "customer products", "category level 1", "category level 2", "ucm", "units per case", "unit size",
                    "kgs per case", "case price", "price per kg/l", "price per unit", "item", "product", "description"
                }

                supplier_price_cols = []
                single_price_col = None

                for c_idx, c_name in enumerate(current_cols):
                    cn = c_name.strip()
                    if not cn or cn.lower().startswith("unnamed"):
                        continue
                    if cn.lower() not in ignore_kws and not cn.lower().startswith("code"):
                        supplier_price_cols.append((c_idx, cn))
                    elif any(p_kw in cn.lower() for p_kw in ["price per unit", "deal price", "best price", "price", "april", "january", "february", "cost", "rate", "amount", "zar", "r"]):
                        if single_price_col is None:
                            single_price_col = (c_idx, cn)

                i += 1
                while i < n_rows:
                    data_row = [str(val).strip() if not pd.isna(val) else "" for val in rows[i]]
                    data_row_str = " ".join(data_row).lower()

                    if any(kw in data_row_str for kw in ["stock item", "item description", "product description", "deal price", "preferred supplier"]):
                        break

                    if len(data_row) > item_col_idx:
                        raw_item = data_row[item_col_idx]
                        if raw_item and not raw_item.lower().startswith("category") and not raw_item.lower().startswith("product") and not raw_item.lower().startswith("region") and not raw_item.lower().startswith("item"):
                            category = data_row[cat_col_idx] if cat_col_idx is not None and len(data_row) > cat_col_idx else "General"
                            uom = data_row[uom_col_idx] if uom_col_idx is not None and len(data_row) > uom_col_idx else ""

                            display_name = raw_item
                            if uom and uom.strip().lower() not in raw_item.strip().lower():
                                display_name = f"{raw_item} ({uom})"

                            is_perish = _infer_perishable(category)

                            if supplier_price_cols:
                                for s_idx, s_name in supplier_price_cols:
                                    if len(data_row) > s_idx:
                                        p_raw = data_row[s_idx]
                                        p_val = _parse_price_val(p_raw)
                                        if p_val is not None:
                                            s_key = s_name.strip()
                                            if s_key not in suppliers:
                                                suppliers[s_key] = {
                                                    "moq": 1000.0 if "GROCERY" in s_key.upper() else 500.0,
                                                    "free_delivery_threshold": 1000.0 if "GROCERY" in s_key.upper() else 500.0,
                                                    "delivery_fee": 150.0 if "GROCERY" in s_key.upper() else 80.0,
                                                    "catalog": {}
                                                }
                                            suppliers[s_key]["catalog"][display_name] = {
                                                "price": p_val,
                                                "category": category,
                                                "perishable": is_perish
                                            }
                            elif single_price_col is not None:
                                sp_idx, sp_name = single_price_col
                                if len(data_row) > sp_idx:
                                    p_val = _parse_price_val(data_row[sp_idx])
                                    if p_val is not None:
                                        s_key = default_supplier_name
                                        if s_key not in suppliers:
                                            suppliers[s_key] = {
                                                "moq": 500.0,
                                                "free_delivery_threshold": 500.0,
                                                "delivery_fee": 80.0,
                                                "catalog": {}
                                            }
                                        suppliers[s_key]["catalog"][display_name] = {
                                            "price": p_val,
                                            "category": category,
                                            "perishable": is_perish
                                        }

                    i += 1
                continue

            i += 1

    # Secondary Pass: Direct tabular DataFrame fallback if matrix parsing yields 0 suppliers
    if not suppliers:
        try:
            if is_excel:
                try:
                    df_tab = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
                except Exception:
                    df_tab = pd.read_excel(io.BytesIO(file_bytes), engine='xlrd')
            else:
                try:
                    df_tab = pd.read_csv(io.BytesIO(file_bytes))
                except Exception:
                    df_tab = pd.read_csv(io.BytesIO(file_bytes), sep=";")

            if df_tab is not None and not df_tab.empty:
                cols = [str(c).strip() for c in df_tab.columns]
                item_col, price_col, cat_col, uom_col = None, None, None, None

                for c in cols:
                    cl = c.lower()
                    if any(k in cl for k in ["item", "product", "description", "name", "particular"]):
                        item_col = c
                        break
                for c in cols:
                    cl = c.lower()
                    if any(k in cl for k in ["price", "cost", "rate", "amount", "zar", "r"]):
                        price_col = c
                        break
                for c in cols:
                    cl = c.lower()
                    if any(k in cl for k in ["cat", "dept", "group"]):
                        cat_col = c
                        break
                for c in cols:
                    cl = c.lower()
                    if any(k in cl for k in ["uom", "unit", "pack", "size"]):
                        uom_col = c
                        break

                if item_col and price_col:
                    s_key = default_supplier_name
                    suppliers[s_key] = {
                        "moq": 500.0,
                        "free_delivery_threshold": 500.0,
                        "delivery_fee": 80.0,
                        "catalog": {}
                    }
                    for _, row in df_tab.iterrows():
                        raw_item = str(row[item_col]).strip() if pd.notna(row[item_col]) else ""
                        p_val = _parse_price_val(row[price_col])
                        if raw_item and p_val is not None:
                            cat = str(row[cat_col]).strip() if cat_col and pd.notna(row[cat_col]) else "General"
                            uom = str(row[uom_col]).strip() if uom_col and pd.notna(row[uom_col]) else ""
                            disp = f"{raw_item} ({uom})" if uom and uom.lower() not in raw_item.lower() else raw_item
                            suppliers[s_key]["catalog"][disp] = {
                                "price": p_val,
                                "category": cat,
                                "perishable": _infer_perishable(cat)
                            }
        except Exception:
            pass

    if not suppliers and pdf_parse_error:
        return {}, pdf_parse_error
    return suppliers, None


# ======================================================================
# 2. MILP OPTIMIZATION SOLVER ENGINE
# ======================================================================

VMAX_DEFAULT = 1_000_000.0
DEFAULT_BUFFER_CAP = 5.0
DEFAULT_HOLDING_COST_FACTOR = 0.02


def build_and_solve(suppliers, required_demand,
                     buffer_caps=None, vmax=VMAX_DEFAULT,
                     buffer_holding_cost_factor=DEFAULT_HOLDING_COST_FACTOR,
                     integer_required=False):
    """
    integer_required: if True, required-item purchase quantities must also be
    integers. Defaults to False because required demand is often fractional
    (e.g. 2.5 kg) and forcing integrality there can silently make the MILP
    infeasible. Buffer items are always kept integer since they represent
    discrete restock units.
    """
    if not suppliers:
        raise ValueError("No suppliers provided for optimization.")
    if not required_demand:
        raise ValueError("No required demand provided for optimization.")

    buffer_caps = buffer_caps or {}
    supplier_names = list(suppliers.keys())
    n_suppliers = len(supplier_names)

    required_demand = {k: float(v) for k, v in required_demand.items() if float(v) > 0}
    if not required_demand:
        raise ValueError("Required demand must contain at least one item with quantity > 0.")

    buffer_items = set()
    for s in supplier_names:
        for item, meta in suppliers[s]["catalog"].items():
            if item in required_demand:
                continue
            if not meta.get("perishable", True):
                buffer_items.add(item)

    pairs = []
    for s in supplier_names:
        for item in suppliers[s]["catalog"]:
            if item in required_demand or item in buffer_items:
                pairs.append((item, s))

    if not pairs:
        raise ValueError("No valid item-supplier pairs found to satisfy required demand.")

    n_x = len(pairs)
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}
    n_y = n_suppliers
    n_z = n_suppliers
    n_vars = n_x + n_y + n_z
    y_offset = n_x
    z_offset = n_x + n_y

    dynamic_vmax = max(vmax, 1000000.0)

    c = np.zeros(n_vars)
    integrality = np.zeros(n_vars)

    for (item, s), idx in pair_index.items():
        price = float(suppliers[s]["catalog"][item]["price"])
        if item in buffer_items:
            c[idx] = float(buffer_holding_cost_factor) * price
            integrality[idx] = 1
        else:
            c[idx] = price
            if integer_required:
                integrality[idx] = 1

    for j, s in enumerate(supplier_names):
        f_s = float(suppliers[s]["delivery_fee"])
        c[y_offset + j] = f_s
        c[z_offset + j] = -f_s
        integrality[y_offset + j] = 1
        integrality[z_offset + j] = 1

    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)

    for (item, s), idx in pair_index.items():
        if item in buffer_items:
            ub[idx] = float(buffer_caps.get(item, DEFAULT_BUFFER_CAP))
        else:
            ub[idx] = 100_000.0

    for j in range(n_suppliers):
        lb[y_offset + j], ub[y_offset + j] = 0, 1
        lb[z_offset + j], ub[z_offset + j] = 0, 1

    bounds = Bounds(lb, ub)
    constraints = []

    # 1. Demand Satisfaction
    for item, d_i in required_demand.items():
        row = np.zeros(n_vars)
        found = False
        for s in supplier_names:
            if item in suppliers[s]["catalog"] and (item, s) in pair_index:
                row[pair_index[(item, s)]] = 1.0
                found = True
        if not found:
            raise ValueError(f"Required item '{item}' not found in any supplier catalog.")
        constraints.append(LinearConstraint(row, d_i, d_i))

    # 2. Supplier Active & Free Delivery Threshold Linkage
    for j, s in enumerate(supplier_names):
        spend_row = np.zeros(n_vars)
        for item in suppliers[s]["catalog"]:
            if (item, s) in pair_index:
                spend_row[pair_index[(item, s)]] = float(suppliers[s]["catalog"][item]["price"])

        T_s = float(suppliers[s].get("free_delivery_threshold", suppliers[s].get("moq", 500.0)))

        # Big-M constraint
        row2 = spend_row.copy()
        row2[y_offset + j] = -dynamic_vmax
        constraints.append(LinearConstraint(row2, -np.inf, 0))

        # Free Delivery constraint
        row4a = -spend_row.copy()
        row4a[z_offset + j] = T_s
        constraints.append(LinearConstraint(row4a, -np.inf, 0))

        # Logic constraint: z_s <= y_s
        row4b = np.zeros(n_vars)
        row4b[z_offset + j] = 1
        row4b[y_offset + j] = -1
        constraints.append(LinearConstraint(row4b, -np.inf, 0))

    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    return result, pairs, pair_index, supplier_names, y_offset, z_offset, buffer_items


def report(result, pairs, pair_index, supplier_names, y_offset, z_offset,
           suppliers, buffer_items):
    if not result or not result.success or result.x is None:
        return None

    x = result.x
    breakdown = {}
    total_required_spend = 0.0
    total_buffer_invest = 0.0
    total_delivery_fees = 0.0

    for j, s in enumerate(supplier_names):
        y_val = x[y_offset + j]
        z_val = x[z_offset + j]
        if y_val < 0.5:
            continue

        items_bought = []
        s_req_spend = 0.0
        s_buf_spend = 0.0

        for (item, sup), idx in pair_index.items():
            if sup != s:
                continue
            qty = x[idx]
            if qty > 1e-6:
                price = float(suppliers[s]["catalog"][item]["price"])
                subtotal = qty * price
                is_buf = item in buffer_items
                kind = "Buffer Stock" if is_buf else "Required"
                if is_buf:
                    s_buf_spend += subtotal
                else:
                    s_req_spend += subtotal
                items_bought.append((item, qty, price, subtotal, kind))

        s_product_spend = s_req_spend + s_buf_spend
        delivery_fee = float(suppliers[s]["delivery_fee"]) if z_val < 0.5 else 0.0
        po_total = s_product_spend + delivery_fee

        total_required_spend += s_req_spend
        total_buffer_invest += s_buf_spend
        total_delivery_fees += delivery_fee

        breakdown[s] = {
            "required_spend": s_req_spend,
            "buffer_spend": s_buf_spend,
            "product_spend": s_product_spend,
            "delivery_fee": delivery_fee,
            "po_total": po_total,
            "free_delivery": z_val > 0.5,
            "items": items_bought
        }

    total_cash_outlay = total_required_spend + total_buffer_invest + total_delivery_fees
    effective_perishable_cost = total_required_spend + total_delivery_fees

    return {
        "suppliers": breakdown,
        "total_required_spend": total_required_spend,
        "total_buffer_invest": total_buffer_invest,
        "total_delivery_fees": total_delivery_fees,
        "total_cash_outlay": total_cash_outlay,
        "effective_perishable_cost": effective_perishable_cost,
    }


# ======================================================================
# 3. STREAMLIT WEB APP UI
# ======================================================================

st.markdown("""
<div style="background-color: #1E293B; padding: 20px; border-radius: 10px; margin-bottom: 25px; color: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
    <h1 style="color: #38BDF8; margin: 0; padding-bottom: 5px;">📊 Procurement Basket Optimizer</h1>
    <p style="color: #94A3B8; margin: 0; font-size: 1.05rem;">Multi-supplier catalog ingestion, demand planning, and MILP basket cost minimization dashboard</p>
</div>
""", unsafe_allow_html=True)

if "suppliers" not in st.session_state:
    st.session_state.suppliers = {}
if "demand" not in st.session_state:
    st.session_state.demand = {}
if "uploaded_raw_bytes" not in st.session_state:
    st.session_state.uploaded_raw_bytes = None
if "uploaded_raw_filename" not in st.session_state:
    st.session_state.uploaded_raw_filename = ""

def load_demo_data():
    st.session_state.suppliers = {
        "Bell Ceres": {
            "moq": 500.0, "free_delivery_threshold": 500.0, "delivery_fee": 80.0,
            "catalog": {
                "Lettuce Fresh (ea)": {"price": 25.0, "category": "Fresh Veg", "perishable": True},
                "Tomatoes (kg)": {"price": 21.0, "category": "Fresh Veg", "perishable": True},
                "Potatoes Large (kg)": {"price": 13.80, "category": "Fresh Veg", "perishable": True},
                "Onions White (kg)": {"price": 13.00, "category": "Fresh Veg", "perishable": True},
                "Cucumbers Fresh (kg)": {"price": 35.00, "category": "Fresh Veg", "perishable": True},
                "Carrots (kg)": {"price": 14.80, "category": "Fresh Veg", "perishable": True},
                "Spinach (kg)": {"price": 32.00, "category": "Fresh Veg", "perishable": True},
            },
        },
        "Grocery Express": {
            "moq": 1000.0, "free_delivery_threshold": 1000.0, "delivery_fee": 150.0,
            "catalog": {
                "Cucumbers Fresh (kg)": {"price": 20.00, "category": "Fresh Veg", "perishable": True},
                "Tomatoes (kg)": {"price": 28.00, "category": "Fresh Veg", "perishable": True},
                "Potatoes Large (kg)": {"price": 18.00, "category": "Fresh Veg", "perishable": True},
                "Cooking Oil 20L (ea)": {"price": 980.00, "category": "Dry Goods", "perishable": False},
                "Rice White 10kg (ea)": {"price": 220.00, "category": "Dry Goods", "perishable": False},
                "Flour Cake 12.5kg (ea)": {"price": 246.10, "category": "Dry Goods", "perishable": False},
                "Sugar White 25kg (ea)": {"price": 763.44, "category": "Dry Goods", "perishable": False},
                "Baked Beans A10 (ea)": {"price": 115.22, "category": "Dry Goods", "perishable": False},
            },
        },
        "Cuyler Butchery": {
            "moq": 800.0, "free_delivery_threshold": 800.0, "delivery_fee": 100.0,
            "catalog": {
                "Beef Mince (kg)": {"price": 99.94, "category": "Meat", "perishable": True},
                "Chicken Breast Fillet (kg)": {"price": 75.35, "category": "Meat", "perishable": True},
                "Pork Chops (kg)": {"price": 109.53, "category": "Meat", "perishable": True},
                "Beef Boerewors (kg)": {"price": 102.68, "category": "Meat", "perishable": True},
            },
        },
        "Crickley Dairy": {
            "moq": 400.0, "free_delivery_threshold": 400.0, "delivery_fee": 60.0,
            "catalog": {
                "Fresh Milk 2L (ea)": {"price": 31.44, "category": "Dairy", "perishable": True},
                "Cheddar Cheese Bulk (kg)": {"price": 112.22, "category": "Dairy", "perishable": True},
                "Yoghurt Assorted 1kg (ea)": {"price": 27.82, "category": "Dairy", "perishable": True},
            },
        },
        "Unick Foods": {
            "moq": 700.0, "free_delivery_threshold": 700.0, "delivery_fee": 100.0,
            "catalog": {
                "Chicken Thighs (kg)": {"price": 58.84, "category": "Meat", "perishable": True},
                "Chicken Leg Quarters (kg)": {"price": 47.50, "category": "Meat", "perishable": True},
                "Hake Fillets 4-6 (kg)": {"price": 173.00, "category": "Meat", "perishable": True},
            },
        },
    }
    st.session_state.demand = {
        "Lettuce Fresh (ea)": 15.0,
        "Tomatoes (kg)": 30.0,
        "Potatoes Large (kg)": 50.0,
        "Onions White (kg)": 30.0,
        "Cucumbers Fresh (kg)": 20.0,
        "Beef Mince (kg)": 25.0,
        "Chicken Thighs (kg)": 35.0,
        "Fresh Milk 2L (ea)": 20.0,
        "Cheddar Cheese Bulk (kg)": 10.0,
    }

st.sidebar.header("⚙️ Supplier Setup & Manuals")

if st.sidebar.button("⚡ Load Demo Canteen Dataset", use_container_width=True):
    load_demo_data()
    st.sidebar.success("Loaded demo dataset!")
    st.rerun()

st.sidebar.divider()

supplier_name = st.sidebar.text_input("Supplier name (optional for matrix tables)", key="s_name_input")
free_threshold = st.sidebar.number_input("Free-delivery threshold / MOQ (R)", min_value=0.0, value=500.0, step=50.0, key="s_free_input")
delivery_fee = st.sidebar.number_input("Delivery fee (R)", min_value=0.0, value=100.0, step=10.0, key="s_fee_input")

uploaded = st.sidebar.file_uploader(
    "Upload Buying Manual (Excel/CSV/PDF)",
    type=["csv", "xlsx", "xls", "pdf"],
    key="s_file_input"
)

if uploaded is not None:
    st.session_state["uploaded_raw_bytes"] = uploaded.getvalue()
    st.session_state["uploaded_raw_filename"] = uploaded.name

if st.session_state["uploaded_raw_bytes"] is not None:
    s_key = supplier_name.strip() if supplier_name.strip() else "Uploaded Supplier"
    file_bytes = st.session_state["uploaded_raw_bytes"]
    fname = st.session_state["uploaded_raw_filename"]

    st.sidebar.info(f"📎 Attached: {fname} ({len(file_bytes)/1024:.1f} KB)")

    parsed_suppliers, parse_error = parse_buying_manual_bytes(file_bytes, filename=fname, default_supplier_name=s_key)
    if parsed_suppliers:
        count = 0
        for name, s_dict in parsed_suppliers.items():
            if free_threshold > 0:
                s_dict["moq"] = float(free_threshold)
                s_dict["free_delivery_threshold"] = float(free_threshold)
            if delivery_fee > 0:
                s_dict["delivery_fee"] = float(delivery_fee)
            st.session_state.suppliers[name] = s_dict
            count += len(s_dict["catalog"])
        st.sidebar.success(f"✅ Loaded {len(parsed_suppliers)} supplier(s) with {count} items from {fname}!")
    elif parse_error:
        st.sidebar.error(f"⚠️ Could not parse {fname}: {parse_error}")
    else:
        st.sidebar.warning(f"⚠️ Could not detect item or price columns in {fname}. Please check that column headers include 'Item'/'Product' and 'Price'/'Cost'.")

with st.sidebar.expander("📝 Mobile Fallback: Paste CSV/Text Catalog"):
    pasted_csv = st.text_area("Paste CSV (e.g. Item, Price or Matrix)", key="pasted_csv_input")
    if st.button("➕ Process Pasted CSV", type="secondary", use_container_width=True):
        if pasted_csv.strip():
            s_key = supplier_name.strip() if supplier_name.strip() else "Uploaded Supplier"
            parsed, paste_error = parse_buying_manual_bytes(pasted_csv.encode('utf-8'), filename="pasted.csv", default_supplier_name=s_key)
            if parsed:
                count = 0
                for name, s_dict in parsed.items():
                    if free_threshold > 0:
                        s_dict["moq"] = float(free_threshold)
                        s_dict["free_delivery_threshold"] = float(free_threshold)
                    if delivery_fee > 0:
                        s_dict["delivery_fee"] = float(delivery_fee)
                    st.session_state.suppliers[name] = s_dict
                    count += len(s_dict["catalog"])
                st.sidebar.success(f"Added {len(parsed)} supplier(s) with {count} items!")
                st.rerun()
            elif paste_error:
                st.sidebar.error(f"⚠️ Could not parse pasted text: {paste_error}")
            else:
                st.sidebar.warning("⚠️ Could not detect item or price columns in the pasted text.")

st.sidebar.divider()
st.sidebar.subheader("Active Suppliers")

if st.session_state.suppliers:
    for name, data in list(st.session_state.suppliers.items()):
        c1, c2 = st.sidebar.columns(2)
        c1.write(f"**{name}** — {len(data['catalog'])} items")
        if c2.button("✕", key=f"remove_{name}"):
            del st.session_state.suppliers[name]
            st.rerun()
else:
    st.sidebar.info("No suppliers added yet.")

tab1, tab2, tab3 = st.tabs(["📦 Demand Planning", "⚙️ Optimization Settings", "📊 Visual Results & Analytics"])

with tab1:
    st.subheader("Required Item Demand")
    st.write("Enter the required quantities for your purchasing cycle.")

    all_items = sorted({
        item
        for supplier in st.session_state.suppliers.values()
        for item in supplier["catalog"]
    })

    if not all_items:
        st.info("Add at least one supplier or click '⚡ Load Demo Canteen Dataset' in the sidebar.")
    else:
        default_rows = []
        for item in all_items:
            default_rows.append({
                "Item": item,
                "Required quantity": float(st.session_state.demand.get(item, 0.0)),
            })
        df = pd.DataFrame(default_rows)
        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Required quantity": st.column_config.NumberColumn(
                    min_value=0.0, step=1.0, format="%.2f"
                )
            },
            disabled=["Item"],
            key="demand_editor",
        )

        if st.button("💾 Save Demand Matrix", type="primary"):
            st.session_state.demand = {
                row["Item"]: float(row["Required quantity"])
                for _, row in edited.iterrows()
                if float(row["Required quantity"]) > 0
            }
            st.success(f"Saved {len(st.session_state.demand)} required items.")

with tab2:
    st.subheader("Optimization Parameters")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        integer_qty = st.checkbox(
            "Force integer quantities for required items",
            value=False,
            help="Leave off if any required item quantities are fractional (e.g. 2.5 kg) — "
                 "forcing integers on fractional demand makes the solver infeasible."
        )
    with col_b:
        holding_pct = st.number_input("Buffer holding cost factor (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    with col_c:
        buffer_cap = st.number_input("Default buffer cap (units)", min_value=0.0, value=5.0, step=1.0)

    st.write("**Active Demand Summary**")
    if st.session_state.demand:
        st.dataframe(
            pd.DataFrame(
                [{"Item": k, "Quantity": v} for k, v in st.session_state.demand.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No demand has been saved yet.")

    if st.button("🚀 Run MILP Basket Optimization", type="primary", use_container_width=True):
        if not st.session_state.suppliers:
            st.error("Add at least one supplier.")
        elif not st.session_state.demand:
            st.error("Enter and save at least one required item.")
        else:
            try:
                missing = [
                    item for item in st.session_state.demand
                    if not any(item in s["catalog"] for s in st.session_state.suppliers.values())
                ]
                if missing:
                    st.error("Required items missing from all catalogs: " + ", ".join(missing))
                else:
                    buffer_items_check = {
                        item for s in st.session_state.suppliers.values()
                        for item, meta in s["catalog"].items()
                        if item not in st.session_state.demand and not meta.get("perishable", True)
                    }
                    caps_dict = {item: float(buffer_cap) for item in buffer_items_check} if buffer_cap > 0 else None

                    result, pairs, pair_index, supplier_names, y_offset, z_offset, buffer_items = build_and_solve(
                        st.session_state.suppliers,
                        st.session_state.demand,
                        buffer_caps=caps_dict,
                        buffer_holding_cost_factor=float(holding_pct) / 100.0,
                        integer_required=integer_qty,
                    )

                    st.session_state.optimization = {
                        "result": result,
                        "pairs": pairs,
                        "pair_index": pair_index,
                        "supplier_names": supplier_names,
                        "y_offset": y_offset,
                        "z_offset": z_offset,
                        "buffer_items": buffer_items,
                    }
                    if result and result.success:
                        st.success("Optimization complete successfully!")
                    else:
                        msg = result.message if result else "Solver failed."
                        if integer_qty:
                            msg += " (Tip: if any required quantities are fractional, try unchecking 'Force integer quantities'.)"
                        st.error(f"Solver failed: {msg}")
            except Exception as e:
                st.exception(e)

with tab3:
    st.subheader("Optimized Basket & Visual Analytics")
    opt = st.session_state.get("optimization")

    if not opt:
        st.info("Run the optimizer to generate visual reports and breakdown charts.")
    elif not opt["result"] or not opt["result"].success:
        msg = opt["result"].message if opt["result"] else "No result available."
        st.error(msg)
    else:
        result = opt["result"]
        suppliers = st.session_state.suppliers
        pairs = opt["pairs"]
        pair_index = opt["pair_index"]
        supplier_names = opt["supplier_names"]
        y_offset = opt["y_offset"]
        z_offset = opt["z_offset"]
        buffer_items = opt["buffer_items"]

        rep = report(result, pairs, pair_index, supplier_names, y_offset, z_offset,
                    suppliers, buffer_items)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Cash Outlay", f"R{rep['total_cash_outlay']:,.2f}")
        m2.metric("Effective Perishable Cost", f"R{rep['effective_perishable_cost']:,.2f}")
        m3.metric("Required Spend", f"R{rep['total_required_spend']:,.2f}")
        m4.metric("Buffer Investment", f"R{rep['total_buffer_invest']:,.2f}")

        st.divider()

        rows = []
        supplier_summary = rep['suppliers']

        for s, s_data in supplier_summary.items():
            for item, qty, price, subtotal, kind in s_data['items']:
                rows.append({
                    "Supplier": s,
                    "Item": item,
                    "Type": kind,
                    "Quantity": qty,
                    "Unit Price (R)": price,
                    "Subtotal (R)": subtotal,
                })

        st.markdown("### 📈 Visual Cost & Spend Distribution")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### Spend Allocation per Supplier (R)")
            if supplier_summary:
                plot_data = {
                    s: {
                        "Required Spend": data["required_spend"],
                        "Buffer Invest": data["buffer_spend"],
                        "Delivery Fee": data["delivery_fee"]
                    }
                    for s, data in supplier_summary.items()
                }
                summary_df = pd.DataFrame(plot_data).T
                fig, ax = plt.subplots(figsize=(6, 4))
                summary_df.plot(kind="bar", stacked=True, ax=ax, color=["#3B82F6", "#10B981", "#EF4444"])
                ax.set_ylabel("Cost (ZAR)")
                ax.set_title("Cost Components by Supplier", fontsize=11, fontweight="bold")
                plt.xticks(rotation=15, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with chart_col2:
            st.markdown("##### Basket Type Breakdown")
            if rows:
                result_df = pd.DataFrame(rows)
                type_spend = result_df.groupby("Type")["Subtotal (R)"].sum()
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                colors = ["#3B82F6", "#10B981"]
                ax2.pie(type_spend, labels=type_spend.index, autopct="%1.1f%%", startangle=90, colors=colors[:len(type_spend)], wedgeprops=dict(width=0.4, edgecolor='w'))
                ax2.set_title("Required Demand vs Buffer Investment", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

        st.divider()

        st.markdown("### 🏢 Supplier Purchase Orders & Threshold Gauges")
        for s, s_data in supplier_summary.items():
            with st.expander(f"📌 {s.upper()} — Total PO: R{s_data['po_total']:,.2f}", expanded=True):
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Product Spend", f"R{s_data['product_spend']:,.2f}")
                c_b.metric("Free Delivery Threshold", f"R{suppliers[s]['free_delivery_threshold']:,.2f}")
                c_c.metric("Delivery Surcharge", "FREE" if s_data['free_delivery'] else f"R{s_data['delivery_fee']:,.2f}")

                moq_pct = min(1.0, s_data["product_spend"] / suppliers[s]["free_delivery_threshold"]) if suppliers[s]["free_delivery_threshold"] > 0 else 1.0
                st.caption(f"Free Delivery Progress: {moq_pct*100:.1f}% met")
                st.progress(moq_pct)

        if rows:
            st.markdown("### 🛒 Itemized Order Basket")
            result_df = pd.DataFrame(rows)
            st.dataframe(
                result_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Quantity": st.column_config.NumberColumn(format="%.2f"),
                    "Unit Price (R)": st.column_config.NumberColumn(format="R%.2f"),
                    "Subtotal (R)": st.column_config.NumberColumn(format="R%.2f"),
                },
            )

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Purchase Basket CSV",
                csv,
                "optimized_purchase_basket.csv",
                "text/csv",
            )

st.divider()
st.caption("Procurement Optimizer Web Dashboard | SciPy MILP Engine & Streamlit")
import re
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
import pdfplumber

# Page Configuration
st.set_page_config(
    page_title="Procurement Basket Optimizer",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# 1. CATALOG & BUYING MANUAL INGESTION MODULE
# ======================================================================

PERISHABLE_CATEGORIES = {
    "fresh veg", "fresh produce", "produce", "meat", "poultry", "dairy",
    "butchery", "seafood", "fish", "bakery", "fruit", "fresh fruit", "veg",
    "vegetables", "fresh"
}
NON_PERISHABLE_CATEGORIES = {
    "dry goods", "chemicals", "cleaning", "packaging", "beverages", "grocery",
    "canned", "canned goods", "spices", "oils", "frozen", "dry", "non-perishable"
}


def _parse_price_val(raw_val):
    if pd.isna(raw_val):
        return None
    val_str = str(raw_val).strip()
    if not val_str or val_str.lower() in ("nan", "none", "null", "0", "0.0", "-"):
        return None
    cleaned = re.sub(r"[^\d.,\-]", "", val_str)
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(".") > cleaned.rfind(","):
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts) <= 2:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    try:
        val = float(cleaned)
        return val if val > 0 else None
    except ValueError:
        return None


def _infer_perishable(category):
    if not category or not isinstance(category, str):
        return True
    cat_lower = category.strip().lower()
    if any(kw in cat_lower for kw in ["dry", "packaging", "chemical", "cleaning", "beverage", "canned", "frozen", "non-perishable"]):
        return False
    return True


def parse_buying_manual_bytes(file_bytes, filename="uploaded_file.csv", default_supplier_name="Uploaded Supplier"):
    """
    Universal buying manual parser supporting single-supplier sheets, multi-supplier matrix tables,
    Excel workbooks, CSV files, and PDF documents.
    """
    fname_lower = filename.lower()
    is_excel = fname_lower.endswith((".xlsx", ".xls"))
    is_pdf = fname_lower.endswith(".pdf")

    all_dfs = []

    if is_pdf:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pdf_rows = []
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        for tbl in tables:
                            for r in tbl:
                                if r and any(r):
                                    pdf_rows.append([str(c).strip() if c is not None else "" for c in r])
                    else:
                        text = page.extract_text()
                        if text:
                            for line in text.split("\n"):
                                parts = line.split()
                                if parts:
                                    pdf_rows.append(parts)
                if pdf_rows:
                    max_cols = max(len(r) for r in pdf_rows)
                    max_cols = max(max_cols, 60)
                    padded_rows = [r + [""] * (max_cols - len(r)) for r in pdf_rows]
                    all_dfs.append(pd.DataFrame(padded_rows))
        except Exception:
            pass
    elif is_excel:
        try:
            xl = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
            for sheet in xl.sheet_names:
                try:
                    raw_df = xl.parse(sheet, header=None, names=list(range(60)))
                    all_dfs.append(raw_df)
                except Exception:
                    pass
        except Exception:
            try:
                raw_df = pd.read_csv(io.BytesIO(file_bytes), header=None, names=list(range(60)), on_bad_lines='skip')
                all_dfs.append(raw_df)
            except Exception:
                pass
    else:
        parsed = False
        for sep in [",", ";", "\t", "|"]:
            try:
                raw_df = pd.read_csv(io.BytesIO(file_bytes), header=None, sep=sep, names=list(range(60)), on_bad_lines='skip')
                if raw_df.shape > 1:
                    all_dfs.append(raw_df)
                    parsed = True
                    break
            except Exception:
                continue
        if not parsed:
            try:
                text_str = file_bytes.decode('utf-8', errors='ignore') if isinstance(file_bytes, bytes) else file_bytes
                raw_df = pd.read_csv(io.StringIO(text_str), header=None, names=list(range(60)), on_bad_lines='skip')
                all_dfs.append(raw_df)
            except Exception:
                pass

    suppliers = {}

    header_kws = [
        "stock item", "item description", "product description", "product name",
        "deal price", "best price", "price excl", "customer products", "item",
        "product", "description", "particulars", "article", "details"
    ]

    for df in all_dfs:
        rows = df.values.tolist()
        i = 0
        n_rows = len(rows)

        while i < n_rows:
            row_vals = [str(val).strip() if not pd.isna(val) else "" for val in rows[i]]
            row_str_lower = " ".join(row_vals).lower()

            if any(kw in row_str_lower for kw in header_kws):
                current_cols = row_vals

                item_col_idx = None
                cat_col_idx = None
                uom_col_idx = None

                for c_idx, c_name in enumerate(current_cols):
                    cn = c_name.lower().strip()
                    if cn in ["stock item", "item description", "product description", "product name", "stock_item", "description", "customer products", "item", "product", "article"]:
                        item_col_idx = c_idx
                        break
                if item_col_idx is None:
                    for c_idx, c_name in enumerate(current_cols):
                        cn = c_name.lower().strip()
                        if ("item" in cn or "product" in cn or "description" in cn or "particular" in cn) and ("group" not in cn and "code" not in cn and "category" not in cn):
                            item_col_idx = c_idx
                            break
                if item_col_idx is None:
                    item_col_idx = 0

                for c_idx, c_name in enumerate(current_cols):
                    cn = c_name.lower().strip()
                    if "category" in cn or "dept" in cn or "group" in cn:
                        cat_col_idx = c_idx
                        break

                for c_idx, c_name in enumerate(current_cols):
                    cn = c_name.lower().strip()
                    if "uom" in cn or "unit" in cn or "pack" in cn or "measure" in cn or "size" in cn:
                        uom_col_idx = c_idx
                        break

                ignore_kws = {
                    "category", "bm", "stock item", "item group", "uom", "code", "new code",
                    "best price", "preferred supplier", "item description", "product description",
                    "barcode", "product code", "product sub category", "product category", "measure",
                    "pack", "deal price excl", "april", "january", "february", "march", "price excl vat", "price",
                    "customer products", "category level 1", "category level 2", "ucm", "units per case", "unit size",
                    "kgs per case", "case price", "price per kg/l", "price per unit", "item", "product", "description"
                }

                supplier_price_cols = []
                single_price_col = None

                for c_idx, c_name in enumerate(current_cols):
                    cn = c_name.strip()
                    if not cn or cn.lower().startswith("unnamed"):
                        continue
                    if cn.lower() not in ignore_kws and not cn.lower().startswith("code"):
                        supplier_price_cols.append((c_idx, cn))
                    elif any(p_kw in cn.lower() for p_kw in ["price per unit", "deal price", "best price", "price", "april", "january", "february", "cost", "rate", "amount", "zar", "r"]):
                        if single_price_col is None:
                            single_price_col = (c_idx, cn)

                i += 1
                while i < n_rows:
                    data_row = [str(val).strip() if not pd.isna(val) else "" for val in rows[i]]
                    data_row_str = " ".join(data_row).lower()

                    if any(kw in data_row_str for kw in ["stock item", "item description", "product description", "deal price", "preferred supplier"]):
                        break

                    if len(data_row) > item_col_idx:
                        raw_item = data_row[item_col_idx]
                        if raw_item and not raw_item.lower().startswith("category") and not raw_item.lower().startswith("product") and not raw_item.lower().startswith("region") and not raw_item.lower().startswith("item"):
                            category = data_row[cat_col_idx] if cat_col_idx is not None and len(data_row) > cat_col_idx else "General"
                            uom = data_row[uom_col_idx] if uom_col_idx is not None and len(data_row) > uom_col_idx else ""

                            display_name = raw_item
                            if uom and uom.strip().lower() not in raw_item.strip().lower():
                                display_name = f"{raw_item} ({uom})"

                            is_perish = _infer_perishable(category)

                            if supplier_price_cols:
                                for s_idx, s_name in supplier_price_cols:
                                    if len(data_row) > s_idx:
                                        p_raw = data_row[s_idx]
                                        p_val = _parse_price_val(p_raw)
                                        if p_val is not None:
                                            s_key = s_name.strip()
                                            if s_key not in suppliers:
                                                suppliers[s_key] = {
                                                    "moq": 1000.0 if "GROCERY" in s_key.upper() else 500.0,
                                                    "free_delivery_threshold": 1000.0 if "GROCERY" in s_key.upper() else 500.0,
                                                    "delivery_fee": 150.0 if "GROCERY" in s_key.upper() else 80.0,
                                                    "catalog": {}
                                                }
                                            suppliers[s_key]["catalog"][display_name] = {
                                                "price": p_val,
                                                "category": category,
                                                "perishable": is_perish
                                            }
                            elif single_price_col is not None:
                                sp_idx, sp_name = single_price_col
                                if len(data_row) > sp_idx:
                                    p_val = _parse_price_val(data_row[sp_idx])
                                    if p_val is not None:
                                        s_key = default_supplier_name
                                        if s_key not in suppliers:
                                            suppliers[s_key] = {
                                                "moq": 500.0,
                                                "free_delivery_threshold": 500.0,
                                                "delivery_fee": 80.0,
                                                "catalog": {}
                                            }
                                        suppliers[s_key]["catalog"][display_name] = {
                                            "price": p_val,
                                            "category": category,
                                            "perishable": is_perish
                                        }

                    i += 1
                continue

            i += 1

    # Secondary Pass: Direct tabular DataFrame fallback if matrix parsing yields 0 suppliers
    if not suppliers:
        try:
            if is_excel:
                df_tab = pd.read_excel(io.BytesIO(file_bytes))
            else:
                try:
                    df_tab = pd.read_csv(io.BytesIO(file_bytes))
                except Exception:
                    df_tab = pd.read_csv(io.BytesIO(file_bytes), sep=";")

            if df_tab is not None and not df_tab.empty:
                cols = [str(c).strip() for c in df_tab.columns]
                item_col, price_col, cat_col, uom_col = None, None, None, None

                for c in cols:
                    cl = c.lower()
                    if any(k in cl for k in ["item", "product", "description", "name", "particular"]):
                        item_col = c
                        break
                for c in cols:
                    cl = c.lower()
                    if any(k in cl for k in ["price", "cost", "rate", "amount", "zar", "r"]):
                        price_col = c
                        break
                for c in cols:
                    cl = c.lower()
                    if any(k in cl for k in ["cat", "dept", "group"]):
                        cat_col = c
                        break
                for c in cols:
                    cl = c.lower()
                    if any(k in cl for k in ["uom", "unit", "pack", "size"]):
                        uom_col = c
                        break

                if item_col and price_col:
                    s_key = default_supplier_name
                    suppliers[s_key] = {
                        "moq": 500.0,
                        "free_delivery_threshold": 500.0,
                        "delivery_fee": 80.0,
                        "catalog": {}
                    }
                    for _, row in df_tab.iterrows():
                        raw_item = str(row[item_col]).strip() if pd.notna(row[item_col]) else ""
                        p_val = _parse_price_val(row[price_col])
                        if raw_item and p_val is not None:
                            cat = str(row[cat_col]).strip() if cat_col and pd.notna(row[cat_col]) else "General"
                            uom = str(row[uom_col]).strip() if uom_col and pd.notna(row[uom_col]) else ""
                            disp = f"{raw_item} ({uom})" if uom and uom.lower() not in raw_item.lower() else raw_item
                            suppliers[s_key]["catalog"][disp] = {
                                "price": p_val,
                                "category": cat,
                                "perishable": _infer_perishable(cat)
                            }
        except Exception:
            pass

    return suppliers


# ======================================================================
# 2. MILP OPTIMIZATION SOLVER ENGINE
# ======================================================================

VMAX_DEFAULT = 1_000_000.0
DEFAULT_BUFFER_CAP = 5.0
DEFAULT_HOLDING_COST_FACTOR = 0.02


def build_and_solve(suppliers, required_demand,
                     buffer_caps=None, vmax=VMAX_DEFAULT,
                     buffer_holding_cost_factor=DEFAULT_HOLDING_COST_FACTOR,
                     integer_required=True):
    if not suppliers:
        raise ValueError("No suppliers provided for optimization.")
    if not required_demand:
        raise ValueError("No required demand provided for optimization.")

    buffer_caps = buffer_caps or {}
    supplier_names = list(suppliers.keys())
    n_suppliers = len(supplier_names)

    required_demand = {k: float(v) for k, v in required_demand.items() if float(v) > 0}
    if not required_demand:
        raise ValueError("Required demand must contain at least one item with quantity > 0.")

    buffer_items = set()
    for s in supplier_names:
        for item, meta in suppliers[s]["catalog"].items():
            if item in required_demand:
                continue
            if not meta.get("perishable", True):
                buffer_items.add(item)

    pairs = []
    for s in supplier_names:
        for item in suppliers[s]["catalog"]:
            if item in required_demand or item in buffer_items:
                pairs.append((item, s))

    if not pairs:
        raise ValueError("No valid item-supplier pairs found to satisfy required demand.")

    n_x = len(pairs)
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}
    n_y = n_suppliers
    n_z = n_suppliers
    n_vars = n_x + n_y + n_z
    y_offset = n_x
    z_offset = n_x + n_y

    dynamic_vmax = max(vmax, 1000000.0)

    c = np.zeros(n_vars)
    integrality = np.zeros(n_vars)

    for (item, s), idx in pair_index.items():
        price = float(suppliers[s]["catalog"][item]["price"])
        if item in buffer_items:
            c[idx] = float(buffer_holding_cost_factor) * price
            integrality[idx] = 1
        else:
            c[idx] = price
            if integer_required:
                integrality[idx] = 1

    for j, s in enumerate(supplier_names):
        f_s = float(suppliers[s]["delivery_fee"])
        c[y_offset + j] = f_s
        c[z_offset + j] = -f_s
        integrality[y_offset + j] = 1
        integrality[z_offset + j] = 1

    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)

    for (item, s), idx in pair_index.items():
        if item in buffer_items:
            ub[idx] = float(buffer_caps.get(item, DEFAULT_BUFFER_CAP))
        else:
            ub[idx] = 100_000.0

    for j in range(n_suppliers):
        lb[y_offset + j], ub[y_offset + j] = 0, 1
        lb[z_offset + j], ub[z_offset + j] = 0, 1

    bounds = Bounds(lb, ub)
    constraints = []

    # 1. Demand Satisfaction
    for item, d_i in required_demand.items():
        row = np.zeros(n_vars)
        found = False
        for s in supplier_names:
            if item in suppliers[s]["catalog"] and (item, s) in pair_index:
                row[pair_index[(item, s)]] = 1.0
                found = True
        if not found:
            raise ValueError(f"Required item '{item}' not found in any supplier catalog.")
        constraints.append(LinearConstraint(row, d_i, d_i))

    # 2. Supplier Active & Free Delivery Threshold Linkage
    for j, s in enumerate(supplier_names):
        spend_row = np.zeros(n_vars)
        for item in suppliers[s]["catalog"]:
            if (item, s) in pair_index:
                spend_row[pair_index[(item, s)]] = float(suppliers[s]["catalog"][item]["price"])

        T_s = float(suppliers[s].get("free_delivery_threshold", suppliers[s].get("moq", 500.0)))

        # Big-M constraint
        row2 = spend_row.copy()
        row2[y_offset + j] = -dynamic_vmax
        constraints.append(LinearConstraint(row2, -np.inf, 0))

        # Free Delivery constraint
        row4a = -spend_row.copy()
        row4a[z_offset + j] = T_s
        constraints.append(LinearConstraint(row4a, -np.inf, 0))

        # Logic constraint: z_s <= y_s
        row4b = np.zeros(n_vars)
        row4b[z_offset + j] = 1
        row4b[y_offset + j] = -1
        constraints.append(LinearConstraint(row4b, -np.inf, 0))

    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    return result, pairs, pair_index, supplier_names, y_offset, z_offset, buffer_items


def report(result, pairs, pair_index, supplier_names, y_offset, z_offset,
           suppliers, buffer_items):
    if not result or not result.success or result.x is None:
        return None

    x = result.x
    breakdown = {}
    total_required_spend = 0.0
    total_buffer_invest = 0.0
    total_delivery_fees = 0.0

    for j, s in enumerate(supplier_names):
        y_val = x[y_offset + j]
        z_val = x[z_offset + j]
        if y_val < 0.5:
            continue

        items_bought = []
        s_req_spend = 0.0
        s_buf_spend = 0.0

        for (item, sup), idx in pair_index.items():
            if sup != s:
                continue
            qty = x[idx]
            if qty > 1e-6:
                price = float(suppliers[s]["catalog"][item]["price"])
                subtotal = qty * price
                is_buf = item in buffer_items
                kind = "Buffer Stock" if is_buf else "Required"
                if is_buf:
                    s_buf_spend += subtotal
                else:
                    s_req_spend += subtotal
                items_bought.append((item, qty, price, subtotal, kind))

        s_product_spend = s_req_spend + s_buf_spend
        delivery_fee = float(suppliers[s]["delivery_fee"]) if z_val < 0.5 else 0.0
        po_total = s_product_spend + delivery_fee

        total_required_spend += s_req_spend
        total_buffer_invest += s_buf_spend
        total_delivery_fees += delivery_fee

        breakdown[s] = {
            "required_spend": s_req_spend,
            "buffer_spend": s_buf_spend,
            "product_spend": s_product_spend,
            "delivery_fee": delivery_fee,
            "po_total": po_total,
            "free_delivery": z_val > 0.5,
            "items": items_bought
        }

    total_cash_outlay = total_required_spend + total_buffer_invest + total_delivery_fees
    effective_perishable_cost = total_required_spend + total_delivery_fees

    return {
        "suppliers": breakdown,
        "total_required_spend": total_required_spend,
        "total_buffer_invest": total_buffer_invest,
        "total_delivery_fees": total_delivery_fees,
        "total_cash_outlay": total_cash_outlay,
        "effective_perishable_cost": effective_perishable_cost,
    }


# ======================================================================
# 3. STREAMLIT WEB APP UI
# ======================================================================

st.markdown("""
<div style="background-color: #1E293B; padding: 20px; border-radius: 10px; margin-bottom: 25px; color: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
    <h1 style="color: #38BDF8; margin: 0; padding-bottom: 5px;">📊 Procurement Basket Optimizer</h1>
    <p style="color: #94A3B8; margin: 0; font-size: 1.05rem;">Multi-supplier catalog ingestion, demand planning, and MILP basket cost minimization dashboard</p>
</div>
""", unsafe_allow_html=True)

if "suppliers" not in st.session_state:
    st.session_state.suppliers = {}
if "demand" not in st.session_state:
    st.session_state.demand = {}
if "uploaded_raw_bytes" not in st.session_state:
    st.session_state.uploaded_raw_bytes = None
if "uploaded_raw_filename" not in st.session_state:
    st.session_state.uploaded_raw_filename = ""

def load_demo_data():
    st.session_state.suppliers = {
        "Bell Ceres": {
            "moq": 500.0, "free_delivery_threshold": 500.0, "delivery_fee": 80.0,
            "catalog": {
                "Lettuce Fresh (ea)": {"price": 25.0, "category": "Fresh Veg", "perishable": True},
                "Tomatoes (kg)": {"price": 21.0, "category": "Fresh Veg", "perishable": True},
                "Potatoes Large (kg)": {"price": 13.80, "category": "Fresh Veg", "perishable": True},
                "Onions White (kg)": {"price": 13.00, "category": "Fresh Veg", "perishable": True},
                "Cucumbers Fresh (kg)": {"price": 35.00, "category": "Fresh Veg", "perishable": True},
                "Carrots (kg)": {"price": 14.80, "category": "Fresh Veg", "perishable": True},
                "Spinach (kg)": {"price": 32.00, "category": "Fresh Veg", "perishable": True},
            },
        },
        "Grocery Express": {
            "moq": 1000.0, "free_delivery_threshold": 1000.0, "delivery_fee": 150.0,
            "catalog": {
                "Cucumbers Fresh (kg)": {"price": 20.00, "category": "Fresh Veg", "perishable": True},
                "Tomatoes (kg)": {"price": 28.00, "category": "Fresh Veg", "perishable": True},
                "Potatoes Large (kg)": {"price": 18.00, "category": "Fresh Veg", "perishable": True},
                "Cooking Oil 20L (ea)": {"price": 980.00, "category": "Dry Goods", "perishable": False},
                "Rice White 10kg (ea)": {"price": 220.00, "category": "Dry Goods", "perishable": False},
                "Flour Cake 12.5kg (ea)": {"price": 246.10, "category": "Dry Goods", "perishable": False},
                "Sugar White 25kg (ea)": {"price": 763.44, "category": "Dry Goods", "perishable": False},
                "Baked Beans A10 (ea)": {"price": 115.22, "category": "Dry Goods", "perishable": False},
            },
        },
        "Cuyler Butchery": {
            "moq": 800.0, "free_delivery_threshold": 800.0, "delivery_fee": 100.0,
            "catalog": {
                "Beef Mince (kg)": {"price": 99.94, "category": "Meat", "perishable": True},
                "Chicken Breast Fillet (kg)": {"price": 75.35, "category": "Meat", "perishable": True},
                "Pork Chops (kg)": {"price": 109.53, "category": "Meat", "perishable": True},
                "Beef Boerewors (kg)": {"price": 102.68, "category": "Meat", "perishable": True},
            },
        },
        "Crickley Dairy": {
            "moq": 400.0, "free_delivery_threshold": 400.0, "delivery_fee": 60.0,
            "catalog": {
                "Fresh Milk 2L (ea)": {"price": 31.44, "category": "Dairy", "perishable": True},
                "Cheddar Cheese Bulk (kg)": {"price": 112.22, "category": "Dairy", "perishable": True},
                "Yoghurt Assorted 1kg (ea)": {"price": 27.82, "category": "Dairy", "perishable": True},
            },
        },
        "Unick Foods": {
            "moq": 700.0, "free_delivery_threshold": 700.0, "delivery_fee": 100.0,
            "catalog": {
                "Chicken Thighs (kg)": {"price": 58.84, "category": "Meat", "perishable": True},
                "Chicken Leg Quarters (kg)": {"price": 47.50, "category": "Meat", "perishable": True},
                "Hake Fillets 4-6 (kg)": {"price": 173.00, "category": "Meat", "perishable": True},
            },
        },
    }
    st.session_state.demand = {
        "Lettuce Fresh (ea)": 15.0,
        "Tomatoes (kg)": 30.0,
        "Potatoes Large (kg)": 50.0,
        "Onions White (kg)": 30.0,
        "Cucumbers Fresh (kg)": 20.0,
        "Beef Mince (kg)": 25.0,
        "Chicken Thighs (kg)": 35.0,
        "Fresh Milk 2L (ea)": 20.0,
        "Cheddar Cheese Bulk (kg)": 10.0,
    }

st.sidebar.header("⚙️ Supplier Setup & Manuals")

if st.sidebar.button("⚡ Load Demo Canteen Dataset", use_container_width=True):
    load_demo_data()
    st.sidebar.success("Loaded demo dataset!")
    st.rerun()

st.sidebar.divider()

supplier_name = st.sidebar.text_input("Supplier name (optional for matrix tables)", key="s_name_input")
free_threshold = st.sidebar.number_input("Free-delivery threshold / MOQ (R)", min_value=0.0, value=500.0, step=50.0, key="s_free_input")
delivery_fee = st.sidebar.number_input("Delivery fee (R)", min_value=0.0, value=100.0, step=10.0, key="s_fee_input")

uploaded = st.sidebar.file_uploader(
    "Upload Buying Manual (Excel/CSV/PDF)",
    type=["csv", "xlsx", "xls", "pdf"],
    key="s_file_input"
)

if uploaded is not None:
    st.session_state["uploaded_raw_bytes"] = uploaded.getvalue()
    st.session_state["uploaded_raw_filename"] = uploaded.name

if st.session_state["uploaded_raw_bytes"] is not None:
    s_key = supplier_name.strip() if supplier_name.strip() else "Uploaded Supplier"
    file_bytes = st.session_state["uploaded_raw_bytes"]
    fname = st.session_state["uploaded_raw_filename"]

    st.sidebar.info(f"📎 Attached: {fname} ({len(file_bytes)/1024:.1f} KB)")

    parsed_suppliers = parse_buying_manual_bytes(file_bytes, filename=fname, default_supplier_name=s_key)
    if parsed_suppliers:
        count = 0
        for name, s_dict in parsed_suppliers.items():
            if free_threshold > 0:
                s_dict["moq"] = float(free_threshold)
                s_dict["free_delivery_threshold"] = float(free_threshold)
            if delivery_fee > 0:
                s_dict["delivery_fee"] = float(delivery_fee)
            st.session_state.suppliers[name] = s_dict
            count += len(s_dict["catalog"])
        st.sidebar.success(f"✅ Loaded {len(parsed_suppliers)} supplier(s) with {count} items from {fname}!")
    else:
        st.sidebar.warning(f"⚠️ Could not detect item or price columns in {fname}. Please check that column headers include 'Item'/'Product' and 'Price'/'Cost'.")

with st.sidebar.expander("📝 Mobile Fallback: Paste CSV/Text Catalog"):
    pasted_csv = st.text_area("Paste CSV (e.g. Item, Price or Matrix)", key="pasted_csv_input")
    if st.button("➕ Process Pasted CSV", type="secondary", use_container_width=True):
        if pasted_csv.strip():
            s_key = supplier_name.strip() if supplier_name.strip() else "Uploaded Supplier"
            parsed = parse_buying_manual_bytes(pasted_csv.encode('utf-8'), filename="pasted.csv", default_supplier_name=s_key)
            if parsed:
                count = 0
                for name, s_dict in parsed.items():
                    if free_threshold > 0:
                        s_dict["moq"] = float(free_threshold)
                        s_dict["free_delivery_threshold"] = float(free_threshold)
                    if delivery_fee > 0:
                        s_dict["delivery_fee"] = float(delivery_fee)
                    st.session_state.suppliers[name] = s_dict
                    count += len(s_dict["catalog"])
                st.sidebar.success(f"Added {len(parsed)} supplier(s) with {count} items!")
                st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Active Suppliers")

if st.session_state.suppliers:
    for name, data in list(st.session_state.suppliers.items()):
        c1, c2 = st.sidebar.columns()
        c1.write(f"**{name}** — {len(data['catalog'])} items")
        if c2.button("✕", key=f"remove_{name}"):
            del st.session_state.suppliers[name]
            st.rerun()
else:
    st.sidebar.info("No suppliers added yet.")

tab1, tab2, tab3 = st.tabs(["📦 Demand Planning", "⚙️ Optimization Settings", "📊 Visual Results & Analytics"])

with tab1:
    st.subheader("Required Item Demand")
    st.write("Enter the required quantities for your purchasing cycle.")

    all_items = sorted({
        item
        for supplier in st.session_state.suppliers.values()
        for item in supplier["catalog"]
    })

    if not all_items:
        st.info("Add at least one supplier or click '⚡ Load Demo Canteen Dataset' in the sidebar.")
    else:
        default_rows = []
        for item in all_items:
            default_rows.append({
                "Item": item,
                "Required quantity": float(st.session_state.demand.get(item, 0.0)),
            })
        df = pd.DataFrame(default_rows)
        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Required quantity": st.column_config.NumberColumn(
                    min_value=0.0, step=1.0, format="%.2f"
                )
            },
            disabled=["Item"],
            key="demand_editor",
        )

        if st.button("💾 Save Demand Matrix", type="primary"):
            st.session_state.demand = {
                row["Item"]: float(row["Required quantity"])
                for _, row in edited.iterrows()
                if float(row["Required quantity"]) > 0
            }
            st.success(f"Saved {len(st.session_state.demand)} required items.")

with tab2:
    st.subheader("Optimization Parameters")
    col_b, col_c = st.columns(2)
    with col_b:
        holding_pct = st.number_input("Buffer holding cost factor (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    with col_c:
        buffer_cap = st.number_input("Default buffer cap (units)", min_value=0.0, value=5.0, step=1.0)

    st.write("**Active Demand Summary**")
    if st.session_state.demand:
        st.dataframe(
            pd.DataFrame(
                [{"Item": k, "Quantity": v} for k, v in st.session_state.demand.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No demand has been saved yet.")

    if st.button("🚀 Run MILP Basket Optimization", type="primary", use_container_width=True):
        if not st.session_state.suppliers:
            st.error("Add at least one supplier.")
        elif not st.session_state.demand:
            st.error("Enter and save at least one required item.")
        else:
            try:
                missing = [
                    item for item in st.session_state.demand
                    if not any(item in s["catalog"] for s in st.session_state.suppliers.values())
                ]
                if missing:
                    st.error("Required items missing from all catalogs: " + ", ".join(missing))
                else:
                    buffer_items_check = {
                        item for s in st.session_state.suppliers.values()
                        for item, meta in s["catalog"].items()
                        if item not in st.session_state.demand and not meta.get("perishable", True)
                    }
                    caps_dict = {item: float(buffer_cap) for item in buffer_items_check} if buffer_cap > 0 else None

                    result, pairs, pair_index, supplier_names, y_offset, z_offset, buffer_items = build_and_solve(
                        st.session_state.suppliers,
                        st.session_state.demand,
                        buffer_caps=caps_dict,
                        buffer_holding_cost_factor=float(holding_pct) / 100.0,
                    )

                    st.session_state.optimization = {
                        "result": result,
                        "pairs": pairs,
                        "pair_index": pair_index,
                        "supplier_names": supplier_names,
                        "y_offset": y_offset,
                        "z_offset": z_offset,
                        "buffer_items": buffer_items,
                    }
                    if result and result.success:
                        st.success("Optimization complete successfully!")
                    else:
                        msg = result.message if result else "Solver failed."
                        st.error(f"Solver failed: {msg}")
            except Exception as e:
                st.exception(e)

with tab3:
    st.subheader("Optimized Basket & Visual Analytics")
    opt = st.session_state.get("optimization")

    if not opt:
        st.info("Run the optimizer to generate visual reports and breakdown charts.")
    elif not opt["result"] or not opt["result"].success:
        msg = opt["result"].message if opt["result"] else "No result available."
        st.error(msg)
    else:
        result = opt["result"]
        suppliers = st.session_state.suppliers
        pairs = opt["pairs"]
        pair_index = opt["pair_index"]
        supplier_names = opt["supplier_names"]
        y_offset = opt["y_offset"]
        z_offset = opt["z_offset"]
        buffer_items = opt["buffer_items"]

        rep = report(result, pairs, pair_index, supplier_names, y_offset, z_offset,
                    suppliers, buffer_items)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Cash Outlay", f"R{rep['total_cash_outlay']:,.2f}")
        m2.metric("Effective Perishable Cost", f"R{rep['effective_perishable_cost']:,.2f}")
        m3.metric("Required Spend", f"R{rep['total_required_spend']:,.2f}")
        m4.metric("Buffer Investment", f"R{rep['total_buffer_invest']:,.2f}")

        st.divider()

        rows = []
        supplier_summary = rep['suppliers']

        for s, s_data in supplier_summary.items():
            for item, qty, price, subtotal, kind in s_data['items']:
                rows.append({
                    "Supplier": s,
                    "Item": item,
                    "Type": kind,
                    "Quantity": qty,
                    "Unit Price (R)": price,
                    "Subtotal (R)": subtotal,
                })

        st.markdown("### 📈 Visual Cost & Spend Distribution")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### Spend Allocation per Supplier (R)")
            if supplier_summary:
                plot_data = {
                    s: {
                        "Required Spend": data["required_spend"],
                        "Buffer Invest": data["buffer_spend"],
                        "Delivery Fee": data["delivery_fee"]
                    }
                    for s, data in supplier_summary.items()
                }
                summary_df = pd.DataFrame(plot_data).T
                fig, ax = plt.subplots(figsize=(6, 4))
                summary_df.plot(kind="bar", stacked=True, ax=ax, color=["#3B82F6", "#10B981", "#EF4444"])
                ax.set_ylabel("Cost (ZAR)")
                ax.set_title("Cost Components by Supplier", fontsize=11, fontweight="bold")
                plt.xticks(rotation=15, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with chart_col2:
            st.markdown("##### Basket Type Breakdown")
            if rows:
                result_df = pd.DataFrame(rows)
                type_spend = result_df.groupby("Type")["Subtotal (R)"].sum()
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                colors = ["#3B82F6", "#10B981"]
                ax2.pie(type_spend, labels=type_spend.index, autopct="%1.1f%%", startangle=90, colors=colors[:len(type_spend)], wedgeprops=dict(width=0.4, edgecolor='w'))
                ax2.set_title("Required Demand vs Buffer Investment", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

        st.divider()

        st.markdown("### 🏢 Supplier Purchase Orders & Threshold Gauges")
        for s, s_data in supplier_summary.items():
            with st.expander(f"📌 {s.upper()} — Total PO: R{s_data['po_total']:,.2f}", expanded=True):
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Product Spend", f"R{s_data['product_spend']:,.2f}")
                c_b.metric("Free Delivery Threshold", f"R{suppliers[s]['free_delivery_threshold']:,.2f}")
                c_c.metric("Delivery Surcharge", "FREE" if s_data['free_delivery'] else f"R{s_data['delivery_fee']:,.2f}")

                moq_pct = min(1.0, s_data["product_spend"] / suppliers[s]["free_delivery_threshold"]) if suppliers[s]["free_delivery_threshold"] > 0 else 1.0
                st.caption(f"Free Delivery Progress: {moq_pct*100:.1f}% met")
                st.progress(moq_pct)

        if rows:
            st.markdown("### 🛒 Itemized Order Basket")
            result_df = pd.DataFrame(rows)
            st.dataframe(
                result_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Quantity": st.column_config.NumberColumn(format="%.2f"),
                    "Unit Price (R)": st.column_config.NumberColumn(format="R%.2f"),
                    "Subtotal (R)": st.column_config.NumberColumn(format="R%.2f"),
                },
            )

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Purchase Basket CSV",
                csv,
                "optimized_purchase_basket.csv",
                "text/csv",
            )

st.divider()
st.caption("Procurement Optimizer Web Dashboard | SciPy MILP Engine & Streamlit")
