import re
import io
import math
import difflib

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Page Configuration
st.set_page_config(
    page_title="Cherry Picker 🍒",
    page_icon="🍒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# 1. CATALOG & BUYING MANUAL INGESTION MODULE
# ======================================================================

DEFAULT_MOQ = 350.0
DEFAULT_DELIVERY_FEE = 80.0
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


def parse_buying_manual_bytes(file_bytes, filename="uploaded_file.csv", default_supplier_name="Uploaded Supplier",
                               default_moq=DEFAULT_MOQ, default_delivery_fee=DEFAULT_DELIVERY_FEE,
                               default_has_account=True):
    """
    Universal buying manual parser supporting single-supplier sheets, multi-supplier matrix tables,
    Excel workbooks, CSV files, and PDF documents.

    Returns a (suppliers_dict, error_message) tuple. error_message is None on
    success (even if suppliers_dict ends up empty because no headers matched);
    it is set when parsing itself failed outright (e.g. missing pdfplumber).
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

                            if supplier_price_cols:
                                for s_idx, s_name in supplier_price_cols:
                                    if len(data_row) > s_idx:
                                        p_raw = data_row[s_idx]
                                        p_val = _parse_price_val(p_raw)
                                        if p_val is not None:
                                            s_key = s_name.strip()
                                            if s_key not in suppliers:
                                                suppliers[s_key] = {
                                                    "moq": default_moq,
                                                    "delivery_fee": default_delivery_fee,
                                                    "has_account": default_has_account,
                                                    "catalog": {}
                                                }
                                            suppliers[s_key]["catalog"][display_name] = {
                                                "price": p_val,
                                                "category": category,
                                            }
                            elif single_price_col is not None:
                                sp_idx, sp_name = single_price_col
                                if len(data_row) > sp_idx:
                                    p_val = _parse_price_val(data_row[sp_idx])
                                    if p_val is not None:
                                        s_key = default_supplier_name
                                        if s_key not in suppliers:
                                            suppliers[s_key] = {
                                                "moq": default_moq,
                                                "delivery_fee": default_delivery_fee,
                                                "has_account": default_has_account,
                                                "catalog": {}
                                            }
                                        suppliers[s_key]["catalog"][display_name] = {
                                            "price": p_val,
                                            "category": category,
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
                        "moq": default_moq,
                        "delivery_fee": default_delivery_fee,
                        "has_account": default_has_account,
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
                            }
        except Exception:
            pass

    if not suppliers and pdf_parse_error:
        return {}, pdf_parse_error
    return suppliers, None


# ======================================================================
# 2. BASKET / MOQ ANALYSIS ENGINE
# ======================================================================

def compute_basket_summary(suppliers, basket):
    """
    basket: dict supplier_name -> dict item_name -> quantity
    Returns dict supplier_name -> summary dict (rows, subtotal, moq, gap, delivery fee, total)
    """
    summary = {}
    for supplier_name, items in basket.items():
        if supplier_name not in suppliers:
            continue
        catalog = suppliers[supplier_name]["catalog"]
        rows = []
        subtotal = 0.0
        for item, qty in items.items():
            if item not in catalog or qty is None or qty <= 0:
                continue
            price = catalog[item]["price"]
            line_total = price * qty
            subtotal += line_total
            rows.append({
                "Item": item,
                "Quantity": qty,
                "Unit Price (R)": price,
                "Subtotal (R)": line_total,
            })
        if not rows:
            continue
        moq = float(suppliers[supplier_name]["moq"])
        delivery_fee_full = float(suppliers[supplier_name]["delivery_fee"])
        meets_moq = subtotal >= moq
        delivery_fee_charged = 0.0 if meets_moq else delivery_fee_full
        summary[supplier_name] = {
            "rows": rows,
            "subtotal": subtotal,
            "moq": moq,
            "meets_moq": meets_moq,
            "gap": max(0.0, moq - subtotal),
            "delivery_fee_full": delivery_fee_full,
            "delivery_fee_charged": delivery_fee_charged,
            "total": subtotal + delivery_fee_charged,
        }
    return summary


def get_topup_suggestions(supplier_name, suppliers, basket, gap, max_suggestions=4):
    """Cheapest items from this supplier NOT already in the basket, with the
    quantity needed of each to close the MOQ gap on its own."""
    catalog = suppliers[supplier_name]["catalog"]
    current_items = basket.get(supplier_name, {})
    candidates = [
        (item, meta["price"])
        for item, meta in catalog.items()
        if item not in current_items and meta["price"] > 0
    ]
    candidates.sort(key=lambda x: x[1])
    suggestions = []
    for item, price in candidates[:max_suggestions]:
        qty_needed = max(1, math.ceil(gap / price))
        suggestions.append({
            "item": item,
            "price": price,
            "qty": qty_needed,
            "cost": qty_needed * price,
        })
    return suggestions


def get_alt_supplier_suggestions(supplier_name, suppliers, basket):
    """For each item currently sitting in supplier_name's basket, find the
    cheapest alternate supplier (from the given `suppliers` dict only — callers
    pass an account-only subset so out-of-account suppliers never get
    recommended here) who sells the same item for less, and work out what it
    would take to hit THAT supplier's MOQ too."""
    results = []
    current_items = basket.get(supplier_name, {})
    for item, qty in current_items.items():
        if supplier_name not in suppliers or item not in suppliers[supplier_name]["catalog"]:
            continue
        current_price = suppliers[supplier_name]["catalog"][item]["price"]
        best = None
        for alt_name, alt_data in suppliers.items():
            if alt_name == supplier_name:
                continue
            if item in alt_data["catalog"]:
                alt_price = alt_data["catalog"][item]["price"]
                if alt_price < current_price and (best is None or alt_price < best[1]):
                    best = (alt_name, alt_price)
        if best:
            alt_name, alt_price = best
            alt_moq = float(suppliers[alt_name]["moq"])
            alt_current_subtotal = sum(
                suppliers[alt_name]["catalog"][i]["price"] * q
                for i, q in basket.get(alt_name, {}).items()
                if i in suppliers[alt_name]["catalog"] and q > 0
            )
            hypothetical_subtotal = alt_current_subtotal + alt_price * qty
            alt_gap = max(0.0, alt_moq - hypothetical_subtotal)
            results.append({
                "item": item,
                "qty": qty,
                "current_price": current_price,
                "alt_supplier": alt_name,
                "alt_price": alt_price,
                "total_savings": (current_price - alt_price) * qty,
                "alt_subtotal_if_moved": hypothetical_subtotal,
                "alt_moq": alt_moq,
                "alt_gap": alt_gap,
            })
    results.sort(key=lambda r: -r["total_savings"])
    return results


# ======================================================================
# 3. SHOPPING LIST MATCHING & ACCOUNT-AWARE SOURCING
# ======================================================================

def find_best_item_match(typed_name, catalog_items):
    """Match free-typed text against a list of known catalog item names.
    Tries exact (case-insensitive), then substring, then fuzzy matching."""
    typed_clean = typed_name.strip().lower()
    if not typed_clean:
        return None

    catalog_items = list(catalog_items)

    for item in catalog_items:
        if item.lower() == typed_clean:
            return item

    substr_matches = [item for item in catalog_items if typed_clean in item.lower() or item.lower() in typed_clean]
    if substr_matches:
        substr_matches.sort(key=len)
        return substr_matches[0]

    close = difflib.get_close_matches(typed_name, catalog_items, n=1, cutoff=0.6)
    if close:
        return close[0]

    return None


def process_shopping_list(rows, suppliers):
    """
    rows: list of {"Item": str, "Quantity": float}
    suppliers: full suppliers dict (each with a has_account flag)

    Returns (results, smart_basket):
      results: one dict per requested row describing the match outcome
      smart_basket: dict supplier_name -> {item: qty}, built only from
                    suppliers the user has an account with
    """
    all_item_index = {}
    account_item_index = {}
    for s_name, s_data in suppliers.items():
        for item, meta in s_data["catalog"].items():
            all_item_index.setdefault(item, []).append((s_name, meta["price"]))
            if s_data.get("has_account", True):
                account_item_index.setdefault(item, []).append((s_name, meta["price"]))

    results = []
    smart_basket = {}

    for row in rows:
        typed_item = str(row.get("Item", "") or "").strip()
        raw_qty = row.get("Quantity", 0)
        try:
            qty = float(raw_qty) if raw_qty is not None else 0.0
        except (TypeError, ValueError):
            qty = 0.0

        if not typed_item or qty <= 0:
            continue

        matched_account_item = find_best_item_match(typed_item, account_item_index.keys())
        matched_any_item = find_best_item_match(typed_item, all_item_index.keys())

        if matched_account_item:
            candidates = account_item_index[matched_account_item]
            best_supplier, best_price = min(candidates, key=lambda x: x[1])
            subtotal = best_price * qty

            smart_basket.setdefault(best_supplier, {})
            smart_basket[best_supplier][matched_account_item] = smart_basket[best_supplier].get(matched_account_item, 0.0) + qty

            out_of_account_better = None
            if matched_any_item:
                non_account_candidates = [
                    (s, p) for s, p in all_item_index[matched_any_item]
                    if not suppliers[s].get("has_account", True)
                ]
                if non_account_candidates:
                    cheapest_noacct_supplier, cheapest_noacct_price = min(non_account_candidates, key=lambda x: x[1])
                    if cheapest_noacct_price < best_price:
                        out_of_account_better = {
                            "supplier": cheapest_noacct_supplier,
                            "price": cheapest_noacct_price,
                            "total_savings": (best_price - cheapest_noacct_price) * qty,
                        }

            results.append({
                "typed_item": typed_item,
                "matched_item": matched_account_item,
                "supplier": best_supplier,
                "price": best_price,
                "qty": qty,
                "subtotal": subtotal,
                "matched": True,
                "out_of_account_better": out_of_account_better,
                "out_of_account_only": None,
            })
        else:
            out_of_account_only = None
            if matched_any_item:
                cheapest_supplier, cheapest_price = min(all_item_index[matched_any_item], key=lambda x: x[1])
                out_of_account_only = {"supplier": cheapest_supplier, "price": cheapest_price}
            results.append({
                "typed_item": typed_item,
                "matched_item": matched_any_item,
                "supplier": None,
                "price": None,
                "qty": qty,
                "subtotal": None,
                "matched": False,
                "out_of_account_better": None,
                "out_of_account_only": out_of_account_only,
            })

    return results, smart_basket


def aggregate_out_of_account_savings(results):
    """Build a supplier -> {items, total_savings} table from shopping-list results,
    covering both 'cheaper elsewhere' items and items only available out-of-account."""
    agg = {}
    for r in results:
        oab = r.get("out_of_account_better")
        if oab:
            s = oab["supplier"]
            agg.setdefault(s, {"items": [], "total_savings": 0.0})
            agg[s]["items"].append({
                "item": r["matched_item"], "qty": r["qty"], "your_price": r["price"],
                "their_price": oab["price"], "savings": oab["total_savings"],
            })
            agg[s]["total_savings"] += oab["total_savings"]

        ooa = r.get("out_of_account_only")
        if not r["matched"] and ooa:
            s = ooa["supplier"]
            agg.setdefault(s, {"items": [], "total_savings": 0.0})
            agg[s]["items"].append({
                "item": r["matched_item"] or r["typed_item"], "qty": r["qty"], "your_price": None,
                "their_price": ooa["price"], "savings": None,
            })
    return agg


# ======================================================================
# 4. STREAMLIT WEB APP UI
# ======================================================================

st.markdown("""
<div style="background-color: #1E293B; padding: 20px; border-radius: 10px; margin-bottom: 25px; color: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
    <h1 style="color: #38BDF8; margin: 0; padding-bottom: 5px;">🍒 Cherry Picker</h1>
    <p style="color: #94A3B8; margin: 0; font-size: 1.05rem;">Pick items straight from supplier catalogs and let Cherry Picker chase down their MOQ for you</p>
</div>
""", unsafe_allow_html=True)

if "suppliers" not in st.session_state:
    st.session_state.suppliers = {}
if "basket" not in st.session_state:
    st.session_state.basket = {}
if "uploaded_raw_bytes" not in st.session_state:
    st.session_state.uploaded_raw_bytes = None
if "uploaded_raw_filename" not in st.session_state:
    st.session_state.uploaded_raw_filename = ""
if "shopping_list_df" not in st.session_state:
    st.session_state.shopping_list_df = pd.DataFrame({"Item": [""] * 5, "Quantity": [0.0] * 5})
if "shopping_list_results" not in st.session_state:
    st.session_state.shopping_list_results = None
if "shopping_list_smart_basket" not in st.session_state:
    st.session_state.shopping_list_smart_basket = None


def load_demo_data():
    st.session_state.suppliers = {
        "Bell Ceres": {
            "moq": DEFAULT_MOQ, "delivery_fee": 80.0, "has_account": True,
            "catalog": {
                "Lettuce Fresh (ea)": {"price": 25.0, "category": "Fresh Veg"},
                "Tomatoes (kg)": {"price": 21.0, "category": "Fresh Veg"},
                "Potatoes Large (kg)": {"price": 13.80, "category": "Fresh Veg"},
                "Onions White (kg)": {"price": 13.00, "category": "Fresh Veg"},
                "Cucumbers Fresh (kg)": {"price": 35.00, "category": "Fresh Veg"},
                "Carrots (kg)": {"price": 14.80, "category": "Fresh Veg"},
                "Spinach (kg)": {"price": 32.00, "category": "Fresh Veg"},
            },
        },
        "Grocery Express": {
            "moq": DEFAULT_MOQ, "delivery_fee": 150.0, "has_account": True,
            "catalog": {
                "Cucumbers Fresh (kg)": {"price": 20.00, "category": "Fresh Veg"},
                "Tomatoes (kg)": {"price": 28.00, "category": "Fresh Veg"},
                "Potatoes Large (kg)": {"price": 18.00, "category": "Fresh Veg"},
                "Cooking Oil 20L (ea)": {"price": 980.00, "category": "Dry Goods"},
                "Rice White 10kg (ea)": {"price": 220.00, "category": "Dry Goods"},
                "Flour Cake 12.5kg (ea)": {"price": 246.10, "category": "Dry Goods"},
                "Sugar White 25kg (ea)": {"price": 763.44, "category": "Dry Goods"},
                "Baked Beans A10 (ea)": {"price": 115.22, "category": "Dry Goods"},
            },
        },
        "Cuyler Butchery": {
            "moq": DEFAULT_MOQ, "delivery_fee": 100.0, "has_account": True,
            "catalog": {
                "Beef Mince (kg)": {"price": 99.94, "category": "Meat"},
                "Chicken Breast Fillet (kg)": {"price": 75.35, "category": "Meat"},
                "Pork Chops (kg)": {"price": 109.53, "category": "Meat"},
                "Beef Boerewors (kg)": {"price": 102.68, "category": "Meat"},
            },
        },
        "Crickley Dairy": {
            "moq": DEFAULT_MOQ, "delivery_fee": 60.0, "has_account": True,
            "catalog": {
                "Fresh Milk 2L (ea)": {"price": 31.44, "category": "Dairy"},
                "Cheddar Cheese Bulk (kg)": {"price": 112.22, "category": "Dairy"},
                "Yoghurt Assorted 1kg (ea)": {"price": 27.82, "category": "Dairy"},
            },
        },
        "Unick Foods": {
            "moq": DEFAULT_MOQ, "delivery_fee": 100.0, "has_account": False,
            "catalog": {
                "Chicken Thighs (kg)": {"price": 47.50, "category": "Meat"},
                "Chicken Leg Quarters (kg)": {"price": 47.50, "category": "Meat"},
                "Hake Fillets 4-6 (kg)": {"price": 173.00, "category": "Meat"},
                "Beef Mince (kg)": {"price": 89.00, "category": "Meat"},
            },
        },
    }
    st.session_state.basket = {
        "Bell Ceres": {"Lettuce Fresh (ea)": 15.0, "Tomatoes (kg)": 5.0},
        "Cuyler Butchery": {"Beef Mince (kg)": 2.0},
    }
    st.session_state.shopping_list_df = pd.DataFrame({
        "Item": ["Tomatoes", "Beef Mince", "Chicken Thighs", ""],
        "Quantity": [5.0, 2.0, 3.0, 0.0],
    })
    st.session_state.shopping_list_results = None
    st.session_state.shopping_list_smart_basket = None


st.sidebar.header("⚙️ Supplier Setup & Manuals")

if st.sidebar.button("⚡ Load Demo Canteen Dataset", width='stretch', key="load_demo_btn"):
    load_demo_data()
    st.sidebar.success("Loaded demo dataset!")
    st.rerun()

st.sidebar.divider()

supplier_name = st.sidebar.text_input("Supplier name (optional for matrix tables)", key="s_name_input")
moq_input = st.sidebar.number_input("MOQ (R)", min_value=0.0, value=DEFAULT_MOQ, step=25.0, key="s_moq_input")
delivery_fee_input = st.sidebar.number_input("Delivery fee (R)", min_value=0.0, value=DEFAULT_DELIVERY_FEE, step=10.0, key="s_fee_input")
has_account_input = st.sidebar.checkbox("I have an account with this supplier", value=True, key="s_has_account_input")

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

    parsed_suppliers, parse_error = parse_buying_manual_bytes(
        file_bytes, filename=fname, default_supplier_name=s_key,
        default_moq=moq_input, default_delivery_fee=delivery_fee_input,
        default_has_account=has_account_input,
    )
    if parsed_suppliers:
        count = 0
        for name, s_dict in parsed_suppliers.items():
            s_dict["moq"] = float(moq_input)
            if delivery_fee_input > 0:
                s_dict["delivery_fee"] = float(delivery_fee_input)
            s_dict["has_account"] = bool(has_account_input)
            st.session_state.suppliers[name] = s_dict
            count += len(s_dict["catalog"])
        st.sidebar.success(f"✅ Loaded {len(parsed_suppliers)} supplier(s) with {count} items from {fname}!")
    elif parse_error:
        st.sidebar.error(f"⚠️ Could not parse {fname}: {parse_error}")
    else:
        st.sidebar.warning(f"⚠️ Could not detect item or price columns in {fname}. Please check that column headers include 'Item'/'Product' and 'Price'/'Cost'.")

with st.sidebar.expander("📝 Mobile Fallback: Paste CSV/Text Catalog"):
    pasted_csv = st.text_area("Paste CSV (e.g. Item, Price or Matrix)", key="pasted_csv_input")
    if st.button("➕ Process Pasted CSV", type="secondary", width='stretch', key="process_pasted_csv_btn"):
        if pasted_csv.strip():
            s_key = supplier_name.strip() if supplier_name.strip() else "Uploaded Supplier"
            parsed, paste_error = parse_buying_manual_bytes(
                pasted_csv.encode('utf-8'), filename="pasted.csv", default_supplier_name=s_key,
                default_moq=moq_input, default_delivery_fee=delivery_fee_input,
                default_has_account=has_account_input,
            )
            if parsed:
                count = 0
                for name, s_dict in parsed.items():
                    s_dict["moq"] = float(moq_input)
                    if delivery_fee_input > 0:
                        s_dict["delivery_fee"] = float(delivery_fee_input)
                    s_dict["has_account"] = bool(has_account_input)
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
        acct_badge = "✅ Account" if data.get("has_account", True) else "🚫 No account"
        with st.sidebar.expander(f"🏢 {name} — {len(data['catalog'])} items · {acct_badge}"):
            acct_toggle = st.checkbox(
                "I have an account with this supplier",
                value=data.get("has_account", True),
                key=f"acct_toggle_{name}",
            )
            if acct_toggle != data.get("has_account", True):
                data["has_account"] = acct_toggle
                st.rerun()

            if data["catalog"]:
                basket_items = st.session_state.basket.get(name, {})
                items_df = pd.DataFrame(
                    [
                        {
                            "In Basket": "✅" if item in basket_items else "",
                            "Item": item,
                            "Price (R)": meta["price"],
                            "Category": meta.get("category", ""),
                        }
                        for item, meta in data["catalog"].items()
                    ]
                ).sort_values("Item").reset_index(drop=True)

                st.caption("Click an item to add it to your Basket")
                event = st.dataframe(
                    items_df,
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "Price (R)": st.column_config.NumberColumn(format="R%.2f"),
                    },
                    on_select="rerun",
                    selection_mode="single-row",
                    key=f"supplier_items_{name}",
                )

                selected_rows = event.selection.rows if event and event.selection else []
                if selected_rows:
                    clicked_item = items_df.iloc[selected_rows[0]]["Item"]
                    last_clicked_key = f"_last_clicked_{name}"
                    if st.session_state.get(last_clicked_key) != clicked_item:
                        st.session_state.basket.setdefault(name, {})
                        st.session_state.basket[name].setdefault(clicked_item, 1.0)
                        st.session_state[last_clicked_key] = clicked_item
                        st.toast(f"Added '{clicked_item}' to your Basket")
                        st.rerun()
            else:
                st.caption("No items in this catalog yet.")

            m1, m2 = st.columns(2)
            m1.caption(f"MOQ: R{data['moq']:,.0f}")
            m2.caption(f"Delivery fee: R{data['delivery_fee']:,.0f}")

            if st.button("✕ Remove supplier", key=f"remove_{name}", width='stretch'):
                del st.session_state.suppliers[name]
                st.session_state.basket.pop(name, None)
                st.rerun()
else:
    st.sidebar.info("No suppliers added yet.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧺 Your Basket", "📝 Shopping List", "📊 Basket Summary & Export", "🔎 Out-of-Account Savings"
])

account_suppliers = {n: d for n, d in st.session_state.suppliers.items() if d.get("has_account", True)}

with tab1:
    st.subheader("Your Basket")
    st.write("Click items in the sidebar to add them here. Each supplier needs to reach its own MOQ before delivery is free.")

    if not st.session_state.basket or all(not v for v in st.session_state.basket.values()):
        st.info("Your basket is empty — pick some items from a supplier in the sidebar to get started.")
    else:
        summary = compute_basket_summary(st.session_state.suppliers, st.session_state.basket)

        for supplier_name, data in summary.items():
            supplier_moq = data["moq"]
            st.markdown(f"### 🏢 {supplier_name}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Basket Subtotal", f"R{data['subtotal']:,.2f}")
            m2.metric("MOQ", f"R{supplier_moq:,.0f}")
            m3.metric("Delivery", "FREE ✅" if data["meets_moq"] else f"R{data['delivery_fee_charged']:,.2f} fee")

            progress = min(1.0, data["subtotal"] / supplier_moq) if supplier_moq > 0 else 1.0
            st.progress(progress, text=f"{progress*100:.0f}% of MOQ met")

            edit_df = pd.DataFrame(data["rows"])
            edited = st.data_editor(
                edit_df,
                hide_index=True,
                width='stretch',
                column_config={
                    "Quantity": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.2f"),
                    "Unit Price (R)": st.column_config.NumberColumn(format="R%.2f", disabled=True),
                    "Subtotal (R)": st.column_config.NumberColumn(format="R%.2f", disabled=True),
                },
                disabled=["Item", "Unit Price (R)", "Subtotal (R)"],
                key=f"basket_editor_{supplier_name}",
            )

            if st.button("💾 Update Quantities", key=f"update_basket_{supplier_name}"):
                new_items = {
                    row["Item"]: float(row["Quantity"])
                    for _, row in edited.iterrows()
                    if float(row["Quantity"]) > 0
                }
                st.session_state.basket[supplier_name] = new_items
                st.rerun()

            if not data["meets_moq"]:
                st.warning(
                    f"⚠️ {supplier_name} needs **R{data['gap']:,.2f} more** to hit its R{supplier_moq:,.0f} MOQ "
                    f"— otherwise a R{data['delivery_fee_full']:,.2f} delivery fee applies."
                )

                topups = get_topup_suggestions(supplier_name, st.session_state.suppliers, st.session_state.basket, data["gap"])
                if topups:
                    st.caption(f"Option 1 — add one of these items from {supplier_name} to close the gap:")
                    for t in topups:
                        c1, c2 = st.columns([4, 1])
                        c1.write(f"**{t['item']}** — R{t['price']:,.2f} each → add **{t['qty']}** (R{t['cost']:,.2f}) to meet MOQ")
                        if c2.button("Add", key=f"topup_{supplier_name}_{t['item']}"):
                            st.session_state.basket.setdefault(supplier_name, {})
                            st.session_state.basket[supplier_name][t["item"]] = (
                                st.session_state.basket[supplier_name].get(t["item"], 0.0) + t["qty"]
                            )
                            st.rerun()

                alts = get_alt_supplier_suggestions(supplier_name, account_suppliers, st.session_state.basket)
                if alts:
                    st.caption("Option 2 — buy the same item cheaper from another supplier you have an account with:")
                    for a in alts:
                        moq_note = (
                            "already meets their MOQ ✅" if a["alt_gap"] <= 0
                            else f"needs R{a['alt_gap']:,.2f} more to hit their R{a['alt_moq']:,.0f} MOQ"
                        )
                        st.write(
                            f"**{a['item']}** ({a['qty']:.0f}x): {a['alt_supplier']} sells it for "
                            f"R{a['alt_price']:,.2f} vs R{a['current_price']:,.2f} here — "
                            f"save R{a['total_savings']:,.2f}. Moving it there would bring their order to "
                            f"R{a['alt_subtotal_if_moved']:,.2f} ({moq_note})."
                        )
                        if st.button(f"Move to {a['alt_supplier']}", key=f"move_{supplier_name}_{a['item']}_{a['alt_supplier']}"):
                            qty_moved = st.session_state.basket[supplier_name].pop(a["item"], None)
                            if qty_moved:
                                st.session_state.basket.setdefault(a["alt_supplier"], {})
                                st.session_state.basket[a["alt_supplier"]][a["item"]] = (
                                    st.session_state.basket[a["alt_supplier"]].get(a["item"], 0.0) + qty_moved
                                )
                            st.rerun()
            else:
                st.success(f"✅ {supplier_name}'s MOQ is met — delivery is free.")

            st.divider()

with tab2:
    st.subheader("Type Your Shopping List")
    st.write(
        "List what you need — one item per row, with a quantity. We'll match each item to the "
        "**cheapest supplier you have an account with** and check their MOQ."
    )

    edited_list = st.data_editor(
        st.session_state.shopping_list_df,
        num_rows="dynamic",
        width='stretch',
        column_config={
            "Item": st.column_config.TextColumn(help="Type the product name — doesn't need to be exact"),
            "Quantity": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.2f"),
        },
        key="shopping_list_editor",
    )

    if st.button("🔍 Find Cheapest Suppliers", type="primary", key="process_shopping_list_btn"):
        st.session_state.shopping_list_df = edited_list
        rows = edited_list.to_dict("records")
        results, smart_basket = process_shopping_list(rows, st.session_state.suppliers)
        st.session_state.shopping_list_results = results
        st.session_state.shopping_list_smart_basket = smart_basket

    results = st.session_state.shopping_list_results
    smart_basket = st.session_state.shopping_list_smart_basket

    if results:
        st.divider()
        st.markdown("### Matched Items")
        display_rows = []
        for r in results:
            if r["matched"]:
                display_rows.append({
                    "You Typed": r["typed_item"],
                    "Matched Item": r["matched_item"],
                    "Cheapest Supplier": r["supplier"],
                    "Price (R)": r["price"],
                    "Qty": r["qty"],
                    "Subtotal (R)": r["subtotal"],
                })
            else:
                display_rows.append({
                    "You Typed": r["typed_item"],
                    "Matched Item": "⚠️ Not found from your suppliers",
                    "Cheapest Supplier": "-",
                    "Price (R)": None,
                    "Qty": r["qty"],
                    "Subtotal (R)": None,
                })
        st.dataframe(
            pd.DataFrame(display_rows),
            hide_index=True,
            width='stretch',
            column_config={
                "Price (R)": st.column_config.NumberColumn(format="R%.2f"),
                "Subtotal (R)": st.column_config.NumberColumn(format="R%.2f"),
            },
        )

        unmatched = [r for r in results if not r["matched"]]
        if unmatched:
            st.warning(
                f"⚠️ {len(unmatched)} item(s) weren't found from any supplier you have an account with. "
                f"Check the 'Out-of-Account Savings' tab to see if another supplier carries them."
            )

        if not smart_basket:
            st.info("No items matched an account supplier — nothing to check against MOQ.")
        else:
            st.divider()
            st.markdown("### Supplier Breakdown & MOQ Check")

            smart_summary = compute_basket_summary(account_suppliers, smart_basket)

            for supplier_name, data in smart_summary.items():
                st.markdown(f"#### 🏢 {supplier_name}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Subtotal", f"R{data['subtotal']:,.2f}")
                m2.metric("MOQ", f"R{data['moq']:,.0f}")
                m3.metric("Delivery", "FREE ✅" if data["meets_moq"] else f"R{data['delivery_fee_charged']:,.2f} fee")
                progress = min(1.0, data["subtotal"] / data["moq"]) if data["moq"] > 0 else 1.0
                st.progress(progress, text=f"{progress*100:.0f}% of MOQ met")

                if not data["meets_moq"]:
                    st.warning(f"⚠️ Needs **R{data['gap']:,.2f} more** to hit the R{data['moq']:,.0f} MOQ.")

                    catalog = account_suppliers[supplier_name]["catalog"]
                    consolidated_items = {}
                    for r in results:
                        if r["matched"] and r["matched_item"] in catalog:
                            consolidated_items[r["matched_item"]] = consolidated_items.get(r["matched_item"], 0.0) + r["qty"]
                    consolidated_subtotal = sum(catalog[i]["price"] * q for i, q in consolidated_items.items())
                    consolidated_meets = consolidated_subtotal >= data["moq"]
                    total_matched = sum(1 for r in results if r["matched"])

                    st.caption(
                        f"**Option 1 — Single-supplier basket:** buy everything on your list that "
                        f"{supplier_name} stocks, all from {supplier_name} (even where it's pricier "
                        f"elsewhere): R{consolidated_subtotal:,.2f}, covering {len(consolidated_items)}/"
                        f"{total_matched} matched items — "
                        f"{'meets MOQ ✅' if consolidated_meets else 'still short of MOQ'}."
                    )
                    if st.button(f"Use {supplier_name} for all available items", key=f"consolidate_{supplier_name}"):
                        for other_supplier in list(smart_basket.keys()):
                            if other_supplier == supplier_name:
                                continue
                            for item in list(smart_basket[other_supplier].keys()):
                                if item in consolidated_items:
                                    del smart_basket[other_supplier][item]
                        smart_basket[supplier_name] = consolidated_items
                        st.session_state.shopping_list_smart_basket = smart_basket
                        st.rerun()

                    topups = get_topup_suggestions(supplier_name, account_suppliers, smart_basket, data["gap"])
                    if topups:
                        st.caption("**Option 2 — add a top-up item from the same supplier:**")
                        for t in topups:
                            c1, c2 = st.columns([4, 1])
                            c1.write(f"**{t['item']}** — R{t['price']:,.2f} each → add **{t['qty']}** (R{t['cost']:,.2f})")
                            if c2.button("Add", key=f"sl_topup_{supplier_name}_{t['item']}"):
                                smart_basket.setdefault(supplier_name, {})
                                smart_basket[supplier_name][t['item']] = smart_basket[supplier_name].get(t['item'], 0.0) + t['qty']
                                st.session_state.shopping_list_smart_basket = smart_basket
                                st.rerun()

                    alts = get_alt_supplier_suggestions(supplier_name, account_suppliers, smart_basket)
                    if alts:
                        st.caption("**Option 3 — move an item to another of your suppliers to help reach ITS MOQ instead:**")
                        for a in alts:
                            moq_note = (
                                "meets their MOQ ✅" if a["alt_gap"] <= 0
                                else f"needs R{a['alt_gap']:,.2f} more for their R{a['alt_moq']:,.0f} MOQ"
                            )
                            st.write(
                                f"**{a['item']}** ({a['qty']:.0f}x): move to {a['alt_supplier']} at "
                                f"R{a['alt_price']:,.2f} (save R{a['total_savings']:,.2f}) — {moq_note}"
                            )
                            if st.button(f"Move to {a['alt_supplier']}", key=f"sl_move_{supplier_name}_{a['item']}_{a['alt_supplier']}"):
                                qty_moved = smart_basket[supplier_name].pop(a['item'], None)
                                if qty_moved:
                                    smart_basket.setdefault(a['alt_supplier'], {})
                                    smart_basket[a['alt_supplier']][a['item']] = smart_basket[a['alt_supplier']].get(a['item'], 0.0) + qty_moved
                                st.session_state.shopping_list_smart_basket = smart_basket
                                st.rerun()
                else:
                    st.success(f"✅ MOQ met for {supplier_name}.")
                st.divider()

            if st.button("✅ Add Entire Shopping List to Your Basket", type="primary", key="commit_shopping_list_btn"):
                for supplier_name, items in smart_basket.items():
                    st.session_state.basket.setdefault(supplier_name, {})
                    for item, qty in items.items():
                        st.session_state.basket[supplier_name][item] = st.session_state.basket[supplier_name].get(item, 0.0) + qty
                st.success("Added to your basket! Check the 'Your Basket' tab.")

with tab3:
    st.subheader("Basket Summary & Export")
    summary = compute_basket_summary(st.session_state.suppliers, st.session_state.basket)

    if not summary:
        st.info("Add items to your basket to see a summary here.")
    else:
        grand_subtotal = sum(d["subtotal"] for d in summary.values())
        grand_delivery = sum(d["delivery_fee_charged"] for d in summary.values())
        grand_total = grand_subtotal + grand_delivery
        suppliers_meeting_moq = sum(1 for d in summary.values() if d["meets_moq"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Grand Total", f"R{grand_total:,.2f}")
        m2.metric("Item Subtotal", f"R{grand_subtotal:,.2f}")
        m3.metric("Delivery Fees", f"R{grand_delivery:,.2f}")
        m4.metric("Suppliers Meeting MOQ", f"{suppliers_meeting_moq} / {len(summary)}")

        st.divider()

        st.markdown("##### Spend per Supplier (R)")
        plot_data = {
            s: {"Item Spend": d["subtotal"], "Delivery Fee": d["delivery_fee_charged"]}
            for s, d in summary.items()
        }
        plot_df = pd.DataFrame(plot_data).T
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df.plot(kind="bar", stacked=True, ax=ax, color=["#3B82F6", "#EF4444"])
        ax.set_ylabel("Cost (ZAR)")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.markdown("### 🛒 Itemized Basket")
        rows = []
        for supplier_name, d in summary.items():
            for r in d["rows"]:
                rows.append({"Supplier": supplier_name, **r})
        result_df = pd.DataFrame(rows)
        st.dataframe(
            result_df,
            hide_index=True,
            width='stretch',
            column_config={
                "Quantity": st.column_config.NumberColumn(format="%.2f"),
                "Unit Price (R)": st.column_config.NumberColumn(format="R%.2f"),
                "Subtotal (R)": st.column_config.NumberColumn(format="R%.2f"),
            },
        )

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Basket CSV",
            csv,
            "cherry_picker_basket.csv",
            "text/csv",
            key="download_basket_csv_btn",
        )

with tab4:
    st.subheader("🔎 Out-of-Account Supplier Savings")
    st.write("Suppliers you don't have an account with, and what your last shopping list would have cost through them instead.")

    non_account_suppliers = {n: d for n, d in st.session_state.suppliers.items() if not d.get("has_account", True)}

    if not non_account_suppliers:
        st.info("All your suppliers are marked as account suppliers — nothing to compare yet. Toggle a supplier's account status in the sidebar to see this in action.")
    else:
        st.caption("Suppliers currently marked without an account:")
        st.dataframe(
            pd.DataFrame([{"Supplier": n, "Items in Catalog": len(d["catalog"])} for n, d in non_account_suppliers.items()]),
            hide_index=True, width='stretch'
        )
        st.divider()

        results = st.session_state.shopping_list_results
        if not results:
            st.info("Process a shopping list in the '📝 Shopping List' tab first to see potential savings here.")
        else:
            agg = aggregate_out_of_account_savings(results)
            if not agg:
                st.success("Your account suppliers already had the best prices for everything on your list! 🎉")
            else:
                total_all_savings = sum(d["total_savings"] for d in agg.values())
                st.metric("Total Potential Savings Across All Out-of-Account Suppliers", f"R{total_all_savings:,.2f}")
                st.divider()
                for supplier_name, data in sorted(agg.items(), key=lambda kv: -kv[1]["total_savings"]):
                    st.markdown(f"#### 🏢 {supplier_name} (no account)")
                    if data["total_savings"] > 0:
                        st.metric("Potential Savings", f"R{data['total_savings']:,.2f}")
                    rows = []
                    for it in data["items"]:
                        rows.append({
                            "Item": it["item"],
                            "Qty": it["qty"],
                            "Your Price (R)": it["your_price"],
                            f"{supplier_name} Price (R)": it["their_price"],
                            "Savings (R)": it["savings"],
                        })
                    st.dataframe(
                        pd.DataFrame(rows), hide_index=True, width='stretch',
                        column_config={
                            "Your Price (R)": st.column_config.NumberColumn(format="R%.2f"),
                            f"{supplier_name} Price (R)": st.column_config.NumberColumn(format="R%.2f"),
                            "Savings (R)": st.column_config.NumberColumn(format="R%.2f"),
                        },
                    )
                    st.divider()

st.divider()
st.caption("🍒 Cherry Picker | Streamlit")
