import base64
import io
import os
from typing import List, Optional
from google import genai
from google.genai import types
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
import streamlit as st

st.set_page_config(
    page_title="Food Catalog Scanner",
    page_icon="🍒",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# Function to inject custom CSS with opaque cherry background and yellow theme
def apply_custom_theme():
  bg_base64 = ""
  if os.path.exists("cherry.jpg"):
    with open("cherry.jpg", "rb") as img_file:
      bg_base64 = base64.b64encode(img_file.read()).decode()

  bg_style = (
      f"""
    <style>
    .stApp {{
        background-color: #FFFDF0;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-image: url("data:image/jpeg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.12;
        pointer-events: none;
        z-index: 0;
    }}
    h1, h2, h3, p, span, label {{
        color: #3B2E05 !important;
    }}
    .stButton>button {{
        background-color: #EAB308 !important;
        color: #1F1600 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(234, 179, 8, 0.35) !important;
    }}
    .stButton>button:hover {{
        background-color: #CA8A04 !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: #854D0E !important;
    }}
    </style>
    """
      if bg_base64
      else """
    <style>
    .stApp { background-color: #FFFDF0; }
    h1, h2, h3, p, span, label { color: #3B2E05 !important; }
    .stButton>button { background-color: #EAB308 !important; color: #1F1600 !important; font-weight: 700 !important; }
    </style>
    """
  )
  st.markdown(bg_style, unsafe_allow_html=True)


apply_custom_theme()

st.title("🍒 Food Catalog Scanner")
st.caption(
    "Snap or upload supplier catalog pages to extract, categorize, and price"
    " items automatically."
)


# Structured data output models
class CatalogItem(BaseModel):
  item_name: str = Field(
      description="Normalized product name without promo text."
  )
  category: str = Field(
      description=(
          "Must be one of: Produce, Proteins, Dairy, Pantry & Dry Goods,"
          " Beverages, Packaging."
      )
  )
  pack_size: str = Field(
      description="Pack size or weight, e.g. 5kg, 2L, Tray of 30."
  )
  price: float = Field(description="Numerical unit price.")
  confidence: float = Field(default=0.95)


class CatalogScanResponse(BaseModel):
  supplier_name: Optional[str] = Field(
      "Supplier Catalog", description="Detected vendor name."
  )
  currency: str = Field(default="ZAR")
  items: List[CatalogItem]


# Initialize Gemini API
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
  st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
  st.stop()

client = genai.Client(api_key=api_key)

# Input Selector
upload_mode = st.radio(
    "Choose Input Method:",
    ["📁 Upload Image", "📷 Camera Snap"],
    horizontal=True,
)
uploaded_file = None

if upload_mode == "📷 Camera Snap":
  uploaded_file = st.camera_input("Take a photo of the catalog page")
else:
  uploaded_file = st.file_uploader(
      "Choose a catalog image", type=["png", "jpg", "jpeg", "webp"]
  )

# Process Uploaded Image
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(
      image,
      caption="Uploaded Catalog Page",
      use_container_width=True,
  )

  scan_triggered = st.button(
      "⚡ Scan & Categorize Catalog", type="primary", use_container_width=True
  )

  if scan_triggered:
    with st.spinner("Analyzing items, prices, and categories with AI..."):
      try:
        prompt = """
                Extract all food catalog and inventory items from this image:
                - Product Name (clean, remove unnecessary promotional noise)
                - Category: Produce, Proteins, Dairy, Pantry & Dry Goods, Beverages, or Packaging
                - Pack size / Unit (e.g., 5kg, 10x1L, Case of 24, Tray of 30)
                - Unit Price (numerical value)
                - Supplier name and currency if visible.
                """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CatalogScanResponse,
                temperature=0.1,
            ),
        )

        data = CatalogScanResponse.model_validate_json(response.text)
        st.session_state["catalog_data"] = data
        st.session_state["supplier"] = data.supplier_name
        st.session_state["currency"] = data.currency
        st.success(f"Successfully extracted {len(data.items)} items!")

      except Exception as e:
        st.error(f"Error scanning catalog: {str(e)}")

# Display & Edit Catalog Results
if "catalog_data" in st.session_state and st.session_state["catalog_data"]:
  data = st.session_state["catalog_data"]

  st.divider()
  st.subheader(f"🏢 {st.session_state.get('supplier', 'Supplier Catalog')}")

  items_list = [item.model_dump() for item in data.items]
  df = pd.DataFrame(items_list)

  st.write("### 📋 Extracted Items (Tap any cell to edit)")
  edited_df = st.data_editor(
      df,
      column_config={
          "item_name": st.column_config.TextColumn("Item Name", required=True),
          "category": st.column_config.SelectboxColumn(
              "Category",
              options=[
                  "Produce",
                  "Proteins",
                  "Dairy",
                  "Pantry & Dry Goods",
                  "Beverages",
                  "Packaging",
              ],
              required=True,
          ),
          "pack_size": st.column_config.TextColumn("Pack Size"),
          "price": st.column_config.NumberColumn(
              f"Price ({st.session_state.get('currency', 'ZAR')})",
              format="%.2f",
          ),
          "confidence": st.column_config.ProgressColumn(
              "Confidence", min_value=0.0, max_value=1.0
          ),
      },
      num_rows="dynamic",
      use_container_width=True,
  )

  # Total metrics calculation
  total_val = edited_df["price"].sum() if "price" in edited_df else 0.0
  st.metric(
      label="Total Catalog Basket Sum",
      value=f"{st.session_state.get('currency', 'ZAR')} {total_val:,.2f}",
  )

  # Export options
  csv_data = edited_df.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="📥 Export Extracted CSV / Spreadsheet",
      data=csv_data,
      file_name="catalog_scanned_items.csv",
      mime="text/csv",
      use_container_width=True,
  )
