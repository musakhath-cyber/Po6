import io
import os
from typing import List, Optional
from google import genai
from google.genai import types
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
import streamlit as st

# Configure mobile page layout
st.set_page_config(
    page_title="Food Catalog Scanner", page_icon="📦", layout="centered"
)

st.title("📦 Food Catalog Scanner")
st.caption(
    "Take a photo or upload supplier sheets to scan, categorize, and price items automatically."
)


# Define structured data model
class CatalogItem(BaseModel):
  item_name: str = Field(description="Product name without promo clutter.")
  category: str = Field(
      description=(
          "Strict category: Produce, Proteins, Dairy, Pantry & Dry Goods,"
          " Beverages, or Packaging."
      )
  )
  pack_size: str = Field(
      description="Pack size or weight, e.g. 5kg, 2L, Tray of 30."
  )
  price: float = Field(description="Unit price.")
  confidence: float = Field(default=0.95)


class CatalogScanResponse(BaseModel):
  supplier_name: Optional[str] = Field(
      "Unknown Supplier", description="Vendor name if detected."
  )
  currency: str = Field(default="ZAR")
  items: List[CatalogItem]


# Initialize Gemini API
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
  st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
  st.stop()

client = genai.Client(api_key=api_key)

# Mobile Input: Camera or File Upload
upload_mode = st.radio(
    "Select Input Method:",
    ["📷 Camera Snap", "📁 Upload File"],
    horizontal=True,
)
uploaded_file = None

if upload_mode == "📷 Camera Snap":
  uploaded_file = st.camera_input("Snap a photo of the catalog page")
else:
  uploaded_file = st.file_uploader(
      "Choose a catalog image", type=["png", "jpg", "jpeg", "webp"]
  )

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Catalog Scan Preview", use_container_width=True)

  if st.button(
      "⚡ Scan & Categorize Items", type="primary", use_container_width=True
  ):
    with st.spinner("Analyzing text, prices, and categories with AI..."):
      try:
        prompt = """
                Extract all food catalog items from this image:
                - Product name (clean and normalized)
                - Pack size (weight, volume, count)
                - Item price
                - Category: Produce, Proteins, Dairy, Pantry & Dry Goods, Beverages, or Packaging.
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

        # Store in session state for editing
        st.session_state["catalog_data"] = data
        st.session_state["supplier"] = data.supplier_name
        st.session_state["currency"] = data.currency
        st.success(f"Detected {len(data.items)} items!")

      except Exception as e:
        st.error(f"Error scanning catalog: {str(e)}")

# Display & Edit Extracted Results
if "catalog_data" in st.session_state and st.session_state["catalog_data"]:
  data = st.session_state["catalog_data"]

  st.divider()
  st.subheader(f"🏢 {st.session_state.get('supplier', 'Supplier Catalog')}")

  # Convert to Pandas DataFrame for interactive mobile table
  items_list = [item.model_dump() for item in data.items]
  df = pd.DataFrame(items_list)

  st.write("### 📋 Extracted Items (Tap to Edit)")
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

  # Total metrics
  total_val = edited_df["price"].sum() if "price" in edited_df else 0.0
  st.metric(
      label="Total Basket Estimate",
      value=f"{st.session_state.get('currency', 'ZAR')} {total_val:,.2f}",
  )

  # Download CSV
  csv_data = edited_df.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="📥 Download Extracted Catalog (CSV)",
      data=csv_data,
      file_name="scanned_catalog.csv",
      mime="text/csv",
      use_container_width=True,
  )
