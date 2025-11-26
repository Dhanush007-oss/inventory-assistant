"""app.py

Inventory Assistant – Streamlit + sklearn Pipeline + Firestore
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

import joblib

# sklearn base classes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

import firebase_admin
from firebase_admin import credentials, firestore

from dotenv import load_dotenv

# Load environment variables from .env when running locally
load_dotenv()

# ----------------------------------------------------------
# CUSTOM TRANSFORMERS (needed so joblib can load the pipeline)
# ----------------------------------------------------------
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper around LabelEncoder so it can be used in ColumnTransformer / Pipeline.
    This version is defensive so it works with the already-saved inventory_pipeline.joblib.
    """

    def __init__(self):
        # no explicit encoder here; we’ll detect it from __dict__ on unpickle
        pass

    def _get_encoder(self):
        # Try to find an existing LabelEncoder attached during training
        for v in self.__dict__.values():
            if isinstance(v, LabelEncoder):
                return v

        # Fallback if none found (should only happen if something is odd)
        if not hasattr(self, "_fallback_enc"):
            self._fallback_enc = LabelEncoder()
        return self._fallback_enc

    def fit(self, X, y=None):
        enc = self._get_encoder()
        # X will be 2D array from ColumnTransformer; flatten to 1D
        s = pd.Series(np.array(X).ravel()).astype(str)
        enc.fit(s)
        return self

    def transform(self, X):
        enc = self._get_encoder()
        s = pd.Series(np.array(X).ravel()).astype(str)
        out = enc.transform(s)
        return out.reshape(-1, 1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Simple frequency encoder: maps each category to its normalized frequency.
    """

    def __init__(self):
        # freq_map_ will be set in fit
        pass

    def _get_mapping(self):
        # Try to find existing dict (freq_map_) attached during training
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                return v

        if not hasattr(self, "_fallback_map"):
            self._fallback_map = {}
        return self._fallback_map

    def fit(self, X, y=None):
        s = pd.Series(np.array(X).ravel())
        self.freq_map_ = s.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        mapping = self._get_mapping()
        s = pd.Series(np.array(X).ravel())
        out = s.map(mapping).fillna(0.0).to_numpy()
        return out.reshape(-1, 1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ----------------------------------------------------------
# STREAMLIT PAGE CONFIG  (must be first Streamlit call)
# ----------------------------------------------------------
st.set_page_config(
    page_title="Inventory Assistant",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------
MODEL_PATH = Path("./inventory_pipeline.joblib")
DATASET_PATH = Path("./bike_sales_data_world_2013_2023 (1).csv")

# ----------------------------------------------------------
# FIREBASE INITIALIZATION (SAFE ON RE-RUNS)
#   - Uses st.secrets["FIREBASE"] on Streamlit Cloud
#   - Falls back to environment variables (.env) locally
# ----------------------------------------------------------
db = None


def _build_firebase_credentials_from_secrets():
    """Build Firebase credentials dict from Streamlit secrets."""
    s = st.secrets["FIREBASE"]
    return {
        "type": s["TYPE"],
        "project_id": s["PROJECT_ID"],
        "private_key_id": s["PRIVATE_KEY_ID"],
        "private_key": s["PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": s["CLIENT_EMAIL"],
        "client_id": s["CLIENT_ID"],
        "auth_uri": s["AUTH_URI"],
        "token_uri": s["TOKEN_URI"],
        "auth_provider_x509_cert_url": s["AUTH_PROVIDER_X509_CERT_URL"],
        "client_x509_cert_url": s["CLIENT_X509_CERT_URL"],
        "universe_domain": s.get("UNIVERSE_DOMAIN", "googleapis.com"),
    }


def _build_firebase_credentials_from_env():
    """Build Firebase credentials dict from environment variables (.env)."""
    return {
        "type": os.getenv("FIREBASE_TYPE"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": (os.getenv("FIREBASE_PRIVATE_KEY") or "").replace("\\n", "\n"),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
        "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN", "googleapis.com"),
    }


def _get_firebase_credentials():
    """Prefer Streamlit secrets when present; otherwise use env variables."""
    try:
        if "FIREBASE" in st.secrets:
            return _build_firebase_credentials_from_secrets()
    except Exception:
        # st.secrets may not be available locally
        pass
    return _build_firebase_credentials_from_env()


if not firebase_admin._apps:
    try:
        firebase_credentials = _get_firebase_credentials()
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    except Exception as e:
        st.warning(f"⚠️ Firestore initialization failed: {e}")
        db = None
else:
    try:
        firebase_admin.get_app()
        db = firestore.client()
    except Exception:
        db = None

# ----------------------------------------------------------
# LOAD PIPELINE MODEL
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        pipe = joblib.load(str(MODEL_PATH))
        return pipe
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at: {DATASET_PATH}")
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]

    # 🔥 REMOVE DUPLICATES AT THE BEGINNING
    df = df.drop_duplicates().reset_index(drop=True)

    # Normalize string columns to lowercase
    text_cols = [
        "Product_Category",
        "Sub_Category",
        "Product",
        "Material",
        "Color",
        "Size",
        "Manufacturer",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Rating" in df.columns and "Rating_Class" not in df.columns:
        df["Rating_Class"] = df["Rating"].apply(lambda r: 1 if r >= 4 else 0)

    return df



# Initialise model + dataframe
pipeline = load_model()
df = load_dataset()

# This is the set of raw input columns the pipeline expects (if available)
if pipeline is not None and hasattr(pipeline, "feature_names_in_"):
    MODEL_INPUT_COLS = list(pipeline.feature_names_in_)
else:
    MODEL_INPUT_COLS = list(df.columns)


# ==========================================================
# Firestore helpers, session state, sidebar
# ==========================================================

# ----------------------------------------------------------
# FIRESTORE HELPERS
# ----------------------------------------------------------
def save_prediction_to_manufacturer(inputs, decision_plain, matches_df):
    """
    Save prediction under:
      manufacturers/{manufacturer}/predictions/{auto-id}
    """
    if db is None:
        return

    manu = inputs.get("Manufacturer", "unknown")
    if isinstance(manu, str):
        manu = manu.lower()
    else:
        manu = str(manu).lower()

    # Ensure manufacturer document exists
    db.collection("manufacturers").document(manu).set(
        {"name": manu, "created_at": firestore.SERVER_TIMESTAMP},
        merge=True,
    )

    data = {
        "inputs": inputs,
        "decision": decision_plain,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "match_count": (
            len(matches_df) if isinstance(matches_df, pd.DataFrame) else 0
        ),
    }

    try:
        (
            db.collection("manufacturers")
            .document(manu)
            .collection("predictions")
            .add(data)
        )
    except Exception:
        pass  # fail silently


def get_manufacturer_predictions_df(manu):
    if db is None:
        return pd.DataFrame()

    manu = str(manu).lower()

    try:
        docs = (
            db.collection("manufacturers")
            .document(manu)
            .collection("predictions")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .stream()
        )
    except Exception:
        docs = (
            db.collection("manufacturers")
            .document(manu)
            .collection("predictions")
            .stream()
        )

    rows = []
    for d in docs:
        item = d.to_dict() or {}
        inputs = item.get("inputs", {})
        r = inputs.copy()
        r["Decision"] = item.get("decision")
        r["Matches"] = item.get("match_count", 0)
        r["Timestamp"] = item.get("timestamp")
        rows.append(r)

    if not rows:
        return pd.DataFrame()

    df_pred = pd.DataFrame(rows).drop_duplicates()

    if "Timestamp" in df_pred.columns:
        df_pred = df_pred.sort_values("Timestamp", ascending=False)

    return df_pred


def get_all_manufacturers_from_firestore():
    if db is None:
        return []
    try:
        docs = db.collection("manufacturers").stream()
        return sorted([d.id for d in docs])
    except Exception:
        return []


# ----------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------
default_state = {
    "current_page": "chat",
    "product_category": None,
    "sub_category": None,
    "product": None,
    "material": None,
    "size": None,
    "manufacturer": None,
    "prediction_done": False,
    "prediction_html": "",
    "match_rows": pd.DataFrame(),
}

for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_prediction_state():
    """Clear only form fields + prediction info."""
    for k in [
        "product_category",
        "sub_category",
        "product",
        "material",
        "size",
        "manufacturer",
    ]:
        st.session_state[k] = None

    st.session_state.prediction_done = False
    st.session_state.prediction_html = ""
    st.session_state.match_rows = pd.DataFrame()


# ----------------------------------------------------------
# HELPER: build a row DataFrame for prediction
# ----------------------------------------------------------
def build_row_from_inputs(inputs):
    """
    Build a DataFrame with exactly MODEL_INPUT_COLS, using:
      - df.iloc[0] as a base template
      - overwriting with user inputs where column names match
    This guarantees the pipeline receives the same structure as training.
    """
    if df.empty:
        base = pd.DataFrame(columns=MODEL_INPUT_COLS)
        base.loc[0, :] = np.nan
    else:
        base = df.iloc[[0]].copy()
        # Align to model input columns if we know them
        base = base.reindex(columns=MODEL_INPUT_COLS, fill_value=np.nan)

    # Overwrite with user inputs
    for k, v in inputs.items():
        if k in base.columns:
            base.iloc[0, base.columns.get_loc(k)] = v

    return base


def predict_single_row(inputs):
    """
    Use the sklearn pipeline to predict for a single 6-field input dict.
    """
    if pipeline is None:
        return None

    row_df = build_row_from_inputs(inputs)
    try:
        pred = int(pipeline.predict(row_df)[0])
        return pred
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None


def predict_for_match_row(row):
    """
    Take a row from df (matching combination) and run it through the pipeline.
    We still keep all other columns from df to match training schema.
    """
    if pipeline is None:
        return None

    if df.empty:
        return predict_single_row({
            "Product_Category": row.get("Product_Category", ""),
            "Sub_Category": row.get("Sub_Category", ""),
            "Product": row.get("Product", ""),
            "Material": row.get("Material", ""),
            "Size": row.get("Size", ""),
            "Manufacturer": row.get("Manufacturer", ""),
        })

    # Base template: row 0 with proper columns
    base = df.iloc[[0]].copy()
    base = base.reindex(columns=MODEL_INPUT_COLS, fill_value=np.nan)

    # Overwrite with the actual row's values where they exist
    for c in base.columns:
        if c in row.index:
            base.iloc[0, base.columns.get_loc(c)] = row[c]

    try:
        pred = int(pipeline.predict(base)[0])
        return pred
    except Exception as e:
        st.error(f"❌ Prediction error (match row): {e}")
        return None


# ----------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------
with st.sidebar:

    if st.button("🆕 New Prediction", use_container_width=True):
        reset_prediction_state()

    st.markdown("---")

    nav_items = [
        ("💬 Chat Assistant", "chat"),
        ("📜 Prediction History", "history"),
        ("⬇️ Export", "export"),
    ]

    for label, page in nav_items:
        if st.button(label, use_container_width=True):
            st.session_state.current_page = page

    st.markdown("---")
    st.caption("Inventory Assistant • sklearn Pipeline + Firestore")


# ==========================================================
# CHAT PAGE — Filtered Dropdowns + Prediction + Matching Row Viewer
# ==========================================================
if st.session_state.current_page == "chat":

    st.title("📝 Enter Product Details")

    if df.empty:
        st.error("Dataset not loaded. Cannot proceed.")
        st.stop()

    if pipeline is None:
        st.error("Model not loaded. Cannot proceed.")
        st.stop()

    # Always begin with full dataset copy
    data = df.copy()

    # ----------------------------------------------------------
    # FILTERED DROPDOWNS (CASCADING)
    # ----------------------------------------------------------

    # PRODUCT CATEGORY
    cat = sorted(data["Product_Category"].dropna().unique())
    st.session_state.product_category = st.selectbox(
        "Product Category",
        ["Select..."] + cat,
        index=(
            cat.index(st.session_state.product_category) + 1
            if st.session_state.product_category in cat
            else 0
        ),
    )
    if st.session_state.product_category != "Select...":
        data = data[data["Product_Category"] == st.session_state.product_category]

    # SUB-CATEGORY
    sub = sorted(data["Sub_Category"].dropna().unique())
    st.session_state.sub_category = st.selectbox(
        "Sub-Category",
        ["Select..."] + sub,
        index=(
            sub.index(st.session_state.sub_category) + 1
            if st.session_state.sub_category in sub
            else 0
        ),
    )
    if st.session_state.sub_category != "Select...":
        data = data[data["Sub_Category"] == st.session_state.sub_category]

    # PRODUCT
    prods = sorted(data["Product"].dropna().unique())
    st.session_state.product = st.selectbox(
        "Product",
        ["Select..."] + prods,
        index=(
            prods.index(st.session_state.product) + 1
            if st.session_state.product in prods
            else 0
        ),
    )
    if st.session_state.product != "Select...":
        data = data[data["Product"] == st.session_state.product]

    # MATERIAL
    mats = sorted(data["Material"].dropna().unique())
    st.session_state.material = st.selectbox(
        "Material",
        ["Select..."] + mats,
        index=(
            mats.index(st.session_state.material) + 1
            if st.session_state.material in mats
            else 0
        ),
    )
    if st.session_state.material != "Select...":
        data = data[data["Material"] == st.session_state.material]

    # SIZE
    sizes = sorted(data["Size"].dropna().unique())
    st.session_state.size = st.selectbox(
        "Size",
        ["Select..."] + sizes,
        index=(
            sizes.index(st.session_state.size) + 1
            if st.session_state.size in sizes
            else 0
        ),
    )
    if st.session_state.size != "Select...":
        data = data[data["Size"] == st.session_state.size]

    # MANUFACTURER
    manus = sorted(data["Manufacturer"].dropna().unique())
    st.session_state.manufacturer = st.selectbox(
        "Manufacturer",
        ["Select..."] + manus,
        index=(
            manus.index(st.session_state.manufacturer) + 1
            if st.session_state.manufacturer in manus
            else 0
        ),
    )

    st.markdown("---")

    # ----------------------------------------------------------
    # PREDICT BUTTON  (MODEL-BASED + MAJORITY RULE)
    # ----------------------------------------------------------
    if st.button("🚀 Predict", use_container_width=True):

        required = {
            "Product Category": st.session_state.product_category,
            "Sub-Category": st.session_state.sub_category,
            "Product": st.session_state.product,
            "Material": st.session_state.material,
            "Size": st.session_state.size,
            "Manufacturer": st.session_state.manufacturer,
        }

        missing = [k for k, v in required.items() if v in (None, "", "Select...")]

        if missing:
            st.error("❌ Please select: " + ", ".join(missing))
            st.session_state.prediction_done = False

        else:
            # Build model input dict
            inputs = {
                "Product_Category": st.session_state.product_category,
                "Sub_Category": st.session_state.sub_category,
                "Product": st.session_state.product,
                "Material": st.session_state.material,
                "Size": st.session_state.size,
                "Manufacturer": st.session_state.manufacturer,
            }

            # Fallback prediction using only this one combination
            base_pred_val = predict_single_row(inputs)

            # ------------------------------
            # Filter matching dataset rows
            # ------------------------------
            matches = df.copy()
            for key, val in inputs.items():
                matches = matches[matches[key] == val]

            # If no matching rows → use single model prediction
            if matches.empty or base_pred_val is None:
                st.session_state.match_rows = matches.copy()
                final_prediction = base_pred_val

            else:
                # -------------------------------------------
                # Compute MODEL_OUTPUT for ALL matching rows
                # -------------------------------------------
                def compute_model_for_row(r):
                    return predict_for_match_row(r)

                matches_with_pred = matches.copy()
                matches_with_pred["Model_Output"] = matches_with_pred.apply(
                    compute_model_for_row, axis=1
                )

                st.session_state.match_rows = matches_with_pred.copy()

                # -------------------------------------------
                # MAJORITY RULE ON MODEL_OUTPUT
                # -------------------------------------------
                total_rows = len(matches_with_pred)
                ones = (matches_with_pred["Model_Output"] == 1).sum()
                zeros = (matches_with_pred["Model_Output"] == 0).sum()

                if total_rows == 1:
                    final_prediction = int(matches_with_pred["Model_Output"].iloc[0])
                else:
                    if ones > zeros:
                        final_prediction = 1
                    elif zeros > ones:
                        final_prediction = 0
                    else:
                        final_prediction = "UNKNOWN"

            # ----------------------------------------------------------
            # FORMAT FINAL DECISION FOR UI
            # ----------------------------------------------------------
            if final_prediction == 1:
                decision_plain = "STOCK"
                st.session_state.prediction_html = (
                    "<h2 style='color:#4CAF50; font-weight:700;'>✅ STOCK</h2>"
                )
            elif final_prediction == 0:
                decision_plain = "DON'T STOCK"
                st.session_state.prediction_html = (
                    "<h2 style='color:#FF5252; font-weight:700;'>❌ DON'T STOCK</h2>"
                )
            else:
                decision_plain = "UNKNOWN"
                st.session_state.prediction_html = (
                    "<h2 style='color:#FFC107; font-weight:700;'>❓ UNKNOWN</h2>"
                )

            st.session_state.prediction_done = True

            # ----------------------------------------------------------
            # SAVE TO FIRESTORE USING FINAL DECISION
            # ----------------------------------------------------------
            save_prediction_to_manufacturer(inputs, decision_plain, matches)

    # ----------------------------------------------------------
    # SHOW RESULT + MATCHING ROW TABLE
    # ----------------------------------------------------------
    if st.session_state.prediction_done:
        st.success("Prediction Complete!", icon="✅")
        st.markdown(st.session_state.prediction_html, unsafe_allow_html=True)

        display_cols = [
                "Model_Output",
                "Rating_Class",
                "Rating",
                "Product_Category",
                "Sub_Category",
                "Product",
                "Material",
                "Size",
                "Manufacturer",
                "Date",
                "Color",
            ]

# ==========================================================
# HISTORY PAGE + EXPORT PAGE
# ==========================================================

# ----------------------------------------------------------
# PAGE 2 — HISTORY
# ----------------------------------------------------------
elif st.session_state.current_page == "history":

    st.title("📜 Prediction History")

    manu_dataset = sorted(df["Manufacturer"].dropna().unique().tolist()) if not df.empty else []
    manu_firestore = get_all_manufacturers_from_firestore()
    all_manus = sorted(set(manu_dataset) | set(manu_firestore))

    if not all_manus:
        st.info("No manufacturers found.")
    else:
        for manu in all_manus:
            dfm = get_manufacturer_predictions_df(manu)
            with st.expander(f"🏭 {str(manu).title()}"):
                if dfm.empty:
                    st.info("No prediction records yet for this manufacturer.")
                else:
                    st.dataframe(dfm, use_container_width=True)

# ----------------------------------------------------------
# PAGE 3 — EXPORT
# ----------------------------------------------------------
elif st.session_state.current_page == "export":

    st.title("⬇️ Export Predictions")

    manu_dataset = sorted(df["Manufacturer"].dropna().unique().tolist()) if not df.empty else []
    manu_firestore = get_all_manufacturers_from_firestore()
    all_manus = sorted(set(manu_dataset) | set(manu_firestore))

    if not all_manus:
        st.info("No manufacturers available.")
        st.stop()

    selected_manu = st.selectbox(
        "Choose Manufacturer to export",
        all_manus,
        index=0,
    )

    dfm = get_manufacturer_predictions_df(selected_manu)

    if dfm.empty:
        st.warning(f"No prediction data yet for '{selected_manu}'.")
        st.stop()

    st.markdown(f"## 📦 Export – {str(selected_manu).title()}")

    export_choice = st.radio(
        "Choose export type:",
        [
            "Full data",
            "High quality products (STOCK)",
            "Low quality products (DON'T STOCK)",
        ],
        index=0,
    )

    df_export = dfm.copy()

    if export_choice == "High quality products (STOCK)":
        df_export = df_export[df_export["Decision"] == "STOCK"]
    elif export_choice == "Low quality products (DON'T STOCK)":
        df_export = df_export[df_export["Decision"] == "DON'T STOCK"]

    df_export = df_export.drop_duplicates()

    if df_export.empty:
        st.warning("No records match this filter.")
        st.stop()

    csv_data = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        csv_data,
        file_name=f"{selected_manu}_export.csv",
        mime="text/csv",
        use_container_width=True,
    )
