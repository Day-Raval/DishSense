"""
FridgeRAG — Streamlit demo dashboard.
Run: streamlit run dashboard/app.py
Requires the FastAPI server running on localhost:8000.
"""
import requests
import streamlit as st
from PIL import Image

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="FridgeRAG",
    page_icon="🥦",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8fffb 0%, #ffffff 60%);
        }
        .fridge-hero {
            background: linear-gradient(135deg, #e9f9ef 0%, #f4f0ff 100%);
            border: 1px solid #d5eadc;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(40, 64, 53, 0.08);
        }
        .fridge-card {
            border: 1px solid #e6ece7;
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.8rem;
            background: #ffffff;
            box-shadow: 0 8px 18px rgba(14, 30, 20, 0.05);
        }
        .fridge-small {
            color: #557464;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="fridge-hero">
        <h1>🥦 FridgeRAG</h1>
        <p class="fridge-small">📸 Snap a photo of your fridge — get ranked recipe recommendations based on exactly what you have.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar — filters and health check ───────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")
    preferences  = st.text_input(
        "🥗 Dietary preferences",
        placeholder="e.g. high protein, vegetarian, low carb",
    )
    max_calories = st.slider("🔥 Max calories",        0, 1500, 0, step=50,
                             help="0 = no limit")
    max_minutes  = st.slider("⏱️ Max cook time (min)", 0, 120,  0, step=10,
                             help="0 = no limit")
    top_n        = st.slider("📚 Recipes to show",     1, 10,   5)

    st.divider()
    st.caption("🔌 API status")
    try:
        hres  = requests.get(f"{API_BASE}/health", timeout=3)
        hdata = hres.json()
        if hdata["db_ready"]:
            st.success(f"DB ready — {hdata['recipe_count']:,} recipes loaded")
        else:
            st.error("DB not ready — run build_vectordb.py first")
    except Exception:
        st.error("API offline — run: uvicorn api.main:app --reload --port 8000")

# ── Main area — upload and results ───────────────────────────────────────────
uploaded = st.file_uploader(
    "🧊 Upload fridge photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    col_photo, col_results = st.columns([1, 2], gap="large")

    with col_photo:
        st.markdown('<div class="fridge-card">', unsafe_allow_html=True)
        st.image(
            Image.open(uploaded),
            caption="📷 Your fridge",
            use_column_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        with st.spinner("Detecting ingredients and finding recipes..."):
            uploaded.seek(0)
            try:
                response = requests.post(
                    f"{API_BASE}/recommend",
                    files={"photo": (uploaded.name, uploaded.read(), uploaded.type)},
                    data={
                        "preferences":  preferences,
                        "max_calories": max_calories,
                        "max_minutes":  max_minutes,
                        "top_n":        top_n,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Is uvicorn running on port 8000?")
                st.stop()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        if data.get("error"):
            st.warning(data["error"])
            st.stop()

        # ── Detected ingredients ──────────────────────────────────────────────
        ings = data.get("detected_ingredients", [])
        src  = data.get("model_sources", {})

        st.subheader(f"🧪 Detected {len(ings)} ingredients")
        st.write(", ".join(f"`{i}`" for i in ings))

        if src:
            st.caption(
                f"YOLO: {src.get('yolo_count', 0)} hits  |  "
                f"DETR: {src.get('detr_count', 0)} hits  |  "
                f"CLIP: {src.get('clip_count', 0)} hits"
            )

        st.divider()

        # ── Recipe cards ──────────────────────────────────────────────────────
        recs = data.get("recommendations", [])
        st.subheader(f"🍽️ Top {len(recs)} recipes for you")

        for rec in recs:
            coverage = rec.get("coverage_pct", 0)
            name     = rec.get("name", "Recipe")
            rank     = rec.get("rank", "")
            color    = (
                "🟢" if coverage >= 80 else
                "🟡" if coverage >= 50 else
                "🔴"
            )

            with st.expander(
                f"{color}  #{rank}  {name}  —  {coverage}% ingredient match"
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🔥 Calories",  f"{rec.get('calories',    0):.0f} kcal")
                c2.metric("💪 Protein",   f"{rec.get('protein_g',   0):.0f} g")
                c3.metric("⏱️ Cook time", f"{rec.get('minutes',     0)} min")
                c4.metric("📊 Nutrition", f"{rec.get('nutrition_score', 0)} / 10")

                missing = rec.get("missing_ingredients", [])
                if missing:
                    st.write("**Still need to buy:**", ", ".join(missing))
                else:
                    st.write("**You have everything needed.**")

                reason = rec.get("reason", "")
                if reason:
                    st.info(reason)
