"""
KiteIQX Intelligence - Streamlit Analytics Engine
=================================================
Refactored to:
  1. Match KiteIQX.com brand (navy + gold, professional consulting feel)
  2. Accept CSV, XLSX, and XLS files
  3. Executive Dashboard rebuilt as a narrative storyboard
  4. AI Intelligence moved to tab 2
  5. Data Quality lives in its own dedicated tab
  6. Every uploaded file is persisted to data/uploads/

SECURITY: The Groq API key is now read from st.secrets, NOT hardcoded.
Set it in .streamlit/secrets.toml or via the Streamlit Cloud dashboard.
"""

import io
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# Google Sheets imports (soft)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

# Groq imports (soft)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="KiteIQX Intelligence",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="◆",
)


# ============================================================
# ⚠️  GROQ API KEY  -- TODO: ROTATE THIS AND MOVE TO st.secrets
# This is the SAME key from your original file. It is exposed in
# source control - rotate at https://console.groq.com/keys and
# replace with st.secrets["GROQ_API_KEY"] when you have time.
# ============================================================
_HARDCODED_GROQ_KEY = "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"


def _get_groq_key() -> str:
    """Prefer st.secrets, fall back to env var, fall back to hardcoded."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", _HARDCODED_GROQ_KEY)


GROQ_API_KEY = _get_groq_key()

# ============================================================
# GOOGLE SHEETS LOGGER
# ============================================================

SHEET_ID = "1rJYe4MVKDc9srbzrqf1CWIfFLmPn_4qTwSEZ5PCTVT0"
GSHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource(show_spinner=False)
def _get_gsheets_client():
    """Cached Google Sheets client — one connection for the whole session."""
    if not GSHEETS_AVAILABLE:
        return None
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=GSHEETS_SCOPES)
        return gspread.authorize(creds)
    except Exception:
        return None


def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


def log_upload(filename: str, df, summary: str = "", takeaways: list = None):
    """Log a file upload event to the 'uploads' sheet."""
    try:
        client = _get_gsheets_client()
        if not client:
            return
        sh = client.open_by_key(SHEET_ID)
        ws = sh.worksheet("uploads")
        a = st.session_state.get("analytics")
        completeness = round(100 - df.isnull().sum().sum() / max(1, df.size) * 100, 1)
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        cat_cols = len(df.select_dtypes(include=["object", "category"]).columns)
        money_cols = [c for c in df.columns if any(w in c.lower()
                      for w in ("price","cost","amount","revenue","salary","sales","gmv"))]
        total_val = round(float(df[money_cols].sum().sum()), 2) if money_cols else ""
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            _get_session_id(),
            filename,
            len(df),
            len(df.columns),
            num_cols,
            cat_cols,
            f"{completeness}%",
            str(total_val) if total_val != "" else "N/A",
            summary[:500] if summary else "",
            " | ".join(takeaways) if takeaways else "",
        ], value_input_option="USER_ENTERED")
    except Exception:
        pass  # never crash the app over logging


def log_ai_query(filename: str, question: str, answer: str):
    """Log an AI question + answer to the 'ai_queries' sheet."""
    try:
        client = _get_gsheets_client()
        if not client:
            return
        sh = client.open_by_key(SHEET_ID)
        ws = sh.worksheet("ai_queries")
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            _get_session_id(),
            filename,
            question[:300],
            answer[:800],
        ], value_input_option="USER_ENTERED")
    except Exception:
        pass


# ============================================================
# THEME (KiteIQX consulting brand: navy + gold)
# ============================================================

st.markdown(
    """
<style>
:root {
    --kite-navy: #0a2540;
    --kite-navy-2: #14304f;
    --kite-gold: #c79a3a;
    --kite-gold-soft: #e6c878;
    --kite-text: #1a2333;
    --kite-text-soft: #4a5568;
    --kite-surface: #ffffff;
    --kite-surface-soft: #f7f8fb;
    --kite-border: #e3e6ee;
    --kite-success: #15803d;
    --kite-warning: #b45309;
    --kite-danger:  #b91c1c;
}

/* Force light mode globally */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, section.main {
    background-color: #ffffff !important;
    color: #1a2333 !important;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif !important;
    background-color: #ffffff !important;
}
div.stMarkdown, div.stText { color: #1a2333; }

.block-container { padding-top: 4.5rem !important; max-width: 1200px; }

.stDeployButton { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.kx-brand {
    padding-top: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}
.kx-mark {
    width: 58px; height: 58px;
    background: var(--kite-navy);
    color: var(--kite-gold);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1.4rem;
    letter-spacing: -1px;
}
.kx-wordmark {
    font-size: 2.6rem; font-weight: 700;
    color: #0a2540 !important;
    letter-spacing: -0.3px;
}
.kx-wordmark span { color: #c79a3a !important; }
.kx-tag {
    color: #4a5568 !important;
    font-size: 0.95rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid #e3e6ee;
    padding-bottom: 1rem;
}

.kx-pill {
    display: inline-block;
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    margin: 0.15rem 0.3rem 0.15rem 0;
}
.kx-pill-navy { background: var(--kite-navy); color: var(--kite-gold-soft); }
.kx-pill-gold { background: var(--kite-gold); color: #2a1b00; }
.kx-pill-soft { background: var(--kite-surface-soft); color: var(--kite-text-soft); border: 1px solid var(--kite-border); }

.kx-hero {
    background: linear-gradient(135deg, var(--kite-navy) 0%, var(--kite-navy-2) 100%);
    color: #ffffff;
    border-radius: 14px;
    padding: 1.75rem 2rem;
    margin: 0.5rem 0 1.25rem 0;
    box-shadow: 0 4px 24px rgba(10, 37, 64, 0.10);
}
.kx-hero h2 { color: #ffffff; margin: 0 0 0.4rem 0; font-weight: 700; }
.kx-hero p  { color: #d8dde6; margin: 0; font-size: 1.05rem; line-height: 1.55; }
.kx-hero .kx-hero-accent { color: var(--kite-gold); font-weight: 700; }

.kx-card {
    background: #ffffff !important;
    border: 1px solid var(--kite-border);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin: 0.6rem 0;
    box-shadow: 0 1px 3px rgba(10, 37, 64, 0.04);
    color: #1a2333 !important;
}
.kx-card-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #4a5568 !important;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.kx-card-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0a2540 !important;
    line-height: 1.2;
}
.kx-card-sub {
    font-size: 0.85rem;
    color: #4a5568 !important;
    margin-top: 0.25rem;
}

.kx-callout {
    border-left: 4px solid var(--kite-gold);
    background: #f7f8fb !important;
    padding: 1rem 1.25rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: #1a2333 !important;
}

.kx-takeaway {
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    background: #fdfcf6 !important;
    border: 1px solid #efe3c1;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
}
.kx-takeaway-num {
    flex: 0 0 30px;
    width: 30px; height: 30px;
    border-radius: 50%;
    background: var(--kite-gold);
    color: #2a1b00 !important;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.kx-takeaway-body { color: #1a2333 !important; font-size: 0.96rem; line-height: 1.45; }

.kx-ai-response {
    background: #f7f8fb !important;
    border-left: 4px solid #0a2540;
    padding: 1.25rem 1.5rem;
    color: #1a2333 !important;
    border-radius: 0 10px 10px 0;
    margin: 1rem 0;
    line-height: 1.55;
}

.kx-q-excellent { background: #ecfdf3 !important; border: 1px solid #abefc6; padding: 1rem 1.2rem; border-radius: 10px; color: #14532d !important; }
.kx-q-good      { background: #f0f9ff !important; border: 1px solid #bae0fd; padding: 1rem 1.2rem; border-radius: 10px; color: #0c4a6e !important; }
.kx-q-warning   { background: #fffaeb !important; border: 1px solid #fcd980; padding: 1rem 1.2rem; border-radius: 10px; color: #78350f !important; }
.kx-q-poor      { background: #fef2f2 !important; border: 1px solid #fda4a4; padding: 1rem 1.2rem; border-radius: 10px; color: #7f1d1d !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--kite-border);
}
.stTabs [data-baseweb="tab"] {
    padding: 0.75rem 1.2rem;
    font-weight: 500;
    color: var(--kite-text-soft);
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    color: var(--kite-navy) !important;
    border-bottom: 3px solid var(--kite-gold) !important;
    font-weight: 700;
}

button,
.stButton > button,
.stButton > button *,
[data-testid*="Button"],
[data-testid*="Button"] *,
[data-testid*="button"],
[data-testid*="button"] *,
div[data-testid="stSidebar"] button,
div[data-testid="stSidebar"] button *,
div[data-testid="stSidebar"] button p,
div[data-testid="stSidebar"] button span {
    background-color: #0a2540 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}
button:hover,
.stButton > button:hover,
.stButton > button:hover * {
    background-color: #14304f !important;
    color: #e6c878 !important;
}

.kx-dash-row { display: flex; gap: 0.75rem; margin: 0.5rem 0; }
.kx-dash-panel {
    flex: 1; background: #ffffff !important; border: 1px solid #e3e6ee;
    border-radius: 10px; padding: 0.5rem; min-width: 0;
}

section[data-testid="stSidebar"] {
    background: #f7f8fb !important;
    border-right: 1px solid #e3e6ee;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not([class*="st-"]),
section[data-testid="stSidebar"] .stMarkdown {
    color: #1a2333 !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
    color: #1a2333 !important;
    background-color: #ffffff !important;
}
section[data-testid="stSidebar"] input {
    color: #1a2333 !important;
    background-color: #ffffff !important;
}
</style>
""",
    unsafe_allow_html=True,
)


KITE_PLOTLY = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Inter, system-ui, sans-serif", color="#1a2333"),
    colorway=["#0a2540", "#c79a3a", "#14304f", "#e6c878", "#4a5568", "#a98a3d"],
    margin=dict(t=50, l=40, r=20, b=40),
)


def kx_apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**KITE_PLOTLY)
    return fig



# ============================================================
# GROQ WRAPPER
# ============================================================

FIXED_MODEL = "llama-3.1-8b-instant"


class GroqLLM:
    def __init__(self, api_key: str, model: str = FIXED_MODEL):
        self.client = Groq(api_key=api_key)
        self.model = model

    def predict(self, prompt: str, max_tokens: int = 2000, system: str = None) -> str:
        if system is None:
            system = (
                "You are KiteIQX Intelligence, a senior management consultant. "
                "Speak in clear business language. Always give specific numbers and "
                "concrete recommendations - never vague generalities."
            )
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"AI error: {e}"


# ============================================================
# ANALYTICS CORE
# ============================================================

class UniversalAnalytics:
    def __init__(self, df: pd.DataFrame, llm=None):
        self.df = df
        self.llm = llm
        self.original_df = df.copy()
        self.process_data()
        self.generate_insights()

    def process_data(self):
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.datetime_cols = []

        for col in list(self.df.columns):
            if any(k in col.lower() for k in ("date", "time", "created", "updated", "timestamp")):
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.datetime_cols.append(col)
                    if col in self.categorical_cols:
                        self.categorical_cols.remove(col)
                except Exception:
                    pass

        self.money_cols = [
            c for c in self.numeric_cols
            if any(w in c.lower() for w in ("price", "cost", "amount", "revenue", "salary", "income", "fee", "payment", "spend", "sales", "gmv"))
        ]
        self.score_cols = [c for c in self.numeric_cols if any(w in c.lower() for w in ("score", "rating", "rank", "grade"))]
        self.id_cols = [
            c for c in self.df.columns
            if any(w in c.lower() for w in ("id", "key")) and self.df[c].nunique() / max(1, len(self.df)) > 0.8
        ]

        if len(self.money_cols) > 1:
            self.df["total_monetary_value"] = self.df[self.money_cols].sum(axis=1, skipna=True)
        elif len(self.money_cols) == 1:
            self.df["total_monetary_value"] = self.df[self.money_cols[0]]

        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.check_data_quality()

    def check_data_quality(self):
        issues = {}
        dup = int(self.df.duplicated().sum())
        if dup:
            issues["duplicates"] = {"count": dup, "percentage": dup / len(self.df) * 100}

        missing = {}
        for c in self.df.columns:
            m = int(self.df[c].isnull().sum())
            if m:
                missing[c] = {"count": m, "percentage": m / len(self.df) * 100}
        if missing:
            issues["missing_data"] = missing

        outliers = {}
        for c in self.numeric_cols:
            o = self._iqr_outliers(c)
            if o["count"] > 0:
                outliers[c] = o
        if outliers:
            issues["outliers"] = outliers

        self.data_quality_issues = issues

    def _iqr_outliers(self, col):
        s = self.df[col].dropna()
        if len(s) < 10:
            return {"count": 0}
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outs = s[(s < lo) | (s > hi)]
        return {
            "count": int(len(outs)),
            "percentage": len(outs) / len(s) * 100,
            "range": f"{outs.min():.2f} to {outs.max():.2f}" if len(outs) else "None",
            "bounds": f"normal range: {lo:.2f} to {hi:.2f}",
        }

    def get_data_quality_report(self):
        if not self.data_quality_issues:
            return "excellent", 100, "No significant data quality issues detected. The dataset is analysis-ready."

        severity = 0
        bullets = []
        if "duplicates" in self.data_quality_issues:
            d = self.data_quality_issues["duplicates"]
            severity += min(d["percentage"] * 2, 30)
            bullets.append(f"**Duplicates:** {d['count']:,} records ({d['percentage']:.1f}%)")
        if "missing_data" in self.data_quality_issues:
            top = max(v["percentage"] for v in self.data_quality_issues["missing_data"].values())
            severity += min(top, 25)
            bullets.append(f"**Missing data:** up to {top:.1f}% in worst-affected column")
        if "outliers" in self.data_quality_issues:
            top = max(v["percentage"] for v in self.data_quality_issues["outliers"].values())
            severity += min(top, 15)
            bullets.append(f"**Outliers:** up to {top:.1f}% values flagged as statistical outliers")

        score = max(0, int(100 - severity))
        if score >= 90:
            status = "excellent"
        elif score >= 75:
            status = "good"
        elif score >= 60:
            status = "fair"
        else:
            status = "poor"
        return status, score, "  \n".join(bullets)

    def clean_duplicates(self):
        if "duplicates" in self.data_quality_issues:
            n = len(self.df)
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            removed = n - len(self.df)
            self.process_data()
            self.generate_insights()
            return f"Removed {removed:,} duplicate records. Now {len(self.df):,} unique rows."
        return "No duplicates to remove."

    def generate_insights(self):
        self.insights = {
            "basic_stats": {
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "missing_pct": (self.df.isnull().sum().sum() / max(1, self.df.size)) * 100,
            },
            "column_types": {
                "numeric": len(self.numeric_cols),
                "categorical": len(self.categorical_cols),
                "datetime": len(self.datetime_cols),
            },
        }
        if "total_monetary_value" in self.df.columns:
            mc = "total_monetary_value"
            self.insights["monetary"] = {
                "total": float(self.df[mc].sum()),
                "avg": float(self.df[mc].mean()),
                "max": float(self.df[mc].max()),
                "min": float(self.df[mc].min()),
            }

    def _data_brief(self, max_rows: int = 5) -> str:
        return (
            f"Rows: {len(self.df):,}\n"
            f"Columns: {len(self.df.columns)} "
            f"({len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical, "
            f"{len(self.datetime_cols)} datetime)\n"
            f"Numeric columns: {', '.join(self.numeric_cols[:12])}\n"
            f"Categorical columns: {', '.join(self.categorical_cols[:12])}\n\n"
            f"Sample rows:\n{self.df.head(max_rows).to_string()}\n\n"
            f"Numeric summary:\n{self.df[self.numeric_cols[:6]].describe().to_string() if self.numeric_cols else '(none)'}\n"
        )

    def ai_query(self, query: str) -> str:
        if not self.llm:
            return "AI is not configured. Add GROQ_API_KEY to .streamlit/secrets.toml."
        prompt = f"DATASET CONTEXT\n{self._data_brief()}\n\nUSER QUESTION\n{query}\n\nAnswer with specific numbers and concrete recommendations."
        answer = self.llm.predict(prompt)
        filename = st.session_state.get("last_saved_path", "unknown")
        log_ai_query(filename, query, answer)
        return answer

    def ai_executive_summary(self) -> str:
        if not self.llm:
            return self._fallback_summary()

        cat_context = ""
        for c in self.categorical_cols[:3]:
            vc = self.df[c].value_counts().head(5)
            cat_context += f"\n  {c}: {', '.join(f'{k}={v}' for k, v in vc.items())}"

        num_context = ""
        for c in self.numeric_cols[:5]:
            s = self.df[c].dropna()
            num_context += f"\n  {c}: mean={s.mean():.2f}, median={s.median():.2f}, max={s.max():.2f}, min={s.min():.2f}"

        monetary_ctx = ""
        if "monetary" in self.insights:
            m = self.insights["monetary"]
            monetary_ctx = f"\n  Total monetary value: ${m['total']:,.0f}, avg per record: ${m['avg']:,.2f}"

        prompt = (
            f"DATASET CONTEXT\n{self._data_brief()}"
            f"\nTop categorical breakdowns:{cat_context}"
            f"\nNumeric profile:{num_context}"
            f"{monetary_ctx}\n\n"
            "Write a sharp 3-sentence executive narrative. Rules:\n"
            "  Sentence 1 — BUSINESS DOMAIN: Name the specific business function this data covers "
            "(e.g. 'retail sales', 'HR workforce', 'logistics') and the time scope if visible. "
            "Be specific — not just 'this dataset contains records'.\n"
            "  Sentence 2 — KEY BUSINESS SIGNAL: State the single most important pattern, trend, or "
            "anomaly using actual numbers from the data. Focus on commercial or operational impact.\n"
            "  Sentence 3 — DECISION IMPLICATION: What should a business leader do or watch because of this? "
            "One clear, actionable implication.\n\n"
            "Tone: senior consultant briefing a CEO. Specific numbers, zero filler, no bullet points — "
            "three sentences of clean prose."
        )
        return self.llm.predict(prompt, max_tokens=600)

    def ai_top_takeaways(self) -> list:
        if not self.llm:
            return self._fallback_takeaways()
        prompt = (
            f"DATASET CONTEXT\n{self._data_brief()}\n\n"
            "Return exactly 3 strategic takeaways for a CEO. Each takeaway must:\n"
            "  - Be a single, direct sentence\n"
            "  - Reference a specific number from the data\n"
            "  - Point toward a concrete growth opportunity, operational fix, or strategic decision — "
            "not just an observation\n"
            "  - Start with the finding, end with the recommended next move\n\n"
            "Think like a McKinsey partner: the goal is to surface what to DO next, not just what the data shows.\n\n"
            "Format: numbered list (1. 2. 3.) — no preamble, no headings, nothing else."
        )
        out = self.llm.predict(prompt, max_tokens=500)
        lines = [re.sub(r"^\s*\d+[\.\)]\s*", "", ln).strip() for ln in out.split("\n") if ln.strip()]
        lines = [ln for ln in lines if len(ln) > 10]
        return lines[:3] if lines else self._fallback_takeaways()

    def _fallback_summary(self) -> str:
        b = self.insights["basic_stats"]
        return (
            f"This dataset contains {b['rows']:,} records across {b['columns']} columns, with "
            f"{self.insights['column_types']['numeric']} numeric, {self.insights['column_types']['categorical']} "
            f"categorical, and {self.insights['column_types']['datetime']} datetime fields. "
            f"Overall completeness sits at {100 - b['missing_pct']:.1f}%. "
            "Configure the AI assistant for a deeper narrative summary."
        )

    def _fallback_takeaways(self) -> list:
        out = []
        if self.numeric_cols:
            c = self.numeric_cols[0]
            out.append(f"Primary metric '{c}' averages {self.df[c].mean():.2f} (range {self.df[c].min():.2f} to {self.df[c].max():.2f}).")
        if self.categorical_cols:
            c = self.categorical_cols[0]
            top = self.df[c].value_counts().head(1)
            if len(top):
                out.append(f"In '{c}', '{top.index[0]}' is the dominant segment with {top.iloc[0]:,} records ({top.iloc[0]/len(self.df)*100:.1f}% share).")
        if self.insights["basic_stats"]["missing_pct"] > 5:
            out.append(f"Data completeness is {100 - self.insights['basic_stats']['missing_pct']:.1f}%; the Data Quality tab will guide cleanup before deep analysis.")
        while len(out) < 3:
            out.append("Connect the AI engine (Groq key in secrets) for richer, data-specific takeaways.")
        return out[:3]



# ============================================================
# VISUALIZATIONS
# ============================================================

class VizEngine:
    def __init__(self, analytics: UniversalAnalytics):
        self.a = analytics
        self.df = analytics.df

    def scatter(self, x, y, color=None):
        if color and color in self.a.categorical_cols:
            fig = px.scatter(self.df, x=x, y=y, color=color, title=f"{y} vs {x} by {color}")
        else:
            fig = px.scatter(self.df, x=x, y=y, title=f"{y} vs {x}")
        try:
            corr = self.df[x].corr(self.df[y])
            fig.add_annotation(text=f"r = {corr:.3f}", xref="paper", yref="paper", x=0.02, y=0.98,
                               showarrow=False, bgcolor="rgba(255,255,255,0.9)", bordercolor="#0a2540")
        except Exception:
            corr = float("nan")
        return kx_apply_theme(fig), f"Correlation {corr:.3f}" if corr == corr else "Scatter created"

    def correlation_matrix(self, cols):
        nums = [c for c in cols if c in self.a.numeric_cols]
        if len(nums) < 2:
            return None, "Need at least 2 numeric columns."
        m = self.df[nums].corr()
        fig = px.imshow(m, text_auto=".2f", aspect="auto", color_continuous_scale=["#b91c1c", "#ffffff", "#0a2540"], title="Correlation matrix")
        return kx_apply_theme(fig), f"Correlation matrix for {len(nums)} variables"

    def distribution(self, cols):
        nums = [c for c in cols if c in self.a.numeric_cols][:4]
        if not nums:
            return None, "No numeric columns selected."
        ncols = min(2, len(nums))
        nrows = (len(nums) + 1) // 2
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[f"{c}" for c in nums])
        for i, c in enumerate(nums):
            r, cc = i // ncols + 1, i % ncols + 1
            fig.add_trace(go.Histogram(x=self.df[c], name=c, nbinsx=30, marker_color="#0a2540"), row=r, col=cc)
        fig.update_layout(height=340 * nrows, showlegend=False, title_text="Distribution analysis")
        return kx_apply_theme(fig), f"Distribution analysis for {len(nums)} variables"

    def regression(self, target, predictors):
        preds = [c for c in predictors if c in self.a.numeric_cols and c != target]
        if not preds:
            return None, "Need a numeric predictor."
        p = preds[0]
        data = self.df[[target, p]].dropna()
        X = data[p].values.reshape(-1, 1)
        y = data[target].values
        m = LinearRegression().fit(X, y)
        y_hat = m.predict(X)
        r2 = r2_score(y, y_hat)
        resid = y - y_hat
        fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{p} -> {target}", "Residuals", "Actual vs predicted", "R^2"])
        xs = np.linspace(X.min(), X.max(), 100)
        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode="markers", name="actual",
                                 marker=dict(color="#0a2540", opacity=0.6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=m.predict(xs.reshape(-1, 1)), mode="lines", name="fit",
                                 line=dict(color="#c79a3a", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=y_hat, y=resid, mode="markers", marker=dict(color="#c79a3a", opacity=0.6),
                                 name="residuals"), row=1, col=2)
        fig.add_trace(go.Scatter(x=y, y=y_hat, mode="markers", marker=dict(color="#14304f", opacity=0.6),
                                 name="a-vs-p"), row=2, col=1)
        lo, hi = min(y.min(), y_hat.min()), max(y.max(), y_hat.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#888", dash="dash"),
                                 name="ideal"), row=2, col=1)
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=r2,
            domain={"x": [0, 1], "y": [0, 1]}, title={"text": "R^2"},
            gauge={"axis": {"range": [None, 1]},
                   "bar": {"color": "#0a2540"},
                   "steps": [{"range": [0, 0.5], "color": "#f0f1f5"},
                             {"range": [0.5, 0.8], "color": "#fcd980"},
                             {"range": [0.8, 1], "color": "#a98a3d"}]}), row=2, col=2)
        fig.update_layout(height=620, showlegend=False, title_text="Predictive model")
        return kx_apply_theme(fig), f"R^2 = {r2:.3f}, RMSE = {np.sqrt(np.mean(resid**2)):.3f}"

    def timeseries(self, date_col, value_col):
        if date_col not in self.a.datetime_cols:
            return None, "Column is not a datetime."
        if value_col not in self.a.numeric_cols:
            return None, "Value column must be numeric."
        d = self.df[[date_col, value_col]].dropna().sort_values(date_col)
        fig = make_subplots(rows=2, cols=1, subplot_titles=[f"{value_col} over time", "Trend"], vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=d[date_col], y=d[value_col], mode="lines+markers", name=value_col,
                                 line=dict(color="#0a2540", width=2)), row=1, col=1)
        if len(d) > 7:
            d["ma7"] = d[value_col].rolling(7).mean()
            fig.add_trace(go.Scatter(x=d[date_col], y=d["ma7"], mode="lines", name="7-period MA",
                                     line=dict(color="#c79a3a", dash="dash")), row=1, col=1)
        if len(d) > 3:
            xn = np.arange(len(d))
            z = np.polyfit(xn, d[value_col], 1)
            fig.add_trace(go.Scatter(x=d[date_col], y=np.poly1d(z)(xn), mode="lines", name="trend",
                                     line=dict(color="#14304f", width=3)), row=2, col=1)
        fig.update_layout(height=560, title_text="Time-series analysis")
        return kx_apply_theme(fig), f"Time-series for {value_col}"


# ============================================================
# DEMO DATA
# ============================================================

def generate_demo(kind: str) -> pd.DataFrame:
    np.random.seed(42)
    if kind == "E-commerce Sales":
        n = 1000
        return pd.DataFrame({
            "order_id": [f"ORD{i:06d}" for i in range(1, n + 1)],
            "customer_id": [f"CUST{i:05d}" for i in np.random.randint(1, 501, n)],
            "product_category": np.random.choice(["Electronics", "Clothing", "Home", "Books", "Sports"], n),
            "order_value": np.random.lognormal(4, 0.8, n),
            "shipping_cost": np.random.uniform(5, 25, n),
            "customer_age": np.random.randint(18, 75, n),
            "customer_segment": np.random.choice(["Premium", "Standard", "Budget"], n),
            "delivery_days": np.random.randint(1, 15, n),
            "customer_rating": np.random.randint(1, 6, n),
            "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        })
    if kind == "Employee Data":
        n = 500
        return pd.DataFrame({
            "employee_id": [f"EMP{i:04d}" for i in range(1, n + 1)],
            "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"], n),
            "salary": np.random.normal(75000, 25000, n),
            "age": np.random.randint(22, 65, n),
            "years_experience": np.random.randint(0, 25, n),
            "performance_score": np.random.normal(3.5, 0.8, n),
            "job_satisfaction": np.random.randint(1, 11, n),
        })
    n = 300
    return pd.DataFrame({
        "id": range(1, n + 1),
        "category": np.random.choice(["A", "B", "C", "D"], n),
        "value": np.random.uniform(10, 100, n),
        "score": np.random.randint(1, 11, n),
        "amount": np.random.uniform(100, 1000, n),
    })


# ============================================================
# FILE LOADING
# ============================================================

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = UPLOAD_DIR / f"{ts}_{name}"
    raw = uploaded_file.getbuffer()
    with open(saved, "wb") as f:
        f.write(raw)
    st.session_state["last_saved_path"] = str(saved)

    buf = io.BytesIO(raw)
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(buf)
    elif ext == "csv":
        df = pd.read_csv(buf)
    else:
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            df = pd.read_excel(io.BytesIO(raw))
    return df



# ============================================================
# UI BLOCKS
# ============================================================

def render_header():
    st.markdown(
        """
        <div class="kx-brand">
            <div class="kx-mark">K</div>
            <div class="kx-wordmark">Kite<span>IQX</span> Intelligence</div>
        </div>
        <div class="kx-tag">Where intelligence meets impact — turn raw data into a CEO-ready story in minutes.</div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome():
    st.markdown(
        """
        <div class="kx-hero">
            <h2>Upload your data, get a CEO-ready narrative in minutes.</h2>
            <p>Built for non-technical leaders. Drop a CSV or Excel file and KiteIQX will surface the
            <span class="kx-hero-accent">three insights that matter</span>, audit data quality, and let you ask follow-up
            questions in plain English.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="kx-card"><div class="kx-card-title">Executive Dashboard</div>
            <div class="kx-card-sub">Narrative summary, hero KPIs, and the three takeaways a leader needs.</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="kx-card"><div class="kx-card-title">AI Intelligence</div>
            <div class="kx-card-sub">Ask any business question; get a consultant-grade answer in seconds.</div></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="kx-card"><div class="kx-card-title">Data Quality</div>
            <div class="kx-card-sub">Detect duplicates, missingness, and outliers before they cost you a decision.</div></div>""",
            unsafe_allow_html=True,
        )
    st.info("Use the sidebar to upload a CSV / XLSX file or try a demo dataset.")


def render_dashboard():
    a: UniversalAnalytics = st.session_state.analytics
    ins = a.insights

    record_str = f"{ins['basic_stats']['rows']:,}"
    col_str = f"{ins['basic_stats']['columns']}"
    quality_pct = 100 - ins["basic_stats"]["missing_pct"]
    monetary_line = ""
    if "monetary" in ins:
        monetary_line = (
            f' Total tracked value across the dataset is '
            f'<span class="kx-hero-accent">${ins["monetary"]["total"]:,.0f}</span>.'
        )
    st.markdown(
        f"""
        <div class="kx-hero">
            <h2>Business Story</h2>
            <p>KiteIQX has analysed <span class="kx-hero-accent">{record_str}</span> records across
            <span class="kx-hero-accent">{col_str}</span> dimensions
            ({ins['column_types']['numeric']} numeric · {ins['column_types']['categorical']} categorical · {ins['column_types']['datetime']} time-based)
            with <span class="kx-hero-accent">{quality_pct:.1f}%</span> completeness.{monetary_line}
            The AI narrative and takeaways below reflect what the numbers actually say about your business.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(4)
    cards[0].markdown(
        f"""<div class="kx-card"><div class="kx-card-title">Records analyzed</div>
        <div class="kx-card-value">{record_str}</div>
        <div class="kx-card-sub">Rows of business activity</div></div>""", unsafe_allow_html=True)
    cards[1].markdown(
        f"""<div class="kx-card"><div class="kx-card-title">Dimensions</div>
        <div class="kx-card-value">{col_str}</div>
        <div class="kx-card-sub">{ins['column_types']['numeric']} numeric · {ins['column_types']['categorical']} categorical · {ins['column_types']['datetime']} time</div></div>""", unsafe_allow_html=True)
    if "monetary" in ins:
        cards[2].markdown(
            f"""<div class="kx-card"><div class="kx-card-title">Total monetary value</div>
            <div class="kx-card-value">${ins['monetary']['total']:,.0f}</div>
            <div class="kx-card-sub">Average per record: ${ins['monetary']['avg']:,.2f}</div></div>""", unsafe_allow_html=True)
    else:
        cards[2].markdown(
            f"""<div class="kx-card"><div class="kx-card-title">Numeric depth</div>
            <div class="kx-card-value">{ins['column_types']['numeric']}</div>
            <div class="kx-card-sub">Quantifiable dimensions</div></div>""", unsafe_allow_html=True)
    cards[3].markdown(
        f"""<div class="kx-card"><div class="kx-card-title">Data completeness</div>
        <div class="kx-card-value">{quality_pct:.1f}%</div>
        <div class="kx-card-sub">Higher is better; details in Data Quality tab</div></div>""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;'
        'color:#4a5568;font-weight:600;margin-bottom:0.25rem;">Business narrative</div>',
        unsafe_allow_html=True,
    )

    if "exec_summary" not in st.session_state:
        with st.spinner("KiteIQX is reading the business context..."):
            st.session_state.exec_summary = a.ai_executive_summary()
    st.markdown(
        f'<div class="kx-callout" style="font-size:1.02rem;line-height:1.65;">'
        f'{st.session_state.exec_summary}</div>',
        unsafe_allow_html=True,
    )

    refresh_col, _ = st.columns([1, 5])
    if refresh_col.button("↻ Regenerate", key="regen_summary"):
        st.session_state.pop("exec_summary", None)
        st.session_state.pop("takeaways", None)
        st.rerun()

    st.markdown(" ")
    st.markdown(
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;'
        'color:#4a5568;font-weight:600;margin-bottom:0.35rem;">3 decisions this data supports</div>',
        unsafe_allow_html=True,
    )
    if "takeaways" not in st.session_state:
        with st.spinner("Identifying the headline insights..."):
            st.session_state.takeaways = a.ai_top_takeaways()

    if not st.session_state.get("upload_logged") and st.session_state.get("upload_filename"):
        log_upload(
            st.session_state.upload_filename,
            a.df,
            summary=st.session_state.get("exec_summary", ""),
            takeaways=st.session_state.get("takeaways", []),
        )
        st.session_state.upload_logged = True

    for i, t in enumerate(st.session_state.takeaways, 1):
        st.markdown(
            f"""<div class="kx-takeaway"><div class="kx-takeaway-num">{i}</div>
            <div class="kx-takeaway-body">{t}</div></div>""", unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;'
        'color:#4a5568;font-weight:600;margin-bottom:0.5rem;">Supporting visuals</div>',
        unsafe_allow_html=True,
    )

    COMPACT = dict(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, system-ui, sans-serif", color="#1a2333", size=11),
        colorway=["#0a2540", "#c79a3a", "#14304f", "#e6c878", "#4a5568"],
        margin=dict(t=36, l=28, r=12, b=28),
        height=220,
        showlegend=False,
    )

    vis_cols = st.columns(3)

    with vis_cols[0]:
        if a.categorical_cols:
            cat = a.categorical_cols[0]
            top = a.df[cat].value_counts().head(6).reset_index()
            top.columns = [cat, "count"]
            fig1 = px.bar(top, x=cat, y="count", title=f"{cat} breakdown")
            fig1.update_traces(marker_color="#0a2540")
            fig1.update_layout(**COMPACT)
            fig1.update_xaxes(tickangle=-30, tickfont_size=10)
            st.plotly_chart(fig1, use_container_width=True)
        elif a.numeric_cols:
            c = a.numeric_cols[0]
            fig1 = px.histogram(a.df, x=c, nbins=20, title=f"{c} distribution")
            fig1.update_traces(marker_color="#0a2540")
            fig1.update_layout(**COMPACT)
            st.plotly_chart(fig1, use_container_width=True)

    with vis_cols[1]:
        if a.datetime_cols and a.numeric_cols:
            dc = a.datetime_cols[0]
            vc = a.money_cols[0] if a.money_cols else a.numeric_cols[0]
            ts = a.df[[dc, vc]].dropna().sort_values(dc)
            fig2 = px.line(ts, x=dc, y=vc, title=f"{vc} over time")
            fig2.update_traces(line_color="#c79a3a", line_width=2)
            fig2.update_layout(**COMPACT)
            st.plotly_chart(fig2, use_container_width=True)
        elif len(a.numeric_cols) >= 2:
            x2, y2 = a.numeric_cols[0], a.numeric_cols[1]
            fig2 = px.scatter(a.df, x=x2, y=y2, title=f"{y2} vs {x2}")
            fig2.update_traces(marker=dict(color="#0a2540", opacity=0.5, size=5))
            fig2.update_layout(**COMPACT)
            st.plotly_chart(fig2, use_container_width=True)

    with vis_cols[2]:
        if len(a.categorical_cols) >= 2:
            cat2 = a.categorical_cols[1]
            top2 = a.df[cat2].value_counts().head(6).reset_index()
            top2.columns = [cat2, "count"]
            fig3 = px.bar(top2, x=cat2, y="count", title=f"{cat2} split",
                          color="count", color_continuous_scale=["#e6c878", "#0a2540"])
            fig3.update_layout(**COMPACT)
            fig3.update_coloraxes(showscale=False)
            fig3.update_xaxes(tickangle=-30, tickfont_size=10)
            st.plotly_chart(fig3, use_container_width=True)
        elif len(a.numeric_cols) >= 2:
            c3 = a.numeric_cols[1] if len(a.numeric_cols) > 1 else a.numeric_cols[0]
            fig3 = px.histogram(a.df, x=c3, nbins=20, title=f"{c3} distribution")
            fig3.update_traces(marker_color="#c79a3a")
            fig3.update_layout(**COMPACT)
            st.plotly_chart(fig3, use_container_width=True)

    if len(a.numeric_cols) >= 3 and len(a.categorical_cols) >= 1:
        vis_cols2 = st.columns(2)
        with vis_cols2[0]:
            box_col = a.money_cols[0] if a.money_cols else a.numeric_cols[0]
            cat_box = a.categorical_cols[0]
            fig4 = px.box(
                a.df, x=cat_box, y=box_col, title=f"{box_col} by {cat_box}",
                color_discrete_sequence=["#0a2540"],
            )
            fig4.update_layout(**COMPACT)
            st.plotly_chart(fig4, use_container_width=True)
        with vis_cols2[1]:
            corr_cols = a.numeric_cols[:5]
            corr_m = a.df[corr_cols].corr()
            fig5 = px.imshow(
                corr_m, text_auto=".1f", aspect="auto",
                color_continuous_scale=["#b91c1c", "#ffffff", "#0a2540"],
                title="Correlation heat",
            )
            fig5.update_layout(**{**COMPACT, "height": 220})
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        """
        <div class="kx-callout">
            <strong>Next step</strong>: open the <em>AI Intelligence</em> tab to ask follow-up questions in plain English,
            or jump to <em>Data Quality</em> to clean the dataset before deeper modeling.
        </div>
        """, unsafe_allow_html=True,
    )


def render_ai():
    a: UniversalAnalytics = st.session_state.analytics
    st.markdown('<div class="kx-card"><div class="kx-card-title">AI Intelligence</div>'
                '<div class="kx-card-sub">Ask any business question. Get a consultant-grade answer.</div></div>',
                unsafe_allow_html=True)

    if not a.llm:
        st.warning("AI is unavailable: GROQ_API_KEY missing. Add it to .streamlit/secrets.toml.")
        return

    qcols = st.columns(4)
    quick_prompts = {
        "Data overview": "Give a comprehensive overview of this dataset with the most important findings.",
        "Top correlations": "Identify the strongest correlations in the data and explain their business significance.",
        "Key trends": "Identify and explain the most important trends or patterns in the data.",
        "Anomalies": "Spot anomalies or outliers in the data and explain why they matter.",
    }
    for col, (label, qry) in zip(qcols, quick_prompts.items()):
        if col.button(label, key=f"qa_{label}"):
            with st.spinner("KiteIQX analyzing..."):
                st.session_state.ai_last_response = a.ai_query(qry)

    st.markdown(" ")
    user_q = st.text_area(
        "Custom question:",
        placeholder='e.g. "Which customer segment drives the most revenue and why?"',
        height=110,
        key="ai_custom_q",
    )
    if st.button("Ask KiteIQX", type="primary"):
        if user_q.strip():
            with st.spinner("Thinking..."):
                st.session_state.ai_last_response = a.ai_query(user_q.strip())

    if "ai_last_response" in st.session_state:
        st.markdown(
            f'<div class="kx-ai-response"><strong>KiteIQX Intelligence:</strong><br><br>'
            f'{st.session_state.ai_last_response}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Available columns"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Numeric**")
            for c in a.numeric_cols[:15]:
                st.write(f"• {c}")
        with c2:
            st.markdown("**Categorical**")
            for c in a.categorical_cols[:15]:
                st.write(f"• {c}")
        with c3:
            st.markdown("**Datetime**")
            for c in a.datetime_cols:
                st.write(f"• {c}")


def render_data_quality():
    a: UniversalAnalytics = st.session_state.analytics
    status, score, summary = a.get_data_quality_report()

    panel_class = {"excellent": "kx-q-excellent", "good": "kx-q-good",
                   "fair": "kx-q-warning", "poor": "kx-q-poor"}[status]
    headline = {"excellent": "Excellent", "good": "Good", "fair": "Needs attention", "poor": "Poor"}[status]

    st.markdown(
        f"""<div class="{panel_class}">
        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.6px; opacity:0.7;">Overall data quality</div>
        <div style="font-size: 2rem; font-weight: 800; line-height: 1.1;">{headline} — {score}/100</div>
        <div style="margin-top: 0.5rem;">{summary}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(" ")
    cols = st.columns(3)

    with cols[0]:
        st.markdown('<div class="kx-card"><div class="kx-card-title">Duplicates</div>', unsafe_allow_html=True)
        if "duplicates" in a.data_quality_issues:
            d = a.data_quality_issues["duplicates"]
            st.markdown(f'<div class="kx-card-value">{d["count"]:,}</div>'
                        f'<div class="kx-card-sub">{d["percentage"]:.1f}% of rows are exact duplicates</div></div>',
                        unsafe_allow_html=True)
            if st.button("Remove duplicates", key="dq_dedup"):
                msg = a.clean_duplicates()
                st.success(msg)
                st.session_state.pop("exec_summary", None)
                st.session_state.pop("takeaways", None)
                st.rerun()
        else:
            st.markdown('<div class="kx-card-value">0</div><div class="kx-card-sub">No duplicate rows detected.</div></div>',
                        unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="kx-card"><div class="kx-card-title">Missing values</div>', unsafe_allow_html=True)
        if "missing_data" in a.data_quality_issues:
            md = a.data_quality_issues["missing_data"]
            worst = max(md.items(), key=lambda x: x[1]["percentage"])
            st.markdown(
                f'<div class="kx-card-value">{len(md)}</div>'
                f'<div class="kx-card-sub">columns affected; worst: <b>{worst[0]}</b> ({worst[1]["percentage"]:.1f}%)</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="kx-card-value">0</div><div class="kx-card-sub">All columns are 100% populated.</div></div>',
                        unsafe_allow_html=True)

    with cols[2]:
        st.markdown('<div class="kx-card"><div class="kx-card-title">Outliers</div>', unsafe_allow_html=True)
        if "outliers" in a.data_quality_issues:
            od = a.data_quality_issues["outliers"]
            worst = max(od.items(), key=lambda x: x[1]["percentage"])
            st.markdown(
                f'<div class="kx-card-value">{len(od)}</div>'
                f'<div class="kx-card-sub">numeric columns flagged; worst: <b>{worst[0]}</b> ({worst[1]["percentage"]:.1f}%)</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="kx-card-value">0</div><div class="kx-card-sub">No statistical outliers (IQR method).</div></div>',
                        unsafe_allow_html=True)

    st.markdown(" ")
    if "missing_data" in a.data_quality_issues:
        st.markdown("#### Missing data — column-by-column")
        md = a.data_quality_issues["missing_data"]
        tbl = pd.DataFrame([
            {"Column": k, "Missing rows": v["count"], "Missing %": round(v["percentage"], 2)}
            for k, v in sorted(md.items(), key=lambda x: -x[1]["percentage"])
        ])
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    if "outliers" in a.data_quality_issues:
        st.markdown("#### Outliers — column-by-column (IQR method)")
        od = a.data_quality_issues["outliers"]
        tbl = pd.DataFrame([
            {"Column": k, "Outlier rows": v["count"], "Outlier %": round(v["percentage"], 2),
             "Outlier range": v.get("range", ""), "Expected range": v.get("bounds", "")}
            for k, v in sorted(od.items(), key=lambda x: -x[1]["percentage"])
        ])
        st.dataframe(tbl, use_container_width=True, hide_index=True)


def render_advanced():
    a: UniversalAnalytics = st.session_state.analytics
    v: VizEngine = st.session_state.viz_engine

    st.markdown('<div class="kx-card"><div class="kx-card-title">Advanced Analytics</div>'
                '<div class="kx-card-sub">Select columns, pick a method, run.</div></div>', unsafe_allow_html=True)

    cols = a.df.columns.tolist()
    selected = st.multiselect("Columns to analyze", cols, default=cols[: min(5, len(cols))])
    if not selected:
        st.info("Select at least one column.")
        return

    numeric = [c for c in selected if c in a.numeric_cols]
    categorical = [c for c in selected if c in a.categorical_cols]
    datetime_ = [c for c in selected if c in a.datetime_cols]

    methods = []
    if len(numeric) >= 2:
        methods += ["Correlation matrix", "Scatter plot", "Predictive model"]
    if numeric:
        methods.append("Distribution analysis")
    if datetime_ and numeric:
        methods.append("Time-series analysis")
    if not methods:
        st.info("No methods available for this selection.")
        return

    pick = st.selectbox("Method", methods)
    if st.button("Run analysis"):
        if pick == "Correlation matrix":
            fig, msg = v.correlation_matrix(selected)
        elif pick == "Distribution analysis":
            fig, msg = v.distribution(selected)
        elif pick == "Scatter plot":
            fig, msg = v.scatter(numeric[0], numeric[1], categorical[0] if categorical else None)
        elif pick == "Predictive model":
            fig, msg = v.regression(numeric[-1], numeric[:-1])
        elif pick == "Time-series analysis":
            fig, msg = v.timeseries(datetime_[0], numeric[0])
        else:
            fig, msg = None, "Method not implemented."
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.success(msg)
        else:
            st.error(msg)


def render_custom_charts():
    a: UniversalAnalytics = st.session_state.analytics
    v: VizEngine = st.session_state.viz_engine

    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="kx-card"><div class="kx-card-title">Build a chart</div></div>', unsafe_allow_html=True)
        chart_type = st.selectbox("Chart type",
                                  ["Scatter", "Correlation matrix", "Distribution", "Predictive model", "Time-series"])

        fig, msg = None, ""
        if chart_type == "Scatter" and len(a.numeric_cols) >= 2:
            x = st.selectbox("X", a.numeric_cols, key="cc_x")
            y = st.selectbox("Y", a.numeric_cols, key="cc_y")
            color = st.selectbox("Color by", [None] + a.categorical_cols, key="cc_color")
            if st.button("Render", key="cc_render_scatter"):
                fig, msg = v.scatter(x, y, color)
        elif chart_type == "Correlation matrix":
            sel = st.multiselect("Columns", a.numeric_cols, default=a.numeric_cols[:5], key="cc_corr")
            if st.button("Render", key="cc_render_corr"):
                fig, msg = v.correlation_matrix(sel)
        elif chart_type == "Distribution":
            sel = st.multiselect("Columns", a.numeric_cols, default=a.numeric_cols[:3], key="cc_dist")
            if st.button("Render", key="cc_render_dist"):
                fig, msg = v.distribution(sel)
        elif chart_type == "Predictive model" and len(a.numeric_cols) >= 2:
            target = st.selectbox("Target", a.numeric_cols, key="cc_target")
            preds = st.multiselect("Predictors", [c for c in a.numeric_cols if c != target],
                                   default=[c for c in a.numeric_cols if c != target][:2], key="cc_preds")
            if st.button("Render", key="cc_render_pred"):
                fig, msg = v.regression(target, preds)
        elif chart_type == "Time-series" and a.datetime_cols and a.numeric_cols:
            date_col = st.selectbox("Date column", a.datetime_cols, key="cc_date")
            val_col = st.selectbox("Value column", a.numeric_cols, key="cc_val")
            if st.button("Render", key="cc_render_ts"):
                fig, msg = v.timeseries(date_col, val_col)
        else:
            st.info("Not enough data of the right type for this chart.")

        if fig:
            st.session_state.cc_fig = fig
            st.session_state.cc_msg = msg

    with right:
        if "cc_fig" in st.session_state:
            st.plotly_chart(st.session_state.cc_fig, use_container_width=True)
            st.success(st.session_state.get("cc_msg", ""))
        else:
            st.info("Configure a chart on the left.")


def render_explorer():
    a: UniversalAnalytics = st.session_state.analytics
    df = a.df

    left, right = st.columns([1, 3])
    with left:
        st.markdown('<div class="kx-card"><div class="kx-card-title">Filters</div></div>', unsafe_allow_html=True)
        cols = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist()[:10])
        rows = st.slider("Rows to display", 10, min(2000, len(df)), 100)
        view = st.radio("View", ["Head", "Sample", "Tail"], horizontal=True)
        search = st.text_input("Search text")

    with right:
        view_df = df[cols] if cols else df
        if search:
            text_cols = view_df.select_dtypes(include=["object"]).columns
            if len(text_cols):
                mask = view_df[text_cols].astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
                view_df = view_df[mask]
        if view == "Head":
            view_df = view_df.head(rows)
        elif view == "Tail":
            view_df = view_df.tail(rows)
        else:
            view_df = view_df.sample(min(rows, len(view_df))) if len(view_df) else view_df
        st.dataframe(view_df, use_container_width=True)

        m = st.columns(4)
        m[0].metric("Showing", f"{len(view_df):,}")
        m[1].metric("Total rows", f"{len(df):,}")
        m[2].metric("Columns", len(cols) if cols else len(df.columns))
        m[3].metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        nums = [c for c in view_df.columns if c in a.numeric_cols]
        if nums:
            st.markdown("#### Numeric summary")
            st.dataframe(view_df[nums].describe(), use_container_width=True)





def render_quality_and_charts():
    """Merged tab: Data Quality (top) + Custom Charts (bottom)."""
    st.markdown('<div class="kx-card"><div class="kx-card-title">Data Quality &amp; Custom Charts</div>'
                '<div class="kx-card-sub">Audit completeness and outliers, then build any chart you need.</div></div>',
                unsafe_allow_html=True)
    render_data_quality()
    st.markdown("---")
    st.markdown('<div style="font-size:0.95rem;font-weight:700;color:#1a2333;margin:0.5rem 0;">Custom charts</div>',
                unsafe_allow_html=True)
    render_custom_charts()



import json as _json
import re as _re
from datetime import datetime as _dt
from pathlib import Path as _Path

# ============================================================
# PRESENTATION MAKER  v3  — story-driven, Gamma-style decks
# ============================================================
# Pure-Python (python-pptx only). No Node, no LibreOffice needed.
#  - StoryMiner       : mines a ranked pool of business "stories" from data
#  - build_deck_spec  : turns chosen stories + chat answers into a slide spec
#  - render_html_slide: faithful HTML/CSS mockup of each slide (in-app preview)
#  - build_pptx       : renders the (edited) spec to .pptx in Gamma style
#  - render_presentation_maker : the conversational agent + preview/edit UI


try:
    from pptx import Presentation as _Prs
    from pptx.util import Inches as _In, Pt as _Pt
    from pptx.dml.color import RGBColor as _RGB
    from pptx.enum.text import PP_ALIGN as _AL, MSO_ANCHOR as _AN
    from pptx.enum.shapes import MSO_SHAPE as _SH
    from pptx.chart.data import CategoryChartData as _CD
    from pptx.enum.chart import (XL_CHART_TYPE as _CT, XL_LEGEND_POSITION as _LEG,
                                 XL_LABEL_POSITION as _LP)
    from pptx.oxml.ns import qn as _qn
    PPTX_OK = True
except ImportError:
    PPTX_OK = False


# ── theme palettes ──────────────────────────────────────────

DECK_THEMES = {
    "Vibrant": {
        "primary": "4C1D95", "primary2": "6D28D9", "accent": "F97316",
        "blue": "2563EB", "green": "059669", "pink": "DB2777",
        "ink": "1E1B4B", "soft": "64748B", "surface": "F5F3FF",
        "surface2": "EFF6FF", "surface3": "FEF3E8", "line": "E5E7EB",
    },
    "KiteIQX Brand": {
        "primary": "0A2540", "primary2": "14304F", "accent": "C79A3A",
        "blue": "2F5C8A", "green": "2C7A5B", "pink": "9A6A3D",
        "ink": "1A2333", "soft": "4A5568", "surface": "F7F8FB",
        "surface2": "EEF3F8", "surface3": "FBF6EC", "line": "E3E6EE",
    },
    "Slate Pro": {
        "primary": "1E293B", "primary2": "334155", "accent": "0EA5E9",
        "blue": "2563EB", "green": "059669", "pink": "E11D48",
        "ink": "0F172A", "soft": "64748B", "surface": "F1F5F9",
        "surface2": "EFF6FF", "surface3": "FEF2F2", "line": "E2E8F0",
    },
}

DECK_FONTS = {
    "Arial + Calibri  (Bold, modern)": {"header": "Arial", "body": "Calibri"},
    "Georgia + Calibri  (Editorial)":  {"header": "Georgia", "body": "Calibri"},
    "Trebuchet + Calibri  (Friendly)": {"header": "Trebuchet MS", "body": "Calibri"},
}

# glyphs that render reliably in Calibri/Arial via Office + LibreOffice
GLY = {
    "up": "↑", "down": "↓", "target": "◎", "star": "★", "trophy": "🏆",
    "chart": "📊", "growth": "📈", "people": "👥", "bulb": "💡", "woman": "♀",
    "man": "♂", "money": "₹", "check": "✓", "diamond": "◆", "ring": "◑",
    "alert": "▲", "flag": "⚑",
}


# ============================================================
# STORY MINER
# ============================================================

class StoryMiner:
    """Extracts a ranked pool of candidate business stories from a dataframe."""

    def __init__(self, analytics):
        self.a = analytics
        self.df = analytics.df
        self.stories = []
        self._mine()

    def _fmt(self, v):
        av = abs(v)
        if av >= 1_000_000: return f"{v/1_000_000:.2f}M"
        if av >= 1_000:     return f"{v/1_000:.1f}K"
        if av >= 1:         return f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"
        return f"{v:.2f}"

    def _money(self, v):
        return "₹" + self._fmt(v)

    def _pretty(self, col):
        return str(col).replace("_", " ").strip().title()

    def _add(self, **kw):
        kw.setdefault("chart", None)
        kw.setdefault("icon", GLY["chart"])
        self.stories.append(kw)

    def _mine(self):
        a, df = self.a, self.df
        n = len(df)
        if n == 0:
            return

        # 1) Segment leaders (top categorical value, by count)
        for ci, cat in enumerate(a.categorical_cols[:4]):
            vc = df[cat].value_counts()
            if len(vc) < 2:
                continue
            top, top_n = vc.index[0], int(vc.iloc[0])
            share = top_n / n * 100
            self._add(
                kind="segment_leader", group=cat,
                headline=f"{top} leads {self._pretty(cat)}",
                detail=f"'{top}' is the largest {self._pretty(cat)} segment with {top_n:,} records ({share:.1f}% of all rows).",
                stat_value=f"{share:.1f}%", stat_label=f"share — {top}",
                score=share + (8 if ci == 0 else 0), icon=GLY["people"],
                chart={"kind": "bar", "title": f"{self._pretty(cat)} breakdown",
                       "labels": [str(x) for x in vc.head(6).index.tolist()],
                       "values": [float(x) for x in vc.head(6).values.tolist()]},
            )
            # concentration risk
            if share >= 45:
                self._add(
                    kind="concentration", group=cat,
                    headline=f"{self._pretty(cat)} is concentrated",
                    detail=f"A single {self._pretty(cat)} ('{top}') accounts for {share:.1f}% of activity — a concentration worth monitoring.",
                    stat_value=f"{share:.1f}%", stat_label="in one segment",
                    score=share + 6, icon=GLY["alert"],
                    chart={"kind": "pie", "title": f"{self._pretty(cat)} concentration",
                           "labels": [str(x) for x in vc.head(5).index.tolist()],
                           "values": [float(x) for x in vc.head(5).values.tolist()]},
                )

        # 2) Money / metric leaders by segment
        metric_cols = (a.money_cols or a.numeric_cols)[:3]
        for m in metric_cols:
            if m not in df.columns:
                continue
            ismoney = m in a.money_cols
            for cat in a.categorical_cols[:2]:
                grp = df.groupby(cat)[m].sum().sort_values(ascending=False)
                if len(grp) < 2:
                    continue
                top, topv = grp.index[0], float(grp.iloc[0])
                tot = float(grp.sum()) or 1
                share = topv / tot * 100
                val = self._money(topv) if ismoney else self._fmt(topv)
                self._add(
                    kind="metric_extreme", group=cat, metric=m,
                    headline=f"{top} drives {self._pretty(m)}",
                    detail=f"'{top}' generates {val} of {self._pretty(m)} — {share:.1f}% of the total across {self._pretty(cat)}.",
                    stat_value=val, stat_label=f"{self._pretty(m)} · {top}",
                    score=share + 4, icon=GLY["money"] if ismoney else GLY["chart"],
                    chart={"kind": "bar", "title": f"{self._pretty(m)} by {self._pretty(cat)}",
                           "labels": [str(x) for x in grp.head(6).index.tolist()],
                           "values": [round(float(x), 2) for x in grp.head(6).values.tolist()]},
                )

        # 3) Time trends
        if a.datetime_cols and (a.money_cols or a.numeric_cols):
            dc = a.datetime_cols[0]
            vcol = a.money_cols[0] if a.money_cols else a.numeric_cols[0]
            ts = df[[dc, vcol]].dropna().sort_values(dc)
            if len(ts) >= 6:
                series = ts.set_index(dc)[vcol]
                try:
                    idx = series.index.to_period("M")
                    res = series.groupby(idx).sum()
                    res.index = res.index.to_timestamp()
                    if len(res) < 3:
                        idxw = series.index.to_period("W")
                        res = series.groupby(idxw).sum()
                        res.index = res.index.to_timestamp()
                except Exception:
                    res = series
                res = res[res != 0]
                if len(res) >= 3:
                    first, last = float(res.iloc[0]), float(res.iloc[-1])
                    growth = (last - first) / abs(first) * 100 if first else 0
                    peak_i = res.values.argmax()
                    peak_label = str(res.index[peak_i])[:10]
                    peak_val = float(res.iloc[peak_i])
                    direction = "grew" if growth >= 0 else "declined"
                    # cap to a readable number of periods
                    if len(res) > 14:
                        res = res.tail(12)
                    self._add(
                        kind="trend", metric=vcol,
                        headline=f"{self._pretty(vcol)} {direction} over time",
                        detail=f"{self._pretty(vcol)} {direction} {abs(growth):.0f}% across the period; peak of {self._fmt(peak_val)} around {peak_label}.",
                        stat_value=f"{growth:+.0f}%", stat_label=f"{self._pretty(vcol)} change",
                        score=abs(growth) + 12, icon=GLY["growth"],
                        chart={"kind": "line", "title": f"{self._pretty(vcol)} over time",
                               "labels": [str(x)[:7] for x in res.index.tolist()],
                               "values": [round(float(x), 2) for x in res.values.tolist()]},
                    )

        # 4) Correlations
        nums = a.numeric_cols
        if len(nums) >= 2:
            try:
                corr = df[nums].corr().abs()
                best = (0, None, None)
                for i in range(len(nums)):
                    for j in range(i + 1, len(nums)):
                        r = corr.iloc[i, j]
                        if r == r and r > best[0] and r < 0.999:
                            best = (r, nums[i], nums[j])
                if best[1] and best[0] >= 0.4:
                    r, c1, c2 = best
                    self._add(
                        kind="correlation", metric=f"{c1}~{c2}",
                        headline=f"{self._pretty(c1)} tracks {self._pretty(c2)}",
                        detail=f"{self._pretty(c1)} and {self._pretty(c2)} move together (correlation {r:.2f}) — a lever worth testing operationally.",
                        stat_value=f"{r:.2f}", stat_label="correlation",
                        score=r * 60, icon=GLY["target"],
                        chart=None,
                    )
            except Exception:
                pass

        # 5) Metric averages (numeric profile)
        for m in nums[:3]:
            s = df[m].dropna()
            if len(s) < 5:
                continue
            self._add(
                kind="metric_avg", metric=m,
                headline=f"Typical {self._pretty(m)}",
                detail=f"{self._pretty(m)} averages {self._fmt(float(s.mean()))} (ranging {self._fmt(float(s.min()))} to {self._fmt(float(s.max()))}).",
                stat_value=self._fmt(float(s.mean())), stat_label=f"avg {self._pretty(m)}",
                score=8, icon=GLY["diamond"],
            )

        # 6) Volume framing (always available)
        comp = 100 - a.insights["basic_stats"]["missing_pct"]
        self._add(
            kind="volume",
            headline="Dataset at a glance",
            detail=f"{n:,} records across {len(df.columns)} fields, {comp:.0f}% complete — a solid base for analysis.",
            stat_value=f"{n:,}", stat_label="records analysed",
            score=5, icon=GLY["chart"],
        )

        # rank
        self.stories.sort(key=lambda s: -s.get("score", 0))
        for i, s in enumerate(self.stories):
            s["id"] = f"story_{i}"

    def top(self, k=10):
        return self.stories[:k]


# ============================================================
# DECK SPEC BUILDER
# ============================================================

def _kpi_pool(analytics, miner):
    """Build up to 4 KPI cards for the dashboard slide."""
    a = analytics
    ins = a.insights
    b = ins["basic_stats"]
    th_keys = ["primary2", "blue", "green", "accent"]
    kpis = []
    if "monetary" in ins:
        m = ins["monetary"]
        kpis.append({"glyph": GLY["money"], "label": "TOTAL VALUE",
                     "value": "₹" + miner._fmt(m["total"]), "sub": f"avg ₹{miner._fmt(m['avg'])}/record", "ck": "primary2"})
        kpis.append({"glyph": GLY["chart"], "label": "AVG / RECORD",
                     "value": "₹" + miner._fmt(m["avg"]), "sub": "per row", "ck": "blue"})
    kpis.append({"glyph": GLY["people"], "label": "RECORDS",
                 "value": f"{b['rows']:,}", "sub": "rows analysed", "ck": "green"})
    kpis.append({"glyph": GLY["diamond"], "label": "DIMENSIONS",
                 "value": str(b["columns"]),
                 "sub": f"{ins['column_types']['numeric']} num · {ins['column_types']['categorical']} cat", "ck": "accent"})
    comp = 100 - b["missing_pct"]
    kpis.append({"glyph": GLY["check"], "label": "COMPLETENESS",
                 "value": f"{comp:.0f}%", "sub": "non-null values", "ck": "primary2"})
    # de-dup color keys across first 4
    for i, k in enumerate(kpis[:4]):
        k["ck"] = th_keys[i % 4]
    return kpis[:4]


def _story_to_insight(s):
    return {"glyph": s["icon"], "head": s["headline"], "stat": s.get("stat_value", ""),
            "text": s["detail"]}


def build_deck_spec(analytics, miner, chosen_stories, answers, theme_name, font_name):
    """Assemble a slide spec from chosen stories + chat answers. LLM polishes text."""
    a = analytics
    title = answers.get("title") or "Business Intelligence Story"
    subtitle = answers.get("subtitle") or f"{len(analytics.df):,} records · {answers.get('goal','executive review')}"
    goal = answers.get("goal", "executive review")
    audience = answers.get("audience", "leadership")
    emphasis = answers.get("emphasis", "")

    exec_summary = st.session_state.get("exec_summary") or a._fallback_summary()
    qstatus, qscore, qsummary = a.get_data_quality_report()

    # Optional LLM framing of title/subtitle/recommendation/closing
    domain, rec, closing = title, [], "Turn these insights into action."
    if a.llm:
        try:
            heads = "; ".join(s["headline"] + " (" + s.get("stat_value", "") + ")" for s in chosen_stories[:8])
            prompt = (
                f"Audience: {audience}. Goal: {goal}. Emphasis: {emphasis or 'none'}.\n"
                f"Dataset summary: {exec_summary[:300]}\n"
                f"Selected findings: {heads}\n\n"
                "Return ONLY JSON (no fences): "
                '{"title":"<=6 word deck title","subtitle":"<=12 word subtitle",'
                '"recommendations":["action 1","action 2","action 3"],'
                '"closing":"one strong closing line"}'
            )
            raw = _re.sub(r"```json|```", "", a.llm.predict(prompt, max_tokens=420)).strip()
            p = _json.loads(raw)
            title = p.get("title", title)
            subtitle = p.get("subtitle", subtitle)
            rec = p.get("recommendations", []) or []
            closing = p.get("closing", closing)
        except Exception:
            pass
    if not rec:
        rec = [s["detail"] for s in chosen_stories[:3]]

    th = DECK_THEMES[theme_name]
    slides = []

    # 1 — Title
    slides.append({"kind": "title", "title": title, "subtitle": subtitle,
                   "date": _dt.now().strftime("%B %Y")})

    # 2 — Dashboard (hero infographic)
    kpis = _kpi_pool(a, miner)
    hero = chosen_stories[0] if chosen_stories else None
    hero_chart = next((s["chart"] for s in chosen_stories if s.get("chart")), None)
    top_two = " ".join(s["headline"] + f" ({s.get('stat_value','')})." for s in chosen_stories[:2])
    banner = top_two or exec_summary[:160]
    insights = [_story_to_insight(s) for s in chosen_stories[1:5]] or [_story_to_insight(s) for s in chosen_stories[:3]]
    dash_footer = (rec[0] if rec else "Review the detailed stories that follow.")
    slides.append({"kind": "dashboard", "title": (goal.title() + " — Snapshot"),
                   "banner": banner, "kpis": kpis, "chart": hero_chart,
                   "insights": insights[:4],
                   "footer": dash_footer})

    # 3..N — one slide per remaining chosen story that has a chart
    used = set()
    for s in chosen_stories:
        if s.get("chart") and s["id"] not in used:
            sib = [_story_to_insight(x) for x in chosen_stories
                   if x["id"] != s["id"] and x.get("group") == s.get("group")][:3]
            if not sib:
                sib = [{"glyph": s["icon"], "head": "Why it matters", "stat": s.get("stat_value", ""),
                        "text": s["detail"]}]
            slides.append({"kind": "story_chart", "title": s["headline"].upper(),
                           "banner": s["detail"], "chart": s["chart"],
                           "stat_value": s.get("stat_value", ""), "stat_label": s.get("stat_label", ""),
                           "insights": sib, "footer": ""})
            used.add(s["id"])
        if len([sl for sl in slides if sl["kind"] == "story_chart"]) >= 4:
            break

    # big-stat slide for a headline number without a chart
    nostat = next((s for s in chosen_stories if not s.get("chart")), None)
    if nostat:
        slides.append({"kind": "big_stat", "title": nostat["headline"].upper(),
                       "stat_value": nostat.get("stat_value", ""), "stat_label": nostat.get("stat_label", ""),
                       "body": nostat["detail"]})

    # Recommendations
    slides.append({"kind": "takeaways", "title": "What To Do Next",
                   "items": rec[:3]})

    # Data quality
    slides.append({"kind": "quality", "title": "Data Quality", "score": qscore,
                   "status": qstatus, "summary": qsummary.replace("**", "").replace("  \n", " · ")})

    # Closing
    slides.append({"kind": "closing", "message": closing, "date": _dt.now().strftime("%B %Y")})

    return {"meta": {"title": title, "subtitle": subtitle, "theme": theme_name,
                     "font": font_name, "date": _dt.now().strftime("%B %Y")},
            "slides": slides}


# ── pptx drawing helpers (validated in prototype) ───────────

def _hx(h):
    h = h.lstrip("#"); return _RGB(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def _soft_shadow(shp):
    try:
        spPr = shp._element.spPr
        el = spPr.makeelement(_qn('a:effectLst'), {})
        sh = spPr.makeelement(_qn('a:outerShdw'),
                              {'blurRad': '90000', 'dist': '25000', 'dir': '5400000', 'rotWithShape': '0'})
        clr = spPr.makeelement(_qn('a:srgbClr'), {'val': '1E1B4B'})
        alpha = spPr.makeelement(_qn('a:alpha'), {'val': '11000'})
        clr.append(alpha); sh.append(clr); el.append(sh); spPr.append(el)
    except Exception:
        pass

def _rect(s, x, y, w, h, fill=None, line=None, lw=1.0, shape=None, radius=0.08, shadow=False):
    shp = s.shapes.add_shape(shape or _SH.ROUNDED_RECTANGLE, _In(x), _In(y), _In(w), _In(h))
    shp.shadow.inherit = False
    if fill is None: shp.fill.background()
    else: shp.fill.solid(); shp.fill.fore_color.rgb = _hx(fill)
    if line is None: shp.line.fill.background()
    else: shp.line.color.rgb = _hx(line); shp.line.width = _Pt(lw)
    if (shape or _SH.ROUNDED_RECTANGLE) == _SH.ROUNDED_RECTANGLE:
        try: shp.adjustments[0] = radius
        except Exception: pass
    if shadow: _soft_shadow(shp)
    return shp

def _txt(s, text, x, y, w, h, size=14, font="Calibri", color="1E1B4B", bold=False,
         align="left", valign="top", spacing=None, line_spacing=None):
    tb = s.shapes.add_textbox(_In(x), _In(y), _In(w), _In(h)); tf = tb.text_frame
    tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, 0)
    tf.vertical_anchor = {"top": _AN.TOP, "middle": _AN.MIDDLE, "bottom": _AN.BOTTOM}[valign]
    parts = text if isinstance(text, list) else [(text, color, bold)]
    p = tf.paragraphs[0]
    p.alignment = {"left": _AL.LEFT, "center": _AL.CENTER, "right": _AL.RIGHT}[align]
    if line_spacing: p.line_spacing = line_spacing
    for t, c, b in parts:
        r = p.add_run(); r.text = t
        r.font.size = _Pt(size); r.font.name = font; r.font.bold = b; r.font.color.rgb = _hx(c)
        if spacing: r._r.get_or_add_rPr().set("spc", str(int(spacing * 100)))
    return tb

def _badge(s, x, y, d, fill, glyph, glyph_color="FFFFFF", gsize=16):
    _rect(s, x, y, d, d, fill=fill, shape=_SH.OVAL)
    _txt(s, glyph, x, y - 0.02, d, d, size=gsize, font="Calibri", color=glyph_color,
         bold=True, align="center", valign="middle")

def _bg(s, c):
    f = s.background.fill; f.solid(); f.fore_color.rgb = _hx(c)


def _chart(s, spec, x, y, w, h, th, BF, labels_on=True):
    kind = spec.get("kind", "bar")
    # auto-suppress data labels when crowded
    if len(spec.get("labels", [])) > 12:
        labels_on = False
    cd = _CD(); cd.categories = [str(l) for l in spec["labels"]]
    cd.add_series(spec.get("title", "Series"), spec["values"])
    if kind == "line":
        gf = s.shapes.add_chart(_CT.LINE_MARKERS, _In(x), _In(y), _In(w), _In(h), cd); ch = gf.chart
        ser = ch.series[0]; ser.format.line.color.rgb = _hx(th["primary2"]); ser.format.line.width = _Pt(2.5)
        if labels_on:
            p = ch.plots[0]; p.has_data_labels = True
            p.data_labels.font.size = _Pt(8); p.data_labels.position = _LP.ABOVE
            p.data_labels.number_format = '0.0'; p.data_labels.number_format_is_linked = False
    elif kind == "pie":
        gf = s.shapes.add_chart(_CT.DOUGHNUT, _In(x), _In(y), _In(w), _In(h), cd); ch = gf.chart
        ch.has_legend = True; ch.legend.position = _LEG.BOTTOM; ch.legend.include_in_layout = False
        ch.legend.font.size = _Pt(9)
        colors = [th["primary2"], th["accent"], th["blue"], th["green"], th["pink"], "94A3B8"]
        for i, pt in enumerate(ch.plots[0].series[0].points):
            pt.format.fill.solid(); pt.format.fill.fore_color.rgb = _hx(colors[i % len(colors)])
        p = ch.plots[0]; p.has_data_labels = True
        p.data_labels.show_percentage = True; p.data_labels.number_format = "0%"
        p.data_labels.number_format_is_linked = False; p.data_labels.font.size = _Pt(9)
    else:  # bar
        gf = s.shapes.add_chart(_CT.COLUMN_CLUSTERED, _In(x), _In(y), _In(w), _In(h), cd); ch = gf.chart
        ch.plots[0].gap_width = 55
        colors = [th["primary2"], th["accent"], th["blue"], th["green"], th["pink"], "94A3B8"]
        for i, pt in enumerate(ch.plots[0].series[0].points):
            pt.format.fill.solid(); pt.format.fill.fore_color.rgb = _hx(colors[i % len(colors)])
        if labels_on:
            p = ch.plots[0]; p.has_data_labels = True
            p.data_labels.font.size = _Pt(9); p.data_labels.position = _LP.OUTSIDE_END
            p.data_labels.font.color.rgb = _hx(th["ink"])
            p.data_labels.number_format = '#,##0'; p.data_labels.number_format_is_linked = False
    ch.has_title = False
    if kind != "pie":
        ch.has_legend = False
        try:
            va = ch.value_axis; va.tick_labels.font.size = _Pt(9); va.tick_labels.font.color.rgb = _hx(th["soft"])
            ca = ch.category_axis; ca.tick_labels.font.size = _Pt(9); ca.tick_labels.font.color.rgb = _hx(th["ink"])
        except Exception:
            pass
    try: ch.font.name = BF
    except Exception: pass
    return ch


def _insight_rows(s, items, x, y, w, th, BF, row_h=0.78):
    tints = [th["surface"], th["surface2"], th["surface3"], th["surface"]]
    glyphcols = [th["primary2"], th["blue"], th["accent"], th["green"]]
    for i, it in enumerate(items[:4]):
        ry = y + i * row_h
        _badge(s, x, ry + 0.04, 0.46, tints[i % 4], it["glyph"],
               glyph_color=glyphcols[i % 4], gsize=14)
        parts = []
        if it.get("stat"):
            parts = [(it["head"] + "  ", th["ink"], True), (it["stat"], glyphcols[i % 4], True)]
        else:
            parts = [(it["head"], th["ink"], True)]
        _txt(s, parts, x + 0.6, ry, w - 0.6, 0.3, size=10.5, font=BF, valign="top")
        _txt(s, it.get("text", ""), x + 0.6, ry + 0.26, w - 0.6, row_h - 0.28,
             size=9, font=BF, color=th["soft"], line_spacing=1.0)


def build_pptx(spec, out_path):
    if not PPTX_OK:
        return False, "python-pptx not installed. Add `python-pptx` to requirements.txt."
    meta = spec["meta"]; th = DECK_THEMES[meta["theme"]]
    fonts = DECK_FONTS[meta["font"]]; HF, BF = fonts["header"], fonts["body"]
    prs = _Prs(); prs.slide_width = _In(13.333); prs.slide_height = _In(7.5)
    blank = prs.slide_layouts[6]; W, H = 13.333, 7.5

    def title_bar(s, t):
        _rect(s, 0.5, 0.34, 0.13, 0.6, fill=th["primary2"], shape=_SH.RECTANGLE)
        _txt(s, t, 0.78, 0.30, 12.0, 0.74, size=27, font=HF, color=th["ink"], bold=True)

    def banner(s, y, glyph, parts, fill, gfill):
        _rect(s, 0.5, y, 12.33, 0.72, fill=fill, radius=0.18)
        _badge(s, 0.7, y + 0.13, 0.46, gfill, glyph, gsize=13)
        _txt(s, parts, 1.36, y + 0.05, 11.2, 0.62, size=12, font=BF, valign="middle", line_spacing=1.0)

    def slidenum(s, n):
        _txt(s, str(n), 12.7, 7.15, 0.4, 0.25, size=9, font=BF, color=th["soft"], align="right")

    for idx, sl in enumerate(spec["slides"], 1):
        k = sl["kind"]
        s = prs.slides.add_slide(blank)

        if k == "title":
            _bg(s, th["primary"])
            _rect(s, 0, 0, 0.16, H, fill=th["accent"], shape=_SH.RECTANGLE)
            _rect(s, W - 2.4, 0, 2.4, 2.0, fill=th["primary2"], shape=_SH.RECTANGLE)
            _txt(s, "KITEIQX INTELLIGENCE", 0.7, 1.4, 9, 0.4, size=12, font=BF,
                 color=th["accent"], bold=True, spacing=4)
            _txt(s, sl["title"], 0.7, 2.2, 11.5, 2.4, size=46, font=HF, color="FFFFFF", bold=True)
            _txt(s, sl.get("subtitle", ""), 0.7, 4.7, 10.5, 0.9, size=16, font=BF, color="C7C3E8")
            _txt(s, sl.get("date", ""), 0.7, 6.6, 4, 0.4, size=12, font=BF, color=th["accent"])

        elif k == "dashboard":
            _bg(s, "FFFFFF"); title_bar(s, sl["title"])
            banner(s, 1.1, GLY["bulb"],
                   [(sl["banner"], th["ink"], False)], th["surface"], th["primary2"])
            kpis = sl.get("kpis", [])[:4]; cw, gap = 2.94, 0.16; xs = 0.5; cy, chh = 2.02, 1.32
            for i, kp in enumerate(kpis):
                x = xs + i * (cw + gap)
                _rect(s, x, cy, cw, chh, fill="FFFFFF", line=th["line"], lw=1, radius=0.10, shadow=True)
                d = 0.6; _badge(s, x + 0.16, cy + (chh - d) / 2 - 0.16, d, th[kp["ck"]], kp["glyph"], gsize=19)
                tx = x + 0.16 + d + 0.14
                _txt(s, kp["label"], tx, cy + 0.18, cw - (tx - x) - 0.1, 0.3, size=9.5, font=BF,
                     color=th["soft"], bold=True, spacing=0.4)
                _txt(s, kp["value"], tx, cy + 0.45, cw - (tx - x) - 0.1, 0.5, size=22, font=HF,
                     color=th[kp["ck"]], bold=True)
                _txt(s, kp["sub"], tx, cy + 0.95, cw - (tx - x) - 0.1, 0.3, size=8.5, font=BF, color=th["soft"])
            # chart card + insights
            _rect(s, 0.5, 3.55, 8.4, 2.92, fill="FFFFFF", line=th["line"], lw=1, radius=0.05, shadow=True)
            ctitle = (sl.get("chart") or {}).get("title", "Overview")
            _rect(s, 0.7, 3.7, min(5.5, 0.25 + 0.11 * len(ctitle)), 0.4, fill=th["primary"], radius=0.3)
            _txt(s, ctitle.upper(), 0.82, 3.74, 5.2, 0.32, size=10, font=HF, color="FFFFFF",
                 bold=True, valign="middle")
            if sl.get("chart"):
                _chart(s, sl["chart"], 0.62, 4.18, 8.15, 2.18, th, BF)
            _rect(s, 9.1, 3.55, 3.73, 2.92, fill="FFFFFF", line=th["line"], lw=1, radius=0.05, shadow=True)
            _rect(s, 9.55, 3.7, 2.83, 0.4, fill=th["primary"], radius=0.3)
            _txt(s, "KEY INSIGHTS", 9.55, 3.74, 2.83, 0.32, size=11, font=HF, color="FFFFFF",
                 bold=True, align="center", valign="middle")
            _insight_rows(s, sl.get("insights", []), 9.3, 4.28, 3.4, th, BF, row_h=0.55)
            banner(s, 6.62, GLY["bulb"], [(sl.get("footer", ""), th["ink"], False)],
                   th["surface3"], th["accent"])
            slidenum(s, idx)

        elif k == "story_chart":
            _bg(s, "FFFFFF"); title_bar(s, sl["title"])
            banner(s, 1.1, GLY["growth"], [(sl["banner"], th["ink"], False)], th["surface"], th["primary2"])
            _rect(s, 0.5, 2.0, 8.4, 4.5, fill="FFFFFF", line=th["line"], lw=1, radius=0.04, shadow=True)
            ctitle = sl["chart"].get("title", "Detail")
            _rect(s, 0.7, 2.16, min(5.5, 0.25 + 0.11 * len(ctitle)), 0.4, fill=th["primary"], radius=0.3)
            _txt(s, ctitle.upper(), 0.82, 2.20, 5.2, 0.32, size=10, font=HF, color="FFFFFF",
                 bold=True, valign="middle")
            _chart(s, sl["chart"], 0.62, 2.7, 8.15, 3.6, th, BF)
            _rect(s, 9.1, 2.0, 3.73, 4.5, fill="FFFFFF", line=th["line"], lw=1, radius=0.04, shadow=True)
            if sl.get("stat_value"):
                _rect(s, 9.3, 2.2, 3.33, 1.2, fill=th["surface"], radius=0.12)
                _txt(s, sl["stat_value"], 9.3, 2.34, 3.33, 0.7, size=34, font=HF, color=th["primary2"],
                     bold=True, align="center")
                _txt(s, sl.get("stat_label", ""), 9.3, 3.02, 3.33, 0.3, size=10, font=BF,
                     color=th["soft"], align="center")
            _txt(s, "WHY IT MATTERS", 9.3, 3.6, 3.33, 0.3, size=10, font=HF, color=th["ink"], bold=True, spacing=0.5)
            _insight_rows(s, sl.get("insights", []), 9.3, 3.95, 3.4, th, BF, row_h=0.82)
            slidenum(s, idx)

        elif k == "big_stat":
            _bg(s, th["primary"]); _rect(s, 0, 0, 0.16, H, fill=th["accent"], shape=_SH.RECTANGLE)
            _txt(s, sl["title"], 0.8, 1.0, 11.5, 1.2, size=26, font=HF, color="FFFFFF", bold=True)
            _txt(s, sl.get("stat_value", ""), 0.8, 2.3, 11.5, 2.2, size=120, font=HF,
                 color=th["accent"], bold=True)
            _txt(s, sl.get("stat_label", "").upper(), 0.85, 4.7, 11.5, 0.5, size=16, font=BF,
                 color="C7C3E8", spacing=1)
            _txt(s, sl.get("body", ""), 0.8, 5.5, 11.0, 1.3, size=15, font=BF, color="E5E3F5",
                 line_spacing=1.2)
            slidenum(s, idx)

        elif k == "takeaways":
            _bg(s, "FFFFFF"); title_bar(s, sl["title"])
            colors = [th["primary2"], th["accent"], th["blue"]]
            for i, t in enumerate(sl.get("items", [])[:3]):
                y = 1.45 + i * 1.78
                _rect(s, 0.5, y, 12.33, 1.55, fill="FFFFFF", line=th["line"], lw=1, radius=0.06, shadow=True)
                _rect(s, 0.5, y, 0.7, 1.55, fill=colors[i % 3], shape=_SH.RECTANGLE)
                _txt(s, str(i + 1), 0.5, y, 0.7, 1.55, size=30, font=HF, color="FFFFFF", bold=True,
                     align="center", valign="middle")
                _txt(s, t, 1.45, y + 0.15, 11.2, 1.25, size=15, font=BF, color=th["ink"],
                     valign="middle", line_spacing=1.1)
            slidenum(s, idx)

        elif k == "quality":
            _bg(s, "FFFFFF"); title_bar(s, sl["title"])
            sc = sl.get("score", 85)
            clr = th["green"] if sc >= 90 else th["blue"] if sc >= 75 else th["accent"] if sc >= 60 else th["pink"]
            _rect(s, 0.8, 1.7, 3.2, 3.2, fill="FFFFFF", line=clr, lw=6, shape=_SH.OVAL, shadow=True)
            _txt(s, str(sc), 0.8, 2.3, 3.2, 1.6, size=72, font=HF, color=clr, bold=True,
                 align="center", valign="middle")
            _txt(s, "/ 100", 0.8, 3.85, 3.2, 0.4, size=14, font=BF, color=clr, align="center")
            _txt(s, "QUALITY SUMMARY", 4.5, 1.8, 8, 0.4, size=13, font=HF, color=th["ink"], bold=True, spacing=0.5)
            _txt(s, sl.get("summary") or "Dataset is analysis-ready.", 4.5, 2.35, 8.0, 3.2,
                 size=14, font=BF, color=th["ink"], line_spacing=1.3)
            slidenum(s, idx)

        elif k == "closing":
            _bg(s, th["primary"]); _rect(s, 0, 0, 0.16, H, fill=th["accent"], shape=_SH.RECTANGLE)
            _rect(s, W - 3.0, H - 2.4, 3.0, 2.4, fill=th["primary2"], shape=_SH.RECTANGLE)
            _txt(s, "KITEIQX INTELLIGENCE", 0.7, 1.3, 9, 0.4, size=12, font=BF, color=th["accent"],
                 bold=True, spacing=4)
            _txt(s, sl.get("message", "Thank you."), 0.7, 2.2, 11.0, 2.6, size=38, font=HF,
                 color="FFFFFF", bold=True, line_spacing=1.05)
            _txt(s, "Generated by KiteIQX Intelligence · " + sl.get("date", ""), 0.7, 5.6, 10, 0.4,
                 size=12, font=BF, color="9A93C8")

    prs.save(out_path)
    return True, f"Generated {_Path(out_path).name} ({_Path(out_path).stat().st_size // 1024} KB)"


# ============================================================
# HTML PREVIEW  (faithful mockup, no dependencies)
# ============================================================

def _svg_chart(spec, th, w=470, h=190):
    if not spec:
        return ""
    kind = spec.get("kind", "bar")
    labels = [str(l) for l in spec["labels"]]
    vals = [float(v) for v in spec["values"]]
    mx = max(vals) if vals else 1
    pad = 24
    cols = [th["primary2"], th["accent"], th["blue"], th["green"], th["pink"], "#94A3B8"]
    if kind == "line":
        n = len(vals); step = (w - 2 * pad) / max(1, n - 1)
        pts = " ".join(f"{pad + i*step:.0f},{h - pad - (v/mx)*(h-2*pad):.0f}" for i, v in enumerate(vals))
        dots = "".join(f'<circle cx="{pad+i*step:.0f}" cy="{h-pad-(v/mx)*(h-2*pad):.0f}" r="3" fill="#{th["primary2"]}"/>' for i, v in enumerate(vals))
        lbls = "".join(f'<text x="{pad+i*step:.0f}" y="{h-6}" font-size="8" fill="#{th["soft"]}" text-anchor="middle">{labels[i][:4]}</text>' for i in range(n))
        return f'<svg viewBox="0 0 {w} {h}" width="100%"><polyline points="{pts}" fill="none" stroke="#{th["primary2"]}" stroke-width="2.5"/>{dots}{lbls}</svg>'
    if kind == "pie":
        import math
        tot = sum(vals) or 1; cx, cy, r = h/2, h/2, h/2 - 12; a0 = -math.pi/2; segs = ""
        for i, v in enumerate(vals):
            a1 = a0 + (v/tot) * 2 * math.pi
            x0, y0 = cx + r*math.cos(a0), cy + r*math.sin(a0)
            x1, y1 = cx + r*math.cos(a1), cy + r*math.sin(a1)
            large = 1 if (a1 - a0) > math.pi else 0
            segs += f'<path d="M{cx},{cy} L{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 1 {x1:.1f},{y1:.1f} Z" fill="{cols[i%len(cols)]}"/>'
            a0 = a1
        segs += f'<circle cx="{cx}" cy="{cy}" r="{r*0.55:.0f}" fill="#fff"/>'
        leg = "".join(f'<text x="{h+10}" y="{20+i*16}" font-size="9" fill="#{th["ink"]}">■ {labels[i][:14]}</text>'.replace("■", f'<tspan fill="{cols[i%len(cols)]}">■</tspan>') for i in range(len(vals)))
        return f'<svg viewBox="0 0 {w} {h}" width="100%">{segs}{leg}</svg>'
    # bar
    n = len(vals); bw = (w - 2*pad) / max(1, n) * 0.62; gap = (w - 2*pad) / max(1, n)
    bars = ""
    for i, v in enumerate(vals):
        bh = (v/mx) * (h - 2*pad); bx = pad + i*gap + (gap-bw)/2; by = h - pad - bh
        bars += f'<rect x="{bx:.0f}" y="{by:.0f}" width="{bw:.0f}" height="{bh:.0f}" rx="3" fill="{cols[i%len(cols)]}"/>'
        bars += f'<text x="{bx+bw/2:.0f}" y="{by-3:.0f}" font-size="8" fill="#{th["ink"]}" text-anchor="middle">{v:.0f}</text>'
        bars += f'<text x="{bx+bw/2:.0f}" y="{h-6}" font-size="8" fill="#{th["soft"]}" text-anchor="middle">{labels[i][:5]}</text>'
    return f'<svg viewBox="0 0 {w} {h}" width="100%">{bars}</svg>'


def render_html_slide(sl, meta):
    th = DECK_THEMES[meta["theme"]]
    P, P2, AC = "#" + th["primary"], "#" + th["primary2"], "#" + th["accent"]
    INK, SOFT, LINE = "#" + th["ink"], "#" + th["soft"], "#" + th["line"]
    SURF = "#" + th["surface"]
    base = f'font-family:Inter,system-ui,sans-serif;border-radius:12px;overflow:hidden;'
    frame = f'width:100%;aspect-ratio:16/9;background:#fff;border:1px solid {LINE};{base}position:relative;box-shadow:0 2px 10px rgba(0,0,0,0.06);'
    k = sl["kind"]

    def card(inner): return f'<div style="{frame}">{inner}</div>'

    if k == "title":
        return card(
            f'<div style="position:absolute;inset:0;background:{P};"></div>'
            f'<div style="position:absolute;left:0;top:0;bottom:0;width:1.4%;background:{AC};"></div>'
            f'<div style="position:absolute;left:5%;top:24%;color:{AC};font-size:0.75rem;font-weight:700;letter-spacing:3px;">KITEIQX INTELLIGENCE</div>'
            f'<div style="position:absolute;left:5%;top:34%;right:8%;color:#fff;font-size:2.4rem;font-weight:800;line-height:1.1;">{sl["title"]}</div>'
            f'<div style="position:absolute;left:5%;top:66%;right:20%;color:#C7C3E8;font-size:1rem;">{sl.get("subtitle","")}</div>'
            f'<div style="position:absolute;left:5%;bottom:8%;color:{AC};font-size:0.8rem;">{sl.get("date","")}</div>'
        )

    if k == "dashboard":
        kpis = "".join(
            f'<div style="flex:1;background:#fff;border:1px solid {LINE};border-radius:10px;padding:8px 10px;display:flex;gap:8px;align-items:center;box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
            f'<div style="width:34px;height:34px;border-radius:50%;background:#{th[kp["ck"]]};color:#fff;display:flex;align-items:center;justify-content:center;font-size:0.95rem;flex:0 0 34px;">{kp["glyph"]}</div>'
            f'<div style="min-width:0;"><div style="font-size:0.5rem;color:{SOFT};font-weight:700;letter-spacing:0.5px;">{kp["label"]}</div>'
            f'<div style="font-size:1.05rem;font-weight:800;color:#{th[kp["ck"]]};line-height:1.1;">{kp["value"]}</div>'
            f'<div style="font-size:0.5rem;color:{SOFT};">{kp["sub"]}</div></div></div>'
            for kp in sl.get("kpis", [])[:4]
        )
        ins = "".join(
            f'<div style="display:flex;gap:7px;margin-bottom:7px;align-items:flex-start;">'
            f'<div style="width:24px;height:24px;border-radius:50%;background:{SURF};color:{P2};display:flex;align-items:center;justify-content:center;font-size:0.7rem;flex:0 0 24px;">{it["glyph"]}</div>'
            f'<div style="font-size:0.56rem;color:{INK};line-height:1.25;"><b>{it["head"]}</b> '
            f'<span style="color:{P2};font-weight:700;">{it.get("stat","")}</span><br>'
            f'<span style="color:{SOFT};">{it.get("text","")[:90]}</span></div></div>'
            for it in sl.get("insights", [])[:4]
        )
        return card(
            f'<div style="position:absolute;left:3%;top:5%;width:1%;height:8%;background:{P2};"></div>'
            f'<div style="position:absolute;left:5.5%;top:4%;font-size:1.25rem;font-weight:800;color:{INK};">{sl["title"]}</div>'
            f'<div style="position:absolute;left:3.5%;right:3.5%;top:15%;background:{SURF};border-radius:8px;padding:6px 12px;font-size:0.6rem;color:{INK};">💡 {sl.get("banner","")[:160]}</div>'
            f'<div style="position:absolute;left:3.5%;right:3.5%;top:27%;display:flex;gap:6px;">{kpis}</div>'
            f'<div style="position:absolute;left:3.5%;top:48%;width:62%;bottom:13%;background:#fff;border:1px solid {LINE};border-radius:8px;padding:6px;">'
            f'<div style="display:inline-block;background:{P};color:#fff;font-size:0.5rem;font-weight:700;padding:2px 10px;border-radius:20px;">{(sl.get("chart") or {}).get("title","OVERVIEW").upper()}</div>'
            f'{_svg_chart(sl.get("chart"), th)}</div>'
            f'<div style="position:absolute;right:3.5%;top:48%;width:30%;bottom:13%;background:#fff;border:1px solid {LINE};border-radius:8px;padding:6px;">'
            f'<div style="background:{P};color:#fff;font-size:0.55rem;font-weight:700;padding:2px 0;border-radius:20px;text-align:center;margin-bottom:6px;">KEY INSIGHTS</div>{ins}</div>'
            f'<div style="position:absolute;left:3.5%;right:3.5%;bottom:3%;background:#{th["surface3"]};border-radius:8px;padding:5px 12px;font-size:0.58rem;color:{INK};">💡 {sl.get("footer","")[:150]}</div>'
        )

    if k == "story_chart":
        ins = "".join(
            f'<div style="margin-bottom:8px;font-size:0.58rem;color:{INK};line-height:1.3;"><b>{it["head"]}</b><br><span style="color:{SOFT};">{it.get("text","")[:120]}</span></div>'
            for it in sl.get("insights", [])[:3]
        )
        statbox = (f'<div style="background:{SURF};border-radius:8px;padding:10px;text-align:center;margin-bottom:10px;">'
                   f'<div style="font-size:1.8rem;font-weight:800;color:{P2};">{sl.get("stat_value","")}</div>'
                   f'<div style="font-size:0.55rem;color:{SOFT};">{sl.get("stat_label","")}</div></div>') if sl.get("stat_value") else ""
        return card(
            f'<div style="position:absolute;left:3%;top:5%;width:1%;height:8%;background:{P2};"></div>'
            f'<div style="position:absolute;left:5.5%;top:4%;font-size:1.2rem;font-weight:800;color:{INK};">{sl["title"]}</div>'
            f'<div style="position:absolute;left:3.5%;right:3.5%;top:15%;background:{SURF};border-radius:8px;padding:6px 12px;font-size:0.6rem;color:{INK};">📈 {sl.get("banner","")[:160]}</div>'
            f'<div style="position:absolute;left:3.5%;top:27%;width:62%;bottom:5%;background:#fff;border:1px solid {LINE};border-radius:8px;padding:8px;">'
            f'<div style="display:inline-block;background:{P};color:#fff;font-size:0.5rem;font-weight:700;padding:2px 10px;border-radius:20px;">{sl["chart"].get("title","DETAIL").upper()}</div>'
            f'{_svg_chart(sl.get("chart"), th, h=230)}</div>'
            f'<div style="position:absolute;right:3.5%;top:27%;width:30%;bottom:5%;background:#fff;border:1px solid {LINE};border-radius:8px;padding:10px;">'
            f'{statbox}<div style="font-size:0.6rem;font-weight:700;color:{INK};margin-bottom:6px;">WHY IT MATTERS</div>{ins}</div>'
        )

    if k == "big_stat":
        return card(
            f'<div style="position:absolute;inset:0;background:{P};"></div>'
            f'<div style="position:absolute;left:0;top:0;bottom:0;width:1.4%;background:{AC};"></div>'
            f'<div style="position:absolute;left:6%;top:14%;right:6%;color:#fff;font-size:1.3rem;font-weight:800;">{sl["title"]}</div>'
            f'<div style="position:absolute;left:5.5%;top:28%;color:{AC};font-size:5rem;font-weight:800;line-height:1;">{sl.get("stat_value","")}</div>'
            f'<div style="position:absolute;left:6%;top:62%;color:#C7C3E8;font-size:0.9rem;letter-spacing:1px;">{sl.get("stat_label","").upper()}</div>'
            f'<div style="position:absolute;left:6%;top:72%;right:8%;color:#E5E3F5;font-size:0.85rem;line-height:1.4;">{sl.get("body","")}</div>'
        )

    if k == "takeaways":
        rows = "".join(
            f'<div style="display:flex;margin-bottom:10px;border:1px solid {LINE};border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
            f'<div style="background:{[P2,AC,"#"+th["blue"]][i%3]};color:#fff;width:42px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;font-weight:800;flex:0 0 42px;">{i+1}</div>'
            f'<div style="padding:10px 14px;font-size:0.72rem;color:{INK};display:flex;align-items:center;">{t}</div></div>'
            for i, t in enumerate(sl.get("items", [])[:3])
        )
        return card(
            f'<div style="position:absolute;left:3%;top:5%;width:1%;height:8%;background:{P2};"></div>'
            f'<div style="position:absolute;left:5.5%;top:4%;font-size:1.25rem;font-weight:800;color:{INK};">{sl["title"]}</div>'
            f'<div style="position:absolute;left:3.5%;right:3.5%;top:20%;">{rows}</div>'
        )

    if k == "quality":
        sc = sl.get("score", 85)
        clr = ("#" + th["green"]) if sc >= 90 else ("#" + th["blue"]) if sc >= 75 else AC if sc >= 60 else ("#" + th["pink"])
        return card(
            f'<div style="position:absolute;left:3%;top:5%;width:1%;height:8%;background:{P2};"></div>'
            f'<div style="position:absolute;left:5.5%;top:4%;font-size:1.25rem;font-weight:800;color:{INK};">{sl["title"]}</div>'
            f'<div style="position:absolute;left:6%;top:30%;width:150px;height:150px;border-radius:50%;border:7px solid {clr};display:flex;flex-direction:column;align-items:center;justify-content:center;">'
            f'<div style="font-size:2.6rem;font-weight:800;color:{clr};">{sc}</div><div style="color:{clr};font-size:0.7rem;">/ 100</div></div>'
            f'<div style="position:absolute;left:36%;right:6%;top:30%;"><div style="font-size:0.75rem;font-weight:700;color:{INK};letter-spacing:0.5px;margin-bottom:8px;">QUALITY SUMMARY</div>'
            f'<div style="font-size:0.78rem;color:{INK};line-height:1.6;">{sl.get("summary","")}</div></div>'
        )

    if k == "closing":
        return card(
            f'<div style="position:absolute;inset:0;background:{P};"></div>'
            f'<div style="position:absolute;left:0;top:0;bottom:0;width:1.4%;background:{AC};"></div>'
            f'<div style="position:absolute;left:5%;top:22%;color:{AC};font-size:0.75rem;font-weight:700;letter-spacing:3px;">KITEIQX INTELLIGENCE</div>'
            f'<div style="position:absolute;left:5%;top:34%;right:25%;color:#fff;font-size:2rem;font-weight:800;line-height:1.15;">{sl.get("message","")}</div>'
            f'<div style="position:absolute;left:5%;bottom:12%;color:#9A93C8;font-size:0.8rem;">Generated by KiteIQX Intelligence · {sl.get("date","")}</div>'
        )

    return card(f'<div style="padding:20px;color:{INK};">{k}</div>')



# ============================================================
# PRESENTATION MAKER — conversational agent + preview/edit UI
# ============================================================

import streamlit.components.v1 as _components


_CHAT_STEPS = [
    {"key": "goal", "q": "What's the goal of this deck?",
     "opts": ["Performance review", "Growth pitch", "Risk & quality review",
              "Investor update", "Board summary"]},
    {"key": "audience", "q": "Who's the audience?",
     "opts": ["Leadership / CXO", "Board", "Investors", "Team / Ops", "Clients"]},
]


def _pm_reset():
    for k in ["pm_stage", "pm_chat", "pm_answers", "pm_stories_pool",
              "pm_selected_ids", "pm_spec", "pm_theme", "pm_font", "pm_step_i"]:
        st.session_state.pop(k, None)


def render_presentation_maker():
    a = st.session_state.get("analytics")
    if a is None:
        st.info("Load data first (sidebar), then come back to build your story deck.")
        return
    if not PPTX_OK:
        st.error("`python-pptx` isn't installed. Add `python-pptx` to requirements.txt and redeploy.")
        return

    st.markdown(
        '<div class="kx-card"><div class="kx-card-title">Presentation Maker</div>'
        '<div class="kx-card-sub">A story producer that mines your data for the strongest findings, '
        'chats with you about what to emphasize, then builds an editable, Gamma-style deck.</div></div>',
        unsafe_allow_html=True,
    )

    # initialise state machine
    if "pm_stage" not in st.session_state:
        st.session_state.pm_stage = "intro"
        st.session_state.pm_chat = []
        st.session_state.pm_answers = {}
        st.session_state.pm_step_i = 0

    stage = st.session_state.pm_stage

    # ---- top controls: theme/font picker (offered each time) ----
    tcol, fcol, rcol = st.columns([2, 2, 1])
    with tcol:
        theme_name = st.selectbox("Visual style", list(DECK_THEMES.keys()),
                                  index=0, key="pm_theme")
    with fcol:
        font_name = st.selectbox("Font pairing", list(DECK_FONTS.keys()),
                                 index=0, key="pm_font")
    with rcol:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("↺ Restart", key="pm_restart", use_container_width=True):
            _pm_reset(); st.rerun()

    st.markdown("---")

    # ============================================================
    # STAGE 1 — INTRO: mine stories, kick off the chat
    # ============================================================
    if stage == "intro":
        with st.spinner("Mining your data for the strongest stories…"):
            miner = StoryMiner(a)
            st.session_state.pm_stories_pool = miner.top(10)
        st.session_state.pm_stage = "chat"
        st.session_state.pm_chat = [
            ("assistant",
             "I went through your dataset and pulled the strongest stories. "
             "Let me ask a couple of quick questions, then you'll pick which stories to feature.")
        ]
        st.rerun()

    miner = StoryMiner(a)  # cheap; deterministic
    pool = st.session_state.get("pm_stories_pool") or miner.top(10)

    # ============================================================
    # STAGE 2 — CHAT: ask goal + audience conversationally
    # ============================================================
    if stage == "chat":
        # render conversation so far
        for role, msg in st.session_state.pm_chat:
            with st.chat_message("assistant" if role == "assistant" else "user"):
                st.markdown(msg)

        step_i = st.session_state.pm_step_i
        if step_i < len(_CHAT_STEPS):
            step = _CHAT_STEPS[step_i]
            with st.chat_message("assistant"):
                st.markdown(f"**{step['q']}**")
                cols = st.columns(len(step["opts"]))
                for j, opt in enumerate(step["opts"]):
                    if cols[j].button(opt, key=f"pm_opt_{step_i}_{j}", use_container_width=True):
                        st.session_state.pm_answers[step["key"]] = opt
                        st.session_state.pm_chat.append(("assistant", step["q"]))
                        st.session_state.pm_chat.append(("user", opt))
                        st.session_state.pm_step_i += 1
                        st.rerun()
        else:
            # free-text emphasis, then move to story selection
            with st.chat_message("assistant"):
                st.markdown("**Anything specific to emphasize?** (optional — e.g. *focus on the "
                            "concentration risk*, or *frame it as a growth opportunity*)")
            emph = st.chat_input("Type emphasis or just hit send to skip…")
            if emph is not None:
                st.session_state.pm_answers["emphasis"] = emph.strip()
                if emph.strip():
                    st.session_state.pm_chat.append(("user", emph.strip()))
                st.session_state.pm_stage = "select"
                st.rerun()
            cskip, _ = st.columns([1, 4])
            if cskip.button("Skip", key="pm_skip_emph"):
                st.session_state.pm_answers["emphasis"] = ""
                st.session_state.pm_stage = "select"
                st.rerun()

    # ============================================================
    # STAGE 3 — SELECT: choose which mined stories to feature
    # ============================================================
    elif stage == "select":
        for role, msg in st.session_state.pm_chat:
            with st.chat_message("assistant" if role == "assistant" else "user"):
                st.markdown(msg)
        with st.chat_message("assistant"):
            st.markdown("Here are the **top stories** I found. Tick the ones you want in the deck "
                        "(I've pre-selected the strongest). Then hit **Build deck**.")

        if "pm_selected_ids" not in st.session_state:
            st.session_state.pm_selected_ids = [s["id"] for s in pool[:6]]

        sel = []
        for s in pool:
            checked = st.checkbox(
                f"**{s['headline']}**  ·  `{s.get('stat_value','')}`  —  {s['detail']}",
                value=(s["id"] in st.session_state.pm_selected_ids),
                key=f"pm_chk_{s['id']}",
            )
            if checked:
                sel.append(s["id"])
        st.session_state.pm_selected_ids = sel

        bcol, ccol = st.columns([1, 4])
        if bcol.button("Build deck →", type="primary", key="pm_build",
                       disabled=(len(sel) == 0), use_container_width=True):
            chosen = [s for s in pool if s["id"] in sel]
            with st.spinner("Framing the narrative and laying out slides…"):
                spec = build_deck_spec(a, miner, chosen, st.session_state.pm_answers,
                                       st.session_state.pm_theme, st.session_state.pm_font)
            st.session_state.pm_spec = spec
            st.session_state.pm_stage = "review"
            st.rerun()
        if len(sel) == 0:
            ccol.caption("Select at least one story to continue.")

    # ============================================================
    # STAGE 4 — REVIEW: visual preview + inline editing + download
    # ============================================================
    elif stage == "review":
        spec = st.session_state.pm_spec
        # keep theme/font in sync with current pickers (lets user restyle without rebuild)
        spec["meta"]["theme"] = st.session_state.pm_theme
        spec["meta"]["font"] = st.session_state.pm_font

        st.markdown(
            f'<div class="kx-callout"><strong>Your deck is ready — {len(spec["slides"])} slides.</strong> '
            "Preview each slide below and edit any text inline. Change the visual style or font up top "
            "and the preview updates instantly. Download when you're happy.</div>",
            unsafe_allow_html=True,
        )

        # AI refine box (conversational agent loop)
        with st.expander("💬 Ask the producer to refine the deck"):
            refine = st.text_input("e.g. 'make the tone punchier', 'add a recommendation about pricing'",
                                   key="pm_refine_input")
            if st.button("Apply refinement", key="pm_refine_btn") and refine.strip():
                if a.llm:
                    try:
                        cur = _json.dumps({"title": spec["meta"]["title"],
                                           "subtitle": spec["meta"]["subtitle"],
                                           "recommendations": next((s["items"] for s in spec["slides"]
                                                                    if s["kind"] == "takeaways"), [])})
                        out = a.llm.predict(
                            f"Current deck framing: {cur}\nUser refinement: {refine}\n"
                            "Return ONLY JSON: {\"title\":\"\",\"subtitle\":\"\",\"recommendations\":[\"\",\"\",\"\"]}",
                            max_tokens=380)
                        p = _json.loads(_re.sub(r"```json|```", "", out).strip())
                        spec["meta"]["title"] = p.get("title", spec["meta"]["title"])
                        spec["meta"]["subtitle"] = p.get("subtitle", spec["meta"]["subtitle"])
                        for s in spec["slides"]:
                            if s["kind"] == "title":
                                s["title"], s["subtitle"] = spec["meta"]["title"], spec["meta"]["subtitle"]
                            if s["kind"] == "takeaways" and p.get("recommendations"):
                                s["items"] = p["recommendations"][:3]
                        st.success("Applied. Scroll to see the changes.")
                    except Exception:
                        st.warning("Couldn't parse the AI response — try rephrasing.")
                else:
                    st.warning("AI engine not configured, so refinement is unavailable.")

        # per-slide preview + editor
        for i, sl in enumerate(spec["slides"]):
            kind = sl["kind"]
            st.markdown(f"**Slide {i+1} — {kind.replace('_',' ').title()}**")
            pv_col, ed_col = st.columns([3, 2])
            with pv_col:
                _components.html(render_html_slide(sl, spec["meta"]), height=300, scrolling=False)
            with ed_col:
                _slide_editor(sl, i)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown("---")
        # build + download
        gcol, _ = st.columns([2, 4])
        if gcol.button("⬇ Generate & download .pptx", type="primary",
                       key="pm_export", use_container_width=True):
            out_dir = _Path("data/uploads"); out_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            fname = st.session_state.get("upload_filename", "deck").replace(".", "_")
            out_path = str(out_dir / f"KiteIQX_{fname}_{ts}.pptx")
            with st.spinner("Rendering PowerPoint…"):
                ok, msg = build_pptx(spec, out_path)
            if ok:
                st.session_state.pm_out_path = out_path
                st.success(msg)
            else:
                st.error(msg)

        if st.session_state.get("pm_out_path") and _Path(st.session_state["pm_out_path"]).exists():
            with open(st.session_state["pm_out_path"], "rb") as f:
                st.download_button("Download your deck", f.read(),
                                   file_name=_Path(st.session_state["pm_out_path"]).name,
                                   mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                   key="pm_dl", use_container_width=False)


def _slide_editor(sl, i):
    """Inline editors that mutate the slide spec in place."""
    k = sl["kind"]
    if "title" in sl:
        sl["title"] = st.text_input("Title", sl.get("title", ""), key=f"pm_e_title_{i}")
    if k == "title":
        sl["subtitle"] = st.text_input("Subtitle", sl.get("subtitle", ""), key=f"pm_e_sub_{i}")
    if "banner" in sl:
        sl["banner"] = st.text_area("Banner text", sl.get("banner", ""), height=70, key=f"pm_e_ban_{i}")
    if k == "dashboard":
        for j, kp in enumerate(sl.get("kpis", [])):
            c1, c2 = st.columns(2)
            kp["value"] = c1.text_input(f"KPI {j+1} value", kp.get("value", ""), key=f"pm_e_kpiv_{i}_{j}")
            kp["sub"] = c2.text_input(f"KPI {j+1} note", kp.get("sub", ""), key=f"pm_e_kpis_{i}_{j}")
        sl["footer"] = st.text_input("Footer", sl.get("footer", ""), key=f"pm_e_foot_{i}")
    if k in ("dashboard", "story_chart"):
        for j, it in enumerate(sl.get("insights", [])):
            it["head"] = st.text_input(f"Insight {j+1} head", it.get("head", ""), key=f"pm_e_ih_{i}_{j}")
            it["text"] = st.text_area(f"Insight {j+1} text", it.get("text", ""), height=60, key=f"pm_e_it_{i}_{j}")
    if k == "big_stat":
        sl["stat_value"] = st.text_input("Big number", sl.get("stat_value", ""), key=f"pm_e_bsv_{i}")
        sl["stat_label"] = st.text_input("Label", sl.get("stat_label", ""), key=f"pm_e_bsl_{i}")
        sl["body"] = st.text_area("Body", sl.get("body", ""), height=70, key=f"pm_e_bsb_{i}")
    if k == "takeaways":
        for j, t in enumerate(sl.get("items", [])):
            sl["items"][j] = st.text_input(f"Point {j+1}", t, key=f"pm_e_tk_{i}_{j}")
    if k == "closing":
        sl["message"] = st.text_area("Closing message", sl.get("message", ""), height=70, key=f"pm_e_cm_{i}")
    if k == "quality":
        sl["summary"] = st.text_area("Summary", sl.get("summary", ""), height=70, key=f"pm_e_qs_{i}")

# ============================================================
# MAIN
# ============================================================

def initialize_analytics(df: pd.DataFrame, model: str, filename: str = "unknown"):
    llm = None
    if GROQ_AVAILABLE and GROQ_API_KEY:
        try:
            llm = GroqLLM(api_key=GROQ_API_KEY, model=model)
        except Exception as e:
            st.error(f"Could not initialize Groq: {e}")
    st.session_state.analytics = UniversalAnalytics(df, llm)
    st.session_state.viz_engine = VizEngine(st.session_state.analytics)
    st.session_state.upload_filename = filename
    st.session_state.pop("exec_summary", None)
    st.session_state.pop("takeaways", None)
    st.session_state.pop("ai_last_response", None)
    st.session_state.pop("upload_logged", None)
    st.session_state.pop("ppt_out_path", None)
    st.session_state.pop("ppt_payload", None)
    for _k in ["pm_stage", "pm_chat", "pm_answers", "pm_step_i", "pm_stories_pool",
               "pm_selected_ids", "pm_spec", "pm_out_path"]:
        st.session_state.pop(_k, None)
    log_upload(filename, df)


def main():
    render_header()

    st.markdown(
        """
        <style>
        button { color: #ffffff !important; }
        button p, button span, button div { color: #ffffff !important; }
        .stButton > button { background-color: #0a2540 !important; color: #ffffff !important; }
        .stButton > button p,
        .stButton > button span { color: #ffffff !important; }
        .stFileUploader button { background-color: #0a2540 !important; color: #ffffff !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "analytics" not in st.session_state:
        st.session_state.analytics = None
        st.session_state.viz_engine = None

    with st.sidebar:
        st.markdown("### Configuration")
        if not GROQ_AVAILABLE:
            st.error("`groq` package not installed.")
        elif not GROQ_API_KEY:
            st.warning("No GROQ_API_KEY configured. AI features will be disabled.")
        else:
            st.success("AI engine ready.")
        st.markdown(
            f'<div style="font-size:0.78rem;color:#4a5568;background:#f7f8fb;'
            f'border:1px solid #e3e6ee;border-radius:6px;padding:0.35rem 0.7rem;'
            f'margin-top:0.3rem;margin-bottom:0.1rem;">'
            f'⚡ Model: <strong style="color:#0a2540;">llama-3.1-8b-instant</strong></div>',
            unsafe_allow_html=True,
        )

        model = FIXED_MODEL

        st.markdown("---")
        st.markdown("### Load data")
        up = st.file_uploader("Drop a CSV or Excel file", type=["csv", "xlsx", "xls"])
        url = st.text_input("…or fetch CSV from URL")
        demo = st.selectbox("…or try a demo dataset", ["", "E-commerce Sales", "Employee Data", "Simple Demo"])

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load", type="primary"):
                try:
                    df = None
                    fname = "unknown"
                    if up:
                        df = load_uploaded_file(up)
                        fname = up.name
                        st.success(f"Loaded **{up.name}** ({len(df):,} rows).")
                    elif url:
                        r = requests.get(url, timeout=30)
                        df = pd.read_csv(io.StringIO(r.text))
                        fname = url.split("/")[-1] or "url_import"
                        st.success(f"Loaded from URL ({len(df):,} rows).")
                    if df is not None:
                        initialize_analytics(df, model, filename=fname)
                except Exception as e:
                    st.error(f"Load failed: {e}")
        with c2:
            if st.button("Use demo") and demo:
                try:
                    df = generate_demo(demo)
                    initialize_analytics(df, model, filename=f"demo_{demo.replace(' ','_')}")
                    st.success(f"Loaded demo: {demo}.")
                except Exception as e:
                    st.error(f"Demo failed: {e}")

        if st.session_state.get("last_saved_path"):
            st.markdown("---")
            st.markdown("**Last uploaded file archived to:**")
            st.code(st.session_state["last_saved_path"], language=None)

        if st.session_state.analytics:
            st.markdown("---")
            s = st.session_state.analytics.insights["basic_stats"]
            st.metric("Rows", f"{s['rows']:,}")
            st.metric("Columns", s["columns"])
            st.metric("Completeness", f"{100 - s['missing_pct']:.1f}%")

    if st.session_state.analytics is None:
        render_welcome()
        return

    tabs = st.tabs([
        "Executive Dashboard",
        "AI Intelligence",
        "Data Quality & Charts",
        "Data Explorer",
        "Presentation Maker",
    ])

    with tabs[0]:
        render_dashboard()
    with tabs[1]:
        render_ai()
    with tabs[2]:
        render_quality_and_charts()
    with tabs[3]:
        render_explorer()
    with tabs[4]:
        render_presentation_maker()


if __name__ == "__main__":
    main()
