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



# ============================================================
# PRESENTATION MAKER  (python-pptx engine — Streamlit-Cloud safe)
# ============================================================
# Pure-Python deck generator. No Node.js, no npm, no subprocess.
# Only dependency is `python-pptx` (add it to requirements.txt).

import json as _json
from datetime import datetime as _dt
from pathlib import Path as _Path

# Soft import so the app still loads with a friendly message if missing
try:
    from pptx import Presentation as _Prs
    from pptx.util import Inches as _In, Pt as _Pt
    from pptx.dml.color import RGBColor as _RGB
    from pptx.enum.text import PP_ALIGN as _ALIGN, MSO_ANCHOR as _ANCHOR
    from pptx.enum.shapes import MSO_SHAPE as _SHAPE
    from pptx.chart.data import CategoryChartData as _ChartData
    from pptx.enum.chart import (
        XL_CHART_TYPE as _CT,
        XL_LEGEND_POSITION as _LEGPOS,
        XL_LABEL_POSITION as _LABPOS,
    )
    PPTX_PY_AVAILABLE = True
except ImportError:
    PPTX_PY_AVAILABLE = False


# ── theme & font presets ────────────────────────────────────

PPTX_THEMES = {
    "KiteIQX Brand": {
        "primary": "0A2540", "secondary": "14304F", "accent": "C79A3A",
        "background": "FFFFFF", "surface": "F7F8FB", "text": "1A2333", "text_soft": "4A5568",
    },
    "Midnight Executive": {
        "primary": "1E2761", "secondary": "16213E", "accent": "CADCFC",
        "background": "FFFFFF", "surface": "F0F4FF", "text": "1E2761", "text_soft": "4A5880",
    },
    "Coral Energy": {
        "primary": "2F3C7E", "secondary": "3D4F9E", "accent": "F96167",
        "background": "FFFFFF", "surface": "FFF8F8", "text": "1A1A2E", "text_soft": "555577",
    },
    "Charcoal Minimal": {
        "primary": "36454F", "secondary": "263238", "accent": "607D8B",
        "background": "FFFFFF", "surface": "F5F5F5", "text": "212121", "text_soft": "616161",
    },
    "Forest & Moss": {
        "primary": "2C5F2D", "secondary": "1B4332", "accent": "97BC62",
        "background": "FFFFFF", "surface": "F0FFF4", "text": "1B4332", "text_soft": "52796F",
    },
    "Custom": {
        "primary": "0A2540", "secondary": "14304F", "accent": "C79A3A",
        "background": "FFFFFF", "surface": "F7F8FB", "text": "1A2333", "text_soft": "4A5568",
    },
}

PPTX_FONTS = {
    "Georgia + Calibri  (Elegant)":        {"header": "Georgia",      "body": "Calibri"},
    "Arial Black + Arial  (Bold)":         {"header": "Arial Black",  "body": "Arial"},
    "Calibri  (Clean Modern)":             {"header": "Calibri",      "body": "Calibri Light"},
    "Trebuchet + Calibri  (Professional)": {"header": "Trebuchet MS", "body": "Calibri"},
    "Cambria + Calibri  (Classic)":        {"header": "Cambria",      "body": "Calibri"},
}


# ── low-level drawing helpers ───────────────────────────────

def _hx(h):
    h = h.lstrip("#")
    return _RGB(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _bg(slide, hexcolor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = _hx(hexcolor)


def _rect(slide, x, y, w, h, fill=None, line=None, line_w=1.0, shape=None):
    shp = slide.shapes.add_shape(
        shape or _SHAPE.RECTANGLE, _In(x), _In(y), _In(w), _In(h)
    )
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _hx(fill)
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = _hx(line)
        shp.line.width = _Pt(line_w)
    return shp


def _txt(slide, text, x, y, w, h, size=14, font="Calibri", color="1A2333",
         bold=False, align="left", valign="top", spacing=None):
    tb = slide.shapes.add_textbox(_In(x), _In(y), _In(w), _In(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = {
        "top": _ANCHOR.TOP, "middle": _ANCHOR.MIDDLE, "bottom": _ANCHOR.BOTTOM
    }[valign]
    p = tf.paragraphs[0]
    p.alignment = {"left": _ALIGN.LEFT, "center": _ALIGN.CENTER, "right": _ALIGN.RIGHT}[align]
    run = p.add_run()
    run.text = text
    f = run.font
    f.size = _Pt(size)
    f.name = font
    f.bold = bold
    f.color.rgb = _hx(color)
    if spacing:
        run._r.get_or_add_rPr().set("spc", str(int(spacing * 100)))
    return tb


def _color_points(series, colors):
    for i, pt in enumerate(series.points):
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = _hx(colors[i % len(colors)])


def _bar_chart(slide, x, y, w, h, labels, values, colors, body_font, text_soft, text):
    cd = _ChartData()
    cd.categories = labels
    cd.add_series("Series 1", values)
    gf = slide.shapes.add_chart(_CT.COLUMN_CLUSTERED, _In(x), _In(y), _In(w), _In(h), cd)
    chart = gf.chart
    chart.has_legend = False
    chart.has_title = False
    try:
        chart.font.name = body_font
        chart.font.size = _Pt(10)
    except Exception:
        pass
    plot = chart.plots[0]
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = "#,##0"
    dl.number_format_is_linked = False
    dl.position = _LABPOS.OUTSIDE_END
    dl.font.size = _Pt(9)
    dl.font.color.rgb = _hx(text)
    _color_points(plot.series[0], colors)
    ca = chart.category_axis
    va = chart.value_axis
    ca.tick_labels.font.size = _Pt(10)
    ca.tick_labels.font.color.rgb = _hx(text_soft)
    va.tick_labels.font.size = _Pt(9)
    va.tick_labels.font.color.rgb = _hx(text_soft)
    return chart


def _line_chart(slide, x, y, w, h, labels, values, color, body_font, text_soft):
    cd = _ChartData()
    cd.categories = labels
    cd.add_series("Series 1", values)
    gf = slide.shapes.add_chart(_CT.LINE, _In(x), _In(y), _In(w), _In(h), cd)
    chart = gf.chart
    chart.has_legend = False
    chart.has_title = False
    try:
        chart.font.name = body_font
        chart.font.size = _Pt(10)
    except Exception:
        pass
    series = chart.plots[0].series[0]
    series.smooth = True
    series.format.line.color.rgb = _hx(color)
    series.format.line.width = _Pt(2.5)
    ca = chart.category_axis
    va = chart.value_axis
    ca.tick_labels.font.size = _Pt(10)
    ca.tick_labels.font.color.rgb = _hx(text_soft)
    va.tick_labels.font.size = _Pt(9)
    va.tick_labels.font.color.rgb = _hx(text_soft)
    return chart


def _doughnut_chart(slide, x, y, w, h, labels, values, colors, body_font):
    cd = _ChartData()
    cd.categories = labels
    cd.add_series("Series 1", values)
    gf = slide.shapes.add_chart(_CT.DOUGHNUT, _In(x), _In(y), _In(w), _In(h), cd)
    chart = gf.chart
    chart.has_title = False
    chart.has_legend = True
    chart.legend.position = _LEGPOS.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = _Pt(10)
    try:
        chart.font.name = body_font
    except Exception:
        pass
    plot = chart.plots[0]
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.show_percentage = True
    dl.show_value = False
    dl.number_format = "0%"
    dl.number_format_is_linked = False
    dl.font.size = _Pt(10)
    _color_points(plot.series[0], colors)
    return chart


# ── chart data extraction (reused) ──────────────────────────

def _collect_chart_data(analytics) -> dict:
    data: dict = {}
    df = analytics.df

    if analytics.categorical_cols:
        cat = analytics.categorical_cols[0]
        vc = df[cat].value_counts().head(7)
        data["bar_chart"] = {
            "title":  f"{cat} Breakdown",
            "labels": [str(x) for x in vc.index.tolist()],
            "values": [float(x) for x in vc.values.tolist()],
        }

    if len(analytics.categorical_cols) >= 2:
        cat2 = analytics.categorical_cols[1]
        vc2 = df[cat2].value_counts().head(6)
        data["pie_chart"] = {
            "title":  f"{cat2} Distribution",
            "labels": [str(x) for x in vc2.index.tolist()],
            "values": [float(x) for x in vc2.values.tolist()],
        }

    if analytics.datetime_cols and analytics.numeric_cols:
        dc = analytics.datetime_cols[0]
        vc_col = analytics.money_cols[0] if analytics.money_cols else analytics.numeric_cols[0]
        ts = df[[dc, vc_col]].dropna().sort_values(dc)
        if len(ts) > 14:
            try:
                ts = ts.set_index(dc).resample("M")[vc_col].sum().reset_index()
            except Exception:
                ts = ts.iloc[::max(1, len(ts) // 12)]
        data["line_chart"] = {
            "title":  f"{vc_col} Trend",
            "labels": [str(x)[:7] for x in ts[dc].tolist()],
            "values": [float(x) for x in ts[vc_col].fillna(0).tolist()],
        }

    return data


# ── narrative payload (reused; LLM optional) ────────────────

def _build_payload(analytics, emphasis: str, theme_cfg: dict, font_cfg: dict) -> dict:
    a = analytics
    ins = a.insights
    b = ins["basic_stats"]

    exec_summary = st.session_state.get("exec_summary") or a._fallback_summary()
    takeaways    = st.session_state.get("takeaways")    or a._fallback_takeaways()
    quality_status, quality_score, quality_summary = a.get_data_quality_report()

    kpis = [
        {"label": "Records Analyzed", "value": f"{b['rows']:,}",            "desc": "rows of business data"},
        {"label": "Dimensions",        "value": str(b["columns"]),           "desc": f"{ins['column_types']['numeric']} numeric · {ins['column_types']['categorical']} categorical"},
        {"label": "Data Completeness",  "value": f"{100 - b['missing_pct']:.1f}%", "desc": "proportion of non-null values"},
    ]
    if "monetary" in ins:
        m = ins["monetary"]
        kpis.append({"label": "Total Value", "value": f"${m['total']:,.0f}", "desc": f"avg ${m['avg']:,.2f}/record"})
    else:
        kpis.append({"label": "Numeric Columns", "value": str(ins["column_types"]["numeric"]), "desc": "quantifiable dimensions"})

    domain          = "Business Intelligence Report"
    subtitle        = f"AI-powered analysis of {b['rows']:,} records across {b['columns']} dimensions"
    recommendations = list(takeaways)
    closing_message = "Data is the compass. Let KiteIQX Intelligence be your guide."

    if a.llm:
        emphasis_clause = f"User emphasis: {emphasis}" if emphasis.strip() else "General analysis — surface the most strategic angle."
        try:
            prompt = (
                f"Dataset summary: {exec_summary[:350]}\n"
                f"Key takeaways: {'; '.join(takeaways)}\n"
                f"{emphasis_clause}\n\n"
                "Return ONLY valid JSON (no markdown fences, no preamble) with this exact structure:\n"
                '{"domain":"2-4 word business domain","subtitle":"compelling subtitle max 12 words",'
                '"recommendations":["actionable rec 1","actionable rec 2","actionable rec 3"],'
                '"closing_message":"one powerful closing sentence"}'
            )
            raw = a.llm.predict(prompt, max_tokens=450)
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = _json.loads(raw)
            domain          = parsed.get("domain",          domain)
            subtitle        = parsed.get("subtitle",        subtitle)
            recommendations = parsed.get("recommendations", recommendations)
            closing_message = parsed.get("closing_message", closing_message)
        except Exception:
            pass

    return {
        "theme":  theme_cfg,
        "fonts":  font_cfg,
        "charts": _collect_chart_data(analytics),
        "content": {
            "domain":          domain,
            "subtitle":        subtitle,
            "date":            _dt.now().strftime("%B %Y"),
            "exec_summary":    exec_summary,
            "takeaways":       takeaways,
            "kpis":            kpis,
            "quality": {
                "score":   quality_score,
                "status":  quality_status,
                "summary": quality_summary.replace("**", "").replace("  \n", " | "),
            },
            "recommendations": recommendations,
            "closing_message": closing_message,
            "emphasis":        emphasis,
        },
    }


# ── the deck builder ────────────────────────────────────────

def _build_pptx_python(payload: dict, out_path: str):
    """Build the 9-slide deck with python-pptx. Returns (success, message)."""
    if not PPTX_PY_AVAILABLE:
        return False, "python-pptx not installed. Add `python-pptx` to requirements.txt."

    th = payload["theme"]
    fonts = payload["fonts"]
    c = payload["content"]
    charts = payload["charts"]
    HF, BF = fonts["header"], fonts["body"]

    prs = _Prs()
    prs.slide_width = _In(10)
    prs.slide_height = _In(5.625)
    blank = prs.slide_layouts[6]
    H = 5.625

    def num_circle(slide, n, x, y, d, fill, txt_color):
        _rect(slide, x, y, d, d, fill=fill, shape=_SHAPE.OVAL)
        _txt(slide, str(n), x, y, d, d, size=18, font=HF, color=txt_color,
             bold=True, align="center", valign="middle")

    def sec_header(slide, title):
        _txt(slide, title, 0.5, 0.22, 9.0, 0.5, size=24, font=HF,
             color=th["primary"], bold=True)

    def slide_num(slide, n):
        _txt(slide, str(n), 9.55, 5.28, 0.35, 0.25, size=8, font=BF,
             color=th["text_soft"], align="right")

    # ── SLIDE 1: TITLE ──────────────────────────────────────
    s = prs.slides.add_slide(blank)
    _bg(s, th["primary"])
    _rect(s, 0, 0, 0.14, H, fill=th["accent"])
    _rect(s, 8.2, 0, 1.8, 1.6, fill=th["secondary"])
    _txt(s, "KITEIQX INTELLIGENCE", 0.46, 0.85, 7.5, 0.4, size=10, font=BF,
         color=th["accent"], bold=True, spacing=4)
    title_text = c.get("domain") or "Business Intelligence Report"
    _txt(s, title_text, 0.46, 1.4, 9.2, 2.1,
         size=36 if len(title_text) > 35 else 44, font=HF, color="FFFFFF", bold=True)
    _txt(s, c.get("subtitle", ""), 0.46, 3.6, 8.0, 0.85, size=14, font=BF, color="94A3B8")
    _txt(s, c.get("date", ""), 0.46, 5.0, 3.5, 0.36, size=10, font=BF, color=th["accent"])
    _txt(s, "Powered by KiteIQX Intelligence", 6.0, 5.0, 3.85, 0.36,
         size=9, font=BF, color="475569", align="right")

    # ── SLIDE 2: EXECUTIVE SUMMARY ──────────────────────────
    s = prs.slides.add_slide(blank)
    _bg(s, th["background"])
    sec_header(s, "Business Story")
    slide_num(s, 2)
    _rect(s, 0.4, 0.9, 5.9, 4.35, fill=th["surface"], line="E2E8F0", line_w=1)
    _rect(s, 0.4, 0.9, 0.09, 4.35, fill=th["accent"])
    _txt(s, c.get("exec_summary", ""), 0.65, 1.05, 5.45, 4.0, size=13, font=BF,
         color=th["text"], valign="top")
    for i, kpi in enumerate((c.get("kpis") or [])[:4]):
        y = 0.9 + i * 1.12
        _rect(s, 6.55, y, 3.15, 0.98, fill="FFFFFF", line="E2E8F0", line_w=1)
        _rect(s, 6.7, y + 0.22, 0.22, 0.22, fill=th["accent"], shape=_SHAPE.OVAL)
        _txt(s, kpi.get("label", ""), 7.06, y + 0.08, 2.5, 0.26, size=9, font=BF,
             color=th["text_soft"], bold=True)
        _txt(s, kpi.get("value", ""), 6.6, y + 0.36, 3.0, 0.42, size=20, font=HF,
             color=th["primary"], bold=True)
        _txt(s, kpi.get("desc", ""), 6.6, y + 0.75, 3.0, 0.2, size=8, font=BF,
             color=th["text_soft"])

    # ── SLIDE 3: KPI DASHBOARD ──────────────────────────────
    s = prs.slides.add_slide(blank)
    _bg(s, th["primary"])
    _txt(s, "Key Metrics", 0.5, 0.28, 9, 0.65, size=28, font=HF, color="FFFFFF", bold=True)
    _txt(s, "At a glance — the numbers that matter most", 0.5, 0.98, 9, 0.35,
         size=12, font=BF, color=th["accent"])
    slide_num(s, 3)
    kpis = (c.get("kpis") or [])[:4]
    n = len(kpis)
    card_w = 2.1 if n == 4 else 2.8 if n == 3 else 4.2
    gap = 0.22
    total_w = n * card_w + (n - 1) * gap
    start_x = (10 - total_w) / 2
    card_y, card_h = 1.48, 3.6
    for i, kpi in enumerate(kpis):
        x = start_x + i * (card_w + gap)
        _rect(s, x, card_y, card_w, card_h, fill=th["secondary"])
        _rect(s, x, card_y, card_w, 0.08, fill=th["accent"])
        _txt(s, kpi.get("label", ""), x + 0.12, card_y + 0.18, card_w - 0.24, 0.4,
             size=9, font=BF, color="94A3B8", bold=True)
        val = kpi.get("value", "")
        val_fs = 20 if len(val) > 7 else 26 if len(val) > 5 else 34
        _txt(s, val, x + 0.1, card_y + 0.72, card_w - 0.2, 1.4, size=val_fs, font=HF,
             color="FFFFFF", bold=True, align="center", valign="middle")
        _txt(s, kpi.get("desc", ""), x + 0.1, card_y + 2.92, card_w - 0.2, 0.55,
             size=9, font=BF, color="94A3B8", align="center")

    # ── SLIDE 4: CATEGORY ANALYSIS (bar) ────────────────────
    bar = charts.get("bar_chart")
    if bar and bar.get("labels"):
        s = prs.slides.add_slide(blank)
        _bg(s, th["background"])
        sec_header(s, bar.get("title", "Category Analysis"))
        slide_num(s, 4)
        _bar_chart(
            s, 0.5, 0.9, 9, 4.4, bar["labels"], bar["values"],
            [th["primary"], th["secondary"], th["accent"], "64748B", "94A3B8", "CBD5E1"],
            BF, th["text_soft"], th["text"],
        )

    # ── SLIDE 5: TREND or DISTRIBUTION ──────────────────────
    line = charts.get("line_chart")
    pie = charts.get("pie_chart")
    if line and line.get("labels"):
        s = prs.slides.add_slide(blank)
        _bg(s, th["background"])
        sec_header(s, line.get("title", "Trend Analysis"))
        slide_num(s, 5)
        _line_chart(s, 0.5, 0.9, 9, 4.4, line["labels"], line["values"],
                    th["primary"], BF, th["text_soft"])
    elif pie and pie.get("labels"):
        s = prs.slides.add_slide(blank)
        _bg(s, th["background"])
        sec_header(s, pie.get("title", "Distribution"))
        slide_num(s, 5)
        _doughnut_chart(
            s, 0.3, 0.85, 5.5, 4.5, pie["labels"], pie["values"],
            [th["primary"], th["accent"], th["secondary"], "64748B", "94A3B8", "CBD5E1"], BF,
        )
        top_labels = pie["labels"][:4]
        top_vals = pie["values"][:4]
        total = sum(top_vals) or 1
        for i, label in enumerate(top_labels):
            pct = top_vals[i] / total * 100
            y = 1.05 + i * 1.08
            _rect(s, 6.1, y, 3.65, 0.9, fill="FFFFFF", line="E2E8F0", line_w=1)
            _txt(s, str(label), 6.25, y + 0.07, 2.5, 0.3, size=11, font=BF,
                 color=th["text"], bold=True)
            _txt(s, f"{pct:.1f}%", 8.65, y + 0.05, 0.9, 0.38, size=18, font=HF,
                 color=th["primary"], bold=True, align="right")
            _txt(s, f"{round(top_vals[i]):,} records", 6.25, y + 0.54, 3.3, 0.28,
                 size=9, font=BF, color=th["text_soft"])

    # ── SLIDE 6: STRATEGIC TAKEAWAYS ────────────────────────
    s = prs.slides.add_slide(blank)
    _bg(s, th["background"])
    sec_header(s, "3 Decisions This Data Supports")
    slide_num(s, 6)
    circle_colors = [th["primary"], th["accent"], th["secondary"]]
    num_colors = ["FFFFFF", "2A1B00", "FFFFFF"]
    for i, t in enumerate((c.get("takeaways") or [])[:3]):
        y = 0.9 + i * 1.5
        _rect(s, 0.4, y, 9.2, 1.3, fill="FFFFFF", line="E2E8F0", line_w=1)
        num_circle(s, i + 1, 0.54, y + 0.33, 0.66, circle_colors[i], num_colors[i])
        _txt(s, t, 1.38, y + 0.1, 8.1, 1.12, size=13, font=BF, color=th["text"], valign="middle")

    # ── SLIDE 7: DATA QUALITY ───────────────────────────────
    s = prs.slides.add_slide(blank)
    _bg(s, th["background"])
    sec_header(s, "Data Quality Assessment")
    slide_num(s, 7)
    q = c.get("quality", {"score": 85, "status": "good", "summary": ""})
    sc = q.get("score", 85)
    clr = "15803D" if sc >= 90 else "0369A1" if sc >= 75 else "B45309" if sc >= 60 else "B91C1C"
    label = {"excellent": "Excellent", "good": "Good", "fair": "Needs Attention",
             "poor": "Poor"}.get(q.get("status"), "Fair")
    _rect(s, 0.65, 1.0, 3.0, 3.0, fill="FFFFFF", line=clr, line_w=5, shape=_SHAPE.OVAL)
    _txt(s, str(sc), 0.65, 1.5, 3.0, 1.6, size=64, font=HF, color=clr, bold=True,
         align="center", valign="middle")
    _txt(s, "/100", 0.65, 3.0, 3.0, 0.4, size=13, font=BF, color=clr, align="center")
    _txt(s, label, 0.65, 3.88, 3.0, 0.45, size=14, font=HF, color=clr, bold=True, align="center")
    _txt(s, "Quality Summary", 4.1, 1.0, 5.5, 0.42, size=14, font=HF, color=th["primary"], bold=True)
    _txt(s, q.get("summary") or "Dataset is ready for analysis.", 4.1, 1.5, 5.5, 3.0,
         size=12, font=BF, color=th["text"], valign="top")
    _rect(s, 0.4, 4.65, 9.2, 0.58, fill=th["surface"], line="E2E8F0", line_w=1)
    _txt(s, "For column-level detail, open the Data Quality tab in KiteIQX Intelligence.",
         0.6, 4.68, 8.8, 0.5, size=10, font=BF, color=th["text_soft"], valign="middle")

    # ── SLIDE 8: RECOMMENDATIONS ────────────────────────────
    s = prs.slides.add_slide(blank)
    _bg(s, th["background"])
    sec_header(s, "Recommended Next Steps")
    slide_num(s, 8)
    for i, rec in enumerate((c.get("recommendations") or [])[:3]):
        y = 0.9 + i * 1.54
        _rect(s, 0.4, y, 9.2, 1.37, fill="FFFFFF" if i % 2 == 0 else th["surface"],
              line="E2E8F0", line_w=1)
        _rect(s, 0.4, y, 0.55, 1.37, fill=th["primary"])
        _txt(s, str(i + 1), 0.4, y, 0.55, 1.37, size=22, font=HF, color="FFFFFF",
             bold=True, align="center", valign="middle")
        _txt(s, rec, 1.1, y + 0.12, 8.35, 1.12, size=13, font=BF, color=th["text"], valign="middle")

    # ── SLIDE 9: CLOSING ────────────────────────────────────
    s = prs.slides.add_slide(blank)
    _bg(s, th["primary"])
    _rect(s, 0, 0, 0.14, H, fill=th["accent"])
    _rect(s, 10 - 2.8, H - 2.2, 2.8, 2.2, fill=th["secondary"])
    _txt(s, "KITEIQX INTELLIGENCE", 0.46, 0.9, 9, 0.42, size=10, font=BF,
         color=th["accent"], bold=True, spacing=4)
    _txt(s, c.get("closing_message", "Thank You"), 0.46, 1.55, 9, 2.1, size=36, font=HF,
         color="FFFFFF", bold=True)
    _txt(s, "Generated by KiteIQX Intelligence · " + c.get("date", ""), 0.46, 3.95, 9, 0.42,
         size=10, font=BF, color="475569")
    _txt(s, "kiteiqx.com", 0.46, 4.48, 4, 0.42, size=14, font=BF, color=th["accent"], bold=True)

    prs.save(out_path)
    size_kb = _Path(out_path).stat().st_size // 1024
    return True, f"Generated {_Path(out_path).name} ({size_kb} KB)"


# ── Streamlit UI ────────────────────────────────────────────

def render_presentation_maker():
    a = st.session_state.get("analytics")
    if a is None:
        st.info("Upload or load data first (use the sidebar), then come back here to generate your deck.")
        return

    if not PPTX_PY_AVAILABLE:
        st.error(
            "The `python-pptx` package isn't installed. Add a line `python-pptx` to your "
            "requirements.txt and redeploy — that's the only dependency this tab needs."
        )
        return

    st.markdown(
        '<div class="kx-card">'
        '<div class="kx-card-title">Presentation Maker</div>'
        '<div class="kx-card-sub">Transform your analysis into a polished, CEO-ready slide deck — '
        'no PowerPoint experience needed. Customize the story, design, and download in seconds.</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # 1 · Story Focus
    st.markdown(
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;'
        'color:#4a5568;font-weight:700;margin:1.4rem 0 0.5rem;">1 · Story Focus</div>',
        unsafe_allow_html=True,
    )
    existing = st.session_state.get("takeaways", [])
    if existing:
        st.markdown(
            '<div class="kx-callout" style="margin-bottom:0.75rem;">'
            "<strong>Current story from the data:</strong><br>"
            + "<br>".join(f"&nbsp;&nbsp;{i+1}. {t}" for i, t in enumerate(existing))
            + "</div>",
            unsafe_allow_html=True,
        )
    emphasis = st.text_area(
        "Angle or message to emphasize  (optional — leave blank to use the AI-detected story)",
        placeholder='e.g. "Focus on the revenue concentration risk in Electronics."',
        height=88, key="ppt_emphasis",
    )

    # 2 · Design
    st.markdown(
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;'
        'color:#4a5568;font-weight:700;margin:1.4rem 0 0.5rem;">2 · Design & Style</div>',
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    with d1:
        theme_name = st.selectbox("Color theme", list(PPTX_THEMES.keys()), index=0, key="ppt_theme")
    with d2:
        font_name = st.selectbox("Font pairing", list(PPTX_FONTS.keys()), index=0, key="ppt_font")

    theme_cfg = dict(PPTX_THEMES[theme_name])
    if theme_name == "Custom":
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            theme_cfg["primary"] = st.color_picker("Primary", f"#{theme_cfg['primary']}", key="ppt_cp").lstrip("#")
        with cc2:
            theme_cfg["accent"] = st.color_picker("Accent", f"#{theme_cfg['accent']}", key="ppt_ca").lstrip("#")
        with cc3:
            theme_cfg["secondary"] = st.color_picker("Secondary", f"#{theme_cfg['secondary']}", key="ppt_cs").lstrip("#")
    else:
        swatches = [theme_cfg["primary"], theme_cfg["accent"], theme_cfg["surface"], theme_cfg["text"]]
        html = "".join(
            f'<div style="width:30px;height:30px;border-radius:6px;background:#{h};border:1px solid #e3e6ee;"></div>'
            for h in swatches
        )
        st.markdown(
            f'<div style="display:flex;gap:0.45rem;align-items:center;margin-bottom:0.5rem;">{html}'
            f'<span style="color:#4a5568;font-size:0.82rem;margin-left:0.4rem;">Theme palette preview</span></div>',
            unsafe_allow_html=True,
        )

    font_cfg = PPTX_FONTS[font_name]
    st.markdown(
        f'<p style="font-family:{font_cfg["header"]};font-size:1.05rem;font-weight:700;'
        f'color:#1a2333;margin:0 0 0.15rem 0;">Header — {font_cfg["header"]}</p>'
        f'<p style="font-family:{font_cfg["body"]};font-size:0.9rem;color:#4a5568;margin:0 0 1rem 0;">'
        f"Body — {font_cfg['body']}</p>",
        unsafe_allow_html=True,
    )

    # 3 · Custom template
    st.markdown(
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;'
        'color:#4a5568;font-weight:700;margin:1.4rem 0 0.5rem;">3 · Custom Template  (optional)</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Upload your branded .pptx template"):
        st.info(
            "Drop your company template here. KiteIQX will read its color scheme and prioritize "
            "it over the theme selected above."
        )
        tpl_file = st.file_uploader("Choose a .pptx file", type=["pptx"], key="ppt_template")
        if tpl_file is not None:
            try:
                import io as _io
                prs_obj = _Prs(_io.BytesIO(tpl_file.read()))
                st.success(
                    f"Template '{tpl_file.name}' loaded — {len(prs_obj.slide_layouts)} layouts detected. "
                    "Theme selection above will be applied to the generated deck."
                )
            except Exception as e:
                st.warning(f"Could not parse template: {e}. Using the selected theme instead.")

    # 4 · Outline
    st.markdown(
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;'
        'color:#4a5568;font-weight:700;margin:1.4rem 0 0.5rem;">4 · What&apos;s in the Deck</div>',
        unsafe_allow_html=True,
    )
    chart_data = _collect_chart_data(a)
    outline = [
        ("1", "Title", "Dataset name, subtitle, date"),
        ("2", "Business Story", "Executive narrative + KPI sidebar"),
        ("3", "Key Metrics", "4 headline KPIs on a dark dashboard slide"),
    ]
    if chart_data.get("bar_chart"):
        outline.append(("4", "Category Analysis",
                         chart_data["bar_chart"]["title"] + f"  ({len(chart_data['bar_chart']['labels'])} categories)"))
    else:
        outline.append(("4", "Category Analysis", "Generates when categorical data is present"))
    if chart_data.get("line_chart"):
        outline.append(("5", "Trend Analysis", chart_data["line_chart"]["title"]))
    elif chart_data.get("pie_chart"):
        outline.append(("5", "Distribution", chart_data["pie_chart"]["title"] + " — doughnut + share cards"))
    else:
        outline.append(("5", "Trend / Distribution", "Generates when time or 2nd categorical data is present"))
    outline += [
        ("6", "Strategic Takeaways", "3 numbered decision cards"),
        ("7", "Data Quality", "Score gauge + quality summary"),
        ("8", "Recommendations", "3 actionable next steps"),
        ("9", "Closing", "Brand close with key message"),
    ]
    rows = ""
    for num, title, sub in outline:
        rows += (
            f'<div style="display:flex;gap:0.65rem;align-items:center;padding:0.32rem 0;'
            f'border-bottom:1px solid #e3e6ee;">'
            f'<div style="flex:0 0 26px;width:26px;height:26px;border-radius:5px;background:#0a2540;'
            f'color:#c79a3a;font-size:0.72rem;font-weight:700;display:flex;align-items:center;'
            f'justify-content:center;">{num}</div>'
            f'<div><strong style="color:#1a2333;font-size:0.9rem;">{title}</strong> '
            f'<span style="color:#4a5568;font-size:0.82rem;">— {sub}</span></div></div>'
        )
    st.markdown(f'<div class="kx-card" style="padding:0.75rem 1rem;">{rows}</div>', unsafe_allow_html=True)

    # 5 · Generate
    st.markdown(
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.8px;'
        'color:#4a5568;font-weight:700;margin:1.4rem 0 0.5rem;">5 · Generate</div>',
        unsafe_allow_html=True,
    )
    gen_col, _ = st.columns([2, 5])
    with gen_col:
        generate = st.button("Generate Presentation", type="primary", key="ppt_generate", use_container_width=True)

    if generate:
        bar = st.progress(0, text="Drafting narrative with KiteIQX AI…")
        payload = _build_payload(a, emphasis.strip(), theme_cfg, font_cfg)
        bar.progress(55, text="Building slides…")
        out_dir = _Path("data/uploads")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        fname = st.session_state.get("upload_filename", "analysis").replace(".", "_")
        out_path = str(out_dir / f"KiteIQX_{fname}_{ts}.pptx")
        success, msg = _build_pptx_python(payload, out_path)
        bar.progress(100, text="Done!")
        bar.empty()
        if success:
            st.session_state["ppt_out_path"] = out_path
            st.session_state["ppt_payload"] = payload
            st.success(f"Presentation ready  ·  9 slides  ·  {payload['content']['domain']}")
        else:
            st.error(msg)

    # Download
    if st.session_state.get("ppt_out_path") and _Path(st.session_state["ppt_out_path"]).exists():
        out_path = st.session_state["ppt_out_path"]
        with open(out_path, "rb") as f:
            data = f.read()
        st.markdown("---")
        dl_col, sum_col = st.columns([2, 4])
        with dl_col:
            st.download_button(
                "⬇ Download .pptx", data=data,
                file_name=_Path(out_path).name,
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key="ppt_download", use_container_width=True,
            )
        with sum_col:
            p = st.session_state.get("ppt_payload", {})
            if p:
                cc = p.get("content", {})
                st.markdown(
                    f'<div class="kx-card"><div class="kx-card-title">Deck summary</div>'
                    f'<div style="font-size:0.98rem;color:#1a2333;"><strong>{cc.get("domain","")}</strong> '
                    f'&nbsp;·&nbsp; 9 slides &nbsp;·&nbsp; {cc.get("date","")}</div>'
                    f'<div style="font-size:0.85rem;color:#4a5568;margin-top:0.2rem;">{cc.get("subtitle","")}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown(
            '<div class="kx-callout" style="margin-top:0.75rem;">'
            "<strong>Want to tweak the deck?</strong>  Change the theme, font, or story emphasis above "
            "and click <em>Generate Presentation</em> again."
            "</div>",
            unsafe_allow_html=True,
        )


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
        "Data Quality",
        "Advanced Analytics",
        "Custom Charts",
        "Data Explorer",
        "Presentation Maker",
    ])

    with tabs[0]:
        render_dashboard()
    with tabs[1]:
        render_ai()
    with tabs[2]:
        render_data_quality()
    with tabs[3]:
        render_advanced()
    with tabs[4]:
        render_custom_charts()
    with tabs[5]:
        render_explorer()
    with tabs[6]:
        render_presentation_maker()


if __name__ == "__main__":
    main()
