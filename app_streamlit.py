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
    initial_sidebar_state="expanded",
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

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif !important;
    color: var(--kite-text);
}

.block-container { padding-top: 2rem; }

/* Header / branding */
.kx-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}
.kx-mark {
    width: 38px; height: 38px;
    background: var(--kite-navy);
    color: var(--kite-gold);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1.4rem;
    letter-spacing: -1px;
}
.kx-wordmark {
    font-size: 1.6rem; font-weight: 700;
    color: var(--kite-navy);
    letter-spacing: -0.3px;
}
.kx-wordmark span { color: var(--kite-gold); }
.kx-tag {
    color: var(--kite-text-soft);
    font-size: 0.95rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--kite-border);
    padding-bottom: 1rem;
}

/* Pills / badges */
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

/* Storyboard cards */
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
    background: var(--kite-surface);
    border: 1px solid var(--kite-border);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin: 0.6rem 0;
    box-shadow: 0 1px 3px rgba(10, 37, 64, 0.04);
}
.kx-card-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--kite-text-soft);
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.kx-card-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--kite-navy);
    line-height: 1.2;
}
.kx-card-sub {
    font-size: 0.85rem;
    color: var(--kite-text-soft);
    margin-top: 0.25rem;
}

.kx-callout {
    border-left: 4px solid var(--kite-gold);
    background: var(--kite-surface-soft);
    padding: 1rem 1.25rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: var(--kite-text);
}

.kx-takeaway {
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    background: #fdfcf6;
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
    color: #2a1b00;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.kx-takeaway-body { color: var(--kite-text); font-size: 0.96rem; line-height: 1.45; }

.kx-ai-response {
    background: var(--kite-surface-soft);
    border-left: 4px solid var(--kite-navy);
    padding: 1.25rem 1.5rem;
    border-radius: 0 10px 10px 0;
    margin: 1rem 0;
    line-height: 1.55;
}

/* Quality status panels */
.kx-q-excellent { background: #ecfdf3; border: 1px solid #abefc6; padding: 1rem 1.2rem; border-radius: 10px; }
.kx-q-good      { background: #f0f9ff; border: 1px solid #bae0fd; padding: 1rem 1.2rem; border-radius: 10px; }
.kx-q-warning   { background: #fffaeb; border: 1px solid #fcd980; padding: 1rem 1.2rem; border-radius: 10px; }
.kx-q-poor      { background: #fef2f2; border: 1px solid #fda4a4; padding: 1rem 1.2rem; border-radius: 10px; }

/* Tabs */
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

/* Buttons */
.stButton > button {
    background: var(--kite-navy);
    color: #ffffff;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 0.5rem 1.1rem;
}
.stButton > button:hover {
    background: var(--kite-navy-2);
    color: var(--kite-gold);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--kite-surface-soft);
    border-right: 1px solid var(--kite-border);
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: var(--kite-navy);
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
    """Apply consistent KiteIQX styling to any Plotly figure."""
    fig.update_layout(**KITE_PLOTLY)
    return fig


# ============================================================
# GROQ WRAPPER
# ============================================================

class GroqLLM:
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
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

        # Attempt datetime parsing on plausibly-named columns
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

    # ---- AI helpers ----

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
        return self.llm.predict(prompt)

    def ai_executive_summary(self) -> str:
        """Generate the storyboard narrative for the Executive Dashboard."""
        if not self.llm:
            return self._fallback_summary()
        prompt = (
            f"DATASET CONTEXT\n{self._data_brief()}\n\n"
            "Write a 3-paragraph executive narrative for a non-technical business leader. "
            "Structure:\n"
            "  Paragraph 1 - WHAT THIS DATA IS: in one sentence, what business activity does it describe?\n"
            "  Paragraph 2 - WHAT JUMPS OUT: the 2-3 most important patterns or anomalies, with specific numbers.\n"
            "  Paragraph 3 - WHY IT MATTERS: the single biggest business implication.\n\n"
            "Tone: senior consultant briefing a CEO. No jargon. No filler. No bullet points - prose."
        )
        return self.llm.predict(prompt, max_tokens=1200)

    def ai_top_takeaways(self) -> list:
        if not self.llm:
            return self._fallback_takeaways()
        prompt = (
            f"DATASET CONTEXT\n{self._data_brief()}\n\n"
            "Return exactly 3 takeaways for a CEO reading this data. Each takeaway is:\n"
            "  - ONE sentence\n"
            "  - Must reference a specific number from the data\n"
            "  - Must include a clear recommended action\n\n"
            "Format your response as a numbered list (1. 2. 3.) - nothing else, no preamble."
        )
        out = self.llm.predict(prompt, max_tokens=600)
        lines = [re.sub(r"^\s*\d+[\.\)]\s*", "", ln).strip() for ln in out.split("\n") if ln.strip()]
        lines = [ln for ln in lines if len(ln) > 15]
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
# FILE LOADING (CSV + XLSX + save to disk)
# ============================================================

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load CSV/XLSX/XLS from an uploaded file and persist to data/uploads/."""
    name = uploaded_file.name
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    # Save a copy for archival before reading (rewind after)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = UPLOAD_DIR / f"{ts}_{name}"
    raw = uploaded_file.getbuffer()
    with open(saved, "wb") as f:
        f.write(raw)
    st.session_state["last_saved_path"] = str(saved)

    # Read into pandas
    buf = io.BytesIO(raw)
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(buf)
    elif ext == "csv":
        df = pd.read_csv(buf)
    else:
        # Best-effort: try CSV first, then Excel
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

    # --- HERO BAND ---
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
            <h2>Your Data Story</h2>
            <p>This view answers: <em>"what does this dataset actually say?"</em> — in business terms.
            We are looking at <span class="kx-hero-accent">{record_str}</span> records across
            <span class="kx-hero-accent">{col_str}</span> columns, with overall completeness of
            <span class="kx-hero-accent">{quality_pct:.1f}%</span>.{monetary_line}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- HERO KPI CARDS ---
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

    # --- NARRATIVE: FUNCTIONAL SUMMARY ---
    st.markdown(" ")
    st.markdown('<div class="kx-card"><div class="kx-card-title">Functional summary</div></div>', unsafe_allow_html=True)

    if "exec_summary" not in st.session_state:
        with st.spinner("KiteIQX is reading the data..."):
            st.session_state.exec_summary = a.ai_executive_summary()
    st.markdown(f'<div class="kx-callout">{st.session_state.exec_summary}</div>', unsafe_allow_html=True)

    refresh_col, _ = st.columns([1, 5])
    if refresh_col.button("Regenerate narrative", key="regen_summary"):
        st.session_state.pop("exec_summary", None)
        st.session_state.pop("takeaways", None)
        st.rerun()

    # --- TOP 3 TAKEAWAYS ---
    st.markdown(" ")
    st.markdown('<div class="kx-card"><div class="kx-card-title">Top 3 takeaways</div></div>', unsafe_allow_html=True)
    if "takeaways" not in st.session_state:
        with st.spinner("Identifying the headline insights..."):
            st.session_state.takeaways = a.ai_top_takeaways()
    for i, t in enumerate(st.session_state.takeaways, 1):
        st.markdown(
            f"""<div class="kx-takeaway"><div class="kx-takeaway-num">{i}</div>
            <div class="kx-takeaway-body">{t}</div></div>""", unsafe_allow_html=True)

    # --- SUPPORTING VISUALS ---
    st.markdown(" ")
    st.markdown('<div class="kx-card"><div class="kx-card-title">Visuals supporting the story</div></div>', unsafe_allow_html=True)
    left, right = st.columns(2)

    # 1. Top categorical breakdown
    with left:
        if a.categorical_cols:
            cat = a.categorical_cols[0]
            top = a.df[cat].value_counts().head(8).reset_index()
            top.columns = [cat, "count"]
            fig = px.bar(top, x=cat, y="count", title=f"Composition: {cat}")
            fig.update_traces(marker_color="#0a2540")
            st.plotly_chart(kx_apply_theme(fig), use_container_width=True)
        elif a.numeric_cols:
            c = a.numeric_cols[0]
            fig = px.histogram(a.df, x=c, nbins=30, title=f"Distribution: {c}")
            fig.update_traces(marker_color="#0a2540")
            st.plotly_chart(kx_apply_theme(fig), use_container_width=True)

    # 2. Time trend OR top numeric distribution
    with right:
        if a.datetime_cols and a.numeric_cols:
            d, v = a.datetime_cols[0], (a.money_cols[0] if a.money_cols else a.numeric_cols[0])
            ts = a.df[[d, v]].dropna().sort_values(d)
            fig = px.line(ts, x=d, y=v, title=f"{v} over time")
            fig.update_traces(line_color="#c79a3a")
            st.plotly_chart(kx_apply_theme(fig), use_container_width=True)
        elif len(a.numeric_cols) >= 2:
            x, y = a.numeric_cols[:2]
            fig = px.scatter(a.df, x=x, y=y, title=f"{y} vs {x}")
            fig.update_traces(marker=dict(color="#0a2540", opacity=0.55))
            st.plotly_chart(kx_apply_theme(fig), use_container_width=True)

    # --- WHAT TO DO NEXT ---
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

    # Duplicates
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

    # Missing data
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

    # Outliers
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

    # Detailed tables
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
# MAIN
# ============================================================

def initialize_analytics(df: pd.DataFrame, model: str):
    llm = None
    if GROQ_AVAILABLE and GROQ_API_KEY:
        try:
            llm = GroqLLM(api_key=GROQ_API_KEY, model=model)
        except Exception as e:
            st.error(f"Could not initialize Groq: {e}")
    st.session_state.analytics = UniversalAnalytics(df, llm)
    st.session_state.viz_engine = VizEngine(st.session_state.analytics)
    # Reset story cache so dashboard re-runs against new data
    st.session_state.pop("exec_summary", None)
    st.session_state.pop("takeaways", None)
    st.session_state.pop("ai_last_response", None)


def main():
    render_header()

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

        model = st.selectbox("AI model", ["llama-3.1-70b-versatile", "llama-3.1-8b-instant",
                                          "mixtral-8x7b-32768", "gemma2-9b-it"])

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
                    if up:
                        df = load_uploaded_file(up)
                        st.success(f"Loaded **{up.name}** ({len(df):,} rows).")
                    elif url:
                        r = requests.get(url, timeout=30)
                        df = pd.read_csv(io.StringIO(r.text))
                        st.success(f"Loaded from URL ({len(df):,} rows).")
                    if df is not None:
                        initialize_analytics(df, model)
                except Exception as e:
                    st.error(f"Load failed: {e}")
        with c2:
            if st.button("Use demo") and demo:
                try:
                    df = generate_demo(demo)
                    initialize_analytics(df, model)
                    st.success(f"Loaded demo: {demo}.")
                except Exception as e:
                    st.error(f"Demo failed: {e}")

        # Show last saved path
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


if __name__ == "__main__":
    main()
