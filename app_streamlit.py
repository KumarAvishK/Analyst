# ================================
# IMPORTS
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

# LangChain (modified)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="🏈 Intelligent NFL Analytics Copilot", layout="wide")

# Load API key from secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

# ================================
# LLM SETUP (Groq)
# ================================
def get_llm():
    return ChatOpenAI(
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        model="llama3-70b-8192",
        temperature=0.1,
        max_tokens=2000
    )

# ================================
# EMBEDDINGS (FREE)
# ================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ================================
# KNOWLEDGE BASE
# ================================
def build_vector_store(df):
    docs = []

    # Dataset summary
    docs.append(Document(page_content=f"""
    Dataset Overview:
    Rows: {len(df)}
    Columns: {len(df.columns)}
    {', '.join(df.columns)}
    """))

    # Column summaries
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            text = f"{col}: mean={df[col].mean():.2f}, max={df[col].max():.2f}, min={df[col].min():.2f}"
        else:
            text = f"{col}: top={df[col].mode()[0] if not df[col].mode().empty else 'NA'}"
        docs.append(Document(page_content=text))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    split_docs = splitter.split_documents(docs)

    return FAISS.from_documents(split_docs, get_embeddings())

# ================================
# ANALYTICS ENGINE (UNCHANGED LOGIC)
# ================================
def generate_insights(df):
    insights = []

    insights.append(f"📊 Total Records: {len(df)}")

    spend_cols = [c for c in df.columns if "spend" in c.lower()]
    if spend_cols:
        df["total_spend"] = df[spend_cols].sum(axis=1)
        insights.append(f"💰 Total Revenue: {df['total_spend'].sum():,.0f}")
        insights.append(f"📈 Avg Customer Value: {df['total_spend'].mean():.0f}")

    return insights

# ================================
# AI RESPONSE
# ================================
def ask_ai(query, vector_store):
    llm = get_llm()

    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a senior analytics consultant.

    Context:
    {context}

    Question:
    {query}

    Provide concise, business-focused insights.
    """

    return llm.predict(prompt)

# ================================
# UI STARTS (UNCHANGED STYLE)
# ================================
st.markdown('<h1 style="text-align: center; color: #1f77b4;">🏈 Intelligent NFL Analytics Copilot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Data Controls")

    if "df" not in st.session_state:
        st.session_state.df = None

    file = st.file_uploader("📁 Upload CSV", type="csv")

    if st.button("📊 Load Data"):
        if file:
            st.session_state.df = pd.read_csv(file)
            st.success("✅ Data Loaded")

    if st.button("🎲 Demo Data"):
        np.random.seed(42)
        st.session_state.df = pd.DataFrame({
            "age": np.random.randint(18, 70, 500),
            "team": np.random.choice(["A","B","C"], 500),
            "spend": np.random.uniform(50, 400, 500)
        })
        st.success("✅ Demo Loaded")

# ================================
# MAIN APP
# ================================
if st.session_state.df is not None:

    df = st.session_state.df

    # Build vector store once
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = build_vector_store(df)

    # Tabs
    tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 AI Assistant"])

    # ============================
    # DASHBOARD
    # ============================
    with tab1:
        st.subheader("📊 Insights")

        insights = generate_insights(df)
        for i in insights:
            st.write(i)

        st.subheader("📈 Visualization")
        num_cols = df.select_dtypes(include=np.number).columns

        if len(num_cols) > 0:
            col = st.selectbox("Select column", num_cols)
            fig = px.histogram(df, x=col)
            st.plotly_chart(fig, use_container_width=True)

    # ============================
    # AI ASSISTANT
    # ============================
    with tab2:
        st.subheader("🤖 AI Assistant")

        if not GROQ_API_KEY:
            st.error("⚠️ GROQ_API_KEY not set in Streamlit secrets")
        else:
            query = st.text_input("Ask anything about your data")

            if query:
                with st.spinner("Thinking..."):
                    response = ask_ai(query, st.session_state.vector_store)
                    st.write(response)

else:
    st.info("Upload a dataset to begin")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>AI Analytics Copilot</div>",
    unsafe_allow_html=True
)
