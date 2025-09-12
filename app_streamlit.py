# nfl_analytics_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from typing import List, Dict, Optional

# Attempt to import OpenAI/LangChain - graceful fallback if missing
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="🏈 NFL Fan Analytics Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .nfl-card {
        background-color: #e8f5e8;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================
# NFL Data Detection and Analytics
# ====================================================

class NFLDataDetector:
    """Detects and analyzes NFL fan datasets"""
    
    NFL_INDICATORS = {
        'fan_id': ['nfl_fan_id', 'fan_id', 'customer_id', 'cust_id'],
        'digital_platforms': ['digital_fantasy', 'digital_redzone', 'digital_nflplus', 'redzone', 'nflplus'],
        'ticketing': ['tix_prim', 'tix_sec', 'sth', 'sgl', 'post', 'season_ticket', 'single_ticket'],
        'segments': ['nfl_segment', 'cluster', 'high_value', 'fantasy', 'gamer', 'digital_user'],
        'engagement': ['email_open', 'minutes_watched', 'games_attended', 'digital_session'],
        'revenue': ['shop_rev', 'cc_rev', 'sgl_rev', 'season_rev', 'sec_rev'],
        'demographics': ['gender', 'age', 'tenure', 'zip_code', 'state'],
        'behavioral_flags': ['_ind', 'active_', '_flag']
    }
    
    @classmethod
    def detect_nfl_data(cls, df):
        """Detect if dataset contains NFL fan data"""
        columns = [col.lower() for col in df.columns]
        score = 0
        detected_categories = []
        
        for category, patterns in cls.NFL_INDICATORS.items():
            matches = sum(1 for pattern in patterns if any(pattern in col for col in columns))
            if matches > 0:
                score += matches
                detected_categories.append(category)
        
        is_nfl_data = score >= 5 and len(detected_categories) >= 3
        
        return {
            'is_nfl_data': is_nfl_data,
            'confidence_score': min(score / 20.0, 1.0),
            'detected_categories': detected_categories,
            'total_indicators': score
        }
    
    @classmethod
    def categorize_nfl_columns(cls, df):
        """Categorize NFL columns by business function"""
        columns = df.columns.tolist()
        categories = {
            'fan_identifiers': [],
            'digital_engagement': [],
            'ticketing_behavior': [],
            'segmentation': [],
            'revenue_metrics': [],
            'demographics': [],
            'behavioral_flags': [],
            'geographic': []
        }
        
        for col in columns:
            col_lower = col.lower()
            
            if any(pattern in col_lower for pattern in ['fan_id', 'cust_id', 'email_addr']):
                categories['fan_identifiers'].append(col)
            elif any(pattern in col_lower for pattern in ['digital_', 'fantasy', 'redzone', 'nflplus', 'minutes_watched']):
                categories['digital_engagement'].append(col)
            elif any(pattern in col_lower for pattern in ['tix_', 'ticket', 'sth', 'sgl']):
                categories['ticketing_behavior'].append(col)
            elif any(pattern in col_lower for pattern in ['segment', 'cluster', 'high_value']):
                categories['segmentation'].append(col)
            elif any(pattern in col_lower for pattern in ['rev']) and 'ltv' not in col_lower:
                categories['revenue_metrics'].append(col)
            elif any(pattern in col_lower for pattern in ['age', 'gender', 'tenure']):
                categories['demographics'].append(col)
            elif any(pattern in col_lower for pattern in ['zip_code', 'city', 'state']):
                categories['geographic'].append(col)
            elif col_lower.endswith('_ind') or col_lower.endswith('_flag') or 'active_' in col_lower:
                categories['behavioral_flags'].append(col)
        
        return {k: v for k, v in categories.items() if v}


class NFLAnalytics:
    """Specialized analytics for NFL fan data"""
    
    def __init__(self, df, nfl_categories):
        self.df = df
        self.categories = nfl_categories
    
    def analyze_fan_segments(self):
        """Analyze NFL fan segments"""
        segment_cols = [col for col in self.categories.get('segmentation', []) 
                       if 'segment' in col.lower() and col in self.df.columns]
        
        if not segment_cols:
            return None, "No fan segment columns detected"
        
        main_segment_col = segment_cols[0]
        if main_segment_col in self.df.columns:
            segment_dist = self.df[main_segment_col].value_counts()
            
            fig = px.pie(
                values=segment_dist.values,
                names=segment_dist.index,
                title=f"🏈 NFL Fan Segments Distribution ({main_segment_col})",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            return fig, f"Analyzed fan segments with {len(segment_dist)} unique segments"
        
        return None, "Unable to create segment analysis"
    
    def analyze_digital_engagement(self):
        """Analyze digital platform usage"""
        digital_cols = self.categories.get('digital_engagement', [])
        numeric_digital = [col for col in digital_cols if col in self.df.columns and 
                          pd.api.types.is_numeric_dtype(self.df[col])]
        
        if len(numeric_digital) < 2:
            return None, "Insufficient digital engagement data"
        
        engagement_data = []
        for col in numeric_digital[:5]:
            clean_name = col.replace('digital_', '').replace('_qty_prevssn', '').replace('_lst_ssn', '')
            avg_engagement = self.df[col].mean()
            engagement_data.append({'Platform': clean_name.title(), 'Average_Usage': avg_engagement})
        
        if engagement_data:
            df_engagement = pd.DataFrame(engagement_data)
            fig = px.bar(df_engagement, x='Platform', y='Average_Usage',
                        title="🏈 Digital Platform Engagement Comparison",
                        color_discrete_sequence=px.colors.qualitative.Set2)
            return fig, f"Analyzed {len(numeric_digital)} digital engagement metrics"
        
        return None, "Unable to create digital engagement analysis"
    
    def analyze_revenue_streams(self):
        """Analyze revenue patterns"""
        revenue_cols = self.categories.get('revenue_metrics', [])
        numeric_revenue = [col for col in revenue_cols if col in self.df.columns and 
                          pd.api.types.is_numeric_dtype(self.df[col])]
        
        if not numeric_revenue:
            return None, "No revenue data available"
        
        revenue_data = {}
        for col in numeric_revenue:
            revenue_data[col] = self.df[col].sum()
        
        if revenue_data:
            fig = px.pie(
                values=list(revenue_data.values()),
                names=[col.replace('_rev', '').title() for col in revenue_data.keys()],
                title="🏈 Revenue Stream Breakdown",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            total_revenue = sum(revenue_data.values())
            return fig, f"Total revenue analyzed: ${total_revenue:,.2f}"
        
        return None, "Unable to create revenue analysis"


# ====================================================
# Simplified AI Assistant (without FAISS)
# ====================================================

class SimpleNFLAssistant:
    """Simplified AI assistant for NFL analytics without vector dependencies"""
    
    def __init__(self, df, nfl_detection=None, nfl_categories=None, api_key=None):
        self.df = df
        self.nfl_detection = nfl_detection or {}
        self.nfl_categories = nfl_categories or {}
        self.is_nfl_data = nfl_detection.get('is_nfl_data', False) if nfl_detection else False
        
        # Initialize OpenAI client if available
        self.client = None
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                st.warning(f"Could not initialize OpenAI client: {e}")
    
    def get_data_context(self):
        """Get contextual information about the dataset"""
        context = []
        
        # Basic data info
        context.append(f"Dataset contains {len(self.df):,} records with {len(self.df.columns)} columns")
        
        # NFL-specific context
        if self.is_nfl_data:
            context.append(f"NFL fan data detected with {self.nfl_detection.get('confidence_score', 0):.1%} confidence")
            context.append(f"Categories: {', '.join(self.nfl_detection.get('detected_categories', []))}")
        
        # Column types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        context.append(f"Numeric columns: {len(numeric_cols)}")
        context.append(f"Categorical columns: {len(categorical_cols)}")
        
        # Key metrics
        if numeric_cols:
            for col in numeric_cols[:3]:
                mean_val = self.df[col].mean()
                context.append(f"{col}: average {mean_val:.2f}")
        
        return "\n".join(context)
    
    def analyze_query(self, query):
        """Analyze user query and provide insights"""
        query_lower = query.lower()
        
        # NFL-specific analysis
        if self.is_nfl_data:
            if any(word in query_lower for word in ['segment', 'fan', 'customer']):
                return self._analyze_fan_segments()
            elif any(word in query_lower for word in ['digital', 'platform', 'engagement']):
                return self._analyze_digital_metrics()
            elif any(word in query_lower for word in ['revenue', 'money', 'spending']):
                return self._analyze_revenue()
            elif any(word in query_lower for word in ['geographic', 'location', 'state']):
                return self._analyze_geographic()
        
        # General analysis
        if any(word in query_lower for word in ['correlation', 'relationship']):
            return self._analyze_correlations()
        elif any(word in query_lower for word in ['summary', 'overview']):
            return self._provide_summary()
        else:
            return self._provide_general_insights()
    
    def _analyze_fan_segments(self):
        """Analyze fan segmentation"""
        segment_cols = self.nfl_categories.get('segmentation', [])
        
        if not segment_cols:
            return "No fan segment data available in this dataset."
        
        insights = ["🎯 Fan Segmentation Analysis:"]
        
        for col in segment_cols[:3]:
            if col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64'] and self.df[col].max() <= 1:
                    # Binary segment flag
                    active_pct = (self.df[col] == 1).mean() * 100
                    clean_name = col.replace('nfl_segment_', '').replace('_', ' ').title()
                    insights.append(f"• {clean_name}: {active_pct:.1f}% of fans")
                else:
                    # Categorical segments
                    top_segment = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "Unknown"
                    top_pct = (self.df[col] == top_segment).mean() * 100
                    insights.append(f"• Largest segment in {col}: {top_segment} ({top_pct:.1f}%)")
        
        return "\n".join(insights)
    
    def _analyze_digital_metrics(self):
        """Analyze digital engagement"""
        digital_cols = self.nfl_categories.get('digital_engagement', [])
        
        if not digital_cols:
            return "No digital engagement data available."
        
        insights = ["📱 Digital Engagement Analysis:"]
        
        for col in digital_cols[:5]:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                avg_usage = self.df[col].mean()
                adoption_rate = (self.df[col] > 0).mean() * 100
                clean_name = col.replace('digital_', '').replace('_qty_prevssn', '').replace('_lst_ssn', '').title()
                insights.append(f"• {clean_name}: {adoption_rate:.1f}% adoption, avg usage {avg_usage:.1f}")
        
        return "\n".join(insights)
    
    def _analyze_revenue(self):
        """Analyze revenue patterns"""
        revenue_cols = self.nfl_categories.get('revenue_metrics', [])
        
        if not revenue_cols:
            return "No revenue data available."
        
        insights = ["💰 Revenue Analysis:"]
        
        total_revenue = 0
        for col in revenue_cols:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                col_revenue = self.df[col].sum()
                avg_revenue = self.df[col].mean()
                total_revenue += col_revenue
                clean_name = col.replace('_rev', '').title()
                insights.append(f"• {clean_name}: ${col_revenue:,.0f} total (${avg_revenue:.0f} avg per fan)")
        
        if total_revenue > 0:
            insights.append(f"• Total Revenue: ${total_revenue:,.0f}")
        
        return "\n".join(insights)
    
    def _analyze_geographic(self):
        """Analyze geographic distribution"""
        geo_cols = self.nfl_categories.get('geographic', [])
        
        if not geo_cols:
            return "No geographic data available."
        
        insights = ["🗺️ Geographic Analysis:"]
        
        for col in geo_cols:
            if col in self.df.columns:
                if col.lower() == 'state':
                    top_states = self.df[col].value_counts().head(5)
                    insights.append("Top 5 states by fan count:")
                    for state, count in top_states.items():
                        pct = (count / len(self.df)) * 100
                        insights.append(f"  • {state}: {count:,} fans ({pct:.1f}%)")
                else:
                    unique_count = self.df[col].nunique()
                    insights.append(f"• {col}: {unique_count} unique values")
        
        return "\n".join(insights)
    
    def _analyze_correlations(self):
        """Find strong correlations"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return "Need at least 2 numeric columns for correlation analysis."
        
        corr_matrix = self.df[numeric_cols].corr()
        strong_corrs = []
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.6:
                    strong_corrs.append((numeric_cols[i], numeric_cols[j], corr_val))
        
        if strong_corrs:
            insights = ["🔗 Strong Correlations Found:"]
            for col1, col2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                direction = "positive" if corr > 0 else "negative"
                insights.append(f"• {col1} ↔ {col2}: {direction} correlation ({corr:.3f})")
            return "\n".join(insights)
        else:
            return "No strong correlations (>0.6) found between numeric variables."
    
    def _provide_summary(self):
        """Provide dataset summary"""
        insights = ["📊 Dataset Summary:"]
        insights.append(f"• {len(self.df):,} total records")
        insights.append(f"• {len(self.df.columns)} columns")
        
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        insights.append(f"• Data completeness: {100-missing_pct:.1f}%")
        
        if self.is_nfl_data:
            insights.append(f"• NFL fan data confidence: {self.nfl_detection.get('confidence_score', 0):.1%}")
            insights.append(f"• Categories detected: {len(self.nfl_detection.get('detected_categories', []))}")
        
        return "\n".join(insights)
    
    def _provide_general_insights(self):
        """Provide general data insights"""
        insights = ["💡 General Data Insights:"]
        
        # Numeric column insights
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            insights.append(f"• {len(numeric_cols)} numeric metrics available for analysis")
            
            # Find the column with highest variability
            cv_scores = []
            for col in numeric_cols:
                if self.df[col].std() > 0:
                    cv = self.df[col].std() / self.df[col].mean()
                    cv_scores.append((col, cv))
            
            if cv_scores:
                most_variable = max(cv_scores, key=lambda x: x[1])
                insights.append(f"• Most variable metric: {most_variable[0]}")
        
        # Categorical insights
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            insights.append(f"• {len(categorical_cols)} categorical dimensions for segmentation")
        
        return "\n".join(insights)
    
    def chat_with_ai(self, query):
        """Chat with OpenAI if available, otherwise use rule-based analysis"""
        if self.client:
            try:
                # Get data context
                context = self.get_data_context()
                
                # Create system prompt
                system_prompt = f"""You are an expert NFL fan analytics assistant. You have access to a dataset with the following context:

{context}

The dataset {'is NFL fan data' if self.is_nfl_data else 'appears to be general business data'}.

Provide helpful, specific insights based on the user's question. Be conversational but data-driven. Use emojis appropriately."""

                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                
                return response.choices[0].message.content
            
            except Exception as e:
                st.warning(f"AI service temporarily unavailable: {e}")
                return self.analyze_query(query)
        else:
            # Fallback to rule-based analysis
            return self.analyze_query(query)


# ====================================================
# Main Application
# ====================================================

def create_nfl_dashboard(df, nfl_analytics):
    """Create comprehensive NFL analytics dashboard"""
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Fans", f"{len(df):,}")
    
    with col2:
        if 'shop_rev' in df.columns:
            avg_revenue = df['shop_rev'].mean()
            st.metric("Avg Shop Revenue", f"${avg_revenue:.0f}")
        else:
            st.metric("Data Completeness", f"{(1 - df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.1f}%")
    
    with col3:
        if 'minutes_watched' in df.columns:
            avg_watch = df['minutes_watched'].mean()
            st.metric("Avg Minutes Watched", f"{avg_watch:.0f}")
        else:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Metrics", numeric_cols)
    
    with col4:
        if 'games_attended' in df.columns:
            avg_games = df['games_attended'].mean()
            st.metric("Avg Games Attended", f"{avg_games:.1f}")
        else:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("Categories", categorical_cols)
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        # Fan segments analysis
        fig1, msg1 = nfl_analytics.analyze_fan_segments()
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info(msg1)
    
    with col2:
        # Digital engagement analysis
        fig2, msg2 = nfl_analytics.analyze_digital_engagement()
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Fallback chart
            if 'age' in df.columns:
                fig = px.histogram(df, x='age', title="🏈 Fan Age Distribution", nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(msg2)
    
    # Revenue analysis
    fig3, msg3 = nfl_analytics.analyze_revenue_streams()
    if fig3:
        st.plotly_chart(fig3, use_container_width=True)
    else:
        # Alternative analysis - correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 2:
            corr_matrix = df[numeric_cols[:8]].corr()  # Limit for readability
            fig = px.imshow(corr_matrix, 
                          title="🏈 NFL Metrics Correlation Matrix",
                          color_continuous_scale='RdBu_r',
                          aspect="auto")
            st.plotly_chart(fig, use_container_width=True)


def main():
    # Header
    st.markdown('<div class="main-header">🏈 NFL Fan Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Advanced analytics for NFL fan data with AI assistance</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Data Configuration")
        
        # File upload
        uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv")
        
        # URL input
        data_url = st.text_input("🌐 Or enter CSV URL:")
        
        # Demo data
        if st.button("🎲 Load NFL Demo Data"):
            # Create demo NFL dataset
            np.random.seed(42)
            demo_data = pd.DataFrame({
                'nfl_fan_id': [f'FAN{i:06d}' for i in range(1, 1001)],
                'email_addr': [f'fan{i}@email.com' for i in range(1, 1001)],
                'zip_code': np.random.randint(10000, 99999, 1000),
                'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 1000),
                'state': np.random.choice(['NY', 'CA', 'IL', 'TX', 'FL'], 1000),
                'digital_fantasy_page_qty_prevssn': np.random.poisson(5, 1000),
                'digital_redzone_lst_ssn': np.random.binomial(1, 0.3, 1000),
                'digital_nflplus_lst_ssn': np.random.binomial(1, 0.2, 1000),
                'email_open_12mo': np.random.poisson(10, 1000),
                'merch_qty_12mo': np.random.poisson(2, 1000),
                'tix_prim_sgl_qty_prevssn': np.random.poisson(1.2, 1000),
                'gender': np.random.choice(['M', 'F'], 1000),
                'age': np.random.randint(18, 75, 1000),
                'tenure': np.random.randint(1, 25, 1000),
                'nfl_segment_high_value': np.random.binomial(1, 0.15, 1000),
                'nfl_segment_fantasy': np.random.binomial(1, 0.35, 1000),
                'nfl_segment_digital_user': np.random.binomial(1, 0.4, 1000),
                'shop_rev': np.random.gamma(2, 50, 1000),
                'cc_rev': np.random.gamma(1.5, 30, 1000),
                'minutes_watched': np.random.gamma(3, 120, 1000),
                'games_attended': np.random.poisson(2.5, 1000),
                'active_1yr_flag': np.random.binomial(1, 0.7, 1000),
                'fantasy_ind': np.random.binomial(1, 0.35, 1000),
                'digital_ind': np.random.binomial(1, 0.6, 1000)
            })
            
            st.session_state['df'] = demo_data
            st.success("✅ NFL demo data loaded!")
        
        # Load data button
        if st.button("📊 Load Data", type="primary"):
            try:
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    st.session_state['df'] = df
                    st.success("✅ Data loaded successfully!")
                elif data_url:
                    response = requests.get(data_url, timeout=30)
                    df = pd.read_csv(io.StringIO(response.text))
                    st.session_state['df'] = df
                    st.success("✅ Data loaded from URL!")
                else:
                    st.warning("Please upload a file or enter a URL")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    # Main content
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Detect NFL data
        nfl_detection = NFLDataDetector.detect_nfl_data(df)
        nfl_categories = NFLDataDetector.categorize_nfl_columns(df)
        
        # Show detection results
        if nfl_detection['is_nfl_data']:
            st.markdown(f"""
            <div class="nfl-card">
                <h4>🏈 NFL Fan Data Detected!</h4>
                <p><strong>Confidence:</strong> {nfl_detection['confidence_score']:.1%}</p>
                <p><strong>Categories:</strong> {', '.join(nfl_detection['detected_categories'])}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 AI Assistant", "🔍 Data Explorer", "📈 Custom Analysis"])
        
        with tab1:
            st.subheader("📊 Analytics Dashboard")
            
            if nfl_detection['is_nfl_data']:
                nfl_analytics = NFLAnalytics(df, nfl_categories)
                create_nfl_dashboard(df, nfl_analytics)
            else:
                # General dashboard
                col1, col2 = st.columns(2)
                
                with col1:
                    # Basic stats
                    st.metric("Total Records", f"{len(df):,}")
                    st.metric("Columns", len(df.columns))
                    st.metric("Missing Data", f"{df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.1f}%")
                
                with col2:
                    # Data types
                    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                    categorical_cols = len(df.select_dtypes(include=['object']).columns)
                    st.metric("Numeric Columns", numeric_cols)
                    st.metric("Categorical Columns", categorical_cols)
                
                # Create general visualizations
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                
                if len(numeric_columns) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Correlation heatmap
                        corr_matrix = df[numeric_columns[:8]].corr()
                        fig = px.imshow(corr_matrix, 
                                      title="Correlation Matrix",
                                      color_continuous_scale='RdBu_r',
                                      aspect="auto")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Distribution of first numeric column
                        fig = px.histogram(df, x=numeric_columns[0], 
                                         title=f"Distribution of {numeric_columns[0]}")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Categorical analysis
                if categorical_columns:
                    fig = px.bar(df[categorical_columns[0]].value_counts().head(10).reset_index(),
                               x='index', y=categorical_columns[0],
                               title=f"Top 10 {categorical_columns[0]} Categories")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🤖 AI Assistant")
            
            # API Key input
            api_key = st.text_input("🔑 OpenAI API Key (Optional)", type="password", 
                                   help="Enter your OpenAI API key for enhanced AI responses")
            
            # Initialize assistant
            if 'assistant' not in st.session_state:
                st.session_state['assistant'] = SimpleNFLAssistant(
                    df, nfl_detection, nfl_categories, api_key
                )
            
            # Update assistant if API key changes
            if api_key and st.session_state['assistant'].client is None:
                st.session_state['assistant'] = SimpleNFLAssistant(
                    df, nfl_detection, nfl_categories, api_key
                )
            
            # Quick action buttons
            st.markdown("### 💡 Quick Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Dataset Summary", use_container_width=True):
                    response = st.session_state['assistant'].analyze_query("Give me a summary of this dataset")
                    st.markdown(f'<div class="insight-box">{response}</div>', unsafe_allow_html=True)
            
            with col2:
                if st.button("🔗 Find Correlations", use_container_width=True):
                    response = st.session_state['assistant'].analyze_query("Show me correlations in the data")
                    st.markdown(f'<div class="insight-box">{response}</div>', unsafe_allow_html=True)
            
            with col3:
                if nfl_detection['is_nfl_data']:
                    if st.button("🏈 Fan Segments", use_container_width=True):
                        response = st.session_state['assistant'].analyze_query("Analyze fan segments")
                        st.markdown(f'<div class="insight-box">{response}</div>', unsafe_allow_html=True)
                else:
                    if st.button("📈 Key Insights", use_container_width=True):
                        response = st.session_state['assistant'].analyze_query("What are the key insights from this data?")
                        st.markdown(f'<div class="insight-box">{response}</div>', unsafe_allow_html=True)
            
            # NFL-specific buttons
            if nfl_detection['is_nfl_data']:
                st.markdown("### 🏈 NFL-Specific Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📱 Digital Engagement", use_container_width=True):
                        response = st.session_state['assistant'].analyze_query("Analyze digital engagement")
                        st.markdown(f'<div class="insight-box">{response}</div>', unsafe_allow_html=True)
                
                with col2:
                    if st.button("💰 Revenue Analysis", use_container_width=True):
                        response = st.session_state['assistant'].analyze_query("Show revenue patterns")
                        st.markdown(f'<div class="insight-box">{response}</div>', unsafe_allow_html=True)
                
                with col3:
                    if st.button("🗺️ Geographic Distribution", use_container_width=True):
                        response = st.session_state['assistant'].analyze_query("Analyze geographic distribution")
                        st.markdown(f'<div class="insight-box">{response}</div>', unsafe_allow_html=True)
            
            # Chat interface
            st.markdown("---")
            st.markdown("### 💬 Ask Questions About Your Data")
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
            
            # Chat input
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask me anything about your data:",
                    placeholder="e.g., 'What drives customer loyalty?' or 'Show me revenue patterns'",
                    key="chat_input"
                )
            
            with col2:
                send_button = st.button("Send", type="primary")
                clear_button = st.button("Clear Chat")
            
            if clear_button:
                st.session_state['chat_history'] = []
                st.rerun()
            
            if send_button and user_input:
                # Add user message
                st.session_state['chat_history'].append(("user", user_input))
                
                # Get AI response
                with st.spinner("🤔 Analyzing..."):
                    if api_key:
                        response = st.session_state['assistant'].chat_with_ai(user_input)
                    else:
                        response = st.session_state['assistant'].analyze_query(user_input)
                
                st.session_state['chat_history'].append(("assistant", response))
                st.rerun()
            
            # Display chat history
            if st.session_state['chat_history']:
                st.markdown("### 💬 Conversation")
                
                for role, message in st.session_state['chat_history'][-10:]:  # Show last 10 messages
                    if role == "user":
                        st.markdown(f"""
                        <div style="background-color:#e8f4fd; padding:10px; border-radius:8px; margin:8px 0;">
                            <strong>You:</strong> {message}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color:#f1f1f1; padding:10px; border-radius:8px; margin:8px 0;">
                            <strong>Assistant:</strong><br>{message}
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("🔍 Data Explorer")
            
            # Data preview
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("#### 📋 Data Preview")
            
            with col2:
                show_rows = st.slider("Rows to display", 5, 100, 20)
            
            st.dataframe(df.head(show_rows), use_container_width=True)
            
            # Column analysis
            st.markdown("---")
            st.markdown("#### 📊 Column Analysis")
            
            selected_column = st.selectbox("Select column to analyze:", df.columns)
            
            if selected_column:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Data Type", str(df[selected_column].dtype))
                
                with col2:
                    st.metric("Unique Values", df[selected_column].nunique())
                
                with col3:
                    st.metric("Missing Values", df[selected_column].isnull().sum())
                
                # Column visualization
                if pd.api.types.is_numeric_dtype(df[selected_column]):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.markdown("#### 📈 Statistics")
                    st.dataframe(df[selected_column].describe().to_frame().T, use_container_width=True)
                
                else:
                    # Categorical column
                    value_counts = df[selected_column].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(x=value_counts.head(10).index, y=value_counts.head(10).values,
                                   title=f"Top 10 {selected_column} Values")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(values=value_counts.head(8).values, names=value_counts.head(8).index,
                                   title=f"{selected_column} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("📈 Custom Analysis")
            
            # Chart type selection
            chart_type = st.selectbox("Select visualization type:", [
                "Scatter Plot",
                "Bar Chart", 
                "Line Chart",
                "Box Plot",
                "Correlation Heatmap",
                "Distribution Plot"
            ])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_col])
                with col3:
                    color_col = st.selectbox("Color by (optional):", [None] + categorical_cols)
                
                if st.button("Create Scatter Plot"):
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                   title=f"{x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Bar Chart" and categorical_cols and numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    cat_col = st.selectbox("Category:", categorical_cols)
                with col2:
                    num_col = st.selectbox("Value:", numeric_cols)
                
                if st.button("Create Bar Chart"):
                    grouped_data = df.groupby(cat_col)[num_col].mean().reset_index()
                    fig = px.bar(grouped_data, x=cat_col, y=num_col,
                               title=f"Average {num_col} by {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Box Plot" and categorical_cols and numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    cat_col = st.selectbox("Category:", categorical_cols)
                with col2:
                    num_col = st.selectbox("Value:", numeric_cols)
                
                if st.button("Create Box Plot"):
                    fig = px.box(df, x=cat_col, y=num_col,
                               title=f"{num_col} Distribution by {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Correlation Heatmap" and len(numeric_cols) > 1:
                selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:6])
                
                if st.button("Create Heatmap") and selected_cols:
                    corr_matrix = df[selected_cols].corr()
                    fig = px.imshow(corr_matrix, title="Correlation Heatmap",
                                  color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Distribution Plot" and numeric_cols:
                selected_col = st.selectbox("Select column:", numeric_cols)
                
                if st.button("Create Distribution Plot"):
                    fig = px.histogram(df, x=selected_col, marginal="box",
                                     title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Data export
            st.markdown("---")
            st.markdown("### 💾 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download Full Dataset"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name="nfl_analytics_data.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Download Summary Statistics"):
                    summary = df.describe().to_csv()
                    st.download_button(
                        label="📈 Download Summary",
                        data=summary,
                        file_name="data_summary.csv",
                        mime="text/csv"
                    )
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>🏈 Welcome to NFL Fan Analytics Platform!</h2>
            <p style="font-size: 1.3rem; color: #666; margin-bottom: 2rem;">
                Upload your data to unlock powerful NFL fan insights with AI assistance
            </p>
            
            <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; margin: 2rem 0;">
                <h3>✨ Platform Features</h3>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 1rem;">
                    <div style="text-align: center; margin: 1rem;">
                        <h4>🤖 AI Assistant</h4>
                        <p>Ask questions in natural language</p>
                    </div>
                    <div style="text-align: center; margin: 1rem;">
                        <h4>📊 Smart Dashboards</h4>
                        <p>Auto-generated NFL analytics</p>
                    </div>
                    <div style="text-align: center; margin: 1rem;">
                        <h4>🎯 NFL Intelligence</h4>
                        <p>Specialized fan data analysis</p>
                    </div>
                    <div style="text-align: center; margin: 1rem;">
                        <h4>📈 Custom Analysis</h4>
                        <p>Build your own visualizations</p>
                    </div>
                </div>
            </div>
            
            <div style="background-color: #e8f5e8; padding: 1.5rem; border-radius: 8px; margin-top: 2rem;">
                <h4>🚀 Getting Started</h4>
                <p>1. Use the sidebar to upload a CSV file or load demo data</p>
                <p>2. The platform will automatically detect NFL fan data</p>
                <p>3. Explore dashboards, chat with the AI assistant, and create custom analysis</p>
                <p>4. Optional: Add your OpenAI API key for enhanced AI responses</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
