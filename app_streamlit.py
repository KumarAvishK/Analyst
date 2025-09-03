import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, io
import re
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate

# ====================================================
# Enhanced Analytics Class with Intelligence
# ====================================================
class IntelligentNFLAnalytics:
    def __init__(self, df):
        self.df = df
        self.filtered_df = None
        self.insights = {}
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.preprocess()
        self.generate_insights()

    def preprocess(self):
        """Enhanced preprocessing with better data understanding"""
        df = self.df.copy()
        
        # Auto-detect numeric columns and create derived features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Smart total spending calculation
        spend_cols = [col for col in df.columns if 'spend' in col.lower() or 'cost' in col.lower() or 'price' in col.lower()]
        if spend_cols:
            df['total_spend'] = df[spend_cols].sum(axis=1, skipna=True)
        
        # Smart age grouping
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 100], 
                                   labels=['Gen Z (18-25)', 'Millennial (26-35)', 'Gen X (36-50)', 'Boomer (51+)'])
        
        # Auto-detect loyalty/score columns
        score_cols = [col for col in df.columns if 'score' in col.lower() or 'loyalty' in col.lower() or 'rating' in col.lower()]
        if score_cols:
            score_col = score_cols[0]
            df['loyalty_tier'] = pd.cut(df[score_col], 
                                      bins=[df[score_col].min()-1, df[score_col].quantile(0.33), 
                                           df[score_col].quantile(0.67), df[score_col].max()+1], 
                                      labels=['Low Loyalty', 'Medium Loyalty', 'High Loyalty'])
        
        # Auto-detect categorical columns for analysis
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        self.filtered_df = df
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.spend_cols = spend_cols

    def generate_insights(self):
        """Generate intelligent insights from the data"""
        df = self.filtered_df
        insights = {}
        
        # Basic stats
        insights['total_records'] = len(df)
        insights['missing_data_pct'] = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        # Revenue insights
        if 'total_spend' in df.columns:
            insights['total_revenue'] = df['total_spend'].sum()
            insights['avg_customer_value'] = df['total_spend'].mean()
            insights['revenue_std'] = df['total_spend'].std()
            
            # Find high-value segments
            high_spenders = df[df['total_spend'] > df['total_spend'].quantile(0.8)]
            insights['high_spender_count'] = len(high_spenders)
            insights['high_spender_revenue_share'] = (high_spenders['total_spend'].sum() / df['total_spend'].sum()) * 100
        
        # Correlation insights
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            # Find strongest correlations (excluding self-correlation)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            insights['strong_correlations'] = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:3]
        
        # Categorical insights
        for col in self.categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts()
                insights[f'{col}_top_category'] = value_counts.index[0]
                insights[f'{col}_dominance'] = (value_counts.iloc[0] / len(df)) * 100
        
        self.insights = insights

    def create_intelligent_dashboard(self):
        """Create contextually aware dashboard with data stories"""
        df = self.filtered_df
        
        # Determine the most interesting visualizations based on data
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=self.get_intelligent_chart_titles(),
            specs=[[{"type": "bar"}, {"type": "domain"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "heatmap"}]],
            vertical_spacing=0.08
        )
        
        # Chart 1: Most impactful categorical vs numeric relationship
        if self.categorical_cols and self.numeric_cols:
            cat_col = self.find_best_categorical_column()
            num_col = self.find_best_numeric_column()
            
            grouped_data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            colors = self.color_palette[:len(grouped_data)]
            
            fig.add_trace(go.Bar(
                x=grouped_data.index, 
                y=grouped_data.values,
                marker_color=colors,
                name=f'Avg {num_col}',
                text=[f'${x:,.0f}' if 'spend' in num_col.lower() else f'{x:.1f}' for x in grouped_data.values],
                textposition='auto'
            ), row=1, col=1)
        
        # Chart 2: Market share/distribution pie
        if self.categorical_cols:
            main_cat = self.find_most_diverse_category()
            cat_counts = df[main_cat].value_counts().head(8)
            
            fig.add_trace(go.Pie(
                labels=cat_counts.index, 
                values=cat_counts.values,
                hole=0.3,
                marker_colors=self.color_palette[:len(cat_counts)]
            ), row=1, col=2)
        
        # Chart 3: Correlation scatter with trend
        if len(self.numeric_cols) >= 2:
            x_col, y_col = self.find_best_correlation_pair()
            
            fig.add_trace(go.Scatter(
                x=df[x_col], 
                y=df[y_col],
                mode='markers',
                marker=dict(size=8, opacity=0.6, color=self.color_palette[0]),
                name=f'{x_col} vs {y_col}'
            ), row=2, col=1)
            
            # Add trendline
            z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=sorted(df[x_col].dropna()), 
                y=p(sorted(df[x_col].dropna())),
                mode='lines',
                line=dict(color='red', width=2),
                name='Trend'
            ), row=2, col=1)
        
        # Chart 4: Performance comparison
        if 'total_spend' in df.columns and len(self.categorical_cols) > 0:
            cat_col = self.find_best_categorical_column()
            perf_data = df.groupby(cat_col)['total_spend'].agg(['mean', 'count']).reset_index()
            perf_data = perf_data.sort_values('mean', ascending=True)
            
            fig.add_trace(go.Bar(
                y=perf_data[cat_col],
                x=perf_data['mean'],
                orientation='h',
                marker_color=self.color_palette[1],
                text=[f'${x:,.0f} (n={n})' for x, n in zip(perf_data['mean'], perf_data['count'])],
                textposition='auto'
            ), row=2, col=2)
        
        # Chart 5: Distribution analysis
        if self.numeric_cols:
            main_numeric = self.find_best_numeric_column()
            if len(self.categorical_cols) > 0:
                cat_col = self.find_best_categorical_column()
                
                for i, category in enumerate(df[cat_col].unique()[:5]):
                    subset = df[df[cat_col] == category]
                    fig.add_trace(go.Box(
                        y=subset[main_numeric],
                        name=category,
                        marker_color=self.color_palette[i % len(self.color_palette)]
                    ), row=3, col=1)
        
        # Chart 6: Correlation heatmap
        if len(self.numeric_cols) >= 3:
            corr_cols = self.numeric_cols[:6]  # Limit for readability
            corr_data = df[corr_cols].corr()
            
            fig.add_trace(go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_data.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ), row=3, col=2)
        
        fig.update_layout(
            height=1200, 
            showlegend=True, 
            title_text="🏈 Intelligent NFL Fan Analytics Dashboard",
            title_x=0.5,
            font=dict(size=12)
        )
        
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    def get_intelligent_chart_titles(self):
        """Generate contextual chart titles based on data insights"""
        titles = [
            f"Revenue by {self.find_best_categorical_column()}" if self.categorical_cols else "Performance Analysis",
            f"{self.find_most_diverse_category()} Distribution" if self.categorical_cols else "Market Share",
            "Revenue Correlation Analysis" if 'total_spend' in self.filtered_df.columns else "Correlation Analysis",
            "Performance Ranking",
            f"{self.find_best_numeric_column()} Distribution by Category" if self.numeric_cols else "Distribution Analysis",
            "Feature Correlation Matrix"
        ]
        return titles

    def find_best_categorical_column(self):
        """Find the most analytically interesting categorical column"""
        if not self.categorical_cols:
            return None
        
        # Prefer columns with moderate cardinality (not too many, not too few categories)
        scores = {}
        for col in self.categorical_cols:
            if col in self.filtered_df.columns:
                unique_count = self.filtered_df[col].nunique()
                # Sweet spot: 2-8 categories
                if 2 <= unique_count <= 8:
                    scores[col] = 10 - abs(5 - unique_count)  # Prefer ~5 categories
                else:
                    scores[col] = max(1, 10 - unique_count * 0.5)  # Penalize too many categories
        
        return max(scores.keys(), key=scores.get) if scores else self.categorical_cols[0]

    def find_best_numeric_column(self):
        """Find the most interesting numeric column (usually revenue/spend related)"""
        if not self.numeric_cols:
            return None
        
        # Prioritize spend/revenue columns
        for col in ['total_spend', 'revenue', 'spend', 'cost', 'price']:
            matching_cols = [c for c in self.numeric_cols if col in c.lower()]
            if matching_cols:
                return matching_cols[0]
        
        return self.numeric_cols[0]

    def find_most_diverse_category(self):
        """Find categorical column with good diversity"""
        if not self.categorical_cols:
            return None
        
        diversity_scores = {}
        for col in self.categorical_cols:
            if col in self.filtered_df.columns:
                # Calculate entropy-like measure
                value_counts = self.filtered_df[col].value_counts()
                proportions = value_counts / len(self.filtered_df)
                entropy = -sum(p * np.log(p) for p in proportions if p > 0)
                diversity_scores[col] = entropy
        
        return max(diversity_scores.keys(), key=diversity_scores.get) if diversity_scores else self.categorical_cols[0]

    def find_best_correlation_pair(self):
        """Find the pair with strongest correlation"""
        if len(self.numeric_cols) < 2:
            return self.numeric_cols[0], self.numeric_cols[0] if self.numeric_cols else (None, None)
        
        corr_matrix = self.filtered_df[self.numeric_cols].corr()
        max_corr = 0
        best_pair = (self.numeric_cols[0], self.numeric_cols[1])
        
        for i in range(len(self.numeric_cols)):
            for j in range(i+1, len(self.numeric_cols)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > max_corr:
                    max_corr = corr_val
                    best_pair = (self.numeric_cols[i], self.numeric_cols[j])
        
        return best_pair

    def get_data_story(self):
        """Generate intelligent data story"""
        insights = self.insights
        story = []
        
        story.append(f"📊 **Dataset Overview**: {insights['total_records']:,} records with {insights['missing_data_pct']:.1f}% missing data")
        
        if 'total_revenue' in insights:
            story.append(f"💰 **Revenue**: ${insights['total_revenue']:,.0f} total, ${insights['avg_customer_value']:.0f} avg per customer")
            story.append(f"🎯 **Key Insight**: Top 20% customers generate {insights['high_spender_revenue_share']:.1f}% of revenue ({insights['high_spender_count']} high-value customers)")
        
        if 'strong_correlations' in insights and insights['strong_correlations']:
            corr = insights['strong_correlations'][0]
            story.append(f"🔗 **Strongest Relationship**: {corr[0]} ↔ {corr[1]} (correlation: {corr[2]:.2f})")
        
        # Add category insights
        for col in self.categorical_cols[:2]:  # Limit to avoid clutter
            if f'{col}_dominance' in insights:
                story.append(f"📈 **{col.title()}**: {insights[f'{col}_top_category']} dominates ({insights[f'{col}_dominance']:.1f}% market share)")
        
        return story


# ====================================================
# Enhanced Visualization Engine
# ====================================================
class SmartVizEngine:
    def __init__(self, df):
        self.df = df
        self.color_palette = px.colors.qualitative.Set2

    def auto_visualize(self, user_query):
        """Intelligently create visualizations based on user intent"""
        query = user_query.lower()
        
        # Extract column names mentioned in query
        mentioned_cols = [col for col in self.df.columns if col.lower() in query or 
                         any(word in col.lower() for word in query.split())]
        
        # Intent detection
        if any(word in query for word in ['compare', 'vs', 'versus', 'between']):
            return self.create_comparison_chart(mentioned_cols)
        elif any(word in query for word in ['trend', 'over time', 'timeline', 'evolution']):
            return self.create_trend_chart(mentioned_cols)
        elif any(word in query for word in ['distribution', 'spread', 'histogram']):
            return self.create_distribution_chart(mentioned_cols)
        elif any(word in query for word in ['correlation', 'relationship', 'related']):
            return self.create_correlation_chart(mentioned_cols)
        elif any(word in query for word in ['segment', 'group', 'cluster', 'breakdown']):
            return self.create_segmentation_chart(mentioned_cols)
        else:
            return self.create_smart_default_chart(mentioned_cols)

    def create_comparison_chart(self, cols):
        if len(cols) >= 2:
            # Find best categorical and numeric pair
            cat_col = None
            num_col = None
            
            for col in cols:
                if self.df[col].dtype == 'object' and cat_col is None:
                    cat_col = col
                elif pd.api.types.is_numeric_dtype(self.df[col]) and num_col is None:
                    num_col = col
            
            if cat_col and num_col:
                fig = px.box(self.df, x=cat_col, y=num_col, 
                           title=f"📊 {num_col} Distribution by {cat_col}",
                           color_discrete_sequence=self.color_palette)
                fig.update_layout(height=500)
                return fig, f"Created comparison chart showing {num_col} across {cat_col} categories"
        
        return None, "Need categorical and numeric columns for comparison"

    def create_trend_chart(self, cols):
        # Look for date/time columns
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower()]
        if date_cols and cols:
            date_col = date_cols[0]
            value_col = cols[0] if cols else None
            
            if value_col and pd.api.types.is_numeric_dtype(self.df[value_col]):
                trend_data = self.df.groupby(date_col)[value_col].mean().reset_index()
                fig = px.line(trend_data, x=date_col, y=value_col,
                            title=f"📈 {value_col} Trend Over Time",
                            markers=True)
                fig.update_layout(height=500)
                return fig, f"Created trend analysis for {value_col} over {date_col}"
        
        return None, "Need date/time column for trend analysis"

    def create_distribution_chart(self, cols):
        if cols:
            col = cols[0]
            if pd.api.types.is_numeric_dtype(self.df[col]):
                fig = px.histogram(self.df, x=col, 
                                 title=f"📊 Distribution of {col}",
                                 marginal="box",
                                 color_discrete_sequence=self.color_palette)
                fig.update_layout(height=500)
                return fig, f"Created distribution analysis for {col}"
            else:
                fig = px.bar(self.df[col].value_counts().reset_index(), 
                           x='index', y=col,
                           title=f"📊 {col} Frequency Distribution",
                           color_discrete_sequence=self.color_palette)
                fig.update_layout(height=500)
                return fig, f"Created frequency distribution for {col}"
        
        return None, "Need a column name for distribution analysis"

    def create_correlation_chart(self, cols):
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(self.df[col])]
        if len(numeric_cols) >= 2:
            fig = px.scatter(self.df, x=numeric_cols[0], y=numeric_cols[1],
                           title=f"🔗 {numeric_cols[0]} vs {numeric_cols[1]} Correlation",
                           trendline="ols",
                           color_discrete_sequence=self.color_palette)
            fig.update_layout(height=500)
            
            # Calculate correlation
            corr = self.df[numeric_cols[0]].corr(self.df[numeric_cols[1]])
            return fig, f"Created correlation chart (r={corr:.3f}) between {numeric_cols[0]} and {numeric_cols[1]}"
        
        return None, "Need at least 2 numeric columns for correlation analysis"

    def create_segmentation_chart(self, cols):
        if cols:
            # Create customer segments using K-means if numeric data available
            numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(self.df[col])]
            if len(numeric_cols) >= 2:
                # Simple 3-cluster segmentation
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(self.df[numeric_cols[:2]].fillna(0))
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                plot_df = self.df.copy()
                plot_df['Segment'] = [f'Segment {i+1}' for i in clusters]
                
                fig = px.scatter(plot_df, x=numeric_cols[0], y=numeric_cols[1], 
                               color='Segment',
                               title=f"🎯 Customer Segmentation: {numeric_cols[0]} vs {numeric_cols[1]}",
                               color_discrete_sequence=self.color_palette)
                fig.update_layout(height=500)
                return fig, f"Created customer segmentation based on {numeric_cols[0]} and {numeric_cols[1]}"
        
        return None, "Need numeric columns for segmentation analysis"

    def create_smart_default_chart(self, cols):
        if cols:
            col = cols[0]
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Create histogram with insights
                fig = px.histogram(self.df, x=col, 
                                 title=f"📊 {col} Analysis",
                                 marginal="box",
                                 color_discrete_sequence=self.color_palette)
                
                # Add statistics annotations
                mean_val = self.df[col].mean()
                median_val = self.df[col].median()
                
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                            annotation_text=f"Mean: {mean_val:.1f}")
                fig.add_vline(x=median_val, line_dash="dash", line_color="blue", 
                            annotation_text=f"Median: {median_val:.1f}")
                
                fig.update_layout(height=500)
                return fig, f"Created smart analysis for {col} with key statistics"
            else:
                # Create enhanced bar chart
                counts = self.df[col].value_counts().head(10)
                fig = px.bar(x=counts.index, y=counts.values,
                           title=f"📊 Top {col} Categories",
                           color=counts.values,
                           color_continuous_scale='Blues')
                fig.update_layout(height=500)
                return fig, f"Created top categories analysis for {col}"
        
        return None, "Please specify a column or analysis type"


# ====================================================
# Enhanced Chatbot with Intelligence
# ====================================================
def create_enhanced_agent(llm, df, analytics):
    """Create an enhanced agent with data context"""
    
    # Enhanced system prompt with data understanding
    system_prompt = f"""
    You are an intelligent NFL Fan Analytics Assistant. You have access to a dataset with {len(df)} records and the following key information:

    DATASET OVERVIEW:
    - Columns: {', '.join(df.columns.tolist())}
    - Numeric columns: {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}
    - Categorical columns: {', '.join(df.select_dtypes(include=['object']).columns.tolist())}
    
    KEY INSIGHTS:
    {chr(10).join([f"- {insight}" for insight in analytics.get_data_story()])}
    
    CAPABILITIES:
    1. Answer questions about the data with specific numbers and insights
    2. Perform statistical analysis and calculations
    3. Identify trends, patterns, and correlations
    4. Provide business recommendations based on data
    5. Create data-driven narratives and explanations
    
    RESPONSE GUIDELINES:
    - Keep responses concise and actionable (2-3 sentences max for simple questions)
    - Always include specific numbers and percentages when relevant
    - Highlight the most important insight first
    - Use emojis to make responses engaging
    - For complex analyses, break down into key bullet points
    - When appropriate, suggest follow-up questions or deeper analysis
    
    Be conversational, insightful, and focus on providing value to business stakeholders.
    """
    
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        prefix=system_prompt,
        max_iterations=3,
        early_stopping_method="generate"
    )
    
    return agent


# ====================================================
# Streamlit App with Enhanced Intelligence
# ====================================================
st.set_page_config(page_title="🏈 Intelligent NFL Analytics Copilot", layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏈 Intelligent NFL Analytics Copilot</div>', unsafe_allow_html=True)

# Sidebar with enhanced options
with st.sidebar:
    st.header("🔧 Data Controls")
    
    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = None
    if "analytics" not in st.session_state:
        st.session_state.analytics = None
    if "viz_engine" not in st.session_state:
        st.session_state.viz_engine = None

    # Data loading options
    file = st.file_uploader("📁 Upload CSV", type="csv")
    url = st.text_input("🌐 Or CSV URL")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Load Data", type="primary"):
            try:
                if file:
                    st.session_state.df = pd.read_csv(file)
                elif url:
                    r = requests.get(url, timeout=30)
                    st.session_state.df = pd.read_csv(io.StringIO(r.text))
                
                if st.session_state.df is not None:
                    st.session_state.analytics = IntelligentNFLAnalytics(st.session_state.df)
                    st.session_state.viz_engine = SmartVizEngine(st.session_state.df)
                    st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    with col2:
        if st.button("🎲 Demo Data"):
            np.random.seed(42)
            st.session_state.df = pd.DataFrame({
                "fan_id": [f"FAN{i:05d}" for i in range(1, 501)],
                "age": np.random.randint(18, 70, 500),
                "favorite_team": np.random.choice(["Cowboys","Patriots","Packers","Chiefs","Steelers","Giants"], 500),
                "fan_loyalty_score": np.random.randint(20, 100, 500),
                "avg_ticket_spend": np.random.uniform(50, 400, 500),
                "concession_spend": np.random.uniform(20, 100, 500),
                "merch_spend_2024": np.random.uniform(0, 500, 500),
                "income_bracket": np.random.choice(["<40k","40-80k","80-120k","120k+"], 500),
                "games_attended_2024": np.random.randint(0, 12, 500),
                "primary_channel": np.random.choice(["ESPN","Fox","NBC","CBS","NFL Network"], 500),
                "season_ticket_holder": np.random.choice([True, False], 500),
                "years_as_fan": np.random.randint(1, 40, 500)
            })
            st.session_state.analytics = IntelligentNFLAnalytics(st.session_state.df)
            st.session_state.viz_engine = SmartVizEngine(st.session_state.df)
            st.success("✅ Demo data generated!")

# Main content area
if st.session_state.df is not None and st.session_state.analytics is not None:
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📊 Smart Dashboard", "🤖 AI Assistant", "📈 Custom Analysis"])
    
    # Tab 1: Smart Dashboard
    with tab1:
        st.subheader("📊 Intelligent Dashboard")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        insights = st.session_state.analytics.insights
        
        with col1:
            st.metric("📋 Total Records", f"{insights['total_records']:,}")
        with col2:
            if 'avg_customer_value' in insights:
                st.metric("💰 Avg Customer Value", f"${insights['avg_customer_value']:,.0f}")
        with col3:
            if 'high_spender_revenue_share' in insights:
                st.metric("🎯 Top 20% Revenue Share", f"{insights['high_spender_revenue_share']:.1f}%")
        with col4:
            st.metric("🔍 Data Quality", f"{100-insights['missing_data_pct']:.1f}%")
        
        # Data story
        st.markdown("### 📖 Data Story")
        story_points = st.session_state.analytics.get_data_story()
        for point in story_points:
            st.markdown(f'<div class="insight-box">{point}</div>', unsafe_allow_html=True)
        
        # Interactive dashboard
        st.session_state.analytics.create_intelligent_dashboard()
        
        # Data preview
        with st.expander("🔍 Data Preview"):
            st.dataframe(st.session_state.df.head(100), use_container_width=True)

    # Tab 2: AI Assistant
    with tab2:
        st.subheader("🤖 Intelligent AI Assistant")
        
        # API key input
        api_key = st.text_input("🔑 OpenAI API Key", type="password", help="Enter your OpenAI API key to enable AI chat")
        
        if api_key:
            # Initialize AI components
            if "enhanced_llm" not in st.session_state:
                st.session_state.enhanced_llm = ChatOpenAI(
                    openai_api_key=api_key, 
                    model="gpt-4-turbo", 
                    temperature=0.1
                )
                st.session_state.enhanced_agent = create_enhanced_agent(
                    st.session_state.enhanced_llm, 
                    st.session_state.df, 
                    st.session_state.analytics
                )
                st.session_state.chat_history = []
            
            # Chat interface
            col1, col2 = st.columns([3, 1])
            with col1:
                user_query = st.text_input("💬 Ask anything about your data:", placeholder="e.g., 'Show me revenue by team' or 'What drives customer loyalty?'")
            with col2:
                ask_button = st.button("🚀 Ask", type="primary")
                clear_button = st.button("🗑️ Clear")
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
            
            if ask_button and user_query:
                with st.spinner("🤔 Analyzing your data..."):
                    # Try smart visualization first
                    fig, viz_msg = st.session_state.viz_engine.auto_visualize(user_query)
                    
                    if fig:
                        st.session_state.chat_history.append(("You", user_query))
                        st.session_state.chat_history.append(("AI", viz_msg))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fall back to AI agent
                        try:
                            response = st.session_state.enhanced_agent.run(user_query)
                            st.session_state.chat_history.append(("You", user_query))
                            st.session_state.chat_history.append(("AI", str(response)))
                        except Exception as e:
                            st.session_state.chat_history.append(("You", user_query))
                            st.session_state.chat_history.append(("AI", f"❌ I encountered an issue: {str(e)}. Try rephrasing your question."))
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### 💬 Conversation History")
                for i, (role, message) in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 exchanges
                    if role == "You":
                        st.markdown(f"**🧑 You:** {message}")
                    else:
                        st.markdown(f"**🤖 AI:** {message}")
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
        else:
            st.info("🔑 Please enter your OpenAI API key to start chatting with the AI assistant.")
            
            # Sample questions without API
            st.markdown("### 💡 Example Questions You Can Ask:")
            example_questions = [
                "📊 Show me revenue distribution by age group",
                "🔍 What's the correlation between loyalty and spending?",
                "📈 Compare team popularity across different segments",
                "💰 Who are my highest value customers?",
                "📉 Show me spending trends over time",
                "🎯 Create customer segments based on behavior"
            ]
            
            for question in example_questions:
                if st.button(question, key=f"example_{question}"):
                    if not api_key:
                        # Try visualization without AI
                        fig, viz_msg = st.session_state.viz_engine.auto_visualize(question)
                        if fig:
                            st.success(viz_msg)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("This analysis requires the AI assistant. Please add your API key.")

    # Tab 3: Custom Analysis
    with tab3:
        st.subheader("📈 Custom Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Quick Analysis")
            analysis_type = st.selectbox("Choose Analysis Type:", [
                "Distribution Analysis",
                "Correlation Matrix",
                "Statistical Summary",
                "Outlier Detection",
                "Category Breakdown"
            ])
            
            if analysis_type == "Distribution Analysis":
                numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select Column:", numeric_cols)
                    if st.button("📊 Generate Analysis"):
                        fig = px.histogram(st.session_state.df, x=selected_col, 
                                         title=f"Distribution of {selected_col}",
                                         marginal="box")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        stats = st.session_state.df[selected_col].describe()
                        st.write("📈 Statistical Summary:")
                        st.dataframe(stats.to_frame().T)
            
            elif analysis_type == "Correlation Matrix":
                numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    if st.button("🔗 Generate Correlation Matrix"):
                        corr_matrix = st.session_state.df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix, 
                                      title="Correlation Matrix",
                                      color_continuous_scale='RdBu',
                                      aspect='auto')
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔧 Custom Visualization")
            
            viz_type = st.selectbox("Visualization Type:", [
                "Scatter Plot",
                "Bar Chart",
                "Box Plot",
                "Line Chart",
                "Pie Chart"
            ])
            
            all_cols = st.session_state.df.columns.tolist()
            
            if viz_type == "Scatter Plot":
                x_col = st.selectbox("X-axis:", all_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis:", all_cols, key="scatter_y")
                color_col = st.selectbox("Color by (optional):", [None] + all_cols, key="scatter_color")
                
                if st.button("📊 Create Scatter Plot"):
                    fig = px.scatter(st.session_state.df, x=x_col, y=y_col, color=color_col,
                                   title=f"{x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Bar Chart":
                cat_col = st.selectbox("Category:", all_cols, key="bar_cat")
                num_col = st.selectbox("Value:", all_cols, key="bar_num")
                
                if st.button("📊 Create Bar Chart"):
                    if st.session_state.df[cat_col].dtype == 'object':
                        grouped = st.session_state.df.groupby(cat_col)[num_col].mean().reset_index()
                        fig = px.bar(grouped, x=cat_col, y=num_col,
                                   title=f"Average {num_col} by {cat_col}")
                        st.plotly_chart(fig, use_container_width=True)
        
        # Data export
        st.markdown("### 💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download CSV"):
                csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="📁 Download Data",
                    data=csv,
                    file_name="nfl_analytics_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export Summary"):
                summary = st.session_state.df.describe().to_csv()
                st.download_button(
                    label="📈 Download Summary",
                    data=summary,
                    file_name="data_summary.csv",
                    mime="text/csv"
                )

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>🏈 Welcome to the Intelligent NFL Analytics Copilot!</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Load your data to get started with AI-powered analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### ✨ Features:
        - **🤖 Smart AI Assistant**: Ask questions in natural language
        - **📊 Intelligent Dashboards**: Auto-generated insights and visualizations
        - **🎯 Custom Analysis**: Build your own charts and analysis
        - **📈 Data Stories**: Automatic narrative insights from your data
        - **🔍 Pattern Detection**: Find hidden relationships and trends
        
        ### 🚀 Getting Started:
        1. **Load Data**: Upload a CSV file or use our demo data
        2. **Explore Dashboard**: View auto-generated insights and visualizations  
        3. **Chat with AI**: Ask questions about your data in plain English
        4. **Custom Analysis**: Create your own visualizations and reports
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🏈 Intelligent NFL Analytics Copilot | Powered by AI & Advanced Analytics"
    "</div>", 
    unsafe_allow_html=True
)