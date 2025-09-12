# app_streamlit.py
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
from datetime import datetime
from typing import List, Optional

# Attempt to import LangChain / OpenAI related libs. If missing, continue with placeholders.
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain.vectorstores import FAISS
    from langchain.tools import BaseTool
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.agents import initialize_agent, AgentType
    from langchain import OpenAI
except Exception:
    # We'll create placeholders if libs missing so script still runs for non-RAG parts.
    ChatOpenAI = None
    OpenAIEmbeddings = None
    RecursiveCharacterTextSplitter = None
    Document = None
    FAISS = None
    BaseTool = object
    ConversationBufferWindowMemory = None
    initialize_agent = None
    AgentType = None

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
        insights['missing_data_pct'] = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0.0
        
        # Revenue insights
        if 'total_spend' in df.columns:
            insights['total_revenue'] = df['total_spend'].sum()
            insights['avg_customer_value'] = df['total_spend'].mean()
            insights['revenue_std'] = df['total_spend'].std()
            
            # Find high-value segments
            high_spenders = df[df['total_spend'] > df['total_spend'].quantile(0.8)]
            insights['high_spender_count'] = len(high_spenders)
            insights['high_spender_revenue_share'] = (high_spenders['total_spend'].sum() / df['total_spend'].sum()) * 100 if df['total_spend'].sum() != 0 else 0.0
        
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
                if not value_counts.empty:
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
            
            if cat_col and num_col:
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
            if main_cat:
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
            
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[x_col], 
                    y=df[y_col],
                    mode='markers',
                    marker=dict(size=8, opacity=0.6, color=self.color_palette[0]),
                    name=f'{x_col} vs {y_col}'
                ), row=2, col=1)
                
                # Add trendline
                common_index = df[[x_col, y_col]].dropna()
                if len(common_index) > 1:
                    z = np.polyfit(common_index[x_col], common_index[y_col], 1)
                    p = np.poly1d(z)
                    xs = sorted(common_index[x_col])
                    fig.add_trace(go.Scatter(
                        x=xs, 
                        y=p(xs),
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Trend'
                    ), row=2, col=1)
        
        # Chart 4: Performance comparison
        if 'total_spend' in df.columns and len(self.categorical_cols) > 0:
            cat_col = self.find_best_categorical_column()
            if cat_col:
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
            if main_numeric and len(self.categorical_cols) > 0:
                cat_col = self.find_best_categorical_column()
                
                if cat_col:
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
        
        story.append(f"📊 **Dataset Overview**: {insights.get('total_records', 0):,} records with {insights.get('missing_data_pct', 0):.1f}% missing data")
        
        if 'total_revenue' in insights:
            story.append(f"💰 **Revenue**: ${insights['total_revenue']:,.0f} total, ${insights['avg_customer_value']:.0f} avg per customer")
            story.append(f"🎯 **Key Insight**: Top 20% customers generate {insights.get('high_spender_revenue_share', 0):.1f}% of revenue ({insights.get('high_spender_count', 0)} high-value customers)")
        
        if 'strong_correlations' in insights and insights['strong_correlations']:
            corr = insights['strong_correlations'][0]
            story.append(f"🔗 **Strongest Relationship**: {corr[0]} ↔ {corr[1]} (correlation: {corr[2]:.2f})")
        
        # Add category insights
        for col in self.categorical_cols[:2]:  # Limit to avoid clutter
            if f'{col}_dominance' in insights:
                story.append(f"📈 **{col.title()}**: {insights.get(f'{col}_top_category','N/A')} dominates ({insights.get(f'{col}_dominance',0):.1f}% market share)")
        
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
# Intelligent Knowledge Base & Vector Store
# ====================================================
class DataKnowledgeBase:
    def __init__(self, df: pd.DataFrame, embeddings_model):
        self.df = df
        self.embeddings = embeddings_model
        self.vector_store = None
        self.documents = []
        if Document is None:
            # Can't build knowledge base without langchain Document
            return
        self.build_knowledge_base()
    
    def build_knowledge_base(self):
        """Build comprehensive knowledge base from the dataset"""
        documents = []
        
        # 1. Dataset Overview Document
        overview = f"""
        Dataset Overview:
        - Total Records: {len(self.df):,}
        - Columns: {len(self.df.columns)}
        - Column Names: {', '.join(self.df.columns)}
        - Numeric Columns: {', '.join(self.df.select_dtypes(include=[np.number]).columns)}
        - Categorical Columns: {', '.join(self.df.select_dtypes(include=['object']).columns)}
        - Missing Data: {self.df.isnull().sum().sum()} total missing values
        - Date Range: {datetime.now().strftime('%Y-%m-%d')}
        """
        documents.append(Document(page_content=overview, metadata={"type": "overview"}))
        
        # 2. Statistical Summaries for Numeric Columns
        for col in self.df.select_dtypes(include=[np.number]).columns:
            stats_summary = f"""
            Column: {col}
            Data Type: Numeric
            Statistics:
            - Mean: {self.df[col].mean():.2f}
            - Median: {self.df[col].median():.2f}
            - Standard Deviation: {self.df[col].std():.2f}
            - Min: {self.df[col].min():.2f}
            - Max: {self.df[col].max():.2f}
            - Missing Values: {self.df[col].isnull().sum()}
            - Unique Values: {self.df[col].nunique()}
            
            Business Context:
            {self._get_business_context(col)}
            """
            documents.append(Document(page_content=stats_summary, metadata={"type": "column_stats", "column": col}))
        
        # 3. Categorical Analysis
        for col in self.df.select_dtypes(include=['object']).columns:
            value_counts = self.df[col].value_counts().head(10)
            cat_summary = f"""
            Column: {col}
            Data Type: Categorical
            Unique Categories: {self.df[col].nunique()}
            Most Common Values:
            {chr(10).join([f"- {val}: {count} ({count/len(self.df)*100:.1f}%)" for val, count in value_counts.items()])}
            Missing Values: {self.df[col].isnull().sum()}
            
            Business Context:
            {self._get_business_context(col)}
            """
            documents.append(Document(page_content=cat_summary, metadata={"type": "categorical_analysis", "column": col}))
        
        # 4. Correlation Insights
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:  # Moderate to strong correlation
                        strength = "strong" if abs(corr_val) > 0.7 else "moderate"
                        direction = "positive" if corr_val > 0 else "negative"
                        strong_correlations.append(f"- {corr_matrix.columns[i]} and {corr_matrix.columns[j]}: {direction} {strength} correlation ({corr_val:.3f})")
            
            corr_doc = f"""
            Correlation Analysis:
            Key Relationships Found:
            {chr(10).join(strong_correlations) if strong_correlations else "No strong correlations detected"}
            
            Business Implications:
            {self._get_correlation_insights(strong_correlations)}
            """
            documents.append(Document(page_content=corr_doc, metadata={"type": "correlations"}))
        
        # 5. Segment Analysis (if applicable)
        segments_doc = self._create_segment_analysis()
        if segments_doc:
            documents.append(segments_doc)
        
        # 6. Business Insights
        insights_doc = self._create_business_insights()
        documents.append(insights_doc)
        
        # Create vector store
        if RecursiveCharacterTextSplitter is None or FAISS is None:
            # can't proceed without langchain components
            self.documents = documents
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        split_docs = text_splitter.split_documents(documents)
        self.documents = split_docs
        
        # Build FAISS vector store
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
    
    def _get_business_context(self, column_name: str) -> str:
        """Generate business context for columns based on naming patterns"""
        col_lower = column_name.lower()
        
        if any(word in col_lower for word in ['spend', 'cost', 'price', 'revenue', 'value']):
            return f"This is a financial metric. High values indicate customers with greater monetary contribution. Useful for revenue analysis and customer segmentation."
        elif any(word in col_lower for word in ['age', 'years']):
            return f"Demographic information. Different age groups may have distinct behaviors and preferences. Useful for targeted marketing and product development."
        elif any(word in col_lower for word in ['loyalty', 'score', 'rating']):
            return f"Customer engagement metric. Higher values suggest stronger brand affinity. Critical for retention strategies and lifetime value prediction."
        elif any(word in col_lower for word in ['team', 'favorite', 'preference']):
            return f"Customer preference data. Shows brand or category affinity. Useful for personalization and market share analysis."
        elif any(word in col_lower for word in ['channel', 'source', 'medium']):
            return f"Customer acquisition or engagement channel. Important for marketing attribution and channel optimization."
        elif any(word in col_lower for word in ['income', 'bracket', 'salary']):
            return f"Economic demographic indicator. Influences purchasing power and product preferences. Key for pricing strategies."
        else:
            return f"This column contains valuable information about customer characteristics or behaviors."
    
    def _get_correlation_insights(self, correlations: List[str]) -> str:
        """Generate business insights from correlations"""
        if not correlations:
            return "No significant correlations detected. Variables appear to be independent."
        
        insights = [
            "Strong correlations suggest:",
            "- Potential opportunities for cross-selling or bundling",
            "- Key drivers of customer behavior",
            "- Areas where improving one metric may positively impact others",
            "- Important variables for predictive modeling"
        ]
        return "\n".join(insights)
    
    def _create_segment_analysis(self) -> Optional[Document]:
        """Create customer segmentation analysis"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        
        # Simple segmentation based on key metrics
        spend_cols = [col for col in numeric_cols if 'spend' in col.lower() or 'cost' in col.lower() or 'revenue' in col.lower()]
        loyalty_cols = [col for col in numeric_cols if 'loyalty' in col.lower() or 'score' in col.lower()]
        
        segment_analysis = "Customer Segmentation Analysis:\n"
        
        if spend_cols:
            spend_col = spend_cols[0]
            high_spenders = self.df[self.df[spend_col] > self.df[spend_col].quantile(0.8)]
            segment_analysis += f"""
            High Value Segment (Top 20% by {spend_col}):
            - Count: {len(high_spenders)} customers ({len(high_spenders)/len(self.df)*100:.1f}%)
            - Average {spend_col}: ${high_spenders[spend_col].mean():.2f}
            - Revenue Share: {high_spenders[spend_col].sum()/self.df[spend_col].sum()*100:.1f}%
            """
        
        if loyalty_cols:
            loyalty_col = loyalty_cols[0]
            high_loyalty = self.df[self.df[loyalty_col] > self.df[loyalty_col].quantile(0.8)]
            segment_analysis += f"""
            High Loyalty Segment (Top 20% by {loyalty_col}):
            - Count: {len(high_loyalty)} customers
            - Average {loyalty_col}: {high_loyalty[loyalty_col].mean():.2f}
            """
        
        return Document(page_content=segment_analysis, metadata={"type": "segmentation"})
    
    def _create_business_insights(self) -> Document:
        """Generate strategic business insights"""
        insights = []
        
        # Revenue concentration
        spend_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                     if 'spend' in col.lower() or 'revenue' in col.lower()]
        
        if spend_cols:
            total_spend = self.df[spend_cols].sum(axis=1)
            top_20_pct = total_spend.quantile(0.8)
            top_customers = len(self.df[total_spend > top_20_pct])
            revenue_concentration = self.df[total_spend > top_20_pct][spend_cols].sum().sum() / self.df[spend_cols].sum().sum() * 100 if self.df[spend_cols].sum().sum() != 0 else 0.0
            
            insights.append(f"Revenue Concentration: Top {top_customers} customers generate {revenue_concentration:.1f}% of total revenue")
        
        # Market distribution
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:2]:  # Analyze first 2 categorical columns
            top_category = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "N/A"
            market_share = (self.df[col] == top_category).mean() * 100 if top_category != "N/A" else 0.0
            insights.append(f"Market Leadership: {top_category} leads {col} with {market_share:.1f}% market share")
        
        # Data quality
        missing_pct = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100 if len(self.df) > 0 else 0.0
        insights.append(f"Data Quality: {100-missing_pct:.1f}% complete data across all fields")
        
        business_doc = f"""
        Strategic Business Insights:
        
        Key Findings:
        {chr(10).join([f"• {insight}" for insight in insights])}
        
        Recommendations:
        • Focus retention efforts on high-value customer segments
        • Investigate drivers of top-performing categories
        • Develop strategies to increase engagement across all segments
        • Consider loyalty programs for high-spending customers
        • Optimize marketing spend based on channel performance
        
        Risk Factors:
        • High revenue concentration increases customer loss risk
        • Monitor competitive threats in dominant segments
        • Track changing customer preferences over time
        """
        
        return Document(page_content=business_doc, metadata={"type": "business_insights"})
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant documents"""
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []


# ====================================================
# Advanced AI Tools for Data Analysis
# ====================================================
class DataAnalysisTool(BaseTool):
    # Annotate class fields to satisfy Pydantic v2 if BaseTool is a Pydantic model
    name: str = "data_analysis"
    description: str = "Perform advanced statistical analysis, generate insights, and answer complex data questions"
    
    def __init__(self, df: pd.DataFrame, knowledge_base: DataKnowledgeBase):
        # Call parent init if available (guarded)
        try:
            super().__init__()
        except Exception:
            pass
        self.df = df
        self.knowledge_base = knowledge_base
    
    def _run(self, query: str) -> str:
        """Execute data analysis based on query"""
        try:
            # Get relevant context from knowledge base
            relevant_docs = self.knowledge_base.similarity_search(query, k=3) if self.knowledge_base else []
            context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
            
            # Perform specific analysis based on query type
            if any(word in query.lower() for word in ['correlation', 'relationship', 'related']):
                return self._correlation_analysis(query, context)
            elif any(word in query.lower() for word in ['segment', 'group', 'cluster']):
                return self._segmentation_analysis(query, context)
            elif any(word in query.lower() for word in ['trend', 'pattern', 'change']):
                return self._trend_analysis(query, context)
            elif any(word in query.lower() for word in ['top', 'best', 'highest', 'lowest']):
                return self._ranking_analysis(query, context)
            else:
                return self._general_analysis(query, context)
        
        except Exception as e:
            return f"I encountered an issue with the analysis: {str(e)}. Let me try a different approach."
    
    def _correlation_analysis(self, query: str, context: str) -> str:
        """Perform correlation analysis"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return "Not enough numeric columns for correlation analysis."
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find strongest correlations
        strong_corr = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strength = "very strong" if abs(corr_val) > 0.8 else "strong"
                    direction = "positive" if corr_val > 0 else "negative"
                    strong_corr.append((numeric_cols[i], numeric_cols[j], corr_val, f"{direction} {strength}"))
        
        if strong_corr:
            result = "🔗 **Strong Correlations Found:**\n\n"
            for col1, col2, corr_val, desc in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)[:3]:
                result += f"• **{col1} ↔ {col2}**: {desc} correlation (r={corr_val:.3f})\n"
            
            result += f"\n💡 **Business Insight**: "
            if 'spend' in str(strong_corr[0][:2]).lower():
                result += "Strong spending correlations suggest opportunities for bundling or cross-selling strategies."
            elif 'loyalty' in str(strong_corr[0][:2]).lower():
                result += "Loyalty correlations indicate key drivers of customer retention."
            else:
                result += "These relationships can guide strategic decisions and predictive modeling."
            
            return result
        else:
            return "No strong correlations detected. Variables appear to be relatively independent."
    
    def _segmentation_analysis(self, query: str, context: str) -> str:
        """Perform customer segmentation"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Focus on key business metrics
        spend_cols = [col for col in numeric_cols if 'spend' in col.lower() or 'revenue' in col.lower()]
        loyalty_cols = [col for col in numeric_cols if 'loyalty' in col.lower() or 'score' in col.lower()]
        
        result = "🎯 **Customer Segmentation Analysis:**\n\n"
        
        if spend_cols:
            spend_col = spend_cols[0]
            # Create spending tiers
            high_spend = self.df[self.df[spend_col] > self.df[spend_col].quantile(0.75)]
            med_spend = self.df[(self.df[spend_col] >= self.df[spend_col].quantile(0.25)) & 
                             (self.df[spend_col] <= self.df[spend_col].quantile(0.75))]
            low_spend = self.df[self.df[spend_col] < self.df[spend_col].quantile(0.25)]
            
            result += f"**💰 Spending Segments:**\n"
            result += f"• High Spenders: {len(high_spend)} customers (${high_spend[spend_col].mean():.0f} avg)\n"
            result += f"• Medium Spenders: {len(med_spend)} customers (${med_spend[spend_col].mean():.0f} avg)\n"
            result += f"• Low Spenders: {len(low_spend)} customers (${low_spend[spend_col].mean():.0f} avg)\n\n"
            
            # Revenue contribution
            total_revenue = self.df[spend_col].sum()
            high_revenue_share = (high_spend[spend_col].sum() / total_revenue) * 100 if total_revenue != 0 else 0.0
            result += f"📊 High spenders contribute **{high_revenue_share:.1f}%** of total revenue\n\n"
        
        if loyalty_cols:
            loyalty_col = loyalty_cols[0]
            high_loyalty = self.df[self.df[loyalty_col] > self.df[loyalty_col].quantile(0.75)]
            result += f"**⭐ High Loyalty Segment:** {len(high_loyalty)} customers with avg {loyalty_col} of {high_loyalty[loyalty_col].mean():.1f}\n\n"
        
        result += "💡 **Strategic Recommendations:**\n"
        result += "• Target high-value segments with premium offerings\n"
        result += "• Develop retention programs for at-risk customers\n"
        result += "• Create loyalty incentives to move customers up-tier"
        
        return result
    
    def _trend_analysis(self, query: str, context: str) -> str:
        """Analyze trends and patterns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        result = "📈 **Trend Analysis:**\n\n"
        
        # Look for time-based columns
        date_cols = [col for col in self.df.columns if any(word in col.lower() for word in ['date', 'time', 'year', 'month'])]
        
        if date_cols:
            result += f"Time-based analysis available for: {', '.join(date_cols)}\n\n"
        
        # Analyze distributions and outliers
        for col in numeric_cols[:3]:  # Limit to top 3 numeric columns
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = len(self.df[(self.df[col] < q1 - 1.5*iqr) | (self.df[col] > q3 + 1.5*iqr)])
            
            result += f"**{col}:**\n"
            result += f"• Range: {self.df[col].min():.1f} to {self.df[col].max():.1f}\n"
            result += f"• Outliers detected: {outliers} ({outliers/len(self.df)*100:.1f}%)\n\n"
        
        result += "💡 **Pattern Insights:** Monitor outliers for anomalies or high-value opportunities"
        
        return result
    
    def _ranking_analysis(self, query: str, context: str) -> str:
        """Perform ranking and top/bottom analysis"""
        result = "🏆 **Performance Ranking:**\n\n"
        
        # Find categorical columns for grouping
        cat_cols = self.df.select_dtypes(include=['object']).columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(cat_cols) > 0 and len(num_cols) > 0:
            cat_col = cat_cols[0]
            
            # Revenue ranking if available
            revenue_cols = [col for col in num_cols if 'spend' in col.lower() or 'revenue' in col.lower()]
            if revenue_cols:
                revenue_col = revenue_cols[0]
                ranking = self.df.groupby(cat_col)[revenue_col].agg(['mean', 'count', 'sum']).round(2)
                ranking = ranking.sort_values('mean', ascending=False)
                
                result += f"**💰 Revenue Performance by {cat_col}:**\n"
                for i, (category, row) in enumerate(ranking.head(5).iterrows()):
                    result += f"{i+1}. **{category}**: ${row['mean']:.0f} avg (${row['sum']:.0f} total, n={row['count']})\n"
                
                result += f"\n🎯 **Winner:** {ranking.index[0]} leads with ${ranking.iloc[0]['mean']:.0f} average revenue\n\n"
        
        # Volume analysis
        if len(cat_cols) > 0:
            cat_col = cat_cols[0]
            volume_ranking = self.df[cat_col].value_counts()
            
            result += f"**📊 Volume Leadership by {cat_col}:**\n"
            for i, (category, count) in enumerate(volume_ranking.head(3).items()):
                result += f"{i+1}. {category}: {count:,} ({count/len(self.df)*100:.1f}%)\n"
        
        return result
    
    def _general_analysis(self, query: str, context: str) -> str:
        """General analysis for other queries"""
        # Use context from knowledge base
        if context:
            lines = context.split('\n')
            key_info = [line for line in lines if any(word in line.lower() for word in 
                       ['mean', 'total', 'count', 'correlation', 'insight', 'recommendation'])][:5]
            
            result = "📊 **Key Data Insights:**\n\n"
            for info in key_info:
                if info.strip():
                    result += f"• {info.strip()}\n"
            
            result += f"\n💡 Based on your data with {len(self.df):,} records across {len(self.df.columns)} variables."
            return result
        
        return "I'd be happy to analyze your data! Try asking about correlations, segments, trends, or rankings for more specific insights."


# ====================================================
# Claude-like AI Assistant with Vector RAG
# ====================================================
class IntelligentAIAssistant:
    def __init__(self, df: pd.DataFrame, api_key: str):
        self.df = df
        self.api_key = api_key
        
        # Initialize models (guarded)
        if ChatOpenAI is None:
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                openai_api_key=api_key,
                model="gpt-4o-mini",  # or "gpt-4-turbo" depending on availability
                temperature=0.1,
                max_tokens=2000
            )
        
        if OpenAIEmbeddings is None:
            self.embeddings = None
        else:
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Build knowledge base (only if embeddings available)
        if self.embeddings is not None and Document is not None:
            self.knowledge_base = DataKnowledgeBase(df, self.embeddings)
        else:
            self.knowledge_base = None
        
        # Initialize memory
        if ConversationBufferWindowMemory is None:
            self.memory = None
        else:
            self.memory = ConversationBufferWindowMemory(
                k=10,  # Remember last 10 exchanges
                return_messages=True,
                memory_key="chat_history",
                output_key="output"
            )
        
        # Create advanced tools
        self.tools = []
        if self.knowledge_base is not None:
            self.tools.append(DataAnalysisTool(df, self.knowledge_base))
        
        # Create the agent if initialize_agent available
        self.agent = self._create_intelligent_agent() if initialize_agent is not None and AgentType is not None else None
    
    def _create_intelligent_agent(self):
        """Create Claude-like intelligent agent"""
        if self.llm is None:
            return None
        
        system_prompt = f"""You are an expert data analyst AI assistant with deep knowledge of the provided dataset. You have access to comprehensive data insights through a vector knowledge base and advanced analysis tools.

DATASET CONTEXT:
- {len(self.df):,} records across {len(self.df.columns)} variables
- Columns: {', '.join(self.df.columns.tolist())}
- Key metrics: {', '.join([col for col in self.df.columns if any(word in col.lower() for word in ['spend', 'revenue', 'score', 'loyalty'])])}

YOUR CAPABILITIES:
🔍 Advanced Analysis: Correlations, segmentation, trends, statistical insights
📊 Smart Visualizations: Auto-generate relevant charts and graphs  
💡 Business Intelligence: Strategic recommendations based on data patterns
🎯 Predictive Insights: Identify opportunities and risk factors
📈 Performance Analysis: Rankings, benchmarks, and KPI analysis

RESPONSE GUIDELINES:
1. **Be Concise**: Lead with the key insight, then provide supporting details
2. **Be Specific**: Include actual numbers, percentages, and concrete findings
3. **Be Actionable**: Provide business recommendations when relevant
4. **Be Contextual**: Reference specific data points and relationships
5. **Be Conversational**: Maintain a helpful, professional tone with emojis for clarity

ANALYSIS APPROACH:
- Always search your knowledge base first for relevant context
- Use statistical analysis tools for complex questions
- Provide confidence levels when making predictions
- Highlight both opportunities and risks
- Connect findings to business outcomes

Remember: You're like Claude but specialized for data analysis - intelligent, thorough, and always helpful!"""

        try:
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True,
                memory=self.memory,
                max_iterations=3,
                early_stopping_method="generate",
                agent_kwargs={
                    "system_message": system_prompt,
                }
            )
            return agent
        except Exception:
            return None
    
    def chat(self, user_input: str) -> str:
        """Process user input and generate intelligent response"""
        try:
            # Get relevant context from knowledge base
            relevant_context = self.knowledge_base.similarity_search(user_input, k=2) if self.knowledge_base else []
            context_text = "\n".join([doc.page_content for doc in relevant_context]) if relevant_context else ""
            
            # Enhance the query with context
            enhanced_query = f"""
            User Question: {user_input}
            
            Relevant Context from Knowledge Base:
            {context_text}
            
            Please provide a comprehensive, actionable response based on the data and context available.
            """
            
            # Get response from agent if available
            if self.agent is not None:
                response = self.agent.run(enhanced_query)
            elif self.llm is not None:
                # fallback to direct LLM call if agent not available
                try:
                    response_obj = self.llm.generate([{"role": "user", "content": enhanced_query}])
                    try:
                        response = response_obj.generations[0][0].text
                    except Exception:
                        response = str(response_obj)
                except Exception as e:
                    response = f"LLM call failed: {e}"
            else:
                # Fallback to DataAnalysisTool if nothing else
                analysis_tool = DataAnalysisTool(self.df, self.knowledge_base) if self.knowledge_base else None
                if analysis_tool:
                    return analysis_tool._run(user_input)
                return "AI backend not available. Please configure your OpenAI / LangChain environment."
            
            return response
            
        except Exception as e:
            # Fallback to direct analysis
            try:
                analysis_tool = DataAnalysisTool(self.df, self.knowledge_base)
                return analysis_tool._run(user_input)
            except Exception:
                return f"I apologize, but I encountered an issue processing your request: {str(e)}. Could you please rephrase your question or try asking about specific aspects like 'revenue analysis' or 'customer segments'?"
    
    def get_suggested_questions(self) -> List[str]:
        """Generate contextual question suggestions"""
        suggestions = []
        
        # Revenue-focused questions
        revenue_cols = [col for col in self.df.columns if 'spend' in col.lower() or 'revenue' in col.lower()]
        if revenue_cols:
            suggestions.append(f"💰 What drives {revenue_cols[0]} in our customer base?")
            suggestions.append(f"🎯 Which customer segments generate the most {revenue_cols[0]}?")
        
        # Loyalty questions
        loyalty_cols = [col for col in self.df.columns if 'loyalty' in col.lower() or 'score' in col.lower()]
        if loyalty_cols:
            suggestions.append(f"⭐ What factors correlate with {loyalty_cols[0]}?")
        
        # Category questions
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            suggestions.append(f"📊 Show me the performance breakdown by {cat_cols[0]}")
            suggestions.append(f"🏆 Which {cat_cols[0]} categories are most valuable?")
        
        # General insights
        suggestions.extend([
            "🔍 What are the most important insights from this data?",
            "📈 Show me key correlations and relationships",
            "🎯 How should we segment our customers?",
            "⚡ What quick wins can you identify?"
        ])
        
        return suggestions[:8]  # Return top 8 suggestions


# ====================================================
# Streamlit App with Enhanced Intelligence
# ====================================================
st.set_page_config(page_title="🏈 Intelligent NFL Analytics Copilot", layout="wide")
st.markdown('<h1 style="text-align: center; color: #1f77b4;">🏈 Intelligent NFL Analytics Copilot</h1>', unsafe_allow_html=True)

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
            st.metric("📋 Total Records", f"{insights.get('total_records',0):,}")
        with col2:
            if 'avg_customer_value' in insights:
                st.metric("💰 Avg Customer Value", f"${insights['avg_customer_value']:,.0f}")
        with col3:
            if 'high_spender_revenue_share' in insights:
                st.metric("🎯 Top 20% Revenue Share", f"{insights['high_spender_revenue_share']:.1f}%")
        with col4:
            st.metric("🔍 Data Quality", f"{100-insights.get('missing_data_pct',0):.1f}%")
        
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
        st.subheader("🤖 Claude-like AI Assistant")
        
        # API key input
        api_key = st.text_input("🔑 OpenAI API Key", type="password", help="Enter your OpenAI API key to enable AI chat")
        
        if api_key:
            # Initialize intelligent AI assistant
            if "intelligent_assistant" not in st.session_state or st.session_state.get("api_key") != api_key:
                try:
                    with st.spinner("🧠 Initializing intelligent AI assistant with vector knowledge base..."):
                        st.session_state.intelligent_assistant = IntelligentAIAssistant(st.session_state.df, api_key)
                        st.session_state.api_key = api_key
                        st.session_state.chat_history = []
                    st.success("✅ AI Assistant ready with advanced analytics capabilities!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize AI assistant: {str(e)}")
                    st.info("Please check your API key and try again.")
                    st.stop()
            
            # Suggested questions
            st.markdown("### 💡 Suggested Questions")
            if st.session_state.intelligent_assistant:
                try:
                    suggestions = st.session_state.intelligent_assistant.get_suggested_questions()
                except Exception:
                    suggestions = [
                        "💰 Show revenue distribution",
                        "🎯 Which team has highest avg spend?",
                        "🔗 Show correlation matrix",
                        "🎲 Segment customers"
                    ]
            else:
                suggestions = [
                    "💰 Show revenue distribution",
                    "🎯 Which team has highest avg spend?",
                    "🔗 Show correlation matrix",
                    "🎲 Segment customers"
                ]
            
            # Display suggestions in columns
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                col_idx = i % 2
                with cols[col_idx]:
                    if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                        # Process suggested question
                        with st.spinner("🤔 Analyzing..."):
                            try:
                                # First try visualization
                                fig, viz_msg = st.session_state.viz_engine.auto_visualize(suggestion)
                                if fig:
                                    st.session_state.chat_history.append(("You", suggestion))
                                    st.session_state.chat_history.append(("AI", viz_msg))
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Then get AI insights
                                ai_response = st.session_state.intelligent_assistant.chat(suggestion) if st.session_state.intelligent_assistant else "AI not initialized"
                                st.session_state.chat_history.append(("AI", ai_response))
                                
                            except Exception as e:
                                error_msg = f"I encountered an issue: {str(e)}. Let me try a different approach."
                                st.session_state.chat_history.append(("You", suggestion))
                                st.session_state.chat_history.append(("AI", error_msg))
            
            st.markdown("---")
            
            # Chat interface
            col1, col2 = st.columns([4, 1])
            with col1:
                user_query = st.text_input(
                    "💬 Ask me anything about your data:", 
                    placeholder="e.g., 'What drives customer loyalty?' or 'Show me revenue patterns'",
                    key="main_chat_input"
                )
            with col2:
                ask_button = st.button("🚀 Ask", type="primary")
                clear_button = st.button("🗑️ Clear Chat")
            
            if clear_button:
                st.session_state.chat_history = []
                st.experimental_rerun()
            
            if (ask_button and user_query) or (user_query and st.session_state.get("last_query") != user_query):
                st.session_state.last_query = user_query
                
                with st.spinner("🧠 Thinking..."):
                    try:
                        # Add user message
                        st.session_state.chat_history.append(("You", user_query))
                        
                        # Check if this is a visualization request
                        viz_keywords = ['show', 'plot', 'chart', 'graph', 'visualize', 'display']
                        if any(keyword in user_query.lower() for keyword in viz_keywords):
                            # Try visualization first
                            fig, viz_msg = st.session_state.viz_engine.auto_visualize(user_query)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                st.session_state.chat_history.append(("AI", f"📊 {viz_msg}"))
                        
                        # Get AI response
                        ai_response = st.session_state.intelligent_assistant.chat(user_query) if st.session_state.intelligent_assistant else "AI not initialized"
                        st.session_state.chat_history.append(("AI", ai_response))
                        
                    except Exception as e:
                        error_response = f"I apologize for the technical difficulty. Let me provide some general insights about your data instead.\n\n"
                        
                        # Fallback to basic analysis
                        try:
                            insights = st.session_state.analytics.get_data_story()
                            error_response += "📊 **Key Data Insights:**\n" + "\n".join([f"• {insight}" for insight in insights[:3]])
                        except:
                            error_response += "Please try rephrasing your question or ask about specific metrics like 'revenue analysis' or 'customer segments'."
                        
                        st.session_state.chat_history.append(("AI", error_response))
            
            # Display chat history with better formatting
            if st.session_state.get("chat_history"):
                st.markdown("### 💬 Conversation")
                
                # Create a container for chat messages
                chat_container = st.container()
                
                with chat_container:
                    # Show recent messages (last 20 for performance)
                    recent_messages = st.session_state.chat_history[-20:]
                    
                    for i, (role, message) in enumerate(recent_messages):
                        # safe stringify
                        safe_message = str(message).replace("<", "&lt;").replace(">", "&gt;")
                        if role == "You":
                            st.markdown(f"""
                                <div style="background-color:#e8f4fd; padding:10px; border-radius:8px; margin:8px 0;">
                                    <strong>You:</strong><br>{safe_message}
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div style="background-color:#f1f1f1; padding:10px; border-radius:8px; margin:8px 0;">
                                    <strong>Assistant:</strong><br>{safe_message}
                                </div>
                            """, unsafe_allow_html=True)
                
                # Auto-scroll to bottom
                st.markdown("""
                    <script>
                    try {
                        var element = document.querySelector('[data-testid="stVerticalBlock"]');
                        if(element) { element.scrollTop = element.scrollHeight; }
                    } catch(e) {}
                    </script>
                """, unsafe_allow_html=True)
        
        else:
            # API key required message
            st.markdown("""
                <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h4>🔑 OpenAI API Key Required</h4>
                    <p>To unlock the full power of the Claude-like AI assistant, please enter your OpenAI API key above.</p>
                    <p><strong>Features you'll get:</strong></p>
                    <ul>
                        <li>🧠 Advanced data analysis with vector knowledge base</li>
                        <li>💬 Natural language conversations about your data</li>
                        <li>📊 Intelligent insights and recommendations</li>
                        <li>🎯 Context-aware responses with memory</li>
                        <li>⚡ Lightning-fast pattern recognition</li>
                    </ul>
                    <p><em>Don't have an API key? Get one at <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI Platform</a></em></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show some visualizations without API
            st.markdown("### 📊 Preview Analysis (No API Required)")
            preview_questions = [
                "Show revenue distribution",
                "Display team popularity", 
                "Analyze customer segments",
                "Show correlation matrix"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(preview_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"preview_{i}"):
                        fig, viz_msg = st.session_state.viz_engine.auto_visualize(question)
                        if fig:
                            st.success(viz_msg)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("💡 This analysis would work better with the AI assistant enabled!")

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
