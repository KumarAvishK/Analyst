import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from difflib import SequenceMatcher
import re
import plotly.figure_factory as ff

# Groq imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.error("⚠️ Groq library not installed. Run: pip install groq")

# Internal Groq API Key
GROQ_API_KEY = "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"

# Set page config
st.set_page_config(
    page_title="KiteIQX Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .kite-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        display: inline-block;
        margin: 0.5rem 0;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .clean-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .data-quality-excellent {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .data-quality-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .data-quality-poor {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .method-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border: 1px solid #bbdefb;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .ai-response-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eaf6 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# GROQ LLM WRAPPER
# -------------------------

class GroqLLM:
    """Wrapper for Groq API for analytics"""
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def predict(self, prompt: str, max_tokens: int = 2000) -> str:
        """Make a prediction/completion"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are KiteIQX Intelligence, an expert data analyst. Provide clear, actionable insights with specific numbers and recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def run(self, prompt: str) -> str:
        """Alias for predict"""
        return self.predict(prompt)


# -------------------------
# SMART VISUALIZATION ENGINE
# -------------------------

class SmartVisualizationEngine:
    def __init__(self, analytics):
        self.analytics = analytics
        self.df = analytics.df
        self.color_palette = px.colors.qualitative.Set3

    def create_scatter_plot(self, x_col, y_col, color_col=None):
        """Create enhanced scatter plot"""
        try:
            if color_col and color_col in self.analytics.categorical_cols:
                fig = px.scatter(
                    self.df, x=x_col, y=y_col, color=color_col,
                    title=f"{y_col} vs {x_col} (colored by {color_col})",
                    hover_data=[col for col in self.df.columns if col not in [x_col, y_col, color_col]][:3]
                )
            else:
                fig = px.scatter(
                    self.df, x=x_col, y=y_col,
                    title=f"{y_col} vs {x_col}",
                    hover_data=[col for col in self.df.columns if col not in [x_col, y_col]][:3]
                )
            
            # Add correlation info
            corr = self.df[x_col].corr(self.df[y_col])
            fig.add_annotation(
                text=f"Correlation: {corr:.3f}",
                xref="paper", yref="paper", x=0.02, y=0.98,
                showarrow=False, bgcolor="white", bordercolor="black"
            )
            
            # Add trend line
            if not self.df[x_col].isnull().all() and not self.df[y_col].isnull().all():
                z = np.polyfit(self.df[x_col].dropna(), self.df[y_col].dropna(), 1)
                p = np.poly1d(z)
                x_trend = np.linspace(self.df[x_col].min(), self.df[x_col].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_trend, y=p(x_trend),
                    mode='lines', name='Trend Line',
                    line=dict(color='red', dash='dash')
                ))
            
            return fig, f"Scatter plot created with correlation: {corr:.3f}"
        except Exception as e:
            return None, f"Error creating scatter plot: {str(e)}"

    def create_correlation_matrix(self, selected_cols):
        """Create correlation matrix heatmap"""
        try:
            numeric_cols = [col for col in selected_cols if col in self.analytics.numeric_cols]
            if len(numeric_cols) < 2:
                return None, "Need at least 2 numeric columns for correlation matrix"
            
            corr_matrix = self.df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title="Correlation Matrix - KiteIQX Analysis"
            )
            
            return fig, f"Correlation matrix for {len(numeric_cols)} variables"
        except Exception as e:
            return None, f"Error creating correlation matrix: {str(e)}"

    def create_distribution_analysis(self, selected_cols):
        """Create distribution analysis"""
        try:
            numeric_cols = [col for col in selected_cols if col in self.analytics.numeric_cols][:4]
            if not numeric_cols:
                return None, "No numeric columns selected"
            
            n_cols = min(2, len(numeric_cols))
            n_rows = (len(numeric_cols) + 1) // 2
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f"Distribution: {col}" for col in numeric_cols]
            )
            
            for i, col in enumerate(numeric_cols):
                row = (i // n_cols) + 1
                col_pos = (i % n_cols) + 1
                
                fig.add_trace(go.Histogram(
                    x=self.df[col], name=col,
                    nbinsx=30, opacity=0.7
                ), row=row, col=col_pos)
            
            fig.update_layout(height=400*n_rows, showlegend=False, title_text="Distribution Analysis - KiteIQX")
            
            return fig, f"Distribution analysis for {len(numeric_cols)} variables"
        except Exception as e:
            return None, f"Error creating distribution analysis: {str(e)}"

    def create_predictive_model(self, target_col, predictor_cols):
        """Create predictive model visualization"""
        try:
            if target_col not in self.analytics.numeric_cols:
                return None, "Target must be numeric"
            
            predictor_cols = [col for col in predictor_cols if col in self.analytics.numeric_cols and col != target_col]
            if not predictor_cols:
                return None, "Need at least one numeric predictor"
            
            # Use first predictor for visualization
            predictor = predictor_cols[0]
            
            # Prepare data
            data = self.df[[target_col, predictor]].dropna()
            X = data[predictor].values.reshape(-1, 1)
            y = data[target_col].values
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions
            X_pred = np.linspace(X.min(), X.max(), 100)
            y_pred = model.predict(X_pred.reshape(-1, 1))
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    f"Regression: {predictor} → {target_col}",
                    "Residuals Analysis",
                    "Actual vs Predicted",
                    "Model Performance"
                ]
            )
            
            # Scatter plot with regression line
            fig.add_trace(go.Scatter(
                x=X.flatten(), y=y, mode='markers', name='Actual Data',
                marker=dict(color='blue', opacity=0.6)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=X_pred.flatten(), y=y_pred, mode='lines', name='Predicted',
                line=dict(color='red', width=2)
            ), row=1, col=1)
            
            # Residuals
            y_pred_actual = model.predict(X)
            residuals = y - y_pred_actual
            fig.add_trace(go.Scatter(
                x=y_pred_actual, y=residuals, mode='markers',
                name='Residuals', marker=dict(color='green', opacity=0.6)
            ), row=1, col=2)
            
            # Actual vs Predicted
            fig.add_trace(go.Scatter(
                x=y, y=y_pred_actual, mode='markers',
                name='Actual vs Predicted', marker=dict(color='purple', opacity=0.6)
            ), row=2, col=1)
            
            # Perfect prediction line
            min_val, max_val = min(y.min(), y_pred_actual.min()), max(y.max(), y_pred_actual.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(color='black', dash='dash')
            ), row=2, col=1)
            
            # R-squared indicator
            r2 = r2_score(y, y_pred_actual)
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=r2,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "R² Score"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "yellow"},
                                {'range': [0.8, 1], 'color': "green"}]}
            ), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=True, title_text="Predictive Model - KiteIQX Intelligence")
            
            return fig, f"Predictive model: R² = {r2:.3f}, RMSE = {np.sqrt(np.mean(residuals**2)):.3f}"
        except Exception as e:
            return None, f"Error creating predictive model: {str(e)}"

    def create_time_series_analysis(self, date_col, value_col):
        """Create time series analysis"""
        try:
            if date_col not in self.analytics.datetime_cols:
                return None, "Selected column is not datetime"
            if value_col not in self.analytics.numeric_cols:
                return None, "Value column must be numeric"
            
            # Prepare data
            ts_data = self.df[[date_col, value_col]].dropna().sort_values(date_col)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[f"Time Series: {value_col}", "Trend Analysis"],
                vertical_spacing=0.1
            )
            
            # Main time series
            fig.add_trace(go.Scatter(
                x=ts_data[date_col], y=ts_data[value_col],
                mode='lines+markers', name='Actual',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            # Moving average if enough data
            if len(ts_data) > 7:
                ts_data['ma7'] = ts_data[value_col].rolling(window=7).mean()
                fig.add_trace(go.Scatter(
                    x=ts_data[date_col], y=ts_data['ma7'],
                    mode='lines', name='7-period MA',
                    line=dict(color='red', dash='dash')
                ), row=1, col=1)
            
            # Trend analysis
            if len(ts_data) > 3:
                x_numeric = np.arange(len(ts_data))
                z = np.polyfit(x_numeric, ts_data[value_col], 1)
                trend = np.poly1d(z)(x_numeric)
                
                fig.add_trace(go.Scatter(
                    x=ts_data[date_col], y=trend,
                    mode='lines', name='Trend',
                    line=dict(color='green', width=3)
                ), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True, title_text="Time Series Analysis - KiteIQX")
            
            return fig, f"Time series analysis for {value_col}"
        except Exception as e:
            return None, f"Error creating time series: {str(e)}"


# -------------------------
# UNIVERSAL ANALYTICS CLASS
# -------------------------

class UniversalAnalytics:
    def __init__(self, df, llm=None):
        self.df = df
        self.llm = llm
        self.original_df = df.copy()
        self.color_palette = px.colors.qualitative.Set2
        self.process_data()
        self.generate_insights()

    def process_data(self):
        """Intelligent data processing that adapts to any dataset"""
        # Detect column types
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = []
        
        # Data quality checks
        self.data_quality_issues = {}
        self.check_data_quality()
        
        # Try to detect datetime columns
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.datetime_cols.append(col)
                except:
                    pass
        
        # Detect special column types
        self.money_cols = [col for col in self.numeric_cols 
                          if any(word in col.lower() for word in 
                                ['price', 'cost', 'amount', 'revenue', 'salary', 'income', 'fee', 'payment', 'spend'])]
        
        self.score_cols = [col for col in self.numeric_cols 
                          if any(word in col.lower() for word in 
                                ['score', 'rating', 'rank', 'grade', 'satisfaction', 'performance'])]
        
        self.id_cols = [col for col in self.df.columns 
                       if any(word in col.lower() for word in ['id', 'key', 'index']) and 
                       self.df[col].nunique() / len(self.df) > 0.8]
        
        # Create derived features
        if len(self.money_cols) > 1:
            self.df['total_monetary_value'] = self.df[self.money_cols].sum(axis=1, skipna=True)
        elif len(self.money_cols) == 1:
            self.df['total_monetary_value'] = self.df[self.money_cols[0]]
        
        # Update numeric columns after adding derived features
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

    def check_data_quality(self):
        """Comprehensive data quality assessment"""
        issues = {}
        
        # 1. Duplicate detection
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            issues['duplicates'] = {
                'count': duplicate_count,
                'percentage': (duplicate_count / len(self.df)) * 100
            }
        
        # 2. Missing data patterns
        missing_data = {}
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_data[col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(self.df)) * 100
                }
        
        if missing_data:
            issues['missing_data'] = missing_data
        
        # 3. Outlier detection for numeric columns
        outlier_summary = {}
        for col in self.numeric_cols:
            if col in self.df.columns:
                outliers = self.detect_outliers_advanced(col)
                if outliers['count'] > 0:
                    outlier_summary[col] = outliers
        
        if outlier_summary:
            issues['outliers'] = outlier_summary
        
        self.data_quality_issues = issues

    def detect_outliers_advanced(self, column):
        """Advanced outlier detection with context"""
        if column not in self.numeric_cols:
            return {'count': 0}
        
        data = self.df[column].dropna()
        if len(data) < 10:
            return {'count': 0}
        
        # Statistical outliers (IQR method)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        statistical_outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'count': len(statistical_outliers),
            'percentage': (len(statistical_outliers) / len(data)) * 100,
            'range': f"{statistical_outliers.min():.2f} to {statistical_outliers.max():.2f}" if len(statistical_outliers) > 0 else "None"
        }

    def get_data_quality_report(self):
        """Generate comprehensive data quality report"""
        if not self.data_quality_issues:
            return "excellent", "✅ No significant data quality issues detected. Your dataset is ready for analysis!"
        
        report = "**KiteIQX Data Quality Assessment**\n\n"
        
        # Calculate quality score
        total_issues = len(self.data_quality_issues)
        severity_score = 0
        
        # Duplicates
        if 'duplicates' in self.data_quality_issues:
            dup_info = self.data_quality_issues['duplicates']
            severity_score += min(dup_info['percentage'] * 2, 30)
            report += f"**Duplicates:** {dup_info['count']:,} records ({dup_info['percentage']:.1f}%)\n"
        
        # Missing data
        if 'missing_data' in self.data_quality_issues:
            missing_severity = max([info['percentage'] for info in self.data_quality_issues['missing_data'].values()])
            severity_score += min(missing_severity, 25)
            report += f"**Missing Data:** Up to {missing_severity:.1f}% missing in some columns\n"
        
        # Outliers
        if 'outliers' in self.data_quality_issues:
            outlier_severity = max([info['percentage'] for info in self.data_quality_issues['outliers'].values()])
            severity_score += min(outlier_severity, 15)
            report += f"**Outliers:** Up to {outlier_severity:.1f}% outliers detected\n"
        
        # Overall quality score
        quality_score = max(0, 100 - severity_score)
        
        if quality_score >= 90:
            status = "excellent"
        elif quality_score >= 75:
            status = "good"
        elif quality_score >= 60:
            status = "fair"
        else:
            status = "poor"
        
        report += f"\n**Quality Score: {quality_score:.0f}/100**"
        
        return status, report

    def clean_duplicates(self):
        """Remove duplicate records"""
        if 'duplicates' in self.data_quality_issues:
            original_count = len(self.df)
            self.df = self.df.drop_duplicates()
            removed_count = original_count - len(self.df)
            
            # Re-process data
            self.process_data()
            self.generate_insights()
            
            return f"✅ Removed {removed_count:,} duplicate records. Dataset now has {len(self.df):,} unique records."
        return "ℹ️ No duplicates found to remove."

    def generate_insights(self):
        """Generate comprehensive insights about the dataset"""
        self.insights = {
            'basic_stats': {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'missing_pct': (self.df.isnull().sum().sum() / self.df.size) * 100,
            },
            'column_types': {
                'numeric': len(self.numeric_cols),
                'categorical': len(self.categorical_cols),
                'datetime': len(self.datetime_cols)
            }
        }
        
        # Monetary insights
        if 'total_monetary_value' in self.df.columns:
            money_col = 'total_monetary_value'
            self.insights['monetary'] = {
                'total_value': self.df[money_col].sum(),
                'avg_value': self.df[money_col].mean(),
                'max_value': self.df[money_col].max(),
            }

    def query_data_with_ai(self, query):
        """Enhanced AI query handler with Groq"""
        if not self.llm:
            return "⚠️ KiteIQX Intelligence AI not available."

        try:
            # Create context about the data
            context = f"""
            KiteIQX Intelligence Dataset Analysis:
            
            Dataset Overview:
            - Total Records: {len(self.df):,}
            - Total Columns: {len(self.df.columns)}
            
            Numeric Columns ({len(self.numeric_cols)}): {', '.join(self.numeric_cols[:10])}
            Categorical Columns ({len(self.categorical_cols)}): {', '.join(self.categorical_cols[:10])}
            DateTime Columns ({len(self.datetime_cols)}): {', '.join(self.datetime_cols)}
            
            Data Preview (First 3 rows):
            {self.df.head(3).to_string()}
            
            Statistical Summary (Numeric columns):
            {self.df[self.numeric_cols[:5]].describe().to_string() if self.numeric_cols else 'No numeric columns'}
            
            User Query: {query}
            
            Provide a clear, actionable analysis with specific insights and recommendations.
            """
            
            response = self.llm.predict(context, max_tokens=2000)
            return response
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"


# -------------------------
# UTILITY FUNCTIONS
# -------------------------

def suggest_analysis_methods(analytics, selected_cols):
    """Suggest optimal analysis methods based on selected columns"""
    suggestions = []
    
    numeric_cols = [col for col in selected_cols if col in analytics.numeric_cols]
    categorical_cols = [col for col in selected_cols if col in analytics.categorical_cols]
    datetime_cols = [col for col in selected_cols if col in analytics.datetime_cols]
    
    # Correlation Analysis
    if len(numeric_cols) >= 2:
        suggestions.append({
            'method': 'Correlation Analysis',
            'description': f'Analyze relationships between {len(numeric_cols)} numeric variables',
            'confidence': 'High',
            'use_case': 'Identify which variables move together'
        })
    
    # Scatter Plot Analysis
    if len(numeric_cols) >= 2:
        suggestions.append({
            'method': 'Scatter Plot Analysis',
            'description': 'Visualize relationships between pairs of variables',
            'confidence': 'High',
            'use_case': 'Spot trends, outliers, and patterns'
        })
    
    # Distribution Analysis
    if len(numeric_cols) >= 1:
        suggestions.append({
            'method': 'Distribution Analysis',
            'description': f'Examine the distribution shape of {len(numeric_cols)} variables',
            'confidence': 'High',
            'use_case': 'Understand data spread and identify skewness'
        })
    
    # Predictive Modeling
    if len(numeric_cols) >= 2:
        suggestions.append({
            'method': 'Predictive Modeling',
            'description': 'Build regression models to predict one variable from others',
            'confidence': 'Medium',
            'use_case': 'Forecast values and understand driver relationships'
        })
    
    # Time Series Analysis
    if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
        suggestions.append({
            'method': 'Time Series Analysis',
            'description': 'Analyze trends and patterns over time',
            'confidence': 'High',
            'use_case': 'Identify seasonal patterns and forecast future values'
        })
    
    # Segmentation Analysis
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        suggestions.append({
            'method': 'Segmentation Analysis',
            'description': 'Compare performance across different categories',
            'confidence': 'High',
            'use_case': 'Identify top/bottom performing segments'
        })
    
    return suggestions


def generate_demo_data(demo_type):
    """Generate different types of demo data"""
    np.random.seed(42)
    
    if demo_type == "E-commerce Sales":
        n_records = 1000
        return pd.DataFrame({
            'order_id': [f'ORD{i:06d}' for i in range(1, n_records + 1)],
            'customer_id': [f'CUST{i:05d}' for i in np.random.randint(1, 501, n_records)],
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_records),
            'order_value': np.random.lognormal(4, 0.8, n_records),
            'shipping_cost': np.random.uniform(5, 25, n_records),
            'customer_age': np.random.randint(18, 75, n_records),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Budget'], n_records),
            'delivery_days': np.random.randint(1, 15, n_records),
            'customer_rating': np.random.randint(1, 6, n_records),
        })
    
    elif demo_type == "Employee Data":
        n_records = 500
        return pd.DataFrame({
            'employee_id': [f'EMP{i:04d}' for i in range(1, n_records + 1)],
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_records),
            'salary': np.random.normal(75000, 25000, n_records),
            'age': np.random.randint(22, 65, n_records),
            'years_experience': np.random.randint(0, 25, n_records),
            'performance_score': np.random.normal(3.5, 0.8, n_records),
            'job_satisfaction': np.random.randint(1, 11, n_records),
        })
    
    else:  # Default simple demo
        n_records = 300
        return pd.DataFrame({
            'id': range(1, n_records + 1),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_records),
            'value': np.random.uniform(10, 100, n_records),
            'score': np.random.randint(1, 11, n_records),
            'amount': np.random.uniform(100, 1000, n_records)
        })


def render_welcome_screen():
    """Render welcome screen when no data is loaded"""
    st.markdown("### Welcome to KiteIQX Intelligence")
    st.markdown('<div class="kite-badge">⚡ Powered by Groq Ultra-Fast AI</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **KiteIQX Intelligence** is your AI-powered data analysis companion, delivering lightning-fast insights 
    with the power of Groq's LPU™ inference technology.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🧠 Smart Analytics**
        - Groq-powered AI insights
        - Automatic pattern detection
        - Professional visualizations
        """)
    
    with col2:
        st.markdown("""
        **🔍 Data Quality**
        - Duplicate detection
        - Missing data analysis
        - Outlier identification
        """)
    
    with col3:
        st.markdown("""
        **📊 Advanced Visualization**
        - Predictive modeling
        - Time series analysis
        - Interactive dashboards
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.info("📁 Upload a CSV file or try our demo data in the sidebar to begin your analysis!")


def main():
    # Header
    st.markdown('<div class="main-header">🪁 KiteIQX Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ultra-Fast AI-Powered Data Analytics Platform</div>', unsafe_allow_html=True)
    
    # Groq status badge
    if GROQ_AVAILABLE:
        st.markdown('<div class="kite-badge">✅ Groq AI Active - Lightning Speed Analytics</div>', unsafe_allow_html=True)
    else:
        st.error("⚠️ Groq library not available. Install with: pip install groq")

    # Initialize session state
    if 'analytics' not in st.session_state:
        st.session_state.analytics = None
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = None

    # Sidebar
    with st.sidebar:
        st.header("🔧 KiteIQX Configuration")
        
        # Groq model selection
        st.markdown("### ⚡ Groq AI Engine")
        groq_model = st.selectbox(
            "Select AI Model",
            [
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            help="Choose the Groq model for AI analysis"
        )
        
        st.success("🔑 Groq API: Internal (Free & Fast)")
        
        st.markdown("---")
        st.subheader("📊 Load Your Data")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        
        # URL input
        data_url = st.text_input("Or enter CSV URL:")
        
        # Demo data selector
        demo_options = ["", "E-commerce Sales", "Employee Data", "Simple Demo"]
        selected_demo = st.selectbox("Or try demo data:", demo_options)
        
        # Load data buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Load Data", type="primary"):
                df = None
                try:
                    if uploaded_file:
                        df = pd.read_csv(uploaded_file)
                        st.success("✅ File uploaded!")
                    elif data_url:
                        response = requests.get(data_url, timeout=30)
                        df = pd.read_csv(io.StringIO(response.text))
                        st.success("✅ Data loaded from URL!")
                    
                    if df is not None:
                        initialize_analytics(df, groq_model)
                        
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
        
        with col2:
            if st.button("🎲 Demo Data") and selected_demo:
                try:
                    df = generate_demo_data(selected_demo)
                    initialize_analytics(df, groq_model)
                    st.success(f"✅ Loaded {selected_demo}!")
                except Exception as e:
                    st.error(f"❌ Error loading demo: {str(e)}")
        
        # Data info
        if st.session_state.analytics:
            st.markdown("---")
            st.subheader("📈 Dataset Info")
            stats = st.session_state.analytics.insights['basic_stats']
            st.metric("Rows", f"{stats['rows']:,}")
            st.metric("Columns", stats['columns'])
            st.metric("Data Quality", f"{100 - stats['missing_pct']:.1f}%")

    # Main content
    if st.session_state.analytics is not None:
        render_main_content()
    else:
        render_welcome_screen()


def initialize_analytics(df, groq_model="llama-3.1-70b-versatile"):
    """Initialize analytics engine with Groq"""
    llm = None
    if GROQ_AVAILABLE and GROQ_API_KEY:
        try:
            llm = GroqLLM(api_key=GROQ_API_KEY, model=groq_model)
            st.success(f"🧠 KiteIQX Intelligence initialized with {groq_model}!")
        except Exception as e:
            st.error(f"❌ Error initializing Groq: {str(e)}")
    
    st.session_state.analytics = UniversalAnalytics(df, llm)
    st.session_state.viz_engine = SmartVisualizationEngine(st.session_state.analytics)


def render_main_content():
    """Render the main content tabs"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🚀 Advanced Analytics", 
        "🧠 AI Intelligence", 
        "🎨 Custom Charts", 
        "🗂️ Data Explorer"
    ])
    
    with tab1:
        render_dashboard()
    
    with tab2:
        render_advanced_analytics()
    
    with tab3:
        render_ai_assistant()
    
    with tab4:
        render_custom_charts()
    
    with tab5:
        render_data_explorer()


def render_dashboard():
    """Clean, focused dashboard"""
    analytics = st.session_state.analytics
    insights = analytics.insights
    
    # Key Metrics
    st.subheader("📊 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{insights['basic_stats']['rows']:,}")
    
    with col2:
        st.metric("Columns", insights['basic_stats']['columns'])
    
    with col3:
        if 'monetary' in insights:
            st.metric("Total Value", f"${insights['monetary']['total_value']:,.0f}")
        else:
            st.metric("Numeric Cols", insights['column_types']['numeric'])
    
    with col4:
        missing_pct = insights['basic_stats']['missing_pct']
        st.metric("Data Quality", f"{100 - missing_pct:.1f}%")

    # Data Quality Section
    st.markdown("---")
    st.subheader("🔍 Data Quality Assessment")
    
    quality_status, quality_report = analytics.get_data_quality_report()
    
    if quality_status == "excellent":
        st.markdown(f'<div class="data-quality-excellent">{quality_report}</div>', unsafe_allow_html=True)
    elif quality_status in ["good", "fair"]:
        st.markdown(f'<div class="data-quality-warning">{quality_report}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="data-quality-poor">{quality_report}</div>', unsafe_allow_html=True)
    
    # Quick actions
    if quality_status != "excellent":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clean Duplicates"):
                result = analytics.clean_duplicates()
                st.success(result)
                st.rerun()
        with col2:
            if st.button("🔄 Refresh Quality Check"):
                analytics.check_data_quality()
                st.success("✅ Quality check refreshed!")
                st.rerun()

    # Quick Insights
    st.markdown("---")
    st.subheader("💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Column Summary**")
        st.write(f"📊 Numeric columns: {len(analytics.numeric_cols)}")
        st.write(f"🏷️ Categorical columns: {len(analytics.categorical_cols)}")
        st.write(f"📅 DateTime columns: {len(analytics.datetime_cols)}")
    
    with col2:
        st.markdown("**📈 Data Summary**")
        if analytics.numeric_cols:
            primary_col = analytics.numeric_cols[0]
            mean_val = analytics.df[primary_col].mean()
            st.write(f"Average {primary_col}: {mean_val:.2f}")


def render_advanced_analytics():
    """Enhanced advanced analytics tab"""
    st.subheader("🚀 Advanced Analytics")
    
    analytics = st.session_state.analytics
    viz_engine = st.session_state.viz_engine
    
    # Column Selection
    st.markdown("### Step 1: Select Columns for Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        available_cols = analytics.df.columns.tolist()
        selected_cols = st.multiselect(
            "Choose columns to analyze:",
            available_cols,
            default=available_cols[:5],
            help="Select the columns you want to include in your analysis"
        )
    
    with col2:
        if selected_cols:
            st.markdown("**Selected Columns:**")
            for col in selected_cols:
                col_type = "📊 Numeric" if col in analytics.numeric_cols else "🏷️ Categorical" if col in analytics.categorical_cols else "📅 DateTime" if col in analytics.datetime_cols else "❓ Other"
                st.write(f"• {col} ({col_type})")

    if not selected_cols:
        st.warning("⚠️ Please select at least one column to proceed with analysis.")
        return

    # Method Suggestions
    st.markdown("---")
    st.markdown("### Step 2: Recommended Analysis Methods")
    
    suggestions = suggest_analysis_methods(analytics, selected_cols)
    
    if not suggestions:
        st.info("ℹ️ No analysis methods available for the selected columns. Try selecting different column types.")
        return
    
    # Display methods
    for i, suggestion in enumerate(suggestions):
        with st.expander(f"📊 {suggestion['method']} (Confidence: {suggestion['confidence']})", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Description:** {suggestion['description']}")
                st.markdown(f"**Use Case:** {suggestion['use_case']}")
            
            with col2:
                if st.button(f"▶️ Run", key=f"run_{i}", type="primary"):
                    with st.spinner(f"⚡ Running {suggestion['method']}..."):
                        try:
                            fig, message = run_simple_analysis(suggestion['method'], selected_cols, viz_engine, analytics)
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                                
                        except Exception as e:
                            st.error(f"❌ Error running analysis: {str(e)}")


def run_simple_analysis(method, selected_cols, viz_engine, analytics):
    """Run simple analysis methods"""
    numeric_cols = [col for col in selected_cols if col in analytics.numeric_cols]
    categorical_cols = [col for col in selected_cols if col in analytics.categorical_cols]
    datetime_cols = [col for col in selected_cols if col in analytics.datetime_cols]
    
    if method == 'Correlation Analysis':
        return viz_engine.create_correlation_matrix(selected_cols)
    
    elif method == 'Distribution Analysis':
        return viz_engine.create_distribution_analysis(selected_cols)
    
    elif method == 'Scatter Plot Analysis' and len(numeric_cols) >= 2:
        color_col = categorical_cols[0] if categorical_cols else None
        return viz_engine.create_scatter_plot(numeric_cols[0], numeric_cols[1], color_col)
    
    elif method == 'Predictive Modeling' and len(numeric_cols) >= 2:
        target = numeric_cols[-1]
        predictors = numeric_cols[:-1]
        return viz_engine.create_predictive_model(target, predictors)
    
    elif method == 'Time Series Analysis':
        if datetime_cols and numeric_cols:
            return viz_engine.create_time_series_analysis(datetime_cols[0], numeric_cols[0])
    
    return None, "Unable to run analysis with current selection"


def render_ai_assistant():
    """AI assistant tab with Groq"""
    st.subheader("🧠 KiteIQX Intelligence AI Assistant")
    
    analytics = st.session_state.analytics
    
    if not analytics.llm:
        st.warning("⚠️ AI Assistant requires Groq initialization.")
        return
    
    st.markdown("""
    ### ⚡ Ask KiteIQX Intelligence to Analyze Your Data
    
    The AI can provide deep insights, identify patterns, and answer complex questions about your data 
    at lightning speed with Groq's ultra-fast inference.
    """)
    
    # Quick action buttons
    st.markdown("**🚀 Quick Actions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Data Overview"):
            query = "Give me a comprehensive overview of this dataset with key insights"
            with st.spinner("⚡ KiteIQX analyzing..."):
                response = analytics.query_data_with_ai(query)
                st.markdown('<div class="ai-response-box">', unsafe_allow_html=True)
                st.markdown("**KiteIQX Intelligence Response:**")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔗 Find Correlations"):
            query = "Identify the strongest correlations in the data and explain their significance"
            with st.spinner("⚡ KiteIQX analyzing..."):
                response = analytics.query_data_with_ai(query)
                st.markdown('<div class="ai-response-box">', unsafe_allow_html=True)
                st.markdown("**KiteIQX Intelligence Response:**")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("📈 Identify Trends"):
            query = "Identify and explain key trends and patterns in the data"
            with st.spinner("⚡ KiteIQX analyzing..."):
                response = analytics.query_data_with_ai(query)
                st.markdown('<div class="ai-response-box">', unsafe_allow_html=True)
                st.markdown("**KiteIQX Intelligence Response:**")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if st.button("⚠️ Spot Outliers"):
            query = "Find and explain any outliers or anomalies in the data"
            with st.spinner("⚡ KiteIQX analyzing..."):
                response = analytics.query_data_with_ai(query)
                st.markdown('<div class="ai-response-box">', unsafe_allow_html=True)
                st.markdown("**KiteIQX Intelligence Response:**")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)

    # Custom query input
    st.markdown("---")
    st.markdown("**💬 Custom Analysis Request:**")
    
    user_input = st.text_area(
        "Describe what you want to analyze:",
        placeholder="""Examples:
• "What are the main factors driving sales performance?"
• "Identify customer segments and their characteristics"
• "Find anomalies in transaction patterns"
• "Predict which products are likely to perform best next quarter"
""",
        height=120
    )
    
    if st.button("⚡ Send to KiteIQX Intelligence", type="primary") and user_input:
        with st.spinner("⚡ KiteIQX Intelligence processing at lightning speed..."):
            try:
                response = analytics.query_data_with_ai(user_input)
                st.markdown("---")
                st.markdown('<div class="ai-response-box">', unsafe_allow_html=True)
                st.markdown("**🧠 KiteIQX Intelligence Response:**")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Available columns reference
    with st.expander("📋 Available Columns Reference"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Numeric Columns:**")
            for col in analytics.numeric_cols[:10]:
                st.write(f"• {col}")
        
        with col2:
            st.markdown("**🏷️ Categorical Columns:**")
            for col in analytics.categorical_cols[:10]:
                st.write(f"• {col}")
        
        with col3:
            st.markdown("**📅 DateTime Columns:**")
            for col in analytics.datetime_cols:
                st.write(f"• {col}")


def render_custom_charts():
    """Render custom charts interface"""
    st.subheader("🎨 Custom Chart Builder")
    
    analytics = st.session_state.analytics
    viz_engine = st.session_state.viz_engine
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**⚙️ Chart Configuration**")
        
        chart_type = st.selectbox("Chart Type", [
            "Scatter Plot",
            "Correlation Matrix",
            "Distribution Analysis",
            "Predictive Model",
            "Time Series"
        ])
        
        if chart_type == "Scatter Plot":
            if len(analytics.numeric_cols) >= 2:
                x_col = st.selectbox("X-axis", analytics.numeric_cols)
                y_col = st.selectbox("Y-axis", analytics.numeric_cols)
                color_col = st.selectbox("Color by (optional)", [None] + analytics.categorical_cols)
                
                if st.button("🎨 Create Chart", type="primary"):
                    fig, message = viz_engine.create_scatter_plot(x_col, y_col, color_col)
                    if fig:
                        st.session_state.current_chart = fig
                        st.session_state.chart_message = message
            else:
                st.warning("⚠️ Need at least 2 numeric columns")
        
        elif chart_type == "Correlation Matrix":
            selected_cols = st.multiselect("Select columns", analytics.numeric_cols, default=analytics.numeric_cols[:5])
            
            if st.button("🎨 Create Chart", type="primary"):
                fig, message = viz_engine.create_correlation_matrix(selected_cols)
                if fig:
                    st.session_state.current_chart = fig
                    st.session_state.chart_message = message
        
        elif chart_type == "Distribution Analysis":
            selected_cols = st.multiselect("Select columns", analytics.numeric_cols, default=analytics.numeric_cols[:3])
            
            if st.button("🎨 Create Chart", type="primary"):
                fig, message = viz_engine.create_distribution_analysis(selected_cols)
                if fig:
                    st.session_state.current_chart = fig
                    st.session_state.chart_message = message
        
        elif chart_type == "Predictive Model":
            if len(analytics.numeric_cols) >= 2:
                target_col = st.selectbox("Target variable", analytics.numeric_cols)
                predictor_cols = st.multiselect("Predictor variables", 
                                              [col for col in analytics.numeric_cols if col != target_col],
                                              default=[col for col in analytics.numeric_cols if col != target_col][:2])
                
                if st.button("🎨 Create Chart", type="primary"):
                    if predictor_cols:
                        fig, message = viz_engine.create_predictive_model(target_col, predictor_cols)
                        if fig:
                            st.session_state.current_chart = fig
                            st.session_state.chart_message = message
                    else:
                        st.error("❌ Select at least one predictor")
            else:
                st.warning("⚠️ Need at least 2 numeric columns")
        
        elif chart_type == "Time Series":
            if analytics.datetime_cols and analytics.numeric_cols:
                date_col = st.selectbox("Date column", analytics.datetime_cols)
                value_col = st.selectbox("Value column", analytics.numeric_cols)
                
                if st.button("🎨 Create Chart", type="primary"):
                    fig, message = viz_engine.create_time_series_analysis(date_col, value_col)
                    if fig:
                        st.session_state.current_chart = fig
                        st.session_state.chart_message = message
            else:
                st.warning("⚠️ Need datetime and numeric columns")
    
    with col2:
        if hasattr(st.session_state, 'current_chart'):
            st.plotly_chart(st.session_state.current_chart, use_container_width=True)
            if hasattr(st.session_state, 'chart_message'):
                st.success(f"✅ {st.session_state.chart_message}")
        else:
            st.info("ℹ️ Configure and create a chart to see it here")


def render_data_explorer():
    """Render data explorer"""
    st.subheader("🗂️ Data Explorer")
    
    analytics = st.session_state.analytics
    df = analytics.df
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**🔍 Filters**")
        
        selected_cols = st.multiselect("Columns to view", 
                                     df.columns.tolist(), 
                                     default=df.columns.tolist()[:10])
        
        row_limit = st.slider("Rows to show", 10, min(1000, len(df)), 100)
        
        view_type = st.radio("View type", ["Head", "Sample", "Tail"])
        
        search_term = st.text_input("🔎 Search in data:")
    
    with col2:
        try:
            display_df = df[selected_cols] if selected_cols else df
            
            if search_term:
                text_cols = display_df.select_dtypes(include=['object']).columns
                if len(text_cols) > 0:
                    mask = display_df[text_cols].astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False, na=False)
                    ).any(axis=1)
                    display_df = display_df[mask]
            
            if view_type == "Head":
                display_df = display_df.head(row_limit)
            elif view_type == "Tail":
                display_df = display_df.tail(row_limit)
            else:
                display_df = display_df.sample(min(row_limit, len(display_df))) if len(display_df) > 0 else display_df
            
            st.dataframe(display_df, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Showing Rows", len(display_df))
            with col2:
                st.metric("Total Rows", len(df))
            with col3:
                st.metric("Columns", len(selected_cols) if selected_cols else len(df.columns))
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Quick statistics
            numeric_display_cols = [col for col in display_df.columns if col in analytics.numeric_cols]
            if numeric_display_cols:
                st.markdown("### 📊 Quick Statistics")
                st.dataframe(display_df[numeric_display_cols].describe(), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error displaying data: {str(e)}")


if __name__ == "__main__":
    main()
