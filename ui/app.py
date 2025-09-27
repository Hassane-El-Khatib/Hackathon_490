"""
CardGuard - Credit Card Fraud Detection System
An interface for fraud detection with EDA visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="CardGuard - Fraud Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load external CSS
def load_css():
    with open("ui/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Load and process data
@st.cache_data
def load_transaction_data():
    """Load the cleaned and processed transaction data"""
    try:
        # Load the cleaned dataset from EDA.py
        df = pd.read_csv("bank_transactions_processed_final.csv")
        
        # Convert TransactionDate to datetime if it exists
        if 'TransactionDate' in df.columns:
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please run EDA.py first to generate the processed data.")
        return pd.DataFrame()

# Load data
data = load_transaction_data()

# Header
st.markdown("""
<div class="main-title">🛡️ CardGuard</div>
<div class="subtitle">Advanced Fraud Detection System</div>
""", unsafe_allow_html=True)

# Welcome message
st.markdown("""
<div class="welcome-message">
    Welcome to CardGuard, your intelligent fraud detection companion. 
    Protect your transactions with cutting-edge machine learning technology.
</div>
""", unsafe_allow_html=True)

if not data.empty:
    # Data Overview Metrics
    st.markdown("### 📈 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon" style="font-size: 2.5rem;">📊</div>
            <div class="feature-title">Total Transactions</div>
            <div class="feature-desc">{len(data):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fraud_count = data['fraud_label'].sum() if 'fraud_label' in data.columns else 0
        fraud_rate = (fraud_count / len(data) * 100) if len(data) > 0 else 0
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon" style="font-size: 2.5rem;">🔍</div>
            <div class="feature-title">Fraud Detected</div>
            <div class="feature-desc">{fraud_count} ({fraud_rate:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_amount = data['TransactionAmount'].mean() if 'TransactionAmount' in data.columns else 0
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon" style="font-size: 2.5rem;">💰</div>
            <div class="feature-title">Average Amount</div>
            <div class="feature-desc">${avg_amount:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'MerchantID' in data.columns:
            unique_merchants = data['MerchantID'].nunique()
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon" style="font-size: 2.5rem;">🏪</div>
                <div class="feature-title">Unique Merchants</div>
                <div class="feature-desc">{unique_merchants:,}</div>
            </div>
            """, unsafe_allow_html=True)
        elif 'AccountID' in data.columns:
            unique_accounts = data['AccountID'].nunique()
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon" style="font-size: 2.5rem;">👥</div>
                <div class="feature-title">Unique Accounts</div>
                <div class="feature-desc">{unique_accounts:,}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback to average transaction duration if available
            if 'TransactionDuration' in data.columns:
                avg_duration = data['TransactionDuration'].mean()
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon" style="font-size: 2.5rem;">⏱️</div>
                    <div class="feature-title">Avg Duration</div>
                    <div class="feature-desc">{avg_duration:.1f}s</div>
                </div>
                """, unsafe_allow_html=True)

    # Transaction Analytics Section
    st.markdown("### 📊 Transaction Analytics")

    # Transaction Amount Analysis
    st.markdown("### 💰 Transaction Amount Distribution")
    col1, col2 = st.columns(2)

    with col1:
        # Histogram
        fig_hist = px.histogram(
            data,
            x='TransactionAmount',
            nbins=30,
            title="Transaction Amount Distribution",
            labels={'TransactionAmount': 'Transaction Amount ($)', 'count': 'Frequency'},
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_family="Poppins",
            title_font_color="#f7fafc",
            font=dict(family="Inter", color="#e2e8f0")
        )
        fig_hist.update_xaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        fig_hist.update_yaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Boxplot
        fig_box = px.box(
            data,
            y='TransactionAmount',
            title="Transaction Amount Distribution",
            labels={'TransactionAmount': 'Transaction Amount ($)'},
            color_discrete_sequence=['#764ba2']
        )
        fig_box.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_family="Poppins",
            title_font_color="#f7fafc",
            font=dict(family="Inter", color="#e2e8f0")
        )
        fig_box.update_xaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        fig_box.update_yaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        st.plotly_chart(fig_box, use_container_width=True)

# Daily/Weekly Transaction Volume
if not data.empty and 'TransactionDate' in data.columns and not data['TransactionDate'].isna().all():
    st.markdown("### 📈 Transaction Volume Over Time")
    
    # Create daily aggregation
    daily_data = data.groupby(data['TransactionDate'].dt.date).agg({
        'TransactionAmount': ['count', 'sum']
    }).reset_index()
    
    daily_data.columns = ['Date', 'Transaction_Count', 'Total_Amount']
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])
    
    # Create dual-axis chart
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Add transaction count
    fig.add_trace(
        go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Transaction_Count'],
            mode='lines+markers',
            name='Transaction Count',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        ),
        secondary_y=False
    )
    
    # Add total amount
    fig.add_trace(
        go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Total_Amount'],
            mode='lines+markers',
            name='Total Amount ($)',
            line=dict(color='#f093fb', width=2),
            marker=dict(size=4)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Daily Transaction Volume and Count",
        height=500,
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=0, r=0, b=0),
        title_font_family="Poppins",
        title_font_color="#f7fafc",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(
            font_color="#e2e8f0",
            bgcolor="rgba(0,0,0,0)"
        )
    )
    
    fig.update_xaxes(title_text="Date", title_font_color="#cbd5e0", tickfont_color="#a0aec0")
    fig.update_yaxes(title_text="Number of Transactions", secondary_y=False, title_font_color="#cbd5e0", tickfont_color="#a0aec0")
    fig.update_yaxes(title_text="Total Amount ($)", secondary_y=True, title_font_color="#cbd5e0", tickfont_color="#a0aec0")
    
    st.plotly_chart(fig, use_container_width=True)

# Weekly analysis
if not data.empty and 'TransactionDate' in data.columns:
    st.markdown("### 📅 Weekly Transaction Pattern")
    
    # Add day of week analysis
    data_with_weekday = data.copy()
    data_with_weekday['DayOfWeek'] = data_with_weekday['TransactionDate'].dt.day_name()
    data_with_weekday['Hour'] = data_with_weekday['TransactionDate'].dt.hour
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekly pattern
        weekly_data = data_with_weekday.groupby('DayOfWeek')['TransactionAmount'].agg(['count', 'sum']).reset_index()
        
        # Order days correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_data['DayOfWeek'] = pd.Categorical(weekly_data['DayOfWeek'], categories=day_order, ordered=True)
        weekly_data = weekly_data.sort_values('DayOfWeek')
        
        fig_week_count = px.bar(
            weekly_data,
            x='DayOfWeek',
            y='count',
            title="Transactions by Day of Week",
            labels={'count': 'Number of Transactions', 'DayOfWeek': 'Day'},
            color='count',
            color_continuous_scale='viridis'
        )
        fig_week_count.update_layout(
            height=400, 
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_family="Poppins",
            title_font_color="#f7fafc",
            font=dict(family="Inter", color="#e2e8f0")
        )
        fig_week_count.update_xaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        fig_week_count.update_yaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        st.plotly_chart(fig_week_count, use_container_width=True)
    
    with col2:
        # Hourly pattern
        hourly_data = data_with_weekday.groupby('Hour')['TransactionAmount'].count().reset_index()
        
        fig_hour = px.line(
            hourly_data,
            x='Hour',
            y='TransactionAmount',
            title="Transactions by Hour of Day",
            labels={'TransactionAmount': 'Number of Transactions'},
            markers=True
        )
        fig_hour.update_traces(line_color='#764ba2', marker_color='#667eea')
        fig_hour.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_family="Poppins",
            title_font_color="#f7fafc",
            font=dict(family="Inter", color="#e2e8f0")
        )
        fig_hour.update_xaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        fig_hour.update_yaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        st.plotly_chart(fig_hour, use_container_width=True)

# Transaction Type Analysis
if not data.empty and 'TransactionType' in data.columns:
    st.markdown("### 💳 Transaction Type Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction Type Distribution
        type_counts = data['TransactionType'].value_counts()
        fig_type = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title="Transactions by Type",
            labels={'x': 'Transaction Type', 'y': 'Count'},
            color=type_counts.values,
            color_continuous_scale='viridis'
        )
        fig_type.update_layout(
            height=400, 
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_family="Poppins",
            title_font_color="#f7fafc",
            font=dict(family="Inter", color="#e2e8f0")
        )
        fig_type.update_xaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        fig_type.update_yaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
        st.plotly_chart(fig_type, use_container_width=True)
    
    with col2:
        # Fraud rate by transaction type
        if 'fraud_label' in data.columns:
            fraud_by_type = data.groupby('TransactionType')['fraud_label'].agg(['count', 'sum']).reset_index()
            fraud_by_type['fraud_rate'] = (fraud_by_type['sum'] / fraud_by_type['count'] * 100)
            
            fig_fraud_type = px.bar(
                fraud_by_type,
                x='TransactionType',
                y='fraud_rate',
                title="Fraud Rate by Transaction Type",
                labels={'fraud_rate': 'Fraud Rate (%)', 'TransactionType': 'Transaction Type'},
                color='fraud_rate',
                color_continuous_scale='Reds'
            )
            fig_fraud_type.update_layout(
                height=400, 
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_family="Poppins",
                title_font_color="#f7fafc",
                font=dict(family="Inter", color="#e2e8f0")
            )
            fig_fraud_type.update_xaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
            fig_fraud_type.update_yaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
            st.plotly_chart(fig_fraud_type, use_container_width=True)

# Advanced Fraud Insights
if not data.empty and 'fraud_label' in data.columns and 'LoginAttempts' in data.columns:
    st.markdown("### 🔬 Advanced Fraud Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Login Attempts vs Fraud
        fig_login = px.box(
            data,
            x='fraud_label',
            y='LoginAttempts',
            title="Login Attempts: Legitimate vs Fraudulent",
            labels={'fraud_label': 'Classification', 'LoginAttempts': 'Login Attempts'},
            color='fraud_label',
            color_discrete_sequence=['#667eea', '#ff6b6b']
        )
        
        fig_login.update_xaxes(
            tickvals=[0, 1], 
            ticktext=['Legitimate', 'Fraudulent'],
            title_font=dict(family="Poppins", size=14, color="#cbd5e0"),
            tickfont_color="#a0aec0"
        )
        
        fig_login.update_yaxes(
            title_font=dict(family="Poppins", size=14, color="#cbd5e0"),
            tickfont_color="#a0aec0"
        )
        
        fig_login.update_layout(
            height=400,
            showlegend=False,
            title_font_family="Poppins",
            title_font_size=16,
            title_font_color="#f7fafc",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12, color="#e2e8f0"),
            margin=dict(t=50, l=20, r=20, b=20)
        )
        
        st.plotly_chart(fig_login, use_container_width=True)
    
    with col2:
        # Channel Distribution for Fraud
        if 'Channel' in data.columns:
            channel_fraud = data[data['fraud_label'] == 1]['Channel'].value_counts()
            if not channel_fraud.empty:
                fig_channel = px.pie(
                    values=channel_fraud.values,
                    names=channel_fraud.index,
                    title="Fraud Distribution by Channel",
                    color_discrete_sequence=['#667eea', '#f093fb', '#4facfe']
                )
                fig_channel.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_family="Poppins",
                    title_font_color="#f7fafc",
                    font=dict(family="Inter", color="#e2e8f0"),
                    showlegend=True,
                    legend=dict(
                        font_color="#e2e8f0",
                        bgcolor="rgba(0,0,0,0)"
                    )
                )
                fig_channel.update_traces(
                    textfont_color="#f7fafc",
                    textinfo='label+percent'
                )
                st.plotly_chart(fig_channel, use_container_width=True)

# Additional Fraud Analysis
if not data.empty and 'fraud_label' in data.columns:
    st.markdown("### 🎯 Detailed Fraud Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud vs Amount Analysis
        if 'TransactionAmount' in data.columns:
            fig_fraud_amount = px.violin(
                data,
                x='fraud_label',
                y='TransactionAmount',
                title="Transaction Amount by Fraud Status",
                labels={'fraud_label': 'Fraud Status', 'TransactionAmount': 'Transaction Amount ($)'},
                color='fraud_label',
                color_discrete_sequence=['#667eea', '#ff6b6b']
            )
            fig_fraud_amount.update_xaxes(
                tickvals=[0, 1], 
                ticktext=['Legitimate', 'Fraudulent'],
                title_font_color="#cbd5e0", 
                tickfont_color="#a0aec0"
            )
            fig_fraud_amount.update_yaxes(
                title_font_color="#cbd5e0", 
                tickfont_color="#a0aec0",
                range=[0, data['TransactionAmount'].max() * 1.05]
            )
            fig_fraud_amount.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_family="Poppins",
                title_font_color="#f7fafc",
                font=dict(family="Inter", color="#e2e8f0")
            )
            st.plotly_chart(fig_fraud_amount, use_container_width=True)
    
    with col2:
        # Time-based fraud analysis
        if 'TransactionDate' in data.columns:
            fraud_by_date = data.groupby(data['TransactionDate'].dt.date)['fraud_label'].agg(['count', 'sum']).reset_index()
            fraud_by_date['fraud_rate'] = (fraud_by_date['sum'] / fraud_by_date['count'] * 100)
            fraud_by_date['TransactionDate'] = pd.to_datetime(fraud_by_date['TransactionDate'])
            
            fig_fraud_time = px.line(
                fraud_by_date,
                x='TransactionDate',
                y='fraud_rate',
                title="Daily Fraud Rate Trends",
                labels={'fraud_rate': 'Fraud Rate (%)', 'TransactionDate': 'Date'},
                markers=True
            )
            fig_fraud_time.update_traces(line_color='#ff6b6b', marker_color='#e74c3c')
            fig_fraud_time.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_family="Poppins",
                title_font_color="#f7fafc",
                font=dict(family="Inter", color="#e2e8f0")
            )
            fig_fraud_time.update_xaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
            fig_fraud_time.update_yaxes(title_font_color="#cbd5e0", tickfont_color="#a0aec0")
            st.plotly_chart(fig_fraud_time, use_container_width=True)

# Statistical Summary
if not data.empty:
    st.markdown("### 📊 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'TransactionAmount' in data.columns:
            st.markdown(f"""
            **Transaction Amount Statistics:**
            - Mean: ${data['TransactionAmount'].mean():.2f}
            - Median: ${data['TransactionAmount'].median():.2f}
            - Std Dev: ${data['TransactionAmount'].std():.2f}
            - Min: ${data['TransactionAmount'].min():.2f}
            - Max: ${data['TransactionAmount'].max():.2f}
            """)
    
    with col2:
        if 'fraud_label' in data.columns:
            fraud_stats = data['fraud_label'].value_counts()
            st.markdown(f"""
            **Fraud Detection Summary:**
            - Total Transactions: {len(data):,}
            - Legitimate: {fraud_stats.get(0, 0):,}
            - Fraudulent: {fraud_stats.get(1, 0):,}
            - Fraud Rate: {(fraud_stats.get(1, 0) / len(data) * 100):.2f}%
            """)
    
    with col3:
        if 'TransactionDate' in data.columns:
            date_stats = data['TransactionDate']
            duration = (date_stats.max() - date_stats.min()).days
            daily_avg = len(data) / max(duration, 1)
            st.markdown(f"""
            **Time Period Analysis:**
            - Start Date: {date_stats.min().strftime('%Y-%m-%d')}
            - End Date: {date_stats.max().strftime('%Y-%m-%d')}
            - Duration: {duration} days
            - Avg Daily Transactions: {daily_avg:.1f}
            """)


# Interactive Data Explorer
if not data.empty:
    st.markdown("### 🔍 Interactive Data Explorer")
    
    with st.expander("🎛️ Customize Your Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount filter
            if 'TransactionAmount' in data.columns:
                min_amount, max_amount = st.slider(
                    "Transaction Amount Range ($)",
                    min_value=float(data['TransactionAmount'].min()),
                    max_value=float(data['TransactionAmount'].max()),
                    value=(float(data['TransactionAmount'].min()), float(data['TransactionAmount'].max())),
                    step=1.0
                )
                
                filtered_data = data[
                    (data['TransactionAmount'] >= min_amount) & 
                    (data['TransactionAmount'] <= max_amount)
                ]
                
                st.write(f"**Filtered Dataset:** {len(filtered_data):,} transactions")
        
        with col2:
            # Show filtered statistics
            if 'fraud_label' in data.columns and len(filtered_data) > 0:
                filtered_fraud_rate = (filtered_data['fraud_label'].sum() / len(filtered_data) * 100)
                st.metric(
                    "Fraud Rate in Filtered Data",
                    f"{filtered_fraud_rate:.2f}%",
                    delta=f"{filtered_fraud_rate - (data['fraud_label'].sum() / len(data) * 100):.2f}% vs overall"
                )

# Risk Assessment Summary
if not data.empty and 'fraud_label' in data.columns:
    st.markdown("### ⚠️ Risk Assessment Summary")
    
    fraud_rate = (data['fraud_label'].sum() / len(data) * 100)
    
    if fraud_rate > 5:
        risk_level = "🔴 HIGH RISK"
        risk_color = "#e74c3c"
        recommendations = [
            "Implement additional security measures",
            "Review high-value transactions manually",
            "Consider implementing real-time fraud detection",
            "Enhance customer authentication protocols"
        ]
    elif fraud_rate > 2:
        risk_level = "🟡 MEDIUM RISK"
        risk_color = "#f39c12"
        recommendations = [
            "Monitor transaction patterns closely",
            "Implement automated fraud alerts",
            "Review authentication procedures",
            "Consider transaction limits for new accounts"
        ]
    else:
        risk_level = "🟢 LOW RISK"
        risk_color = "#27ae60"
        recommendations = [
            "Continue current security measures",
            "Regular monitoring of patterns",
            "Maintain updated fraud detection models",
            "Periodic security audits"
        ]
    
    st.markdown(f"""
    <div style="padding: 20px; border-left: 5px solid {risk_color}; background-color: rgba(255,255,255,0.1); border-radius: 10px; margin: 10px 0;">
        <h4 style="color: {risk_color}; margin: 0 0 10px 0;">{risk_level}</h4>
        <p style="color: #e2e8f0; margin: 0;">Current fraud rate: {fraud_rate:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Recommended Actions:**")
    for rec in recommendations:
        st.markdown(f"• {rec}")

else:
    # Fallback for when no data is available
    st.markdown("""
    ### 🚀 Get Started
    
    To view transaction analytics:
    1. Run the EDA.py script to process your transaction data
    2. Refresh this page to see detailed visualizations
    3. Explore fraud detection insights and patterns
    """)

# Footer info
st.markdown("""
<div class="footer-section">
    <div class="footer-text">
        🔒 Your data is secure and encrypted<br>
        Built for EECE490 Hackathon
    </div>
</div>
""", unsafe_allow_html=True)