"""
CardGuard - Credit Card Fraud Detection System
A simple, elegant interface for fraud detection
"""

import streamlit as st

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

# Feature cards
col1, col2, col3 = st.columns(3)

# Footer info
st.markdown("""
<div class="footer-section">
    <div class="footer-text">
        🔒 Your data is secure and encrypted<br>
        Built for EECE490 Hackathon
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
