import streamlit as st

# TEST 1: Absolute Minimum
# This is the simplest possible Streamlit app to verify cloud deployment works

st.set_page_config(
    page_title="SHL Test",
    page_icon="🏒",
    layout="wide"
)

st.title("🏒 SHL Monte Carlo Forecast - Test 1")
st.success("✅ TEST 1 PASSED: Minimal Streamlit app is running!")

st.info("""
This is Test 1 of incremental deployment testing.
If you can see this message, Streamlit Cloud is working!

Next test will add basic imports.
""")

st.write("Python imports working:")
st.write("- streamlit ✓")

# Show environment info
import sys
st.write(f"Python version: {sys.version}")
