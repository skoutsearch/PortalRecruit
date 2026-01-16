import streamlit as st

st.title("SKOUT - Moneyball for Hoops")
query = st.text_input("Describe the play you're looking for:")
if query:
    st.write(f"Searching for: {query}...")