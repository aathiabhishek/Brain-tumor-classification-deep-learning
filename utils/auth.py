import streamlit as st


def check_login(username, password):
    """Validate login credentials stored in Streamlit Secrets."""

    try:
        return (
            username == st.secrets["auth"]["username"]
            and password == st.secrets["auth"]["password"]
        )

    except KeyError:
        st.error("Authentication credentials are not configured.")
        return False