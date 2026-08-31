import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Login — RAG Assistant",
    page_icon  = "🔐",
    layout     = "centered"   # centered for login form
)

# ─────────────────────────────────────────
# Imports
# ─────────────────────────────────────────
from auth.auth import (
    init_auth_state,
    is_logged_in,
    get_current_username,
    render_login_page,
    render_auth_sidebar
)

# ─────────────────────────────────────────
# Initialize session state
# ─────────────────────────────────────────
init_auth_state()

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
render_auth_sidebar()

# ─────────────────────────────────────────
# Already logged in → show welcome
# ─────────────────────────────────────────
if is_logged_in():
    st.title("✅ You're logged in!")
    st.success(f"Welcome, **{get_current_username()}**!")

    st.markdown("""
    You are already logged in. Use the sidebar to navigate:

    - 📚 **Documents** — view and manage your PDFs
    - 💬 **Chat** — ask questions about your documents
    - 📊 **Analytics** — view performance metrics
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "📚 Go to My Documents",
            type    = "primary",
            use_container_width = True
        ):
            st.switch_page("pages/1_documents.py")
    with col2:
        if st.button(
            "💬 Go to Chat",
            use_container_width = True
        ):
            st.switch_page("pages/2_chat.py")

# ─────────────────────────────────────────
# Not logged in → show login form
# ─────────────────────────────────────────
else:
    render_login_page()