import streamlit as st
from typing import Optional
from database.connection import get_streamlit_db
from database.queries import UserQueries
from database.models import User


# ─────────────────────────────────────────
# Session state keys
# ─────────────────────────────────────────
# Why constants not raw strings?
#   "user_id" typed wrong → silent bug
#   AUTH_USER_ID typed wrong → NameError (caught immediately)
#   Constants also make refactoring easy
AUTH_USER_ID   = "auth_user_id"
AUTH_USERNAME  = "auth_username"
AUTH_LOGGED_IN = "auth_logged_in"


# ─────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────
def init_auth_state():
    """
    Initialize auth-related session state.
    Call at top of every page.

    Why check before setting?
      Streamlit reruns entire script on every interaction
      Without check: would reset auth state on every rerun
      With check: only sets if key doesn't exist yet
    """
    if AUTH_USER_ID   not in st.session_state:
        st.session_state[AUTH_USER_ID]   = None
    if AUTH_USERNAME  not in st.session_state:
        st.session_state[AUTH_USERNAME]  = None
    if AUTH_LOGGED_IN not in st.session_state:
        st.session_state[AUTH_LOGGED_IN] = False


def is_logged_in() -> bool:
    """Check if user is currently logged in."""
    return st.session_state.get(AUTH_LOGGED_IN, False)


def get_current_user_id() -> Optional[int]:
    """Get current logged-in user's ID."""
    return st.session_state.get(AUTH_USER_ID)


def get_current_username() -> Optional[str]:
    """Get current logged-in user's username."""
    return st.session_state.get(AUTH_USERNAME)


def set_session(user: User):
    """
    Save user info to session state after login.

    Why store user_id and username separately?
      user_id → for all database queries (foreign key)
      username → for display in UI ("Welcome, Sonal!")
      Avoid hitting DB just to show username
    """
    st.session_state[AUTH_USER_ID]   = user.id
    st.session_state[AUTH_USERNAME]  = user.username
    st.session_state[AUTH_LOGGED_IN] = True


def clear_session():
    """
    Clear all session state on logout.

    Why clear everything not just auth keys?
      User A logs out → User B logs in
      User B should NOT see User A's documents/chat
      Full clear ensures clean slate for new user

    Keys to clear:
      auth keys          → login status
      ingested_pdf       → document selection
      chunk_stats        → document stats
      chat_history       → conversation
      query_logs         → analytics data
      current_doc_id     → selected document
      current_conv_id    → selected conversation
    """
    keys_to_clear = [
        AUTH_USER_ID, AUTH_USERNAME, AUTH_LOGGED_IN,
        "ingested_pdf", "chunk_stats", "chat_history",
        "query_logs", "pipeline", "current_doc_id",
        "current_conv_id", "pending_question"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# ─────────────────────────────────────────
# Auth guard
# ─────────────────────────────────────────
def require_login():
    """
    Redirect to login if not authenticated.

    Call at the top of every protected page:
      require_login()
      # rest of page code...

    How it works:
      Not logged in → show warning + stop page execution
      st.stop() halts the rest of the script
      User sees login prompt instead of page content

    Why st.stop() not return?
      return only stops the function
      st.stop() stops the ENTIRE page script
      Nothing after require_login() runs if not logged in
    """
    init_auth_state()

    if not is_logged_in():
        st.warning("⚠️ Please log in to access this page.")
        st.markdown("👈 Go to the **Login** page in the sidebar.")
        st.stop()


# ─────────────────────────────────────────
# Register
# ─────────────────────────────────────────
def register_user(
    username: str,
    email:    str,
    password: str,
    confirm:  str
) -> tuple[bool, str]:
    """
    Register a new user with validation.

    Returns (success, message) tuple.

    Validation order matters:
      1. Client-side checks (no DB hit needed)
      2. DB check (only if basic validation passes)
      This minimizes unnecessary DB queries

    Why return tuple not raise exception?
      Streamlit UI handles success/error differently
      Tuple makes it easy: if success: ... else: show error
    """
    # Basic validation
    if not username or not email or not password:
        return False, "All fields are required"

    if len(username) < 3:
        return False, "Username must be at least 3 characters"

    if len(username) > 50:
        return False, "Username must be under 50 characters"

    if "@" not in email:
        return False, "Invalid email address"

    if len(password) < 6:
        return False, "Password must be at least 6 characters"

    if password != confirm:
        return False, "Passwords do not match"

    # Username can only contain letters, numbers, underscore
    import re
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return False, "Username can only contain letters, numbers, underscore"

    # Create user in database
    try:
        db   = get_streamlit_db()
        user = UserQueries.create_user(db, username, email, password)
        db.commit()
        db.close()
        return True, f"Account created! Welcome, {username}!"

    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Registration failed: {e}"


# ─────────────────────────────────────────
# Login
# ─────────────────────────────────────────
def login_user(username: str, password: str) -> tuple[bool, str]:
    """
    Authenticate user and start session.

    Returns (success, message) tuple.

    Security note:
      On failure: "Invalid username or password"
      Never say which one is wrong (enumeration attack)
    """
    if not username or not password:
        return False, "Username and password are required"

    try:
        db   = get_streamlit_db()
        user = UserQueries.verify_login(db, username, password)
        db.close()

        if user:
            set_session(user)
            return True, f"Welcome back, {user.username}!"
        else:
            return False, "Invalid username or password"

    except Exception as e:
        return False, f"Login failed: {e}"


# ─────────────────────────────────────────
# Logout
# ─────────────────────────────────────────
def logout_user():
    """Log out current user and clear all session data."""
    clear_session()
    st.rerun()


# ─────────────────────────────────────────
# Sidebar auth widget
# ─────────────────────────────────────────
def render_auth_sidebar():
    """
    Render user info and logout button in sidebar.

    Called from every page's sidebar section.
    Shows:
      Logged in:  username + logout button
      Logged out: "Please log in" message
    """
    init_auth_state()

    with st.sidebar:
        if is_logged_in():
            st.markdown("### 👤 Account")
            st.success(f"**{get_current_username()}**")

            if st.button(
                "🚪 Logout",
                use_container_width=True,
                key="logout_btn"
            ):
                logout_user()
        else:
            st.markdown("### 👤 Account")
            st.warning("Not logged in")


# ─────────────────────────────────────────
# Login / Register UI
# ─────────────────────────────────────────
def render_login_page():
    """
    Render the complete login + register UI.

    Uses tabs for clean Login / Register separation.
    Called from app/pages/0_login.py.

    Why tabs not separate pages?
      Login and register are closely related
      Users often switch between them
      Cleaner UX than two separate sidebar pages
    """
    st.title("🔐 RAG PDF Assistant")
    st.markdown(
        "Multimodal Document Q&A — "
        "login to access your documents and conversations."
    )
    st.divider()

    # Center the form
    _, col, _ = st.columns([1, 2, 1])

    with col:
        tab_login, tab_register = st.tabs(["Login", "Register"])

        # ── Login tab ──
        with tab_login:
            st.markdown("#### Welcome back!")

            with st.form("login_form"):
                username = st.text_input(
                    "Username",
                    placeholder="your_username"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="••••••••"
                )
                submitted = st.form_submit_button(
                    "Login",
                    type="primary",
                    use_container_width=True
                )

            if submitted:
                if username and password:
                    success, message = login_user(username, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter username and password")

        # ── Register tab ──
        with tab_register:
            st.markdown("#### Create an account")

            with st.form("register_form"):
                new_username = st.text_input(
                    "Username",
                    placeholder="choose_a_username",
                    help="Letters, numbers, underscore only"
                )
                new_email = st.text_input(
                    "Email",
                    placeholder="you@example.com"
                )
                new_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="at least 6 characters"
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="repeat password"
                )
                submitted_reg = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True
                )

            if submitted_reg:
                success, message = register_user(
                    new_username,
                    new_email,
                    new_password,
                    confirm_password
                )
                if success:
                    st.success(message)
                    st.info("You can now log in with your credentials.")
                else:
                    st.error(message)