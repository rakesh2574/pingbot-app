import streamlit as st
import os
from rag_engine import PingbhaiRAG
from dotenv import load_dotenv
import openai

# 1. Page Configuration
st.set_page_config(
    page_title="PingBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. PROFESSIONAL DARK THEME CSS
st.markdown("""
<style>
    .stApp { background-color: #0E1117 !important; color: #FAFAFA !important; }
    section[data-testid="stSidebar"] { background-color: #262730 !important; border-right: 1px solid #333333; }
    h1, h2, h3 { color: #FFFFFF !important; font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    p, li, span, label, .stCheckbox label { color: #E0E0E0 !important; }
    .stButton > button { background-color: transparent !important; color: #FFFFFF !important; border: 1px solid #555555 !important; border-radius: 5px; }
    .stButton > button:hover { border-color: #FFFFFF !important; background-color: #333333 !important; }
    .stTextInput > div > div > input { color: #FFFFFF !important; background-color: #1E1E1E !important; border: 1px solid #444444; }
    .stChatMessage { background-color: #1E1E1E !important; border: 1px solid #333333 !important; border-radius: 10px; }
    header {visibility: hidden;}
    .welcome-card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; border: 1px solid #333333; margin-bottom: 10px; text-align: center; }
    a[href^="mailto"] { text-decoration: none; }
    a[href^="mailto"] button { background-color: #28a745 !important; color: white !important; border: none !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 3. Authentication & Key Logic
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = None


def validate_openai_key(api_key):
    """
    Tries to list models to verify if the key is valid.
    Returns True if valid, False otherwise.
    """
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except openai.AuthenticationError:
        return False
    except Exception:
        return False


def check_login():
    username = st.session_state.get("username", "")
    password = st.session_state.get("password", "")
    use_own_key = st.session_state.get("use_own_key", False)
    custom_key = st.session_state.get("custom_key_input", "")

    # 1. Check Credentials
    if username == "sherlock" and password == "elementary":

        # 2. Check API Key Logic
        if use_own_key:
            if not custom_key:
                st.error("❌ You checked 'Use my own API Key' but didn't enter one.")
                return

            with st.spinner("Validating API Key..."):
                if validate_openai_key(custom_key):
                    st.session_state.user_api_key = custom_key
                    st.session_state.authenticated = True
                    st.toast("Login & Key Validation Successful!", icon="✅")
                    # Clear history on fresh login
                    if 'rag_engine' in globals():
                        try:
                            rag_engine.clear_history()
                        except:
                            pass
                else:
                    st.error("❌ Invalid OpenAI API Key. Please check and try again.")
                    return
        else:
            # User wants to use System/Default Key
            st.session_state.user_api_key = None  # Will rely on os.getenv
            st.session_state.authenticated = True
            st.toast("Login Successful!", icon="✅")
            if 'rag_engine' in globals():
                try:
                    rag_engine.clear_history()
                except:
                    pass
    else:
        st.error("❌ Invalid Username or Password")


def logout():
    st.session_state.authenticated = False
    st.session_state.messages = []
    st.session_state.user_api_key = None

    # Try to clear history if engine exists
    try:
        tmp_rag = PingbhaiRAG()
        tmp_rag.clear_history()
    except:
        pass

    st.rerun()


# 4. Login Page View
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.title("🔒 PingBot Login")

        with st.form("login"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")

            st.markdown("---")
            # Optional API Key Section
            st.checkbox("I have my own OpenAI API Key", key="use_own_key")

            # Note: We can't conditionally hide the text input inside a form easily in Streamlit
            # without rerun, so we display it but only check it if checkbox is true.
            st.text_input("OpenAI API Key (Optional)", type="password", key="custom_key_input",
                          help="Enter if you checked the box above.")

            st.form_submit_button("ENTER SYSTEM", on_click=check_login, type="primary")
    st.stop()

# --- MAIN APP ---

load_dotenv()


# Initialize RAG Engine with Dynamic Key
@st.cache_resource(show_spinner=False)
def get_rag_engine(user_key=None):
    # If user provided a key, use it. Otherwise, RAG engine looks for env var.
    try:
        rag = PingbhaiRAG(api_key=user_key)
        rag.load_and_process_data()
        return rag
    except ValueError as ve:
        st.error(f"Configuration Error: {ve}")
        st.stop()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()


# Load the engine using the session state key (or None)
rag_engine = get_rag_engine(st.session_state.user_api_key)

with st.sidebar:
    st.title("🤖 PingBot")
    st.caption("ESDM Intelligence v2.4")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        rag_engine.clear_history()
        st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Logout", on_click=logout, use_container_width=True):
        pass

st.title("PingBot")
st.caption(f"Logged in as: **Sherlock**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome Screen
if len(st.session_state.messages) == 0:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 👋 Welcome! I'm ready to help.")
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(
        """<div class="welcome-card"><b>🛠️ Equipment</b><br><span style="font-size:0.8em">"Tell me about the VEEFIX BGA Station"</span></div>""",
        unsafe_allow_html=True)
    with col2: st.markdown(
        """<div class="welcome-card"><b>🔍 Inspection</b><br><span style="font-size:0.8em">"How accurate is the Seamark X-Ray?"</span></div>""",
        unsafe_allow_html=True)
    with col3: st.markdown(
        """<div class="welcome-card"><b>📝 Selling</b><br><span style="font-size:0.8em">"How do I list my products?"</span></div>""",
        unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if "[ACTION:CONTACT_SALES]" in content:
            display_text = content.replace("[ACTION:CONTACT_SALES]", "")
            st.markdown(display_text)
            contact_url = "mailto:ping@pingbhai.com?subject=Sales%20Inquiry%20via%20PingBot&body=Hi%20Team%2C%0A%0AI%20am%20interested%20in%20a%20quote%20for..."
            st.link_button("📧 Draft Email to Sales Team", contact_url, type="primary")
        else:
            st.markdown(content)

# Input Handling
if prompt := st.chat_input("Ask about specifications..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Processing..."):
            response = rag_engine.query(prompt)

            if "[ACTION:CONTACT_SALES]" in response:
                clean_response = response.replace("[ACTION:CONTACT_SALES]", "")
                message_placeholder.markdown(clean_response)
                contact_url = "mailto:ping@pingbhai.com?subject=Sales%20Inquiry%20via%20PingBot&body=Hi%20Team%2C%0A%0AI%20would%20like%20to%20request%20a%20quote."
                st.link_button("📧 Draft Email to Sales Team", contact_url, type="primary")
            else:
                message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})