
import time
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Import backend functions from main.py
try:
    from main import make_llm, get_executor, to_text, format_research, MODES
except ImportError as e:
    st.error(f"Could not import from main.py: {e}. Ensure app.py is in the exact same folder as main.py.")
    st.stop()

# 1. Page Configuration
st.set_page_config(
    page_title="Aura — Personal AI Co-Pilot",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Helper Function for Typing/Streaming Effect
def stream_data(text: str):
    """Streams text word-by-word to create a smooth typewriter effect."""
    words = text.split(" ")
    for word in words:
        yield word + " "
        time.sleep(0.015)

# 3. Initialize Backend LLM & State
if "agent_state" not in st.session_state:
    with st.spinner("Waking up Aura's brain..."):
        try:
            llm = make_llm()
            st.session_state.agent_state = {
                "mode": "assistant",
                "history": [],
                "llm": llm
            }
        except Exception as e:
            st.error(f"Failed to initialize LLM backend: {e}")
            st.stop()

# Initialize Chat UI History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Sidebar UI & Quick Actions
st.sidebar.title("✨ Aura AI")
st.sidebar.caption("Your private financial & productivity co-pilot.")

# Mode Selector
available_modes = [m for m in MODES.keys() if m != "briefing"]
current_mode = st.session_state.agent_state["mode"]
selected_index = available_modes.index(current_mode) if current_mode in available_modes else 0

agent_persona = st.sidebar.selectbox(
    "Active Persona",
    available_modes,
    index=selected_index
)
st.session_state.agent_state["mode"] = agent_persona

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Actions")

# Helper to execute prompts programmatically from quick buttons
def trigger_quick_prompt(prompt_text: str):
    st.session_state.pending_prompt = prompt_text

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("📰 Briefing", use_container_width=True):
        trigger_quick_prompt("Prepare my daily briefing: market news, tech headlines, and watchlist update.")
    if st.button("📁 View Docs", use_container_width=True):
        trigger_quick_prompt("List all files available in my docs folder.")

with col2:
    if st.button("📝 View Notes", use_container_width=True):
        trigger_quick_prompt("Read and list my saved learning notes.")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.agent_state["history"].clear()
        st.session_state.messages = []
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"**Current Mode:** `{agent_persona.upper()}`")

# 5. Main Header
st.title("Aura")
st.caption("Ask questions, analyze stocks, review documents, or practice mock interviews.")

# 6. Render Prior Chat History
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "✨"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 7. Check for Quick Action or Chat Input
user_prompt = st.chat_input("Message Aura...")
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_prompt = st.session_state.pop("pending_prompt")

# 8. Process Prompt & Generate Response
if user_prompt:
    # Render user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # Render assistant response with live streaming
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Aura is reasoning..."):
            try:
                state = st.session_state.agent_state
                executor, parser = get_executor(state["mode"], state["llm"])
                
                # Query LangChain backend with conversation history memory
                result = executor.invoke({
                    "query": user_prompt, 
                    "chat_history": state["history"][-20:]
                })
                
                raw_output = to_text(result["output"])
                response_text = raw_output
                
                # Apply research parser formatting if applicable
                if parser:
                    try:
                        response_text = format_research(parser.parse(raw_output))
                    except Exception:
                        response_text = raw_output
                
                # Stream response dynamically to the UI
                st.write_stream(stream_data(response_text))
                
                # Update backend memory
                state["history"] += [("human", user_prompt), ("ai", raw_output)]
                
            except Exception as e:
                response_text = f"❌ Error executing agent: {e}"
                st.error(response_text)

    # Save to UI history
    st.session_state.messages.append({"role": "assistant", "content": response_text})