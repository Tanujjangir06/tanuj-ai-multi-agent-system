
import streamlit as st
from agents import llm, build_search_agent

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="TANUJ AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 TANUJ AI")
st.caption("Your AI Assistant • Chat • Coding • AI/ML • Cybersecurity")

# -----------------------------
# AI Identity and Instructions
# -----------------------------
SYSTEM_PROMPT = """
You are TANUJ AI, a personal AI assistant developed
as an independent software project.

Your primary job is to answer the user's question directly.

RESPONSE RULES:
- Give a direct answer first.
- Keep answers concise unless the user asks for details.
- Use simple language and match the user's language.
- For coding questions, provide working code and a short explanation.
- For maths, solve the problem directly.
- For greetings and casual conversation, respond naturally.
- Do not create research reports, lengthy analysis, or tables
  unless the user specifically requests them.
- Do not search the web for ordinary questions.
- If information is uncertain, say so honestly.
- Never claim to have searched the web unless you actually did.

SEARCH RULES:
- Search only when the user explicitly asks to search, research,
  find sources, or look up information online.
- Current information that genuinely requires verification may
  also need a web search.
- A simple question containing a word like "who", "price",
  "today", or a year does not automatically require research.
- If the user asks for research, use the available search tools.

IDENTITY:
- Your name is TANUJ AI.
- You are an independent AI assistant project.
- Do not claim to be the official ChatGPT product.
- Do not invent your underlying model or capabilities.
"""

SEARCH_SYSTEM_PROMPT = SYSTEM_PROMPT + """
The user has requested online research or information verification.

Use available search tools when appropriate.
Prefer relevant and reliable sources.
Answer the user's actual question instead of producing a
long research report by default.
Summarize the findings directly and include sources when available.
Be transparent about uncertainty and missing information.
"""

# -----------------------------
# Chat Memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Welcome Message
# -----------------------------
if not st.session_state.messages:
    st.chat_message("assistant").markdown(
        "Namaste! 👋 Main **TANUJ AI** hoon.\n\n"
        "Mujhse coding, AI/ML, cybersecurity, college studies "
        "ya koi bhi general question poochho.\n\n"
        "**Main seedha answer dunga.** 🚀"
    )

# -----------------------------
# Display Previous Messages
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Detect Explicit Search Requests
# -----------------------------
def needs_web_search(text):
    text = text.lower().strip()

    # Search only when the user clearly requests it.
    search_phrases = [
        "search the web",
        "search online",
        "look it up online",
        "find sources",
        "give me sources",
        "research this",
        "do research",
        "web search",
        "internet par search",
        "online search karo",
        "web par search",
        "internet se dhoondo",
        "research karke batao",
        "latest news",
        "current news",
        "aaj ki news",
        "taaza khabar"
    ]

    return any(phrase in text for phrase in search_phrases)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Assistant: TANUJ AI")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by the model configured in agents.py")

# -----------------------------
# Chat Input
# -----------------------------
prompt = st.chat_input(
    "Message TANUJ AI...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
    accept_audio=True,
    audio_sample_rate=16000
)

if prompt:
    user_text = (prompt.text or "").strip()
    files = prompt.files or []
    audio = prompt.audio

    # -------------------------
    # Display User Message
    # -------------------------
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text)

        for file in files:
            if file.type.startswith("image/"):
                st.image(file, width=400)
            else:
                st.write(f"📎 {file.name}")

        if audio:
            st.audio(audio)

    # -------------------------
    # Process Text Questions
    # -------------------------
    if user_text:
        st.session_state.messages.append({
            "role": "user",
            "content": user_text
        })

        with st.chat_message("assistant"):
            with st.spinner("TANUJ AI is thinking..."):
                try:
                    chat_messages = [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        }
                    ]

                    chat_messages.extend(
                        st.session_state.messages
                    )

                    # -------------------------
                    # Search Only When Requested
                    # -------------------------
                    if needs_web_search(user_text):
                        search_agent = build_search_agent()

                        search_messages = [
                            {
                                "role": "system",
                                "content": SEARCH_SYSTEM_PROMPT
                            }
                        ]

                        search_messages.extend(
                            st.session_state.messages
                        )

                        result = search_agent.invoke({
                            "messages": search_messages
                        })

                        answer = result["messages"][-1].content

                    # -------------------------
                    # Normal Direct Answer
                    # -------------------------
                    else:
                        response = llm.invoke(chat_messages)
                        answer = response.content

                    # Normalize model output
                    if isinstance(answer, list):
                        answer = "\n".join(
                            block.get("text", str(block))
                            if isinstance(block, dict)
                            else str(block)
                            for block in answer
                        )

                    answer = str(answer).strip()

                    # -------------------------
                    # Display and Save Answer
                    # -------------------------
                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(
                        "Response generate nahi ho paya. "
                        "agents.py aur model/API configuration check karo."
                    )
                    st.exception(e)

    elif files or audio:
        st.info(
            "File ya audio receive ho gaya hai. "
            "Usko understand karne ke liye agents.py mein "
            "appropriate image/audio model support zaroori hai."
        )
