
import streamlit as st

from main import ask_agent


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RealEstateAI",
    page_icon="🏡",
    layout="centered"
)

st.title("🏡 RealEstateAI")

st.caption(
    "Your AI assistant for property information "
    "from DarGlobal and Wasalt."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("RealEstateAI")

    st.write(
        "Ask questions about properties, "
        "locations, prices, and projects."
    )

    st.divider()

    st.write("**Knowledge sources**")
    st.write("- DarGlobal")
    st.write("- Wasalt")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask about properties or projects..."
)

if question:

    history = st.session_state.messages.copy()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Agent Is Thinking"):

            try:

                answer = ask_agent(
                    question,
                    history
                )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:

                st.error(
                    f"Unable to generate answer: {error}"
                )
