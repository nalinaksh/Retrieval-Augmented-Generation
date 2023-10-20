import streamlit as st
from QA_Langchain_Memory import *
from PIL import Image
import urllib.request

# streamlit_app.py

#User authentication
import hmac

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()

# Main Streamlit app starts here
st.title("\t\tAsk from AOY")
url = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Paramahansa_Yogananda_Standard_Pose.jpg"
urllib.request.urlretrieve(url, "Yogananda.jpg")
image = Image.open("Yogananda.jpg")
new_image = image.resize((200, 300))

#To center the image...
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image(new_image, caption='Sri Sri Paramhansa Yogananda')
with col3:
    st.write(' ')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Clear chat messages
def clear_chat_history():
    st.session_state.messages = []
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# React to user input
if prompt := st.chat_input("Ask any spiritual question"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # response = ask(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain({"question": prompt})
            response = result["answer"]
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
