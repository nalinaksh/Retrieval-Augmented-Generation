import streamlit as st
from QA_Langchain import *
from PIL import Image
import urllib.request

st.title("Ask questions from Autobiography of a Yogi")
url = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Paramahansa_Yogananda_Standard_Pose.jpg"
urllib.request.urlretrieve(url, "Yogananda.jpg")
image = Image.open("Yogananda.jpg")
new_image = image.resize((600, 400))
st.image(new_image, caption='Sri Sri Paramhansa Yogananda')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = ask(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
