import streamlit as st
from Authenticate import *
from QA_Langchain_Memory import *
from PIL import Image
import urllib.request

#Authenticate user
if not check_password():
    st.stop()

# Main Streamlit app starts here
# st.title("Ask from AOY")
st.markdown("<h1 style='text-align: center; color: grey;'>Ask from AOY</h1>", unsafe_allow_html=True)
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

vectorstore = get_vecstore()

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
            answer = result["answer"]
            answer += "\n\nContext: "
            docs = vectorstore.similarity_search(prompt)
            for doc in docs:
                answer += "\n" + doc.page_content
            st.markdown(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
