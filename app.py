!pip install -q gradio
import gradio as gr
import time

#Extract saved embeddings, it takes some time, so wait for around 5 minutes
!jar xvf HuggingFaceEmbeddings.jar
time.sleep(300)

def slow_echo(question, history):
  message = ask(question)
  for i in range(len(message)):
    time.sleep(0.05)
    yield message[: i+1]

title = "Ask anything from \"Autobiography of a Yogi\" by Sri Sri Paramhansa Yogananda!"
description="""
<center><img src="https://upload.wikimedia.org/wikipedia/commons/3/3f/Paramahansa_Yogananda_Standard_Pose.jpg" width=200px></center>
"""

demo = gr.ChatInterface(slow_echo, title=title, description=description).queue()
if __name__ == "__main__":
    demo.launch()
