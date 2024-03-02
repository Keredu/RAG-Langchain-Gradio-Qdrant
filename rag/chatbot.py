import gradio as gr

class Chatbot:
    def __init__(self):
        self.interface = gr.Interface(fn=self.chatbot_response,
                                      inputs=gr.Textbox(lines=2, placeholder="Write your question..."),
                                      outputs="text",
                                      title="RAG-Langchain-Gradio-Qdrant",
                                      description="")

    def run(self):
        self.interface.launch()

    def chatbot_response(self, question):
        return f"Has preguntado: {question}. Lo siento, aún estoy aprendiendo y no tengo una respuesta específica para eso."


