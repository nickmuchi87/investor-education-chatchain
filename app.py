from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import os
from typing import Optional, Tuple
import gradio as gr
import pickle
from threading import Lock

model_options = {'all-mpnet-base-v2': "sentence-transformers/all-mpnet-base-v2",
                'instructor-base': "hkunlp/instructor-base"}

model_options_list = list(model_options.keys())

def load_vectorstore(model):
    '''load embeddings and vectorstore'''

    if 'mpnet' in model:
        
        emb = HuggingFaceEmbeddings(model_name=model)
        print(emb)
        return FAISS.load_local('vanguard-embeddings', emb)

    elif 'instructor'in model:
        
        emb = HuggingFaceInstructEmbeddings(model_name=model,
                                               query_instruction='Represent the Financial question for retrieving supporting paragraphs: ',
                                               embed_instruction='Represent the Financial paragraph for retrieval: ')
        print(emb)
        return FAISS.load_local('vanguard_embeddings_inst', emb)

#default embeddings
vectorstore = load_vectorstore(model_options['all-mpnet-base-v2'])

def on_value_change(change):
    '''When radio changes, change the embeddings'''
    global vectorstore
    vectorstore = load_vectorstore(model_options[change])
    
# vectorstore = load_vectorstore('vanguard-embeddings',sbert_emb)
    
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about investing and the investment management industry.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about investing and the investment management industry.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about investing, politely inform them that you are tuned to only answer questions about investing and the investment management industry.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain

def set_openai_api_key(api_key: str):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = get_chain(vectorstore)
        os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain({"question": inp, "chat_history": history})["answer"]
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Chat-Your-Data (Investor Education)</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    with gr.Row():
        embeddings = gr.Radio(choices=model_options_list,value=model_options_list[0], label='Choose your Embedding Model',
                             interactive=True)
        embeddings.change(on_value_change, embeddings)
    
    vectorstore = load_vectorstore(embeddings.value)

    print(vectorstore)

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask questions about Investing",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What are the benefits of investing in ETFs?",
            "What is the average cost of investing in a managed fund?",
            "At what age can I start investing?",
            "Do you offer investment accounts for kids?"
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a LangChain chain.")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    
    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

    gr.Markdown("![visitor badge](https://visitor-badge.glitch.me/badge?page_id=nickmuchi-investor-chatchain)")

block.launch(debug=True)
