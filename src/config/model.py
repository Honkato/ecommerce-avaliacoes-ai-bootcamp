from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama


def load_model(model_name) -> ChatOllama:
    model = ChatOllama(model=model_name, name=model_name)

    chain = model

    return chain

def load_embedding_model() :
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embeddings
