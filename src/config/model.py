from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import OllamaLLM, OllamaEmbeddings


def load_model(model_name) -> RunnableSerializable[dict, str]:
    template = """
    you are an ecommerce professional that answers about stock, and reviews
    another colleague needs help from you and will ask questions about it
    colleague: {input}
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model=model_name, name=model_name)

    chain = prompt | model

    return chain

def load_embedding_model() :
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embeddings
