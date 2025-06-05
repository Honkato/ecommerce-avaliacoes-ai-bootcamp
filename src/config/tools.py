from typing import List
from model.topics_model import Topics
from langchain_core.tools import tool
from langchain_core.documents import Document

from config.vectorstore import generate_vector_db_to_retrieve_by_product_name
from config.vectorstore import generate_vector_db_to_retrieve_by_product_brand
from config.vectorstore import generate_vector_db_to_retrieve_by_site_category_lv1
from config.vectorstore import generate_vector_db_to_retrieve_by_site_category_lv2



@tool
def create_list_of_topics(list_of_topics: list[str]) -> Topics:
    """
    you should give a list of strings and this funciton will return a list of topics

    :param list_of_topics: a list of topics
    :return: Topic list
    """
    return Topics(topics=list_of_topics)

@tool
def create_summarization():
    """

    :return:
    """
    pass


@tool("busca_por_nome_produto", return_direct=True)
def buscar_por_nome_produto(pergunta: str) -> str:
    """
    Busca produtos similares com base no nome, usando FAISS.
    Retorna até 3 resultados formatados (nome + título e texto da review + rating).
    """
    retriever = generate_vector_db_to_retrieve_by_product_name(top_k=3)

    docs: List[Document] = retriever.get_relevant_documents(pergunta)
    if not docs:
        return "Nenhum produto encontrado relacionado a essa consulta."

    blocos: List[str] = []
    for doc in docs:
        nome = doc.page_content
        meta = doc.metadata
        titulo_review = meta.get("review_title", "—")
        texto_review = meta.get("review_text", "—")
        rating = meta.get("overall_rating", "—")
        bloco = (
            f"Produto: {nome}\n"
            f"Título da Review: {titulo_review}\n"
            f"Avaliação: {rating}\n"
            f"Review: {texto_review}"
        )
        blocos.append(bloco)

    return "\n\n---\n\n".join(blocos)


@tool("busca_por_marca_produto", return_direct=True)
def buscar_por_marca_produto(pergunta: str) -> str:
    """
    Busca produtos similares com base na marca, usando FAISS.
    Retorna até 3 resultados formatados (marca + título e texto da review + rating).
    """
    retriever = generate_vector_db_to_retrieve_by_product_brand(top_k=3)
    docs: List[Document] = retriever.get_relevant_documents(pergunta)
    if not docs:
        return "Nenhum produto (pela marca) encontrado para essa consulta."

    blocos: List[str] = []
    for doc in docs:
        marca = doc.page_content
        meta = doc.metadata
        nome_produto = meta.get("product_name", "—")
        titulo_review = meta.get("review_title", "—")
        texto_review = meta.get("review_text", "—")
        rating = meta.get("overall_rating", "—")
        bloco = (
            f"Marca: {marca}\n"
            f"Produto: {nome_produto}\n"
            f"Título da Review: {titulo_review}\n"
            f"Avaliação: {rating}\n"
            f"Review: {texto_review}"
        )
        blocos.append(bloco)

    return "\n\n---\n\n".join(blocos)


@tool("busca_por_categoria_lv1", return_direct=True)
def buscar_por_categoria_lv1(pergunta: str) -> str:
    """
    Busca produtos similares com base na categoria de nível 1, usando FAISS.
    Retorna até 3 resultados formatados.
    """
    retriever = generate_vector_db_to_retrieve_by_site_category_lv1(top_k=3)
    docs: List[Document] = retriever.get_relevant_documents(pergunta)
    if not docs:
        return "Nenhum produto encontrado para essa categoria (nível 1)."

    blocos: List[str] = []
    for doc in docs:
        categoria = doc.page_content
        meta = doc.metadata
        nome_produto = meta.get("product_name", "—")
        titulo_review = meta.get("review_title", "—")
        texto_review = meta.get("review_text", "—")
        rating = meta.get("overall_rating", "—")
        bloco = (
            f"Categoria LV1: {categoria}\n"
            f"Produto: {nome_produto}\n"
            f"Título da Review: {titulo_review}\n"
            f"Avaliação: {rating}\n"
            f"Review: {texto_review}"
        )
        blocos.append(bloco)

    return "\n\n---\n\n".join(blocos)


@tool("busca_por_categoria_lv2", return_direct=True)
def buscar_por_categoria_lv2(pergunta: str) -> str:
    """
    Busca produtos similares com base na categoria de nível 2, usando FAISS.
    Retorna até 3 resultados formatados.
    """
    retriever = generate_vector_db_to_retrieve_by_site_category_lv2(top_k=3)
    docs: List[Document] = retriever.get_relevant_documents(pergunta)
    if not docs:
        return "Nenhum produto encontrado para essa categoria (nível 2)."

    blocos: List[str] = []
    for doc in docs:
        categoria = doc.page_content
        meta = doc.metadata
        nome_produto = meta.get("product_name", "—")
        titulo_review = meta.get("review_title", "—")
        texto_review = meta.get("review_text", "—")
        rating = meta.get("overall_rating", "—")
        bloco = (
            f"Categoria LV2: {categoria}\n"
            f"Produto: {nome_produto}\n"
            f"Título da Review: {titulo_review}\n"
            f"Avaliação: {rating}\n"
            f"Review: {texto_review}"
        )
        blocos.append(bloco)

    return "\n\n---\n\n".join(blocos)
