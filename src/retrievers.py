from typing import Annotated


from config.vectorstore import generate_vector_db_to_retrieve_by_product_name, \
    generate_vector_db_to_retrieve_by_product_brand, generate_vector_db_to_retrieve_by_site_category_lv1, \
    generate_vector_db_to_retrieve_by_site_category_lv2


def format_docs(documents):
    """
    Formata dinamicamente os campos presentes em cada documento, usando rótulos amigáveis.
    Repete a Categoria Principal no topo e também no metadata (caso exista).
    """

    # Mapeamento de chaves para rótulos amigáveis
    label_map = {
        "product_name": "Produto",
        "product_brand": "Marca",
        "site_category_lv1": "Categoria",
        "site_category_lv2": "Subcategoria",
        "overall_rating": "Avaliação Geral",
        "recommend_to_a_friend": "Recomendaria a um amigo? Nota",
        "review_title": "Título da Avaliação",
        "review_text": "Comentário"
    }

    formatted = []
    for doc in documents:
        lines = []

        # Categoria Principal (do page_content)
        categoria_principal = getattr(doc, "page_content", None)
        if categoria_principal:
            lines.append(f"Categoria Principal: {categoria_principal}")

        # Campos do metadata (com rótulos amigáveis)
        for key, value in doc.metadata.items():
            label = label_map.get(key, key.replace('_', ' ').capitalize())
            lines.append(f"{label}: {value}")

        lines.append("-" * 60)
        formatted.append("\n".join(lines))

    return "\n".join(formatted)


def product_name_retriever(product_name: Annotated[str, 'product name quoted by the employee']):
    """" Retrieves up to 10 database products names matching the user's input.

    Args:
        product_name(str): The name of the product

    returns:
        str: A string with 10 similar product names
    """
    retriever = generate_vector_db_to_retrieve_by_product_name(top_k=10)
    retrieved_docs = format_docs(retriever.invoke(product_name))
    return retrieved_docs


def product_brand_retriever(product_brand: Annotated[str, 'product brand quoted by the employee']):
    """" Retrieves up to 10 database products brands matching the user's input.

    Args:
        product_brand(str): The brand of the product

    returns:
        str: A string with 10 similar product brands
    """
    retriever = generate_vector_db_to_retrieve_by_product_brand(top_k=10)
    retrieved_docs = format_docs(retriever.invoke(product_brand))
    return retrieved_docs


def site_category_lv1_retriever(site_category_lv1: Annotated[str, 'site category lv1 quoted by the employee']):
    """" Retrieves up to 10 database gategory lv1 matching the user's input.

    Args:
        site_category_lv1(str): The gategory lv1 of the product

    returns:
        str: A string with 10 similar category lv1
    """
    retriever = generate_vector_db_to_retrieve_by_site_category_lv1(top_k=10)
    retrieved_docs = format_docs(retriever.invoke(site_category_lv1))
    return retrieved_docs


def site_category_lv2_retriever(site_category_lv2: Annotated[str, 'site category lv2 quoted by the employee']):
    """" Retrieves up to 10 database gategory lv2 matching the user's input.

    Args:
        site_category_lv2(str): The gategory lv2 of the product

    returns:
        str: A string with 10 similar category lv2
    """
    retriever = generate_vector_db_to_retrieve_by_site_category_lv2(top_k=10)
    retrieved_docs = format_docs(retriever.invoke(site_category_lv2))
    print(retrieved_docs)
    return retrieved_docs