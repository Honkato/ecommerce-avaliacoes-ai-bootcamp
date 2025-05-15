from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
import os
from config.database import select_distinct_all
from config.model import load_embedding_model

embedding_model = load_embedding_model()

def generate_vector_db_to_retrieve_by_product_name(top_k: int) -> VectorStoreRetriever:
    other_columns = select_distinct_all()
    documents = []
    if not os.path.exists("./vector_db_product_name"):
        for row in other_columns:
            product_name = row[3]
            metadata = {
                "product_brand": row[4],
                "site_category_lv1": row[5],
                "site_category_lv2": row[6],
                "overall_rating": row[7],
                "recommend_to_a_friend": row[8],
                "review_title": row[9],
                "review_text": row[10],
            }
            doc = Document(page_content=product_name, metadata=metadata)
            documents.append(doc)
        db = FAISS.from_documents(documents=documents, embedding=embedding_model)
        db.save_local('vector_db_product_name')
    else:
        db = FAISS.load_local("./vector_db_product_name", embeddings=embedding_model,
                              allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": top_k})
    return retriever


def generate_vector_db_to_retrieve_by_product_brand(top_k: int) -> VectorStoreRetriever:
    other_columns = select_distinct_all()
    documents = []
    if not os.path.exists("./vector_db_product_brand"):
        for row in other_columns:
            product_name = row[4]
            metadata = {
                "product_name": row[3],
                "site_category_lv1": row[5],
                "site_category_lv2": row[6],
                "overall_rating": row[7],
                "recommend_to_a_friend": row[8],
                "review_title": row[9],
                "review_text": row[10],
            }
            # print(product_name)
            doc = Document(page_content=product_name, metadata=metadata)
            documents.append(doc)
        db = FAISS.from_documents(documents=documents, embedding=embedding_model)
        db.save_local('vector_db_product_brand')
    else:
        db = FAISS.load_local("./vector_db_product_brand", embeddings=embedding_model,
                              allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": top_k})
    return retriever


def generate_vector_db_to_retrieve_by_site_category_lv1(top_k: int) -> VectorStoreRetriever:
    other_columns = select_distinct_all()
    documents = []
    if not os.path.exists("./vector_db_site_category_lv1"):
        for row in other_columns:
            category_lv1 = row[5]  # assumindo que esse é o nome do produto
            metadata = {
                "product_name": row[3],
                "product_brand": row[4],
                "site_category_lv2": row[6],
                "overall_rating": row[7],
                "recommend_to_a_friend": row[8],
                "review_title": row[9],
                "review_text": row[10],
            }
            doc = Document(page_content=category_lv1, metadata=metadata)
            documents.append(doc)
        db = FAISS.from_documents(documents=documents, embedding=embedding_model)
        db.save_local('vector_db_site_category_lv1')
    else:
        db = FAISS.load_local("./vector_db_site_category_lv1", embeddings=embedding_model,
                              allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": top_k})
    return retriever


def generate_vector_db_to_retrieve_by_site_category_lv2(top_k: int) -> VectorStoreRetriever:
    other_columns = select_distinct_all()
    documents = []
    if not os.path.exists("./vector_db_site_category_lv2"):
        for row in other_columns:
            site_category_lv2 = row[6]
            metadata = {
                "product_name": row[3],
                "product_brand": row[4],
                "site_category_lv1": row[5],
                "overall_rating": row[7],
                "recommend_to_a_friend": row[8],
                "review_title": row[9],
                "review_text": row[10],
            }
            doc = Document(page_content=site_category_lv2, metadata=metadata)
            documents.append(doc)
        db = FAISS.from_documents(documents=documents, embedding=embedding_model)
        db.save_local('vector_db_site_category_lv2')
    else:
        db = FAISS.load_local("./vector_db_site_category_lv2", embeddings=embedding_model,
                              allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": top_k})
    return retriever


## ARRUMAR FUNÇÃO AQUI, TA MUITO RUIM! TA CONFUNDINDO RECOMMEND TO A FRIEND COM O OVERALL RATING
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
        "review_title": "Título da Avaliação",
        "overall_rating": "Avaliação Geral",
        "recommend_to_a_friend": "Recomendaria a um amigo?",
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