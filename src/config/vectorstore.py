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
                "product_id": row[2],
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
                "product_id": row[2],
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
                "product_id": row[2],
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
                "product_id": row[2],
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
