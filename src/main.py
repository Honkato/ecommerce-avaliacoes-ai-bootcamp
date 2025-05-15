from typing import Callable
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableSerializable
import sqlite3

from config.agents import get_agent_sumarizacao, get_agent_gerador_topicos
from config.database import csv_to_sqlite
from config.model import load_model
from config.vectorstore import generate_vector_db_to_retrieve_by_product_name, \
    generate_vector_db_to_retrieve_by_product_brand, generate_vector_db_to_retrieve_by_site_category_lv1, \
    generate_vector_db_to_retrieve_by_site_category_lv2, format_docs
from retrievers import product_brand_retriever, product_name_retriever, site_category_lv1_retriever, \
    site_category_lv2_retriever


def get_app() -> FastAPI:
    """
    Generate the FastAPI with custom OpenAPI specifications
    :return: FastAPI application
    :rtype:
    """

    app = FastAPI(
        docs_url="/docs",
        redoc_url="/redoc",
        root_path="/api",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.models = {'name': 'nome_do_modelo'}

    def start_app_handler() -> Callable:
        """
        Startup handle
        :param app: FastAPI
        :type app:
        :return: None
        :rtype:
        """

        def startup() -> None:
            class Consts:
                model: RunnableSerializable[dict, str] = None

            app.consts = Consts()

            csv_filepath = rf'./src/B2W-Reviews01.csv'
            csv_to_sqlite(csv_filepath)
            #carregar os modelos, assim podemos carregar os diferentes modelos aqui e utilizalos quando quisermos
            app.consts.model = load_model('mistral')
            pass

        return startup

    app.add_event_handler("startup", start_app_handler())

    return app


app = get_app()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/model")
async def model():
    return {"message": app.consts.model}


@app.get("/product_brand/:product_brand")
async def brands(product_brand):
    return product_brand_retriever(product_brand)

@app.get("/product_name/:product_name")
async def product_name(product_name):
    return product_name_retriever(product_name)

@app.get("/site_category_lv1/:site_category_lv1")
async def site_category_lv1(site_category_lv1):
    return site_category_lv1_retriever(site_category_lv1)

@app.get("/site_category_lv2/:site_category_lv2")
async def site_category_lv2(site_category_lv2):
    return site_category_lv2_retriever(site_category_lv2)

@app.get("/search/:search_type/:search_query")
async def product_name(search_type, search_query):
    match search_type:
        case 'product_brand':
            retriever_result = product_brand_retriever(search_query)
        case 'product_name':
            retriever_result = product_name_retriever(search_query)
        case 'site_category_lv1':
            retriever_result = site_category_lv1_retriever(search_query)
        case 'site_category_lv2':
            retriever_result = site_category_lv2_retriever(search_query)
        case _:
            return
    model = app.consts.model
    agent_gerador_topicos = get_agent_gerador_topicos(model)
    topicos = agent_gerador_topicos.invoke(retriever_result)
    print(topicos)
    agent_sumarizacao = get_agent_sumarizacao(model, topicos)
    sumarizacao = agent_sumarizacao.invoke(retriever_result)
    return {
        'sumarizacao':sumarizacao,
        'topicos':topicos,
        'retriever_result':retriever_result,
    }




@app.post("/chat")
async def chat(message: str):
    return app.consts.models.invoke({'input': message})

if __name__ == "__main__":
    """
    Runs the API
    """
    import uvicorn

    print("Starting Azure Storage Container")

    uvicorn.run("main:app", port=8081, reload=True)
