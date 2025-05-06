import logging
from typing import Callable
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import OllamaLLM

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
#
# logger = logging.getLogger(__name__)

import csv
import sqlite3

def csv_to_sqlite(csv_filepath, db_filepath, table_name):
    """Imports a CSV file into a SQLite database table.

    Args:
        csv_filepath: Path to the CSV file.
        db_filepath: Path to the SQLite database file.
        table_name: Name of the table to create or append to.
    """
    try:
        with open(csv_filepath, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            data = list(csv_reader)

        with sqlite3.connect(db_filepath) as connection:
            cursor = connection.cursor()

            # Create table if it doesn't exist
            placeholders = ', '.join(['?'] * len(header))
            create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(header)});"
            cursor.execute(create_table_query)

            # Insert data into table
            insert_query = f"INSERT INTO {table_name} VALUES ({placeholders});"
            cursor.executemany(insert_query, data)

            connection.commit()

    except Exception as e:
        print(f"An error occurred: {e}")

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
            #tenho quase certeza q é gambiarra mas funcionou faze oq
            class Consts:
                db_filepath = None
                models: RunnableSerializable[dict, str] = None
            #pegar database
            app.consts = Consts()

            csv_filepath = rf'./src/B2W-Reviews01.csv'
            db_filepath = rf'./my_database.db'
            table_name = 'reviews'
            csv_to_sqlite(csv_filepath, db_filepath, table_name)
            app.consts.db_filepath = db_filepath
            #carregar os modelos, assim podemos carregar os diferentes modelos aqui e utilizalos quando quisermos
            app.consts.models = load_model('mistral')
            pass

        return startup

    app.add_event_handler("startup", start_app_handler())

    return app


app = get_app()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/models")
async def models():
    return {"message": app.consts.models.name}

def select_distinct(columns):
    result = []
    with sqlite3.connect(app.consts.db_filepath) as connection:
        query = f"SELECT DISTINCT {columns} FROM reviews"
        cursor = connection.cursor()
        cursor.execute(query)
        print(cursor.fetchone())

        result = cursor.fetchall()
        print(result)
    return result
@app.get("/product_brand")
async def brands():
    return select_distinct('product_brand')

@app.get("/site_category_lv1")
async def brands():
    return select_distinct('site_category_lv1')

@app.get("/site_category_lv2")
async def brands():
    return select_distinct('site_category_lv2')

@app.get("/site_category_lvs")
async def brands():
    return select_distinct('site_category_lv1, site_category_lv2')

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
