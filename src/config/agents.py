from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from model.summarization import AnaliseCompleta
from model.topics import Topics
from retrievers import product_name_retriever, product_brand_retriever, site_category_lv1_retriever, \
    site_category_lv2_retriever


def get_agent_gerador_topicos(model):

    parser = PydanticOutputParser(pydantic_object=Topics)

    prompt = ChatPromptTemplate.from_template(template=
                                              """
                                          Você é um assistente de IA que analisa avaliações de produtos em um e-commerce. 
                                          
                                          Você receberá uma lista de comentarios de produtos e dever analisar para determinar os topicos abordados em cada comentario
                                          
                                          Você deve seguir essas instrucoes de como formatar sua resposta, ela devera ser uma unica saída com o seguinte formato:
                                          {format_instructions}
                                          
                                          Esses são alguns exemplos:
                                              - Entrega 
                                              - Embalagem  
                                              - Qualidade do Produto 
                                              - Usabilidade 
                                              - Atendimento 
                                              - Preço 
                                          
                                          LISTA DE COMENTARIOS PARA ANALISAR:
                                          {query}
                                      
                                          """,
                                              partial_variables={"format_instructions": parser.get_format_instructions()}

                                          )
    # model_tool = model.bind_tools(Topicos.model_json_schema, tool_choice="required")
    #     [product_name_retriever, product_brand_retriever, site_category_lv1_retriever,
    #      site_category_lv2_retriever], tool_choice="required")
    chain = prompt | model #| parser
    return chain

def get_agent_sumarizacao(model, topicos):

    parser = PydanticOutputParser(pydantic_object=AnaliseCompleta)

    prompt = ChatPromptTemplate.from_template(template=
                                              """
                                                Você é uma inteligência artificial especialista em análise de linguagem natural. Seu objetivo é ler e processar uma lista de comentários de usuários, extraindo as seguintes informações:
                                                
                                                Você deve seguir essas instrucoes de como formatar sua resposta, ela devera ser uma unica saída com o seguinte formato:
                                                {format_instructions}
                                                
                                                Para cada comentário:
                                                
                                                Grupo: Identifique a que grupo pertence (produto/marca/categoria).
                                                
                                                Tópico: Esses serão os possiveis topicos, e voce não deve adicionar outros {topics}.
                                                
                                                Pontos principais: Liste de forma objetiva os principais pontos abordados no comentário.
                                                
                                                Resumo do sentimento: Classifique como positivo, negativo ou neutro e justifique com base no texto.
                                                
                                                Após analisar todos os comentários, gere um Resumo Final contendo:
                                                
                                                Uma síntese geral dos sentimentos predominantes
                                                
                                                Os tópicos mais recorrentes
                                                
                                                Quais grupos mais comentaram e quais seus focos principais
                                                
                                                Sugestões de ação com base nas análises
                                                
                                                COMENTARIOS PARA ANALISAR:
                                                {query}
                                          """,
                                              partial_variables={
                                                  "format_instructions": parser.get_format_instructions(),
                                                    "topics":topicos}

                                              )
    # model_tool = model.bind_tools(
    #     [product_name_retriever, product_brand_retriever, site_category_lv1_retriever,
    #      site_category_lv2_retriever], tool_choice="required")
    chain = prompt | model# | parser
    return chain

