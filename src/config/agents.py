from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

from config.tools import create_list_of_topics
from model.summarization_model import AnaliseCompleta
from model.topics_model import Topics


def get_agent_gerador_topicos(model):

    parser = PydanticOutputParser(pydantic_object=Topics)
        # "Você deve seguir essas instrucoes de como formatar sua resposta, ela devera ser uma unica saída com o seguinte formato:
        #   "{format_instructions}"
    prompt = ChatPromptTemplate.from_template(template=
                                              """
                                          Você é um assistente de IA que analisa avaliações de produtos em um e-commerce. 
                                          
                                          Você receberá uma lista de comentarios de produtos e dever analisar para determinar os topicos abordados em cada comentario
                                          
                                          Esses são alguns exemplos de topicos:
                                              - Entrega 
                                              - Embalagem  
                                              - Qualidade do Produto 
                                              - Usabilidade 
                                              - Atendimento 
                                              - Preço 
                                          
                                          LISTA DE COMENTARIOS PARA ANALISAR:
                                          {query}
                                      
                                          """,
                                              # partial_variables={"format_instructions": parser.get_format_instructions()}

                                          )
    model_tool = model.with_structured_output(Topics)#.bind_tools([create_list_of_topics], tool_choice="required")
    chain = prompt | model_tool #| parser
    return chain

def get_agent_sumarizacao(model, topicos):

    parser = PydanticOutputParser(pydantic_object=AnaliseCompleta)

    prompt = ChatPromptTemplate.from_template(template=
                                              """
                                                Você é uma IA analista de comentários. Seu trabalho é avaliar um conjunto de comentários e gerar uma análise completa estruturada, conforme os seguintes critérios:

                                                1. Para cada comentário:
                                                 -Identifique o grupo ao qual ele pertence (por exemplo: equipe, cliente, fornecedor, etc.).
                                                 -Identifique o tópico principal abordado no comentário, você não deve criar novos topicos apenas usar os ja existente, esses são os possiveis tópicos: {topics}.
                                                 -Extraia os pontos principais mencionados no comentário (insights, sugestões, elogios, críticas).
                                                
                                                2. Após processar todos os comentários, produza um resumo final, contendo:
                                                 -O sentimento predominante dos comentários analisados (positivo, negativo ou neutro).
                                                 -Os tópicos mais mencionados nos comentários.
                                                 -Os grupos mais ativos (grupos com mais comentários).
                                                 -Ações sugeridas com base nas críticas, sugestões e padrões observados.
                                                
                                                3. Formato de saída esperado:
                                                {{
                                                  "comentarios": [
                                                    {{
                                                      "comentario_id": 123,
                                                      "grupo": "cliente",
                                                      "topico": "entrega",
                                                      "pontos_principais": ["atraso na entrega", "falta de informação no rastreamento"]
                                                    }},
                                                    {{
                                                      "comentario_id": 124,
                                                      "grupo": "equipe",
                                                      "topico": "comunicação",
                                                      "pontos_principais": ["falta de alinhamento entre departamentos"]
                                                    }}
                                                  ],
                                                  "resumo_final": {{
                                                    "sentimento_predominante": "negativo",
                                                    "topicos_mais_mencionados": ["entrega", "comunicação"],
                                                    "grupos_mais_ativos": ["cliente", "equipe"],
                                                    "acoes_sugeridas": ["Melhorar o sistema de rastreamento de pedidos", "Criar reuniões semanais entre os departamentos"]
                                                  }}
                                                }}
                                                4. Observações:
                                                 -Caso algum campo não possa ser identificado claramente, utilize valores aproximados ou indique "indefinido".
                                                 -Seja objetivo e coeso na extração de dados.
                                                 -Extraia no máximo 3 pontos principais por comentário.
                                                
                                                COMENTARIOS PARA ANALISAR:
                                                {query}
                                          """,
                                              partial_variables={
                                                  # "format_instructions": parser.get_format_instructions(),
                                                    "topics":topicos}

                                              )
    model_tool = model.with_structured_output(AnaliseCompleta)
    chain = prompt | model_tool# | parser
    return chain

