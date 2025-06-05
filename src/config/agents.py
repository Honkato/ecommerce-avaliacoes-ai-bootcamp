from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

from config.tools import create_list_of_topics
from model.summarization_model import AnaliseCompleta
from model.topics_model import Topics

from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

from config.tools import (
    buscar_por_nome_produto,
    buscar_por_marca_produto,
    buscar_por_categoria_lv1,
    buscar_por_categoria_lv2,
)


def get_agent_gerador_topicos(model):

    parser = PydanticOutputParser(pydantic_object=Topics)

    prompt = ChatPromptTemplate.from_template(template="""
Você é um assistente de IA para e-commerce cujo trabalho é **extrair tópicos principais** de uma lista de comentários.

Siga estas instruções:

1. A entrada será um único texto contendo vários comentários de usuário, separados por linhas ou por marcadores.  
2. Para cada comentário, identifique de forma sucinta **até 3 tópicos** que melhor descrevem o assunto.  
3. Use apenas tópicos presentes nesta lista de exemplos e não crie novos termos:
   - Entrega
   - Embalagem
   - Qualidade do Produto
   - Usabilidade
   - Atendimento
   - Preço

4. **Formato de saída (JSON)**:  
   Utilize estritamente este modelo Pydantic (já gerado por {format_instructions}) e retorne apenas o JSON.  
   Não inclua texto adicional fora do JSON!  

{format_instructions}

---  
**Comentários para analisar (texto único)**:  
{query}
""",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    model_tool = model.with_structured_output(Topics)
    chain = prompt | model_tool

    return chain

def get_agent_sumarizacao(model, topicos: list[str]):

    parser = PydanticOutputParser(pydantic_object=AnaliseCompleta)

    prompt = ChatPromptTemplate.from_template(template="""
Você é uma IA analista de feedback de e-commerce. Receberá um conjunto de comentários já pré-processados, 
onde cada comentário inclui:
  • comentario_id (inteiro)  
  • grupo (por exemplo: “cliente”, “equipe”, “fornecedor”)  
  • topico (um dos tópicos permitidos listados abaixo)  
  • pontos_principais (lista de até 3 frases/resumos)  

Não crie novos tópicos—use apenas estes:
{topics}

Agora, aponte:

1. Para cada comentário:  
   - Verifique se “grupo” e “topico” estão corretos.  
   - Extraia ou valide até 3 pontos principais (insights, sugestões, elogios, críticas) já contidos em “pontos_principais”.  

2. Em seguida, produza um **resumo final** com:  
   - sentimento_predominante (positivo, negativo ou neutro)  
   - topicos_mais_mencionados (lista ordenada por frequência)  
   - grupos_mais_ativos (lista ordenada por número de comentários)  
   - acoes_sugeridas (com base em problemas ou padrões detectados)  

3. **Formato de saída (JSON)**:  
   Use estritamente este modelo Pydantic e retorne somente o JSON, sem texto extra:

{format_instructions}

---  
**Comentários pré-processados (entrada)**:  
{query}
""",
        partial_variables={
            "topics": ", ".join(topicos),
            "format_instructions": parser.get_format_instructions()
        }
    )

    # 3) Encadeia: prompt → modelo com output estruturado
    model_tool = model.with_structured_output(AnaliseCompleta)
    chain = prompt | model_tool

    return chain

def get_agent_chat_rag(model_name: str = "mistral", temperature: float = 0.0):
    """
    Agente RAG com:
     - Prefixo que lida com cumprimentos (greetings)
     - Suffix que instrui quando e como usar as ferramentas FAISS
     - Memória de conversa (ConversationBufferMemory)
     - LLM Ollama(Mistral)
    """
    llm = OllamaLLM(model=model_name, temperature=temperature)

    tools = [
        buscar_por_nome_produto,
        buscar_por_marca_produto,
        buscar_por_categoria_lv1,
        buscar_por_categoria_lv2,
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prefix = """
Você é um assistente conversacional de e-commerce. Siga estas instruções:
1. Se o usuário apenas disser algo como “oi”, “olá”, “bom dia”, etc., responda com uma saudação cordial e não tente usar nenhuma ferramenta FAISS.
   Exemplos:
     Usuário: “oi”
     Assistente: “Olá! Em que posso ajudar você hoje?”
     
     Usuário: “olá, tudo bem?”
     Assistente: “Tudo ótimo, obrigado! Como posso ajudar com seus produtos hoje?”

2. Se o usuário perguntar sobre produtos, marcas ou categorias, considere usar as ferramentas listadas abaixo para buscar reviews semelhantes.  

Ferramentas disponíveis:
- busca_por_nome_produto(pergunta): retorna até 3 produtos similares com base no nome, junto com título da review, avaliação e texto da review.
- busca_por_marca_produto(pergunta): retorna até 3 produtos similares com base na marca, com os mesmos metadados.
- busca_por_categoria_lv1(pergunta): retorna até 3 produtos com base na categoria de nível 1.
- busca_por_categoria_lv2(pergunta): retorna até 3 produtos com base na categoria de nível 2.

Se decidir usar uma ferramenta, chame exatamente desta forma no raciocínio do agente:
@<nome_da_ferramenta>("texto da pergunta")

3. Se a pergunta não estiver relacionada a cumprimentos nem a busca de produtos, responda de forma educada pedindo mais detalhes ou esclarecendo que não entendeu completamente.
    """

    suffix = """
{chat_history}

Se quiser buscar reviews ou informações de produtos, chame uma das ferramentas acima. Se não precisar de ferramenta (por exemplo, é um cumprimento), apenas responda normalmente.

Pergunta atual: "{input}"
{agent_scratchpad}"""

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix,
            "input_variables": ["input", "chat_history", "agent_scratchpad"],
        }
    )

    return agent

