# import pandas as pd
# import ollama
# import time
# import re

# df = pd.read_csv('src/B2W-Reviews01.csv')

# df = df.dropna(subset=['review_text'])
# df['clean_text'] = df['review_text'].str.lower().str.replace('[^\w\s]', ' ', regex=True)

# def montar_prompt_topicos(texto):
#     return f"""
# Você é um assistente de IA que analisa avaliações de produtos em um e-commerce.

# Você receberá uma lista de comentários de produtos e deverá analisá-los para determinar os tópicos abordados em cada comentário.

# Esses são alguns exemplos de tópicos:
# - Entrega
# - Embalagem
# - Qualidade do Produto
# - Usabilidade
# - Atendimento
# - Preço

# {texto}
# """


# def montar_prompt_sumarizacao(lista_de_comentarios, topics):
#     comentarios_formatados = "\n".join([f"- {comentario}" for comentario in lista_de_comentarios])
#     return f"""
#                                                 Você é uma IA analista de comentários. Seu trabalho é avaliar um conjunto de comentários e gerar uma análise completa estruturada, conforme os seguintes critérios:

#                                                 1. Para cada comentário:
#                                                  -Identifique o grupo ao qual ele pertence (por exemplo: equipe, cliente, fornecedor, etc.).
#                                                  -Identifique o tópico principal abordado no comentário, você não deve criar novos topicos apenas usar os ja existente, esses são os possiveis tópicos: {topics}.
#                                                  -Extraia os pontos principais mencionados no comentário (insights, sugestões, elogios, críticas).
                                                
#                                                 2. Após processar todos os comentários, produza um resumo final, contendo:
#                                                  -O sentimento predominante dos comentários analisados (positivo, negativo ou neutro).
#                                                  -Os tópicos mais mencionados nos comentários.
#                                                  -Os grupos mais ativos (grupos com mais comentários).
#                                                  -Ações sugeridas com base nas críticas, sugestões e padrões observados.
                                                
#                                                 3. Formato de saída esperado:
#                                                 {{
#                                                   "comentarios": [
#                                                     {{
#                                                       "comentario_id": 123,
#                                                       "grupo": "cliente",
#                                                       "topico": "entrega",
#                                                       "pontos_principais": ["atraso na entrega", "falta de informação no rastreamento"]
#                                                     }},
#                                                     {{
#                                                       "comentario_id": 124,
#                                                       "grupo": "equipe",
#                                                       "topico": "comunicação",
#                                                       "pontos_principais": ["falta de alinhamento entre departamentos"]
#                                                     }}
#                                                   ],
#                                                   "resumo_final": {{
#                                                     "topicos_mais_mencionados": ["entrega", "comunicação"],
#                                                     "grupos_mais_ativos": ["cliente", "equipe"],
#                                                     "acoes_sugeridas": ["Melhorar o sistema de rastreamento de pedidos", "Criar reuniões semanais entre os departamentos"]
#                                                   }}
#                                                 }}

#                                                 4. Resumo para texto:
#                                                     Após finalizar o resumo final, transforme em um único texto assim como nas avalições do Mercado Livre, por exemplo:
#                                                     "Os usuários acreditam que o produto X não vale a pena seu valor, enquanto o produto Y já é altamente avaliado com 5 estrelas e comprado com maior frequência.
#                                                     O serviço de entrega da empresa deixa a desejar, a educação dos funcionários pode melhorar".

#                                                 5. Observações:
#                                                  -Caso algum campo não possa ser identificado claramente, utilize valores aproximados ou indique "indefinido".
#                                                  -Seja objetivo e coeso na extração de dados.
#                                                  -Extraia no máximo 3 pontos principais por comentário.
                                                
#                                                 COMENTARIOS PARA ANALISAR:
#                                                 {comentarios_formatados}
#                                           """

# def chamar_mistral(prompt):
#     try:
#         response = ollama.chat(model='mistral', messages=[
#             {'role': 'user', 'content': prompt}
#         ])
#         return response['message']['content']
#     except Exception as e:
#         print(f"Erro ao chamar o modelo: {e}")
#         return None

# limite = 3 # limitando pro t14 aguentar
# respostas = []

# for idx, row in df.iloc[:limite].iterrows():
#     print(f"Processando comentário {idx+1}/{limite}")
#     texto = row['clean_text']
#     prompt = montar_prompt_topicos(texto)
#     resposta = chamar_mistral(prompt)
#     respostas.append(resposta)
#     time.sleep(1)  

# df_resultado = df.iloc[:limite].copy()
# df_resultado['analise_topicos'] = respostas

# print("\n\nResultados da análise:\n")
# for i, row in df_resultado.iterrows():
#     print(f"Comentário {i+1}:")
#     print(row['review_text'])
#     print("Análise de Tópicos:")
#     print(row['analise_topicos'])
#     print("=" * 80)

# comentarios_para_resumir = df_resultado['review_text'].dropna().tolist()

# matches = [
#     t.strip()
#     for resp in respostas if resp
#     for t in re.findall(r'"topico"\s*:\s*"([^"]+)"', resp)
# ]

# topics_str = ", ".join(set(matches))

# if len(comentarios_para_resumir) == 0:
#     print("\nNenhum comentário disponível para gerar o resumo.")
# else:
#     prompt_summarizacao = montar_prompt_sumarizacao(comentarios_para_resumir, topics=topics_str)
#     resumo = chamar_mistral(prompt_summarizacao)

#     if resumo:
#         print("\n📋 RESUMO FINAL DOS COMENTÁRIOS ANALISADOS:")
#         print("=" * 80)
#         print(resumo)
#         print("=" * 80)
#     else:
#         print("\nO modelo não retornou um resumo. ")

import pandas as pd
import ollama
import time
import re

df = pd.read_csv('src/B2W-Reviews01.csv')
df = df.dropna(subset=['review_text'])
df['clean_text'] = df['review_text'].str.lower().str.replace('[^\w\s]', ' ', regex=True)

def montar_prompt_topicos(texto):
    return f"""
Você é um assistente de análise de avaliações de produtos. Dado o texto abaixo de uma avaliação de cliente, extraia todos os aspectos distintos mencionados (por exemplo: Entrega, Embalagem, Qualidade do Produto, Usabilidade, etc.). Para cada aspecto, determine o sentimento (Positivo, Negativo ou Neutro) e forneça um pequeno trecho ou justificativa retirado do texto. Se possível, identifique a perspectiva do autor (por exemplo: Cliente, Fornecedor ou Equipe Interna). Formate a saída como uma lista estruturada.

Avaliação: "{texto}"

Exemplo de saída:
- Entrega: Negativo — "O produto chegou com duas semanas de atraso."
- Embalagem: Positivo — "A caixa veio bem embalada e protegida."
- Qualidade do Produto: Neutro — "Funciona como esperado pelo preço."
"""

def montar_prompt_sumarizacao(lista_de_comentarios, topics):
    comentarios_formatados = "\n".join([f"- {comentario}" for comentario in lista_de_comentarios])
    return f"""
Você é uma IA analista de avaliações de produtos. Você receberá uma lista de comentários com tópicos e sentimentos identificados. Sua tarefa é sintetizar essas informações e gerar insights gerais:

- Grupos de Avaliadores: Identifique os grupos envolvidos (Cliente, Equipe Interna, Fornecedor) e os tipos de feedback fornecidos.
- Tópicos & Tendências: Liste os aspectos mais mencionados e o sentimento predominante de cada um (ex: comentários geralmente negativos sobre Entrega).
- Análise de Sentimentos: Resuma a proporção de sentimentos positivos, negativos e neutros.
- Sugestões Práticas: Para cada tema principal, proponha ações ou melhorias (em lista de tópicos).
- Resumo Final: Escreva um parágrafo curto, no estilo de destaque de avaliações de marketplaces, sintetizando os principais pontos e o sentimento geral.

Organize a resposta com subtítulos claros para cada seção e finalize com o parágrafo de resumo.

COMENTÁRIOS PARA ANÁLISE:
{comentarios_formatados}
"""

def chamar_mistral(prompt):
    try:
        response = ollama.chat(model='mistral', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Erro ao chamar o modelo: {e}")
        return None

limite = 3  
respostas = []

for idx, row in df.iloc[:limite].iterrows():
    print(f"Processando comentário {idx+1}/{limite}")
    texto = row['clean_text']
    prompt = montar_prompt_topicos(texto)
    resposta = chamar_mistral(prompt)
    respostas.append(resposta)
    time.sleep(1)

df_resultado = df.iloc[:limite].copy()
df_resultado['analise_topicos'] = respostas

print("\n\nResultados da análise:\n")
for i, row in df_resultado.iterrows():
    print(f"Comentário {i+1}:")
    print(row['review_text'])
    print("Análise de Tópicos:")
    print(row['analise_topicos'])
    print("=" * 80)

comentarios_para_resumir = df_resultado['review_text'].dropna().tolist()
matches = [
    t.strip()
    for resp in respostas if resp
    for t in re.findall(r'(?:-\s*)([\w\s]+):', resp)
]
topics_str = ", ".join(set(matches))

if len(comentarios_para_resumir) == 0:
    print("\nNenhum comentário disponível para gerar o resumo.")
else:
    prompt_summarizacao = montar_prompt_sumarizacao(comentarios_para_resumir, topics=topics_str)
    resumo = chamar_mistral(prompt_summarizacao)

    if resumo:
        print("\n📋 RESUMO FINAL DOS COMENTÁRIOS ANALISADOS:")
        print("=" * 80)
        print(resumo)
        print("=" * 80)
    else:
        print("\nO modelo não retornou um resumo.")

