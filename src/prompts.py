import pandas as pd
import ollama
import time

df = pd.read_csv('src/B2W-Reviews01.csv')

df = df.dropna(subset=['review_text'])
df['clean_text'] = df['review_text'].str.lower().str.replace('[^\w\s]', ' ', regex=True)

def montar_prompt_topicos(texto):
    return f"""
Você é um assistente de IA que analisa avaliações de produtos em e-commerce. 

Dado um comentário de cliente, identifique:
1. Os tópicos abordados (ex: entrega, atendimento, qualidade, preço, etc.)
2. O insights principais para cada tópico (positivo ou negativo)
3. Explique seu raciocínio brevemente antes de apresentar o resultado final 

Exemplo:
Comentário: "A entrega foi super rápida, mas o produto veio riscado. Atendimento nem respondeu."
Resposta: 
- entrega: positiva
- produto: negativa
- atendimento: negativa

Agora analise o seguinte comentário:
Comentário: "{texto}"
Resposta:
"""

def montar_prompt_sumarizacao(lista_de_comentarios):
    comentarios_formatados = "\n".join([f"- {comentario}" for comentario in lista_de_comentarios])
    return f"""
Você é um assistente de IA especializado em analisar comentários de clientes de e-commerce.

Sua tarefa é gerar um relatório estratégico a partir dos comentários listados abaixo, seguindo estas etapas:

1. **Agrupar Comentários**  
   - Organize os comentários em grupos com base em **produto**, **marca**, **categoria** (e quaisquer outros atributos relevantes que você identificar).  
   - Cada comentário deve pertencer a exatamente um grupo.

2. **Extrair Insights por Grupo**  
   Para cada grupo (produto/marca/categoria):  
   a. Identifique os principais tópicos mencionados, tais como:  
      - **Entrega** (prazo, avarias, rastreamento)  
      - **Embalagem** (proteção, apresentação)  
      - **Qualidade do Produto** (acabamento, durabilidade)  
      - **Usabilidade** (funcionalidade, ergonomia)  
      - **Atendimento** (suporte, canal de contato)  
      - **Preço** (custo-benefício, promoções)  
   b. Liste brevemente os *insights* ou reclamações mais frequentes para cada tópico dentro do grupo.

3. **Síntese Estratégica**  
   - Para cada grupo, escreva um parágrafo estratégico que resuma:  
     1. Pontos fortes a serem mantidos ou destacados.  
     2. Principais problemas/áreas de melhoria.  
     3. Sugestões de ação para as áreas de Produto, Logística, Atendimento e Precificação.

4. **Visão Geral Consolidada**  
   - Ao final, apresente uma visão geral com recomendações de alto nível para otimizar a experiência do cliente e as operações, apoiando decisões estratégicas.

Comentários a serem analisados:  
{comentarios_formatados}

Resumo:
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

limite = 3 # limitando pro t14 aguentar
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

if len(comentarios_para_resumir) == 0:
    print("\nNenhum comentário disponível para gerar o resumo.")
else:
    prompt_summarizacao = montar_prompt_sumarizacao(comentarios_para_resumir)
    resumo = chamar_mistral(prompt_summarizacao)

    if resumo:
        print("\n📋 RESUMO FINAL DOS COMENTÁRIOS ANALISADOS:")
        print("=" * 80)
        print(resumo)
        print("=" * 80)
    else:
        print("\nO modelo não retornou um resumo. ")
