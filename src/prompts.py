import pandas as pd
import ollama
import time

df = pd.read_csv('B2W-Reviews01.csv')

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
Você é um assistente de IA que analisa múltiplos comentários de clientes sobre produtos de e-commerce.

Sua tarefa é gerar um resumo estratégico com base nos comentários abaixo. Siga as etapas:

1. Leia todos os comentários cuidadosamente.
2. Identifique os principais *insights* mencionados, como: entrega (prazo, problemas), embalagem, qualidade do produto, usabilidade, atendimento, preço, entre outros.
3. Para cada insight identificado, indique quantas vezes ele foi mencionado e resuma o que foi dito.
4. Por fim, escreva um resumo geral e estratégico com base nessa análise, útil para as áreas de produto, atendimento ou logística da empresa.
Comentários:
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
