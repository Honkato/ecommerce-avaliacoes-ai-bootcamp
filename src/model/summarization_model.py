from pydantic import BaseModel, Field
from typing import List


class ComentarioAnalise(BaseModel):
    comentario_id: int= Field(description="id do comentario se houver, se nao é 0")
    grupo: str
    topico: str
    pontos_principais: List[str]


class ResumoFinal(BaseModel):
    sentimento_predominante: str
    topicos_mais_mencionados: List[str]
    grupos_mais_ativos: List[str]
    acoes_sugeridas: List[str]


class AnaliseCompleta(BaseModel):
    comentarios: List[ComentarioAnalise]
    resumo_final: ResumoFinal