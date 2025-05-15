from pydantic import BaseModel
from typing import List, Literal, Optional


class ComentarioAnalise(BaseModel):
    comentario_id: int
    grupo: str
    topico: str
    pontos_principais: List[str]
    sentimento: Literal["positivo", "negativo", "neutro"]
    justificativa_sentimento: str


class ResumoFinal(BaseModel):
    sentimento_predominante: str  # Ex: "majoritariamente positivo"
    topicos_mais_mencionados: List[str]
    grupos_mais_ativos: List[str]
    acoes_sugeridas: List[str]


class AnaliseCompleta(BaseModel):
    comentarios: List[ComentarioAnalise]
    resumo_final: ResumoFinal