from pydantic import BaseModel, Field

class SentimentosModel(BaseModel):
    Positivos: str = Field(..., description="Porcentagem de sentimentos positivos")
    Negativos: str = Field(..., description="Porcentagem de sentimentos negativos")
    Neutros: str = Field(..., description="Porcentagem de sentimentos neutros")

