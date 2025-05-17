from pydantic import BaseModel
from typing import Optional


class Comentario(BaseModel):
    product_id:Optional[str]
    product_name:Optional[str]
    product_brand:Optional[str]
    site_category_lv1:Optional[str]
    site_category_lv2:Optional[str]
    overall_rating:Optional[str]
    review_title:Optional[str]
    recommend_to_a_friend:Optional[str]
    review_text:Optional[str]
