from pydantic import BaseModel # type: ignore


class RecommendResponse(BaseModel):
    detected_ingredients: list[str]
    recommendations:      list[dict]
    model_sources:        dict | None = None
    error:                str  | None = None


class HealthResponse(BaseModel):
    status:        str
    models_loaded: list[str]
    db_ready:      bool
    recipe_count:  int