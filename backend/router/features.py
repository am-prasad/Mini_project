from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Features"])

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    rank: int

class FeatureImportanceResponse(BaseModel):
    features: List[FeatureImportance]
    total_features: int
    interpretation: Dict[str, str]
    source_model: str = ""            # which model's data is displayed
    source_type: str = ""             # "shap" (live computed) or "json" (stored file)
    available_models: List[str] = []  # all loaded model names for frontend dropdown


QUESTION_DESCRIPTIONS = {
    'Q1': 'Finding ways to overcome (Endurance)',
    'Q2': 'Taking responsibility (Ownership)',
    'Q3': 'Confidence across subjects (Reach)',
    'Q4': 'Problems are temporary (Endurance)',
    'Q5': 'Control under pressure (Control)',
    'Q6': 'Learning from mistakes (Ownership)',
    'Q7': "Failures don't define ability (Reach)",
    'Q8': 'Motivation without visible reward (Control)',
    'Q9': 'Influence on academic outcomes (Control)',
    'Q10': 'Recovery from disappointment (Ownership)',
}


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_global_feature_importance(
    model: Optional[str] = Query(
        None,
        description=(
            "Name of the model whose feature importance to return. "
            "Omit to use the best model automatically."
        )
    )
):
    """
    Return global feature importance (mean |SHAP| across training data).

    - With no ?model= param  → uses the best-accuracy model
    - With ?model=Random Forest → uses that specific model

    Priority for data source:
      1. Live-computed mean |SHAP| via shap_explainer.pkl + X_train.pkl
      2. Fallback: stored feature_importance.json (Gini/tree importance)
    """
    try:
        from main import model_registry

        available_models = sorted(model_registry.models.keys())

        if not available_models:
            raise HTTPException(status_code=404, detail="No models loaded.")

        # ── Resolve target model ───────────────────────────────────────────────
        if model and model in model_registry.models:
            target_model = model
        else:
            # Auto-select best model
            target_model = model_registry.get_best_model_name()
            if target_model is None:
                target_model = available_models[0]

        # ── Compute true mean |SHAP| from explainer + X_train ─────────────────
        fi_data   = model_registry.compute_global_shap(target_model)
        src_type  = "shap"

        # ── Fallback: stored feature_importance.json ───────────────────────────
        if fi_data is None:
            fi_data  = model_registry._all_feature_importances.get(target_model, [])
            src_type = "json"
            logger.warning(
                f"Using stored feature_importance.json for '{target_model}' "
                f"(SHAP computation unavailable)."
            )

        if not fi_data:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No feature importance data available for '{target_model}'. "
                    "Ensure shap_explainer.pkl + X_train.pkl or feature_importance.json exist."
                )
            )

        # ── Build response objects ────────────────────────────────────────────
        features = [
            FeatureImportance(
                feature=f['feature'],
                importance=f['importance'],
                rank=f.get('rank', idx + 1)
            )
            for idx, f in enumerate(fi_data)
        ]

        interpretation = {
            f['feature']: (
                f"{QUESTION_DESCRIPTIONS.get(f['feature'], 'N/A')} "
                f"— Rank #{f.get('rank', idx + 1)}"
            )
            for idx, f in enumerate(fi_data)
        }

        return FeatureImportanceResponse(
            features=features,
            total_features=len(fi_data),
            interpretation=interpretation,
            source_model=target_model,
            source_type=src_type,
            available_models=available_models,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))