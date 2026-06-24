from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
import numpy as np
import logging
from typing import Dict, List, Any

from utils.csv_logger import log_prediction

logger = logging.getLogger(__name__)
class QuestionnaireInput(BaseModel):
    """Input schema for 10-question AQ questionnaire"""
    Q1: int = Field(..., ge=1, le=5, description="Q1 Response (1-5)")
    Q2: int = Field(..., ge=1, le=5, description="Q2 Response (1-5)")
    Q3: int = Field(..., ge=1, le=5, description="Q3 Response (1-5)")
    Q4: int = Field(..., ge=1, le=5, description="Q4 Response (1-5)")
    Q5: int = Field(..., ge=1, le=5, description="Q5 Response (1-5)")
    Q6: int = Field(..., ge=1, le=5, description="Q6 Response (1-5)")
    Q7: int = Field(..., ge=1, le=5, description="Q7 Response (1-5)")
    Q8: int = Field(..., ge=1, le=5, description="Q8 Response (1-5)")
    Q9: int = Field(..., ge=1, le=5, description="Q9 Response (1-5)")
    Q10: int = Field(..., ge=1, le=5, description="Q10 Response (1-5)")
    
    @validator('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10')
    def validate_range(cls, v):
        if not (1 <= v <= 5):
            raise ValueError("Response must be between 1 and 5")
        return v

class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    aq_category: str
    aq_score: float
    confidence: float
    core_scores: Dict[str, float]
    model_predictions: Dict[str, str]
    model_confidences: Dict[str, float]
    feature_importance: List[Dict[str, Any]]
    local_shap: List[Dict[str, Any]]          # signed SHAP per feature for predicted class
    weak_dimensions: List[Dict[str, Any]]
    behavioral_pattern: str
    recommendations: List[Dict[str, Any]]

router = APIRouter(prefix="", tags=["Prediction"])

def calculate_core_scores(responses: Dict) -> Dict:
    """Calculate CORE dimension scores"""
    return {
        'Control': (responses['Q1'] + responses['Q5'] + responses['Q9']) / 3,
        'Ownership': (responses['Q2'] + responses['Q6']) / 2,
        'Reach': (responses['Q3'] + responses['Q7']) / 2,
        'Endurance': (responses['Q4'] + responses['Q8'] + responses['Q10']) / 3,
    }

def identify_weak_dimensions(core_scores: Dict, threshold=3.0) -> List[Dict]:
    """Identify weak CORE dimensions below threshold"""
    weak_dims = []
    for dimension, score in core_scores.items():
        if score < threshold:
            severity = 'Critical' if score < 2.0 else 'High' if score < 2.5 else 'Moderate'
            weak_dims.append({
                'dimension': dimension,
                'score': float(score),
                'severity': severity,
                'target_score': 3.5,
                'improvement_needed': float(3.5 - score)
            })
    return sorted(weak_dims, key=lambda x: x['improvement_needed'], reverse=True)

def get_behavioral_pattern(ml_category: str, core_scores: Dict) -> str:
    """Dynamically generated based strictly on the ML predicted category."""
    strongest = max(core_scores, key=core_scores.get)
    weakest = min(core_scores, key=core_scores.get)
    
    if ml_category == 'High':
        return f"HIGHLY RESILIENT: ML Prediction confirms excellent resilience. Strongest dimension: {strongest}. You maintain confidence under pressure."
    elif ml_category == 'Medium':
        return f"MODERATELY RESILIENT: Solid resilience detected. Strongest: {strongest}, Weakest: {weakest}. Focus on strengthening your {weakest} dimension."
    else:
        return f"BUILDING RESILIENCE: ML flagged low resilience. Priority is to build your {weakest} dimension. External support and mentorship recommended."

def generate_recommendations(core_scores: Dict) -> List[Dict]:
    """Generate personalized improvement recommendations based on weak dimensions."""
    recommendations = []
    weak_dims = identify_weak_dimensions(core_scores)
    
    templates = {
        'Control': {
            'Suggestion': 'Develop personal agency through small wins',
            'Actions': ['Set 3 small achievable academic goals weekly', 'Practice positive self-talk: "I can solve this"', 'Seek mentorship for problem-solving strategies']
        },
        'Ownership': {
            'Suggestion': 'Build responsibility and learning from mistakes',
            'Actions': ['After each test, do a 10-minute reflection', 'Identify 1 specific area to improve next time', 'Create a "Next time I will..." action plan']
        },
        'Reach': {
            'Suggestion': 'Build resilience in self-concept',
            'Actions': ['Practice positive identity statements daily', 'Remember: one grade doesn\'t define your entire identity', 'Compartmentalize failures as specific events']
        },
        'Endurance': {
            'Suggestion': 'Build persistence and long-term motivation',
            'Actions': ['Break large goals into 2-4 week milestones', 'Track progress - even small improvements count', 'Find peers with similar goals for mutual support']
        }
    }
    
    for dim in weak_dims:
        dimension = dim['dimension']
        severity = dim['severity']
        
        if dimension in templates:
            template = templates[dimension]
            recommendations.append({
                'Dimension': dimension,
                'Priority': severity,
                'Suggestion': template['Suggestion'],
                'Actions': template['Actions']
            })
    
    return recommendations
@router.post("/predict", response_model=PredictionResponse)
async def predict_aq(questionnaire: QuestionnaireInput):
    try:
        from main import model_registry
        
       
        if not model_registry.models:
            raise HTTPException(status_code=500, detail="No ML Models are trained/loaded. Cannot make predictions.")

        responses = questionnaire.dict()
        X = np.array([[
            responses['Q1'], responses['Q2'], responses['Q3'], responses['Q4'], 
            responses['Q5'], responses['Q6'], responses['Q7'], responses['Q8'], 
            responses['Q9'], responses['Q10']
        ]])
        
        core_scores = calculate_core_scores(responses)
        
        model_predictions = {}
        model_confidences = {}
        confidences = []
        
        class_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

       
        for model_name in model_registry.models.keys():
            try:
                pred, proba = model_registry.predict(X, model_name)
                pred_str = class_mapping.get(int(pred), str(pred))
                
                model_predictions[model_name] = pred_str
                confidence = float(np.max(proba))
                model_confidences[model_name] = confidence
                confidences.append(confidence)
            except Exception as e:
                logger.error(f"Prediction failed for model {model_name}: {e}")

        if not model_predictions:
            raise HTTPException(status_code=500, detail="All loaded models failed to make a prediction.")

        
        best_model_name = list(model_predictions.keys())[0] 
        if model_registry.evaluation_metrics:
            try:
                
                best_model_data = max(model_registry.evaluation_metrics, key=lambda x: x['accuracy'])
                if best_model_data['model_name'] in model_predictions:
                    best_model_name = best_model_data['model_name']
            except Exception as e:
                logger.warning(f"Failed to find best model from metrics: {e}")
        
        final_aq_category = model_predictions[best_model_name]
        avg_confidence = float(np.mean(confidences))
        
        
        feature_importance = []
        local_shap = []
        shap_explainer = model_registry.shap_explainers.get(best_model_name)
        if shap_explainer:
            try:
                raw_shap = shap_explainer.shap_values(X)
                feature_names = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

                # ── Mean |SHAP| across classes → top-3 local importance (existing field) ──
                if isinstance(raw_shap, list):
                    abs_vals = np.mean([np.abs(sv[0]) for sv in raw_shap], axis=0)
                else:
                    arr = np.array(raw_shap)
                    abs_vals = np.mean(np.abs(arr[0]), axis=-1) if arr.ndim == 3 else np.abs(arr[0])
                fi_dict = dict(zip(feature_names, abs_vals))
                sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
                feature_importance = [
                    {'question': k, 'importance': float(v), 'rank': idx + 1}
                    for idx, (k, v) in enumerate(sorted_fi[:3])
                ]

                # ── Signed SHAP for predicted class → local waterfall chart ──
                pred_class_idx = {v: k for k, v in class_mapping.items()}.get(final_aq_category, 2)
                if isinstance(raw_shap, list):
                    signed_vals = (
                        raw_shap[pred_class_idx][0]
                        if pred_class_idx < len(raw_shap)
                        else raw_shap[-1][0]
                    )
                else:
                    arr = np.array(raw_shap)
                    signed_vals = arr[0, :, pred_class_idx] if arr.ndim == 3 else arr[0]

                local_shap = sorted(
                    [
                        {
                            "feature": f,
                            "shap_value": round(float(v), 6),
                            "direction": (
                                f"pushes toward {final_aq_category} AQ"
                                if float(v) > 0
                                else f"pushes away from {final_aq_category} AQ"
                            ),
                        }
                        for f, v in zip(feature_names, signed_vals)
                    ],
                    key=lambda x: abs(x["shap_value"]),
                    reverse=True,
                )
            except Exception as e:
                logger.warning(f"Failed to calculate SHAP explanation: {e}", exc_info=True)

        weak_dimensions = identify_weak_dimensions(core_scores)
        behavioral_pattern = get_behavioral_pattern(final_aq_category, core_scores)
        recommendations = generate_recommendations(core_scores)
        
        response = PredictionResponse(
            aq_category=final_aq_category,
            aq_score=float(np.mean(list(core_scores.values()))),
            confidence=avg_confidence,
            core_scores={k: float(v) for k, v in core_scores.items()},
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            feature_importance=feature_importance,
            local_shap=local_shap,
            weak_dimensions=weak_dimensions,
            behavioral_pattern=behavioral_pattern,
            recommendations=recommendations
        )

        # ── Persist prediction to data/predictions_log.csv ────────────────────
        # Logs Q1-Q10 (raw inputs), calculated CORE dimensions, AQ score,
        # and numeric Target_Category (Low=0, Medium=1, High=2).
        log_prediction(responses, final_aq_category)

        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")