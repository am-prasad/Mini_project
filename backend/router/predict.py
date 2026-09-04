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

class AcademicIndicator(BaseModel):
    name: str
    score: int          # 0-100
    rating: str         # Excellent / Good / Moderate / Needs Work
    description: str

class AcademicProfile(BaseModel):
    overall_index: int
    outlook: str        # High / Medium / Low
    headline: str
    summary: str
    indicators: List[AcademicIndicator]
    dimension_impact: List[Dict[str, Any]]

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
    academic_profile: AcademicProfile

router = APIRouter(prefix="", tags=["Prediction"])

def calculate_core_scores(responses: Dict) -> Dict:
    """Calculate CORE dimension scores based on form question mapping"""
    return {
        'Control': (responses['Q5'] + responses['Q8'] + responses['Q9']) / 3,
        'Ownership': (responses['Q2'] + responses['Q6'] + responses['Q10']) / 3,
        'Reach': (responses['Q3'] + responses['Q7']) / 2,
        'Endurance': (responses['Q1'] + responses['Q4']) / 2,
    }

def identify_weak_dimensions(core_scores: Dict) -> List[Dict]:
    """Classify all four CORE dimensions by their score severity level."""
    severity_rank = {'Critical': 0, 'High': 1, 'Moderate': 2, 'Growth': 3}
    dim_list = []
    for dimension in ['Control', 'Ownership', 'Reach', 'Endurance']:
        score = float(core_scores.get(dimension, 3.0))
        if score < 2.0:
            severity = 'Critical'
            target = 3.5
        elif score < 2.75:
            severity = 'High'
            target = 3.5
        elif score < 3.5:
            severity = 'Moderate'
            target = 4.0
        else:
            severity = 'Growth'
            target = 5.0

        dim_list.append({
            'dimension': dimension,
            'score': round(score, 2),
            'severity': severity,
            'target_score': target,
            'improvement_needed': round(max(0.0, target - score), 2),
            '_rank': severity_rank[severity]
        })

    # Sort with most critical priorities first, then by score ascending
    sorted_dims = sorted(dim_list, key=lambda x: (x['_rank'], x['score']))
    for d in sorted_dims:
        d.pop('_rank', None)
    return sorted_dims

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

# ─────────────────────────────────────────────────────────────────────────────
# Academic Performance Profile
# Maps CORE dimension scores → research-backed academic indicators
# ─────────────────────────────────────────────────────────────────────────────

_ACADEMIC_DESCRIPTIONS = {
    'Academic Persistence': (
        'Likelihood of persisting through difficult courses, long projects, '
        'and multi-semester challenges without giving up.'
    ),
    'Grade Recovery Speed': (
        'How quickly the student bounces back from low marks by actively '
        'reviewing mistakes and adapting their study strategy.'
    ),
    'Burnout Resistance': (
        'Ability to manage exam-period stress without spillover across '
        'subjects or loss of overall motivation.'
    ),
    'Academic Self-Efficacy': (
        'Belief in one\'s own ability to influence academic outcomes '
        'through focused effort and problem-solving.'
    ),
    'Stress Compartmentalisation': (
        'Capacity to prevent one bad result from affecting confidence or '
        'performance in other academic areas.'
    ),
    'Learning Agility': (
        'Speed of adapting to new topics, new teaching styles, and course '
        'format changes while maintaining accountability.'
    ),
}

_OUTLOOK_TEXT = {
    'High': (
        'Strong Academic Potential',
        'This student is likely to perform well under pressure, recover from '
        'setbacks quickly, and maintain sustained academic engagement.'
    ),
    'Medium': (
        'Developing Academic Resilience',
        'This student shows a reasonable foundation but may struggle during '
        'high-stress periods like exams or project deadlines.'
    ),
    'Low': (
        'Academic Support Recommended',
        'This student may find it difficult to cope with academic adversity. '
        'Early intervention and mentoring is strongly advised.'
    ),
}

_DIMENSION_IMPACT = [
    {
        'dimension': 'Control',
        'icon': '🎯',
        'academic_implication': 'Self-directed study, proactive help-seeking, strong study habits',
        'risk_if_low': 'Learned helplessness, poor time management',
    },
    {
        'dimension': 'Ownership',
        'icon': '🔑',
        'academic_implication': 'Takes responsibility for grades, learns from exam feedback',
        'risk_if_low': 'Blames external factors, repeats same mistakes',
    },
    {
        'dimension': 'Reach',
        'icon': '🌐',
        'academic_implication': 'Isolates subject failures, maintains broad academic confidence',
        'risk_if_low': 'One failure cascades into overall academic decline',
    },
    {
        'dimension': 'Endurance',
        'icon': '⏳',
        'academic_implication': 'Long-term commitment, project completion, low dropout risk',
        'risk_if_low': 'Short-term thinking, prone to abandoning hard courses',
    },
]

def _rating(score: int) -> str:
    if score >= 80: return 'Excellent'
    if score >= 60: return 'Good'
    if score >= 40: return 'Moderate'
    return 'Needs Work'

def compute_academic_profile(core_scores: Dict, aq_category: str) -> AcademicProfile:
    """Derive academic performance indicators from CORE dimension scores."""
    c = float(core_scores.get('Control',   0))
    o = float(core_scores.get('Ownership', 0))
    r = float(core_scores.get('Reach',     0))
    e = float(core_scores.get('Endurance', 0))

    # Each indicator is a weighted blend of the most relevant dimensions,
    # scaled to a 0-100 percentage.
    raw = {
        'Academic Persistence':        int(round(((e * 0.6 + c * 0.4) / 5) * 100)),
        'Grade Recovery Speed':        int(round(((o * 0.55 + c * 0.45) / 5) * 100)),
        'Burnout Resistance':          int(round(((r * 0.6 + e * 0.4) / 5) * 100)),
        'Academic Self-Efficacy':      int(round((c / 5) * 100)),
        'Stress Compartmentalisation': int(round(((r * 0.7 + o * 0.3) / 5) * 100)),
        'Learning Agility':            int(round(((o * 0.5 + e * 0.3 + c * 0.2) / 5) * 100)),
    }

    indicators = [
        AcademicIndicator(
            name=name,
            score=score,
            rating=_rating(score),
            description=_ACADEMIC_DESCRIPTIONS[name],
        )
        for name, score in raw.items()
    ]

    overall_index = int(round(sum(raw.values()) / len(raw)))
    outlook_key   = aq_category if aq_category in _OUTLOOK_TEXT else 'Medium'
    headline, summary = _OUTLOOK_TEXT[outlook_key]

    # Enrich dimension_impact with the student's actual scores
    dimension_impact = [
        {
            **entry,
            'score': round(float(core_scores.get(entry['dimension'], 0)), 2),
        }
        for entry in _DIMENSION_IMPACT
    ]

    return AcademicProfile(
        overall_index=overall_index,
        outlook=outlook_key,
        headline=headline,
        summary=summary,
        indicators=indicators,
        dimension_impact=dimension_impact,
    )

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
    
    growth_templates = {
        'Control': {
            'Suggestion': 'Sustain personal agency and advance proactive problem-solving',
            'Actions': ['Mentor peers struggling with challenging coursework', 'Set advanced stretch goals beyond standard syllabus requirements', 'Maintain structured reflection routines during peak exam seasons']
        },
        'Ownership': {
            'Suggestion': 'Maintain high accountability and continuous learning',
            'Actions': ['Lead collaborative study groups to reinforce mastery', 'Systematically document revision habits that deliver top results', 'Continue viewing unexpected grading setbacks as refined feedback']
        },
        'Reach': {
            'Suggestion': 'Reinforce strong compartmentalization and holistic balance',
            'Actions': ['Cultivate diverse extracurricular interests to maintain balance', 'Support classmates in keeping isolated subject setbacks in perspective', 'Maintain broad perspective during competitive examination cycles']
        },
        'Endurance': {
            'Suggestion': 'Preserve stamina and long-horizon persistence',
            'Actions': ['Pace yourself strategically during intensive multi-semester projects', 'Schedule intentional recovery periods to prevent cognitive fatigue', 'Keep celebrating milestones on long-term career aspirations']
        }
    }
    
    for dim in weak_dims:
        dimension = dim['dimension']
        severity = dim['severity']
        
        tmpl = growth_templates.get(dimension) if severity == 'Growth' else templates.get(dimension)
        if not tmpl and dimension in templates:
            tmpl = templates[dimension]
            
        if tmpl:
            recommendations.append({
                'Dimension': dimension,
                'Priority': severity,
                'Suggestion': tmpl['Suggestion'],
                'Actions': tmpl['Actions']
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
        academic_profile  = compute_academic_profile(core_scores, final_aq_category)
        
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
            recommendations=recommendations,
            academic_profile=academic_profile,
        )

        # ── Persist prediction to data/predictions_log.csv ────────────────────
        # Logs Q1-Q10 (raw inputs), calculated CORE dimensions, AQ score,
        # and numeric Target_Category (Low=0, Medium=1, High=2).
        log_prediction(responses, final_aq_category)

        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")