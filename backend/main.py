from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import joblib
import json
import os

from router import predict, models, features, dimensions, health

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRegistry:
  

    def __init__(self, models_dir='ml_models'):
        self.models_dir = models_dir

        # Primary model store  {model_name: sklearn model}
        self.models = {}

        # Per-model optional components  {model_name: component}
        self.scalers = {}
        self.shap_explainers = {}
        self.X_trains = {}

        # Aggregated metadata
        self.evaluation_metrics = []       # list of dicts, one per model
        self.global_feature_importance = [] # resolved after all models load (best model)
        self._all_feature_importances = {}  # {model_name: [...]} – collected during scan

        self.load_models()

    
    def load_models(self):
        try:
            if not os.path.exists(self.models_dir):
                logger.warning(f"Models directory '{self.models_dir}' not found!")
                return False

            for subfolder in sorted(os.listdir(self.models_dir)):
                subfolder_path = os.path.join(self.models_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                # ── 1. Resolve model_name from evaluation_metrics.json ───────────
                metrics_path = os.path.join(subfolder_path, 'evaluation_metrics.json')
                if not os.path.exists(metrics_path):
                    logger.warning(
                        f"No evaluation_metrics.json in '{subfolder_path}', skipping."
                    )
                    continue

                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                if not metrics:
                    continue

                model_name = metrics[0]['model_name']   

                # ── 2. Find and load *_model.pkl ─────────────────────────────────
                model_file = next(
                    (fn for fn in os.listdir(subfolder_path) if fn.endswith('_model.pkl')),
                    None
                )
                if model_file is None:
                    logger.warning(
                        f"No *_model.pkl in '{subfolder_path}', skipping."
                    )
                    continue

                self.models[model_name] = joblib.load(
                    os.path.join(subfolder_path, model_file)
                )
                logger.info(f"Loaded model '{model_name}' from {model_file}")

                # ── 3. Optional per-model components ─────────────────────────────
                scaler_path = os.path.join(subfolder_path, 'scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scalers[model_name] = joblib.load(scaler_path)
                    logger.info(f"  + scaler loaded for '{model_name}'")

                shap_path = os.path.join(subfolder_path, 'shap_explainer.pkl')
                if os.path.exists(shap_path):
                    self.shap_explainers[model_name] = joblib.load(shap_path)
                    logger.info(f"  + SHAP explainer loaded for '{model_name}'")

                xtrain_path = os.path.join(subfolder_path, 'X_train.pkl')
                if os.path.exists(xtrain_path):
                    self.X_trains[model_name] = joblib.load(xtrain_path)
                    logger.info(f"  + X_train loaded for '{model_name}'")

                
                self.evaluation_metrics.extend(metrics)

                # ── 5. Feature importance – collect per model; resolve best later
                fi_path = os.path.join(subfolder_path, 'feature_importance.json')
                if os.path.exists(fi_path):
                    with open(fi_path, 'r') as f:
                        self._all_feature_importances[model_name] = json.load(f)

            # ── 6. Resolve global feature importance from the BEST model ─────
            # Best = highest accuracy → then f1 → then auc_roc → lowest cv_std
            if self.evaluation_metrics and self._all_feature_importances:
                try:
                    best = max(
                        self.evaluation_metrics,
                        key=lambda x: (
                            x.get('accuracy', 0),
                            x.get('f1_score', 0),
                            x.get('auc_roc', 0),
                            -x.get('cv_std', 1),
                        )
                    )
                    best_name = best['model_name']
                    if best_name in self._all_feature_importances:
                        self.global_feature_importance = self._all_feature_importances[best_name]
                        logger.info(
                            f"Global feature importance sourced from best model: '{best_name}' "
                            f"(accuracy={best.get('accuracy', 0):.4f})"
                        )
                    else:
                        # Fallback: any available
                        fallback_name = next(iter(self._all_feature_importances))
                        self.global_feature_importance = self._all_feature_importances[fallback_name]
                        logger.warning(
                            f"Best model '{best_name}' has no feature_importance.json. "
                            f"Falling back to '{fallback_name}'."
                        )
                except Exception as e:
                    logger.warning(f"Could not resolve best model for feature importance: {e}")
                    if self._all_feature_importances:
                        fallback_name = next(iter(self._all_feature_importances))
                        self.global_feature_importance = self._all_feature_importances[fallback_name]

            logger.info(
                f"ModelRegistry ready – loaded {len(self.models)} model(s): "
                f"{list(self.models.keys())}"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            return False

    
    def predict(self, X, model_name):
        """Return (prediction_label, probabilities) for the given model."""
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)          # None if model doesn't need scaling
        X_processed = scaler.transform(X) if scaler is not None else X
        return model.predict(X_processed)[0], model.predict_proba(X_processed)[0]



app = FastAPI(title="Exploring the Adversity quotient of adoloscents a behavioural pattern")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global model_registry
    model_registry = ModelRegistry()

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(features.router)
app.include_router(dimensions.router)

@app.get("/")
async def root():
    return {"AQ of adoloscents"}