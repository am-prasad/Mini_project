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
    """Manage trained models and dynamic ML metadata.

    Folder layout expected under `models_dir`:
        ml_models/
            logistic_regression/
                logistic_regression_model.pkl
                scaler.pkl              (optional – only for models that need scaling)
                shap_explainer.pkl      (optional)
                X_train.pkl             (optional)
                evaluation_metrics.json (required – contains {"model_name": ...})
                feature_importance.json (optional)
            random_forest/
                random_forest_model.pkl
                shap_explainer.pkl
                X_train.pkl
                evaluation_metrics.json
                feature_importance.json
    """

    def __init__(self, models_dir='ml_models'):
        self.models_dir = models_dir

        # Primary model store  {model_name: sklearn model}
        self.models = {}

        # Per-model optional components  {model_name: component}
        self.scalers = {}
        self.shap_explainers = {}
        self.X_trains = {}

        # Aggregated metadata
        self.evaluation_metrics = []      # list of dicts, one per model
        self.global_feature_importance = []  # from first model found

        self.load_models()

    # ------------------------------------------------------------------
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

                # ── 4. Accumulate evaluation metrics ─────────────────────────────
                self.evaluation_metrics.extend(metrics)

                # ── 5. Feature importance – use first model found as global default
                fi_path = os.path.join(subfolder_path, 'feature_importance.json')
                if os.path.exists(fi_path) and not self.global_feature_importance:
                    with open(fi_path, 'r') as f:
                        self.global_feature_importance = json.load(f)

            logger.info(
                f"ModelRegistry ready – loaded {len(self.models)} model(s): "
                f"{list(self.models.keys())}"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    def predict(self, X, model_name):
        """Return (prediction_label, probabilities) for the given model."""
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)          # None if model doesn't need scaling
        X_processed = scaler.transform(X) if scaler is not None else X
        return model.predict(X_processed)[0], model.predict_proba(X_processed)[0]


# ── FastAPI app ────────────────────────────────────────────────────────────────
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