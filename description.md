# Project: Adversity Quotient Analysis of Adolescents - Multi-Regression Model Performance Evaluation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Use Case & Scope](#use-case--scope)
4. [Architecture Design](#architecture-design)
5. [Frontend Architecture](#frontend-architecture)
6. [Backend Architecture](#backend-architecture)
7. [Machine Learning Pipeline](#machine-learning-pipeline)
8. [Design Decisions](#design-decisions)
9. [Implementation Plan](#implementation-plan)
10. [Metrics & Evaluation Strategy](#metrics--evaluation-strategy)
11. [Database Schema](#database-schema)
12. [API Specifications](#api-specifications)
13. [Risk Management](#risk-management)
14. [Timeline & Milestones](#timeline--milestones)

---

## Executive Summary

This project develops an integrated web application to analyze and compare the performance of multiple regression models in predicting the **Adversity Quotient (AQ)** of adolescents. The system enables researchers and educators to understand how various demographic, psychological, and environmental factors influence adolescents' ability to overcome adversity.

**Key Deliverables:**
- Web-based data analysis platform
- Multiple regression models (Linear, Ridge, Lasso, SVR, Random Forest, Gradient Boosting, Neural Networks)
- Real-time model comparison dashboard
- Comprehensive metrics evaluation system
- Data visualization and insights export

---

## Project Overview

### Objectives

1. **Build a predictive model** that accurately estimates AQ scores based on demographic and psychological factors
2. **Compare regression models** using multiple performance metrics
3. **Provide interpretability** of model predictions and feature importance
4. **Enable data exploration** through interactive visualizations
5. **Support research workflows** with model comparison and export capabilities

### Problem Statement

Adolescents face varying levels of adversity, and understanding their Adversity Quotient is crucial for:
- Educational interventions
- Mental health support planning
- Resource allocation
- Individual development tracking

Current limitations:
- Lack of unified platform for AQ analysis
- No comparative analysis of different modeling approaches
- Limited visualization of factor relationships
- Manual, time-consuming model evaluation

---

## Use Case & Scope

### Primary Use Cases

#### Use Case 1: Data Upload & Preprocessing
**Actor:** Researcher/Administrator
**Flow:**
1. Upload CSV/Excel file containing adolescent survey data
2. System validates data format and quality
3. Automatic preprocessing (missing value handling, normalization)
4. Data quality report generated
5. Proceed to model training

**Success Criteria:** 
- Data correctly parsed and stored
- 95%+ data quality threshold achieved
- Preprocessing results clearly documented

#### Use Case 2: Model Training & Comparison
**Actor:** Data Scientist/Researcher
**Flow:**
1. Select features for model training
2. Choose train-test split ratio
3. Select one or multiple regression models
4. Run training pipeline
5. View comparative performance metrics
6. Analyze model-specific insights

**Success Criteria:**
- All models train successfully
- Real-time progress tracking
- Comparison metrics displayed within 5 seconds

#### Use Case 3: Results Analysis & Interpretation
**Actor:** Educational Administrator/Counselor
**Flow:**
1. View model rankings by performance metric
2. Examine feature importance visualizations
3. Explore prediction distributions
4. Generate summary reports
5. Export insights for stakeholder communication

**Success Criteria:**
- Visual clarity and interpretability
- Accurate predictions within ±0.5 AQ points
- Export functionality functional

#### Use Case 4: Individual Prediction
**Actor:** School Counselor/Educator
**Flow:**
1. Enter adolescent's characteristics (survey responses)
2. Best model predicts AQ score
3. View confidence intervals
4. Generate personalized insights report
5. Track predictions over time

**Success Criteria:**
- Prediction generated in <2 seconds
- Confidence intervals provided
- Actionable recommendations included

### Project Scope

**Included:**
- Data import (CSV, Excel)
- 7+ regression models implementation
- Cross-validation and hyperparameter tuning
- Model comparison dashboard
- Feature importance analysis
- Prediction intervals
- Export functionality (PDF, CSV reports)
- Real-time data visualization
- User authentication & roles
- Audit logging

**Excluded:**
- Real-time streaming data ingestion
- Mobile application (responsive web only)
- Advanced NLP for open-ended responses
- Longitudinal trend analysis (single-point analysis only)
- Integration with external educational systems
- Automated intervention recommendation system

---

## Architecture Design

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER (Frontend)                   │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐   │
│  │  Dashboard   │  Data Upload │ Model Config │ Results View │   │
│  │  (React)     │  & Preview   │  & Training  │  & Export    │   │
│  └──────────────┴──────────────┴──────────────┴──────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API + WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│                     API GATEWAY & AUTH LAYER                     │
│              (JWT Auth, Rate Limiting, Request Routing)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    BACKEND LAYER (Node.js/Python)               │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │   User Mgmt  │   Data Mgmt  │  Model Mgmt  │  Report Mgmt │  │
│  │   Service    │   Service    │   Service    │  Service     │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │    Job Queue (Celery) - Async Task Processing            │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────┬──────────────────────┬──────────────────────────────┘
         │                      │
         │                      │
    ┌────▼──────┐        ┌──────▼────────┐
    │ Database  │        │ ML Pipeline   │
    │(PostgreSQL)       │  (Python ML   │
    └──────────┘        │   Service)    │
                        └───────────────┘
```

### Technology Stack

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Frontend** | UI Framework | React 18+ |
| | State Management | Redux Toolkit |
| | HTTP Client | Axios |
| | Real-time Updates | Socket.io |
| | Visualization | Plotly.js, D3.js, Chart.js |
| | UI Components | Material-UI v5 |
| **Backend** | API Server | Node.js + Express.js / Python Flask |
| | Task Queue | Celery + Redis |
| | Database | PostgreSQL 13+ |
| | Cache | Redis |
| | File Storage | AWS S3 / MinIO |
| **ML** | Core Framework | Python 3.9+ |
| | ML Libraries | Scikit-learn, XGBoost, LightGBM |
| | Deep Learning | TensorFlow/PyTorch |
| | Data Processing | Pandas, NumPy |
| | Visualization | Matplotlib, Seaborn |
| | Experiment Tracking | MLflow / Weights & Biases |
| **DevOps** | Containerization | Docker |
| | Orchestration | Docker Compose / Kubernetes |
| | CI/CD | GitHub Actions / GitLab CI |

---

## Frontend Architecture

### Component Structure

```
src/
├── components/
│   ├── Layout/
│   │   ├── Navbar.jsx
│   │   ├── Sidebar.jsx
│   │   └── MainLayout.jsx
│   ├── Auth/
│   │   ├── LoginForm.jsx
│   │   ├── RegisterForm.jsx
│   │   └── ProtectedRoute.jsx
│   ├── Dashboard/
│   │   ├── OverviewPanel.jsx
│   │   ├── ModelComparisonChart.jsx
│   │   ├── QuickStats.jsx
│   │   └── RecentModels.jsx
│   ├── DataManagement/
│   │   ├── DataUpload.jsx
│   │   ├── DataPreview.jsx
│   │   ├── DataQualityReport.jsx
│   │   └── FeatureSelector.jsx
│   ├── ModelTraining/
│   │   ├── ModelSelector.jsx
│   │   ├── HyperparameterConfig.jsx
│   │   ├── TrainingProgress.jsx
│   │   └── ParameterGrid.jsx
│   ├── ResultsAnalysis/
│   │   ├── ModelRanking.jsx
│   │   ├── PerformanceMetrics.jsx
│   │   ├── FeatureImportance.jsx
│   │   ├── ResidualPlots.jsx
│   │   ├── PredictionDistribution.jsx
│   │   └── ConfusionMatrix.jsx (for classification variants)
│   ├── Visualization/
│   │   ├── InteractiveScatter.jsx
│   │   ├── RegressionLine.jsx
│   │   ├── CorrelationHeatmap.jsx
│   │   └── FeatureRelationship.jsx
│   ├── Prediction/
│   │   ├── PredictionForm.jsx
│   │   ├── PredictionResult.jsx
│   │   └── ConfidenceInterval.jsx
│   ├── Reports/
│   │   ├── ReportGenerator.jsx
│   │   ├── ExportOptions.jsx
│   │   └── ReportTemplate.jsx
│   └── Common/
│       ├── LoadingSpinner.jsx
│       ├── ErrorBoundary.jsx
│       ├── ConfirmDialog.jsx
│       └── Toast.jsx
├── pages/
│   ├── LoginPage.jsx
│   ├── DashboardPage.jsx
│   ├── DataUploadPage.jsx
│   ├── ModelTrainingPage.jsx
│   ├── ResultsPage.jsx
│   ├── PredictionPage.jsx
│   └── NotFoundPage.jsx
├── services/
│   ├── api.js
│   ├── authService.js
│   ├── dataService.js
│   ├── modelService.js
│   └── reportService.js
├── store/
│   ├── slices/
│   │   ├── authSlice.js
│   │   ├── dataSlice.js
│   │   ├── modelSlice.js
│   │   ├── resultsSlice.js
│   │   └── uiSlice.js
│   └── store.js
├── hooks/
│   ├── useAuth.js
│   ├── useData.js
│   ├── useModel.js
│   └── useWebSocket.js
├── utils/
│   ├── formatters.js
│   ├── validators.js
│   ├── constants.js
│   └── helpers.js
├── styles/
│   ├── theme.css
│   ├── globals.css
│   └── components.css
└── App.jsx
```

### Key UI Components & Features

#### 1. Dashboard
- **Real-time model performance tracking**
- Widgets showing:
  - Total models trained
  - Best performing model
  - Average metrics
  - Recent activities timeline
- Quick links to primary workflows
- System health status

**Data Flow:**
```
Redux Store (modelState) 
  → Dashboard Component 
  → WebSocket listeners update in real-time
  → Charts re-render on new data
```

#### 2. Data Upload & Preprocessing
- **Drag-and-drop interface** for CSV/Excel
- **Data preview table** (first 100 rows)
- **Data quality assessment:**
  - Missing value percentages
  - Data type detection
  - Duplicate row identification
  - Statistical summary
- **Feature mapping:** UI to rename/select relevant features
- **Preview processing results** before confirmation

**Validation Rules:**
- File size limit: 100MB
- Minimum rows: 50
- Minimum features: 5
- Supported formats: CSV, XLSX, JSON

#### 3. Model Configuration & Training
- **Model Selector** with descriptions and algorithms
- **Feature Selection Panel:**
  - Checkbox selection
  - Feature statistics display
  - Correlation preview
- **Hyperparameter Configuration:**
  - Pre-populated defaults
  - Visual sliders for numeric parameters
  - Dropdown selections for categorical parameters
- **Cross-Validation Settings:**
  - k-fold selector (3-10)
  - Train/test split ratio selector
- **Real-time Training Progress:**
  - Progress bar with percentage
  - Current fold/epoch display
  - Time elapsed and estimated time
  - Status messages and warnings

#### 4. Results Analysis Dashboard
- **Model Comparison Table:**
  - Columns: Model Name, R², RMSE, MAE, MAPE, CV Score
  - Sortable/filterable
  - Color-coding for best/worst performance
- **Visualizations:**
  - Bar charts comparing metrics
  - Line charts for cross-validation progression
  - Scatter plots (Predicted vs Actual)
  - Residual plots (distribution, Q-Q plots)
- **Feature Importance Charts:**
  - Horizontal bar charts (top 15 features)
  - SHAP values visualization (if available)
  - Feature contribution to predictions
- **Statistical Summary:**
  - Prediction error distribution
  - Confidence/prediction intervals
  - Assumption validation results

#### 5. Prediction Interface
- **Input Form:**
  - Fields for all demographic/psychological features
  - Tooltips and validation rules
  - Form auto-save to local storage
- **Prediction Results:**
  - Predicted AQ score (displayed prominently)
  - Confidence interval (95% by default)
  - Prediction probability/reliability score
  - Comparison to population average
  - Top 5 influential factors for this prediction
- **Interpretation Guide:**
  - What does this AQ score mean?
  - Actionable insights and recommendations
  - Similar cases in database (anonymized)

#### 6. Export & Reporting
- **Export Formats:**
  - PDF report (formatted with charts)
  - CSV (detailed results)
  - PNG/SVG (individual charts)
  - Excel workbook (multiple sheets)
- **Report Templates:**
  - Executive summary
  - Technical analysis
  - Educator summary
  - Research paper format

### Frontend Responsive Design
- **Mobile (< 768px):** Single column, stack panels vertically
- **Tablet (768px - 1024px):** Two-column layout
- **Desktop (> 1024px):** Full multi-panel dashboard

### State Management with Redux

```javascript
// Redux Store Structure
{
  auth: {
    isAuthenticated: bool,
    user: { id, email, role, name },
    token: string
  },
  data: {
    datasets: [{ id, name, rows, cols, uploadDate, status }],
    activeDataset: object,
    preprocessing: { missingHandled, normalized, outliersTreated },
    selectedFeatures: [],
    statistics: {}
  },
  models: {
    trained: [{ id, name, algorithm, hyperparameters, trainedAt }],
    training: {
      inProgress: bool,
      currentModel: string,
      progress: number,
      eta: number
    }
  },
  results: {
    metrics: {
      r2: float,
      rmse: float,
      mae: float,
      crossValScore: float
    },
    predictions: [{ actual, predicted, residual }],
    featureImportance: [{ feature, importance }]
  },
  ui: {
    sidebarOpen: bool,
    currentPage: string,
    notifications: [],
    darkMode: bool
  }
}
```

---

## Backend Architecture

### API Structure

```
/api/
├── /auth
│   ├── POST /register (public)
│   ├── POST /login (public)
│   ├── POST /logout
│   ├── POST /refresh-token
│   └── GET /me (current user)
├── /data
│   ├── POST /upload (file upload)
│   ├── GET / (list datasets)
│   ├── GET /:id (dataset details)
│   ├── POST /:id/preprocess
│   ├── GET /:id/preview
│   ├── GET /:id/quality-report
│   ├── DELETE /:id
│   └── POST /:id/features (feature operations)
├── /models
│   ├── POST /train (start training)
│   ├── GET / (list trained models)
│   ├── GET /:id (model details)
│   ├── GET /:id/metrics
│   ├── GET /:id/predictions
│   ├── POST /:id/predict (single prediction)
│   ├── POST /:id/predict-batch (batch prediction)
│   ├── DELETE /:id
│   └── GET /:id/feature-importance
├── /comparison
│   ├── GET / (get comparison data)
│   ├── POST / (generate comparison)
│   └── GET /:id/export
├── /reports
│   ├── GET / (list reports)
│   ├── POST / (generate report)
│   ├── GET /:id (report details)
│   ├── GET /:id/download (PDF/CSV download)
│   └── DELETE /:id
└── /admin
    ├── GET /users
    ├── GET /system-stats
    ├── POST /audit-logs
    └── POST /backup
```

### Backend Service Architecture

#### 1. Authentication Service
```python
class AuthService:
    def register(email, password, name, role):
        # Validate input
        # Hash password (bcrypt)
        # Create user record
        # Send verification email
        return user_id, verification_token
    
    def login(email, password):
        # Verify credentials
        # Generate JWT token pair (access + refresh)
        # Store refresh token in secure cookie
        return access_token, user_data
    
    def verify_token(token):
        # Decode JWT
        # Check expiration
        # Return user_id if valid
        raise UnauthorizedException if invalid
    
    def refresh_token(refresh_token):
        # Validate refresh token
        # Generate new access token
        return new_access_token
```

#### 2. Data Management Service
```python
class DataService:
    def upload_dataset(file, user_id):
        # Validate file (format, size, content)
        # Parse CSV/Excel
        # Store in S3
        # Create database record
        # Queue preprocessing job
        return dataset_id
    
    def preprocess_data(dataset_id):
        # Handle missing values (mean/median/forward-fill)
        # Remove duplicates
        # Normalize/standardize features
        # Detect and handle outliers
        # Create features view
        return preprocessing_report
    
    def get_statistics(dataset_id):
        # Calculate descriptive statistics
        # Compute correlations
        # Feature distributions
        # Identify categorical vs continuous
        return statistics_dict
    
    def validate_data_quality(dataset_id):
        # Missing percentage per feature
        # Duplicates count
        # Type mismatches
        # Outlier detection
        return quality_score, report
```

#### 3. Model Management Service
```python
class ModelService:
    def train_model(model_type, dataset_id, features, hyperparams):
        # Load dataset
        # Apply feature selection
        # Split data (train/val/test)
        # Train model with hyperparameters
        # Validate on cross-validation folds
        # Store model artifact
        # Calculate metrics
        return model_id, metrics
    
    def get_model_metrics(model_id):
        # Retrieve stored metrics
        # Calculate additional statistics
        # Compute prediction intervals
        return {
            'r2': float,
            'rmse': float,
            'mae': float,
            'mape': float,
            'cv_scores': [float],
            'prediction_interval': float
        }
    
    def predict(model_id, input_features):
        # Load model
        # Validate input
        # Apply same preprocessing as training
        # Generate prediction
        # Calculate confidence interval
        return prediction, confidence_interval
    
    def batch_predict(model_id, batch_data):
        # Load model once
        # Vectorized prediction on batch
        # Calculate intervals for all
        return predictions_array, intervals_array
```

#### 4. Comparison Service
```python
class ComparisonService:
    def compare_models(model_ids):
        # Fetch metrics for all models
        # Normalize metrics to 0-1 scale
        # Calculate comparative statistics
        # Rank models by different criteria
        # Generate comparison visualizations
        return comparison_data, rankings
    
    def statistical_significance_test(model_1_id, model_2_id):
        # Paired t-test on predictions
        # McNemar test
        # Calculate p-value
        return is_significant, p_value, test_results
```

#### 5. Report Service
```python
class ReportService:
    def generate_report(model_id, format='pdf'):
        # Compile model data
        # Generate charts/visualizations
        # Create summary statistics
        # Add interpretations
        # Render template
        # Convert to requested format
        return report_file
    
    def export_predictions(model_id, output_format):
        # Load all predictions
        # Add confidence intervals
        # Include feature values
        # Format and export
        return export_file
```

### Job Queue Processing (Celery)

```python
@app.task(bind=True)
def train_model_async(self, model_type, dataset_id, params):
    try:
        self.update_state(state='PROGRESS', meta={'current': 10})
        
        # Load and preprocess
        data = load_dataset(dataset_id)
        self.update_state(state='PROGRESS', meta={'current': 30})
        
        # Train
        model, metrics = train_regression_model(
            data, model_type, params
        )
        self.update_state(state='PROGRESS', meta={'current': 80})
        
        # Save and store metrics
        save_model(model, model_id)
        store_metrics(model_id, metrics)
        
        return {'status': 'success', 'model_id': model_id}
    
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

# Frontend can poll: GET /api/models/{id}/training-status
# Or use WebSocket for real-time updates
```

### WebSocket Events for Real-time Updates
```javascript
// Server emits
socket.emit('training:start', { model_id, dataset_id })
socket.emit('training:progress', { model_id, progress: 45, eta: 120 })
socket.emit('training:complete', { model_id, metrics })
socket.emit('training:error', { model_id, error: 'message' })

// Client listens
socket.on('training:progress', (data) => {
  dispatch(updateTrainingProgress(data))
  // UI updates automatically
})
```

### Error Handling & Validation

```python
class APIResponse:
    @staticmethod
    def success(data, status_code=200):
        return {
            'success': True,
            'data': data,
            'timestamp': datetime.now()
        }, status_code
    
    @staticmethod
    def error(message, status_code=400, errors=None):
        return {
            'success': False,
            'error': message,
            'details': errors,
            'timestamp': datetime.now()
        }, status_code

# Middleware for error handling
def error_handler(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            return APIResponse.error(str(e), 400, e.messages)
        except UnauthorizedException as e:
            return APIResponse.error(str(e), 401)
        except ForbiddenException as e:
            return APIResponse.error(str(e), 403)
        except NotFoundException as e:
            return APIResponse.error(str(e), 404)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return APIResponse.error(
                "Internal server error",
                500
            )
    return decorated
```

---

## Machine Learning Pipeline

### 1. Data Preprocessing Module

```python
class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.feature_names = None
    
    def fit_transform(self, X, y=None):
        """Fit preprocessor on training data"""
        # Handle missing values
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Remove outliers (optional)
        outlier_indices = self.detect_outliers(X_scaled)
        
        # Feature engineering
        X_engineered = self.create_features(X_scaled)
        
        return X_engineered, outlier_indices
    
    def transform(self, X):
        """Apply fitted transformations to new data"""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        X_engineered = self.create_features(X_scaled)
        return X_engineered
    
    def detect_outliers(self, X, method='iqr', threshold=1.5):
        """Detect outliers using IQR method"""
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        outliers = (X < Q1 - threshold*IQR) | (X > Q3 + threshold*IQR)
        return np.any(outliers, axis=1)
    
    def create_features(self, X):
        """Feature engineering: polynomial, interactions"""
        # Add polynomial features (degree=2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        return X_poly
```

### 2. Regression Models Implementation

```python
class RegressionModelFactory:
    """Factory for instantiating various regression models"""
    
    @staticmethod
    def get_model(model_type, hyperparams):
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(
                alpha=hyperparams.get('alpha', 1.0),
                solver='auto'
            ),
            'lasso': Lasso(
                alpha=hyperparams.get('alpha', 0.1),
                max_iter=hyperparams.get('max_iter', 10000)
            ),
            'elasticnet': ElasticNet(
                alpha=hyperparams.get('alpha', 0.1),
                l1_ratio=hyperparams.get('l1_ratio', 0.5)
            ),
            'svr': SVR(
                kernel=hyperparams.get('kernel', 'rbf'),
                C=hyperparams.get('C', 100),
                epsilon=hyperparams.get('epsilon', 0.1)
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 20),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 5),
                random_state=42
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                num_leaves=hyperparams.get('num_leaves', 31),
                random_state=42
            ),
            'xgboost': XGBRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 6),
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=hyperparams.get('layers', (128, 64, 32)),
                activation=hyperparams.get('activation', 'relu'),
                learning_rate_init=hyperparams.get('lr', 0.001),
                max_iter=hyperparams.get('max_iter', 500),
                random_state=42
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type]

class RegressionPipeline:
    """Complete training and evaluation pipeline"""
    
    def __init__(self, model_type, hyperparams=None):
        self.model_type = model_type
        self.hyperparams = hyperparams or {}
        self.model = RegressionModelFactory.get_model(
            model_type, 
            self.hyperparams
        )
        self.preprocessor = DataPreprocessor()
        self.metrics_history = []
    
    def train(self, X, y, cv_folds=5, test_size=0.2):
        """Train and evaluate model with cross-validation"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42
        )
        
        # Fit preprocessor on training data
        X_train_processed, _ = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Train model
        self.model.fit(X_train_processed, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, 
            X_train_processed, 
            y_train,
            cv=cv_folds,
            scoring='r2'
        )
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_processed)
        
        metrics = self.calculate_metrics(y_test, y_pred, cv_scores)
        
        return {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'metrics': metrics,
            'predictions': {
                'y_test': y_test,
                'y_pred': y_pred,
                'residuals': y_test - y_pred
            }
        }
    
    def calculate_metrics(self, y_true, y_pred, cv_scores):
        """Calculate comprehensive regression metrics"""
        
        # Point metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        
        # Cross-validation
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Prediction interval (95%)
        residuals = y_true - y_pred
        std_residuals = np.std(residuals)
        prediction_interval = 1.96 * std_residuals
        
        # Durbin-Watson for autocorrelation
        dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        
        return {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'medae': float(medae),
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'cv_scores': cv_scores.tolist(),
            'prediction_interval': float(prediction_interval),
            'durbin_watson': float(dw),
            'residual_std': float(std_residuals),
            'train_size': len(y_true),
            'model_type': self.model_type
        }
    
    def predict_with_interval(self, X, confidence=0.95):
        """Make predictions with confidence intervals"""
        X_processed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_processed)
        
        # Calculate prediction intervals
        z_score = norm.ppf((1 + confidence) / 2)
        margin = z_score * self.residual_std
        
        intervals = np.array([
            predictions - margin,
            predictions + margin
        ]).T
        
        return predictions, intervals
```

### 3. Feature Importance & SHAP Values

```python
class FeatureAnalyzer:
    """Analyze feature importance and relationships"""
    
    @staticmethod
    def get_feature_importance(model, X, y, model_type):
        """Extract feature importance based on model type"""
        
        if model_type in ['random_forest', 'gradient_boosting', 'lightgbm', 'xgboost']:
            # Tree-based importance
            importance = model.feature_importances_
            importance_norm = importance / np.sum(importance)
        
        elif model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
            # Coefficient-based importance
            coef = np.abs(model.coef_)
            importance_norm = coef / np.sum(coef)
        
        else:
            # Permutation importance (model-agnostic)
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=10, 
                random_state=42
            )
            importance_norm = perm_importance.importances_mean
        
        return importance_norm
    
    @staticmethod
    def compute_shap_values(model, X, model_type):
        """Compute SHAP values for interpretability"""
        
        if model_type in ['tree', 'random_forest', 'xgboost', 'lightgbm']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(
                model.predict, 
                shap.sample(X, 100)
            )
        
        shap_values = explainer.shap_values(X)
        
        return {
            'shap_values': shap_values,
            'base_value': explainer.expected_value,
            'feature_names': explainer.feature_names
        }
    
    @staticmethod
    def correlation_analysis(X, y):
        """Analyze feature-target correlations"""
        correlations = {}
        for i, feature in enumerate(X.columns):
            corr = np.corrcoef(X.iloc[:, i], y)[0, 1]
            correlations[feature] = corr
        
        return sorted(
            correlations.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
```

### 4. Hyperparameter Tuning

```python
class HyperparameterTuner:
    """Grid search and Bayesian optimization for hyperparameters"""
    
    @staticmethod
    def grid_search(model_type, X, y, param_grid, cv_folds=5):
        """Perform grid search"""
        
        base_model = RegressionModelFactory.get_model(model_type, {})
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    @staticmethod
    def bayesian_optimization(model_type, X, y, space, n_calls=50):
        """Bayesian optimization using hyperopt"""
        
        def objective(params):
            model = RegressionModelFactory.get_model(model_type, params)
            score = -cross_val_score(
                model, X, y, 
                cv=5, 
                scoring='neg_mean_squared_error'
            ).mean()
            return score
        
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=n_calls,
            trials=trials
        )
        
        return best
```

### 5. Model Serialization & Version Control

```python
class ModelRegistry:
    """Store, version, and retrieve trained models"""
    
    def __init__(self, storage_path='./models'):
        self.storage_path = storage_path
        self.mlflow_client = mlflow.tracking.MlflowClient()
    
    def save_model(self, model, preprocessor, metadata):
        """Save model with MLflow"""
        
        mlflow.log_model(
            model,
            artifact_path='models'
        )
        
        mlflow.log_params(metadata['hyperparams'])
        mlflow.log_metrics(metadata['metrics'])
        mlflow.log_dict(metadata, 'metadata.json')
        
        model_uri = mlflow.get_artifact_uri()
        return model_uri
    
    def load_model(self, model_id):
        """Load model by ID"""
        return mlflow.pyfunc.load_model(f'runs:/{model_id}/models')
    
    def list_models(self, experiment_name='adversity_quotient'):
        """List all trained models"""
        exp = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        return runs
```

---

## Design Decisions

### 1. **Architecture & Framework Choices**

| Decision | Rationale |
|----------|-----------|
| **React for Frontend** | Component-based, large ecosystem, strong TypeScript support, excellent visualization libraries |
| **Node.js/Express + Python Flask** | Node for API server (async, fast), Python for ML (rich scientific libraries) |
| **PostgreSQL** | ACID compliance, JSON support, excellent with structured data, mature ecosystem |
| **Celery + Redis** | Robust async task queue, handles long-running ML jobs without blocking API |
| **Docker & K8s** | Easy deployment, scaling, and environment consistency |

### 2. **Data Handling Strategy**

**Decision: CSV/Excel import with in-memory processing**
- Rationale: Dataset expected to be <10GB; allows fast preprocessing
- Alternative considered: Streaming ingestion for real-time data (excluded from scope)

**Decision: Middleware preprocessing pipeline**
- Separate preprocessing from model training ensures:
  - Reproducibility (same preprocessing for all models)
  - Efficiency (single preprocessing, multiple model trains)
  - Auditability (track all transformations)

### 3. **Model Selection Strategy**

**Decision: Multiple regression models for comparison**
- Linear/Ridge/Lasso: Interpretability, baseline
- SVR: Non-linear kernel methods
- Tree-based: Handle interactions naturally
- Neural networks: Capture complex patterns
- Ensemble: Robustness and generalization

**Alternative considered:** Single best model
- Rejected because research requires understanding which approach works best for AQ prediction

### 4. **Evaluation Metrics**

**Primary Metrics:**
- **R² Score:** Explains variance, interpretable (0-1)
- **RMSE:** Penalizes large errors, same units as target
- **MAE:** Robust to outliers, interpretable

**Supporting Metrics:**
- **Cross-validation scores:** Generalization assessment
- **Prediction intervals:** Uncertainty quantification
- **Residual analysis:** Assumption validation

### 5. **Preprocessing Decisions**

| Step | Decision | Rationale |
|------|----------|-----------|
| Missing Values | Median/Mean imputation | Simple, preserves relationships |
| Outliers | IQR method with threshold=1.5 | Robust, interpretable |
| Scaling | StandardScaler | Required for distance-based/tree models, neural networks |
| Feature Engineering | Polynomial features (degree 2) | Capture non-linearities |
| Feature Selection | Manual selection in UI + correlation analysis | Domain knowledge + data-driven |

### 6. **Security & Privacy**

**Decision: Role-based access control (RBAC)**
```
User Roles:
- Viewer: Can view dashboards, results
- Analyst: Can upload data, run models
- Administrator: System management, user management
- Researcher: Full access, export capabilities
```

**Data Protection:**
- PII handling: Anonymization before upload
- Password hashing: bcrypt with salt rounds=12
- API authentication: JWT tokens (15min access, 7d refresh)
- Audit logging: All user actions logged with timestamp
- Data encryption: At rest (AES-256) and in transit (TLS 1.3)

### 7. **Performance Optimization**

| Component | Strategy |
|-----------|----------|
| **Frontend** | Code splitting, lazy loading, memoization, virtual scrolling for large tables |
| **Backend** | Caching (Redis), database indexing, connection pooling, async processing |
| **ML Pipeline** | Parallel training (n_jobs=-1), GPU acceleration (if available), model serialization |

### 8. **Monitoring & Logging**

```python
# Structured logging
logger.info("Model training started", extra={
    'model_type': 'random_forest',
    'dataset_id': 'ds_123',
    'user_id': 'user_456',
    'timestamp': datetime.now()
})

# Metrics tracking
prometheus_metrics = {
    'model_training_duration': Histogram(...),
    'api_request_latency': Histogram(...),
    'data_upload_size': Gauge(...),
    'active_predictions': Gauge(...)
}
```

---

## Implementation Plan

### Phase 1: Foundation (Weeks 1-3)

**Week 1: Project Setup & Infrastructure**
- [ ] Repository setup (Git) with branching strategy
- [ ] Docker setup (Docker Compose for local development)
- [ ] Database schema design and migration scripts
- [ ] Backend project structure (Express.js/Flask)
- [ ] Frontend project setup (React + Redux)
- **Deliverable:** Development environment ready, team can start coding

**Week 2: Authentication & User Management**
- [ ] User registration/login endpoints
- [ ] JWT token implementation
- [ ] Password hashing and validation
- [ ] Email verification system
- [ ] Role-based access control middleware
- [ ] Frontend login/registration UI
- **Deliverable:** Secure authentication system

**Week 3: Database & Data Models**
- [ ] PostgreSQL schema creation
- [ ] ORM models (Sequelize/SQLAlchemy)
- [ ] Migration management
- [ ] Indexing strategy
- [ ] Initial seed data
- **Deliverable:** Database ready for data storage

### Phase 2: Core Features (Weeks 4-8)

**Week 4: Data Upload & Preview**
- [ ] File upload endpoint (multipart form)
- [ ] CSV/Excel parsing with validation
- [ ] Data preview API
- [ ] Data quality assessment
- [ ] React upload component with drag-drop
- [ ] Data preview table component
- **Deliverable:** Data upload workflow functional

**Week 5: Data Preprocessing Pipeline**
- [ ] Implement preprocessor class
- [ ] Missing value imputation
- [ ] Outlier detection and handling
- [ ] Feature scaling and normalization
- [ ] Preprocessing status tracking
- [ ] Preprocessing history/audit trail
- **Deliverable:** Robust data preprocessing

**Week 6-7: ML Models & Training**
- [ ] Implement 7+ regression models
- [ ] Training pipeline with cross-validation
- [ ] Metrics calculation
- [ ] Model serialization
- [ ] Celery job queue setup
- [ ] Real-time progress tracking (WebSocket)
- [ ] Model storage in S3
- **Deliverable:** End-to-end model training

**Week 8: Model Comparison & Analysis**
- [ ] Comparison service
- [ ] Ranking algorithms
- [ ] Statistical tests (paired t-test)
- [ ] Results dashboard component
- [ ] Visualization components (charts)
- [ ] Export functionality
- **Deliverable:** Model comparison workflows

### Phase 3: Advanced Features (Weeks 9-11)

**Week 9: Feature Importance & Interpretability**
- [ ] Feature importance extraction (all model types)
- [ ] SHAP values computation
- [ ] Correlation analysis
- [ ] Feature interaction plots
- [ ] React visualization components
- **Deliverable:** Interpretability features

**Week 10: Single Prediction & Recommendations**
- [ ] Prediction endpoint (best model)
- [ ] Confidence interval calculation
- [ ] Input validation form
- [ ] Result display component
- [ ] Recommendation engine (based on predictions)
- [ ] History tracking
- **Deliverable:** Prediction workflow

**Week 11: Reporting & Export**
- [ ] Report generation (PDF, CSV, Excel)
- [ ] Report templates (executive, technical, educator)
- [ ] Template rendering
- [ ] Export functionality
- [ ] React report generator component
- **Deliverable:** Complete reporting system

### Phase 4: Polish & Deployment (Weeks 12-14)

**Week 12: Testing & Quality Assurance**
- [ ] Unit tests (frontend & backend)
- [ ] Integration tests
- [ ] E2E tests (Selenium/Cypress)
- [ ] Load testing (k6, JMeter)
- [ ] Security testing (OWASP)
- [ ] Performance profiling
- **Deliverable:** Test coverage >80%, no critical bugs

**Week 13: Documentation & Monitoring**
- [ ] API documentation (Swagger/OpenAPI)
- [ ] User documentation
- [ ] Developer guide
- [ ] Deployment guide
- [ ] Monitoring setup (Prometheus, Grafana)
- [ ] Alerting rules
- [ ] Log aggregation (ELK stack)
- **Deliverable:** Complete documentation

**Week 14: Deployment & Launch**
- [ ] Production environment setup
- [ ] Data backup strategy
- [ ] Disaster recovery plan
- [ ] Load balancing setup
- [ ] CDN for static assets
- [ ] SSL certificate setup
- [ ] Beta testing with users
- [ ] Launch & monitoring
- **Deliverable:** Production system live

### Development Workflow

```
Main Branch (Production)
├── Develop Branch (Staging)
│   ├── Feature Branch 1 (feature/auth)
│   ├── Feature Branch 2 (feature/data-upload)
│   └── Feature Branch 3 (feature/models)
└── Hotfix Branch (hotfix/bug-fix)

PR Requirements:
- 2 approvals minimum
- CI/CD pipeline pass
- Test coverage >80%
- No merge conflicts
```

### Team Structure (Recommended)

| Role | Count | Responsibilities |
|------|-------|-----------------|
| Backend Engineer | 2 | API, database, ML integration |
| Frontend Engineer | 2 | React components, UI/UX |
| ML Engineer | 2 | Model implementation, optimization |
| DevOps Engineer | 1 | Infrastructure, deployment, monitoring |
| QA Engineer | 1 | Testing, bug reporting |
| Tech Lead | 1 | Architecture, code review, mentoring |

---

## Metrics & Evaluation Strategy

### 1. Regression Performance Metrics

#### Primary Metrics

```python
class MetricsCalculator:
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """
        R² Score (Coefficient of Determination)
        - Range: -∞ to 1 (higher is better)
        - 1.0: Perfect prediction
        - 0.0: Model explains no variance
        - <0: Worse than horizontal line
        
        Formula: R² = 1 - (SS_res / SS_tot)
        Where SS_res = Σ(y_true - y_pred)²
              SS_tot = Σ(y_true - mean(y_true))²
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def rmse(y_true, y_pred):
        """
        Root Mean Squared Error
        - Range: 0 to ∞ (lower is better)
        - Penalizes large errors heavily
        - Same units as target variable
        
        Formula: RMSE = √(1/n * Σ(y_true - y_pred)²)
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true, y_pred):
        """
        Mean Absolute Error
        - Range: 0 to ∞ (lower is better)
        - Robust to outliers
        - Same units as target variable
        
        Formula: MAE = (1/n) * Σ|y_true - y_pred|
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true, y_pred):
        """
        Mean Absolute Percentage Error
        - Range: 0 to ∞ (lower is better)
        - Scale-independent
        - Percentage error interpretation
        
        Formula: MAPE = (1/n) * Σ|y_true - y_pred|/|y_true| * 100
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def median_ae(y_true, y_pred):
        """
        Median Absolute Error
        - Robust to outliers
        - Less sensitive to distribution
        """
        return np.median(np.abs(y_true - y_pred))
    
    @staticmethod
    def explained_variance(y_true, y_pred):
        """
        Explained Variance Score
        - Similar to R² but uses variance instead of SS
        - Range: -∞ to 1 (higher is better)
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
```

#### Cross-Validation Strategy

```python
def cross_validate_model(model, X, y, cv_folds=5, return_train=True):
    """
    Stratified K-Fold Cross-Validation
    
    Benefits:
    - Estimates generalization error
    - Detects overfitting (train vs test score gap)
    - More reliable than single train-test split
    """
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    results = {
        'train_r2': [],
        'test_r2': [],
        'train_rmse': [],
        'test_rmse': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        
        # Training metrics
        y_train_pred = model.predict(X_train)
        results['train_r2'].append(r2_score(y_train, y_train_pred))
        results['train_rmse'].append(rmse(y_train, y_train_pred))
        
        # Test metrics
        y_test_pred = model.predict(X_test)
        results['test_r2'].append(r2_score(y_test, y_test_pred))
        results['test_rmse'].append(rmse(y_test, y_test_pred))
    
    # Calculate statistics
    return {
        'train_r2_mean': np.mean(results['train_r2']),
        'test_r2_mean': np.mean(results['test_r2']),
        'test_r2_std': np.std(results['test_r2']),
        'overfitting_gap': np.mean(results['train_r2']) - np.mean(results['test_r2']),
        'fold_scores': results
    }
```

### 2. Model Comparison Metrics

```python
class ModelComparator:
    
    @staticmethod
    def rank_models(model_metrics):
        """
        Composite ranking across multiple metrics
        
        Scoring:
        - R² (weight=0.4): Higher is better
        - RMSE (weight=0.3): Lower is better
        - MAE (weight=0.2): Lower is better
        - CV Stability (weight=0.1): Lower std is better
        """
        
        scores = {}
        
        for model_name, metrics in model_metrics.items():
            # Normalize scores to 0-1
            r2_norm = metrics['r2']  # Already 0-1
            rmse_norm = 1 / (1 + metrics['rmse'])  # Lower is better
            mae_norm = 1 / (1 + metrics['mae'])
            cv_stability = 1 / (1 + metrics['cv_std'])
            
            # Weighted composite score
            composite = (
                0.4 * r2_norm +
                0.3 * rmse_norm +
                0.2 * mae_norm +
                0.1 * cv_stability
            )
            
            scores[model_name] = {
                'composite_score': composite,
                'rank': None
            }
        
        # Assign ranks
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )
        
        for rank, (model_name, score_data) in enumerate(ranked, 1):
            scores[model_name]['rank'] = rank
        
        return scores
    
    @staticmethod
    def statistical_significance_test(model1_preds, model2_preds, y_true):
        """
        Paired t-test to determine if one model is significantly
        better than another
        
        H0: Both models have same performance
        H1: Models have different performance
        
        If p-value < 0.05: Reject H0 (significant difference)
        """
        
        model1_errors = np.abs(y_true - model1_preds)
        model2_errors = np.abs(y_true - model2_preds)
        
        t_stat, p_value = ttest_rel(model1_errors, model2_errors)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': (np.mean(model1_errors) - np.mean(model2_errors)) / np.std(model1_errors - model2_errors)
        }
```

### 3. Residual Analysis

```python
class ResidualAnalyzer:
    
    @staticmethod
    def normality_test(residuals):
        """
        Shapiro-Wilk test for normality
        - Assumes residuals are normally distributed
        - p-value > 0.05: Residuals are normal
        """
        statistic, p_value = shapiro(residuals)
        return {'statistic': statistic, 'p_value': p_value, 'normal': p_value > 0.05}
    
    @staticmethod
    def heteroscedasticity_test(y_pred, residuals):
        """
        Breusch-Pagan test for constant variance
        - H0: Residuals have constant variance (homoscedastic)
        - p-value > 0.05: Homoscedastic
        """
        statistic, p_value = het_breuschpagan(residuals, y_pred)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'homoscedastic': p_value > 0.05
        }
    
    @staticmethod
    def autocorrelation_test(residuals):
        """
        Durbin-Watson test for autocorrelation
        - Range: 0-4 (2 = no autocorrelation)
        - <2: Positive autocorrelation
        - >2: Negative autocorrelation
        """
        dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        return {
            'durbin_watson': dw,
            'autocorrelated': dw < 1.5 or dw > 2.5
        }
```

### 4. Prediction Interval Calculation

```python
class PredictionIntervalCalculator:
    
    @staticmethod
    def calculate_pi(model, X_train, y_train, X_test, y_test, confidence=0.95):
        """
        Calculate prediction intervals using residual standard error
        
        For new prediction x:
        ŷ = model.predict(x)
        PI = [ŷ - z*σ_residual, ŷ + z*σ_residual]
        
        Where:
        - z = quantile of normal distribution (1.96 for 95%)
        - σ_residual = standard deviation of training residuals
        """
        
        # Training residuals
        y_train_pred = model.predict(X_train)
        train_residuals = y_train - y_train_pred
        sigma = np.std(train_residuals)
        
        # Test predictions
        y_test_pred = model.predict(X_test)
        
        # z-score
        z_score = norm.ppf((1 + confidence) / 2)
        margin = z_score * sigma
        
        # Intervals
        lower_bound = y_test_pred - margin
        upper_bound = y_test_pred + margin
        
        return {
            'predictions': y_test_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'intervals': np.array([lower_bound, upper_bound]).T,
            'coverage': np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
        }
```

### 5. Business Metrics

```python
class BusinessMetrics:
    
    @staticmethod
    def prediction_accuracy_by_range(y_true, y_pred, tolerance=0.5):
        """
        What percentage of predictions fall within X points of actual?
        
        Important for AQ prediction:
        - ±0.5: High accuracy
        - ±1.0: Acceptable accuracy
        - ±2.0: Useful guidance
        """
        within_range = np.abs(y_true - y_pred) <= tolerance
        accuracy = np.sum(within_range) / len(y_true) * 100
        return accuracy
    
    @staticmethod
    def student_risk_classification(y_true, y_pred, low_aq_threshold=40, high_aq_threshold=70):
        """
        Classify students as Low/Medium/High AQ and check if model
        classifies correctly
        
        Important for educational interventions
        """
        
        # Actual classifications
        actual_low = y_true < low_aq_threshold
        actual_high = y_true > high_aq_threshold
        
        # Predicted classifications
        pred_low = y_pred < low_aq_threshold
        pred_high = y_pred > high_aq_threshold
        
        # Misclassifications (false negatives critical!)
        false_negatives = actual_low & ~pred_low  # Predicted high AQ but actually low
        false_positives = actual_high & ~pred_high  # Predicted low AQ but actually high
        
        return {
            'false_negative_rate': np.sum(false_negatives) / np.sum(actual_low) if np.sum(actual_low) > 0 else 0,
            'false_positive_rate': np.sum(false_positives) / np.sum(actual_high) if np.sum(actual_high) > 0 else 0,
            'correctly_identified_low': np.sum(actual_low & pred_low) / np.sum(actual_low) if np.sum(actual_low) > 0 else 0
        }
```

### Metrics Dashboard Display

**Displayed in Results View:**

```
┌─────────────────────────────────────────┐
│ MODEL PERFORMANCE SUMMARY                │
├─────────────────────────────────────────┤
│ Model: Random Forest Regressor           │
│ ├─ R² Score: 0.87 ✓ (Excellent)        │
│ ├─ RMSE: 3.2 AQ points                 │
│ ├─ MAE: 2.5 AQ points                  │
│ └─ MAPE: 4.1%                          │
├─────────────────────────────────────────┤
│ CROSS-VALIDATION (5-Fold)               │
│ ├─ CV Score (Mean): 0.85 ± 0.02        │
│ └─ Overfitting Gap: 0.02 (✓ Low)       │
├─────────────────────────────────────────┤
│ RESIDUAL ANALYSIS                       │
│ ├─ Normality: Passed (p=0.23)          │
│ ├─ Homoscedasticity: Passed (p=0.18)  │
│ └─ Autocorrelation: DW=2.1 ✓            │
├─────────────────────────────────────────┤
│ PREDICTION QUALITY                      │
│ ├─ Within ±0.5: 78% of predictions     │
│ ├─ Within ±1.0: 92% of predictions     │
│ └─ 95% PI Width: [±6.3 AQ points]      │
└─────────────────────────────────────────┘
```

---

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role ENUM('viewer', 'analyst', 'admin', 'researcher') DEFAULT 'analyst',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
```

### Datasets Table
```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    file_path VARCHAR(512),
    s3_key VARCHAR(512),
    format ENUM('csv', 'xlsx', 'json') NOT NULL,
    row_count INTEGER,
    column_count INTEGER,
    file_size BIGINT,
    preprocessed BOOLEAN DEFAULT FALSE,
    preprocessing_status ENUM('pending', 'processing', 'completed', 'failed'),
    quality_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_datasets_user_id ON datasets(user_id);
CREATE INDEX idx_datasets_status ON datasets(preprocessing_status);
```

### Features Table
```sql
CREATE TABLE features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    data_type ENUM('numeric', 'categorical', 'text') NOT NULL,
    missing_count INTEGER,
    unique_count INTEGER,
    mean_value FLOAT,
    std_value FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    correlation_with_target FLOAT,
    is_selected BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_features_dataset_id ON features(dataset_id);
```

### Models Table
```sql
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    model_type ENUM('linear', 'ridge', 'lasso', 'svr', 'rf', 'gb', 'xgb', 'lgb', 'nn') NOT NULL,
    hyperparameters JSONB,
    model_path VARCHAR(512),
    s3_key VARCHAR(512),
    training_status ENUM('pending', 'training', 'completed', 'failed') DEFAULT 'pending',
    training_error TEXT,
    cv_folds INTEGER DEFAULT 5,
    test_size FLOAT DEFAULT 0.2,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    training_duration_seconds INTEGER
);

CREATE INDEX idx_models_user_id ON models(user_id);
CREATE INDEX idx_models_dataset_id ON models(dataset_id);
CREATE INDEX idx_models_status ON models(training_status);
```

### Model Metrics Table
```sql
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    r2_score FLOAT NOT NULL,
    rmse FLOAT NOT NULL,
    mae FLOAT NOT NULL,
    mape FLOAT,
    median_ae FLOAT,
    cv_mean FLOAT,
    cv_std FLOAT,
    cv_scores FLOAT[],
    prediction_interval FLOAT,
    durbin_watson FLOAT,
    residual_std FLOAT,
    test_set_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_model_id ON model_metrics(model_id);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(id),
    user_id UUID NOT NULL REFERENCES users(id),
    input_features JSONB NOT NULL,
    predicted_value FLOAT NOT NULL,
    confidence_lower FLOAT,
    confidence_upper FLOAT,
    confidence_level FLOAT DEFAULT 0.95,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_model_id ON predictions(model_id);
CREATE INDEX idx_predictions_user_id ON predictions(user_id);
```

### Reports Table
```sql
CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    model_id UUID NOT NULL REFERENCES models(id),
    report_type ENUM('executive', 'technical', 'educator', 'research') NOT NULL,
    format ENUM('pdf', 'csv', 'xlsx', 'html') NOT NULL,
    file_path VARCHAR(512),
    s3_key VARCHAR(512),
    file_size BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reports_user_id ON reports(user_id);
```

### Audit Log Table
```sql
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(255),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_created_at ON audit_logs(created_at);
```

---

## API Specifications

### Authentication Endpoints

```
POST /api/auth/register
Body: { email, password, fullName }
Response: { userId, verificationToken }

POST /api/auth/login
Body: { email, password }
Response: { accessToken, refreshToken, user }

POST /api/auth/refresh
Body: { refreshToken }
Response: { accessToken }

POST /api/auth/logout
Headers: Authorization: Bearer {token}
Response: { success }

GET /api/auth/me
Headers: Authorization: Bearer {token}
Response: { user }
```

### Data Management Endpoints

```
POST /api/data/upload
Headers: Authorization: Bearer {token}, Content-Type: multipart/form-data
Body: { file }
Response: { datasetId, uploadPath, rowCount, columnCount }

GET /api/data
Headers: Authorization: Bearer {token}
Response: { datasets: [...] }

GET /api/data/:id
Headers: Authorization: Bearer {token}
Response: { dataset, features, statistics }

POST /api/data/:id/preprocess
Headers: Authorization: Bearer {token}
Body: { strategy: 'auto' | 'manual', options: {} }
Response: { jobId, estimatedTime }

GET /api/data/:id/preprocess-status/:jobId
Headers: Authorization: Bearer {token}
Response: { progress, status, results }

GET /api/data/:id/quality-report
Headers: Authorization: Bearer {token}
Response: { qualityScore, missingValues, duplicates, outliers }
```

### Model Training Endpoints

```
POST /api/models/train
Headers: Authorization: Bearer {token}
Body: {
    datasetId,
    modelType: 'linear' | 'ridge' | 'lasso' | 'svr' | 'rf' | 'gb' | 'xgb' | 'lgb' | 'nn',
    hyperparameters: { ... },
    selectedFeatures: [...],
    cvFolds: 5,
    testSize: 0.2
}
Response: { modelId, jobId, estimatedTime }

GET /api/models/:id/training-status
Headers: Authorization: Bearer {token}
Response: { status, progress, currentFold, eta, logs }

GET /api/models
Headers: Authorization: Bearer {token}
Query: { datasetId?, status?, sortBy?, limit, offset }
Response: { models: [...], total, page }

GET /api/models/:id
Headers: Authorization: Bearer {token}
Response: { model, metrics, metadata }

GET /api/models/:id/metrics
Headers: Authorization: Bearer {token}
Response: { metrics, predictions, residuals }

GET /api/models/:id/feature-importance
Headers: Authorization: Bearer {token}
Response: { importance: [{feature, importance}], shap: {...} }

POST /api/models/:id/predict
Headers: Authorization: Bearer {token}
Body: { features: { ...input } }
Response: { prediction, confidenceInterval, explanation }

POST /api/models/:id/predict-batch
Headers: Authorization: Bearer {token}
Body: { predictions: [{...}, {...}] }
Response: { predictions: [...] }
```

### Comparison Endpoints

```
GET /api/comparison
Headers: Authorization: Bearer {token}
Query: { datasetId, modelIds: [...] }
Response: { comparison, rankings, significance }

POST /api/comparison/export
Headers: Authorization: Bearer {token}
Body: { modelIds, format: 'pdf' | 'csv' | 'xlsx' }
Response: { fileUrl, expiresIn }
```

### Report Endpoints

```
POST /api/reports
Headers: Authorization: Bearer {token}
Body: { modelId, reportType: 'executive' | 'technical' | 'educator' | 'research', format: 'pdf' | 'csv' | 'xlsx' }
Response: { reportId, fileUrl }

GET /api/reports/:id/download
Headers: Authorization: Bearer {token}
Response: File (PDF/CSV/XLSX)
```

---

## Risk Management

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Model training timeout | High | Medium | Implement async jobs, set max_iter limits |
| Memory overflow on large datasets | High | Low | Implement data streaming, batch processing |
| Slow prediction API | Medium | Medium | Caching, model optimization, load balancing |
| Security breach (user data) | Critical | Low | Encryption, SSL, regular security audits |
| Database failure | Critical | Low | Automated backups, replication, disaster recovery |

### Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Inadequate user adoption | High | Medium | User training, clear documentation, good UX |
| Data quality issues | Medium | High | Validation, quality reports, preprocessing |
| Model overfitting | Medium | Medium | Cross-validation, regularization, monitoring |
| Scalability issues | Medium | Low | Horizontal scaling, caching, optimization |

### Mitigation Strategies

1. **Testing**: Comprehensive test suite (unit, integration, E2E)
2. **Monitoring**: Real-time alerts for system health
3. **Documentation**: Clear setup and usage guides
4. **Backup & Recovery**: Daily automated backups, 4-hour RTO
5. **Version Control**: Git with code review process
6. **User Support**: Help desk and documentation portal

---

## Timeline & Milestones

```
PROJECT TIMELINE (14 Weeks)

Week 1-3: Foundation (Infrastructure & Auth)
├─ Day 1-5: Setup & Planning
├─ Day 6-10: Dev Environment & Deployment Pipeline
└─ Day 11-15: Auth Implementation & Testing
   MILESTONE 1: ✓ Development environment ready

Week 4-5: Core Data Features
├─ Day 16-20: Data Upload & Preview
├─ Day 21-25: Preprocessing Pipeline
└─ Day 26-30: Data Quality & Statistics
   MILESTONE 2: ✓ Data management module functional

Week 6-8: ML Training System
├─ Day 31-35: Model Implementations
├─ Day 36-40: Training Pipeline & Celery
├─ Day 41-45: Cross-Validation & Metrics
└─ Day 46-50: WebSocket Real-time Updates
   MILESTONE 3: ✓ End-to-end model training

Week 9-10: Analysis & Interpretability
├─ Day 51-55: Model Comparison & Ranking
├─ Day 56-60: Feature Importance & SHAP
└─ Day 61-65: Results Dashboard
   MILESTONE 4: ✓ Model analysis complete

Week 11: Advanced Features
├─ Day 66-70: Prediction & Recommendations
└─ Day 71-75: Reporting & Export
   MILESTONE 5: ✓ User-facing features complete

Week 12-14: Testing & Launch
├─ Day 76-80: Testing & QA
├─ Day 81-85: Documentation & Monitoring
└─ Day 86-98: Deployment & Launch
   MILESTONE 6: ✓ Production launch

LAUNCH DATE: End of Week 14
```

---

## Success Criteria

### Functional Requirements
- ✓ All 7+ regression models implemented and trainable
- ✓ Model comparison with statistical tests
- ✓ Feature importance visualization
- ✓ Prediction interface with confidence intervals
- ✓ Report generation and export
- ✓ User authentication and role-based access

### Performance Requirements
- ✓ Data upload: <30 seconds for 10,000 rows
- ✓ Model training: <5 minutes for typical dataset
- ✓ Single prediction: <2 seconds
- ✓ Dashboard load: <3 seconds
- ✓ API response time: <500ms (p99)

### Quality Requirements
- ✓ Test coverage: >80%
- ✓ No critical security vulnerabilities
- ✓ Model R² score: >0.75 for benchmark dataset
- ✓ Zero data loss incidents
- ✓ 99.5% uptime SLA

### User Requirements
- ✓ Intuitive UI (Net Promoter Score >40)
- ✓ Clear documentation
- ✓ Responsive design (mobile + desktop)
- ✓ Accessible (WCAG 2.1 AA compliance)

---

## Conclusion

This comprehensive project documentation provides a complete roadmap for developing an Adversity Quotient analysis platform with multi-model regression comparison. The system is designed to be:

- **Scalable**: Handle growing datasets and users
- **Robust**: Comprehensive error handling and monitoring
- **Interpretable**: Clear visualization of model decisions
- **User-friendly**: Intuitive interface for researchers and educators
- **Research-ready**: Rigorous evaluation metrics and statistical testing

The 14-week timeline, detailed architecture, and clear milestones ensure successful delivery of a production-ready system that enables meaningful insights into adolescent resilience patterns.

---

