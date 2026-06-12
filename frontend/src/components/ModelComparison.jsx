import React from 'react';
import LoadingSpinner from './LoadingSpinner';
import '../styles/ModelComparison.css';

const MODEL_COLORS = ['model-0', 'model-1', 'model-2', 'model-3', 'model-4'];

function ModelComparison({ data, isLoading }) {
  if (isLoading) {
    return <LoadingSpinner message="Loading model comparison data..." />;
  }

  // 1. Target the correct backend key: 'algorithms'
  let rawAlgorithms = [];
  if (data && Array.isArray(data.algorithms)) {
    rawAlgorithms = data.algorithms;
  }

  // 2. Fallback empty state if no array is found
  if (!rawAlgorithms || rawAlgorithms.length === 0) {
    return (
      <div className="model-comparison">
        <div className="section-header">
          <h2>Model Comparison</h2>
          <p>Compare performance metrics across different ML models</p>
        </div>
        <div className="no-data-message">
          <div className="empty-state-icon">📊</div>
          <h3>No Model Data Available</h3>
          <p>Start the backend server to load model comparison data.</p>
        </div>
      </div>
    );
  }

  // 3. Map backend keys ('model_name') safely to your layout variables
  const models = rawAlgorithms.map((m) => ({
    name: m.model_name || 'Unknown Model',
    accuracy: m.accuracy || 0,
    f1_score: m.f1_score || 0,
    auc_roc: m.auc_roc || 0,
    cv_mean: m.cv_mean || 0,
    cv_std: m.cv_std || 0,
  }));

  // 4. Extract global summary values directly from the top level response
  const best_model = data.best_model || 'Logistic Regression';
  const best_accuracy = data.best_accuracy || 0;

  return (
    <div className="model-comparison">
      <div className="section-header">
        <h2> Model Comparison</h2>
        <p>Performance metrics across different machine learning models</p>
      </div>

      {/* Best Model Card */}
      {best_model && (
        <div className="best-model-card">
          <span className="best-model-trophy">🏆</span>
          <div className="best-model-info">
            <h3>Best Performing Model</h3>
            <div className="best-model-name">{best_model}</div>
            {best_accuracy > 0 && (
              <p className="best-model-accuracy">
                Accuracy: <strong>{(best_accuracy * 100).toFixed(2)}%</strong>
              </p>
            )}
          </div>
        </div>
      )}

      {/* Desktop Table */}
      <div className="model-table-wrapper">
        <table className="model-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>F1 Score</th>
              <th>AUC-ROC</th>
              <th>CV Mean</th>
              <th>CV Std</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model, idx) => {
              const isBest = model.name === best_model;
              return (
                <tr key={idx} className={isBest ? 'best-row' : ''}>
                  <td>
                    <span className="model-name-cell">
                      {model.name}
                      {isBest && <span className="best-badge">Best</span>}
                    </span>
                  </td>
                  <td>{(model.accuracy * 100).toFixed(2)}%</td>
                  <td>{(model.f1_score * 100).toFixed(2)}%</td>
                  <td>{model.auc_roc.toFixed(2)}</td>
                  <td>{(model.cv_mean * 100).toFixed(2)}%</td>
                  <td>{model.cv_std.toFixed(4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Mobile Cards */}
      <div className="model-cards-mobile">
        {models.map((model, idx) => {
          const isBest = model.name === best_model;
          return (
            <div key={idx} className={`model-card-mobile${isBest ? ' best' : ''}`}>
              <div className="model-card-mobile-header">
                <span className="model-card-mobile-name">{model.name}</span>
                {isBest && <span className="best-badge">Best</span>}
              </div>
              <div className="model-card-mobile-metrics">
                <div className="metric-item">
                  <span className="metric-label">Accuracy</span>
                  <span className="metric-value">{(model.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">F1 Score</span>
                  <span className="metric-value">{(model.f1_score * 100).toFixed(2)}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">AUC-ROC</span>
                  <span className="metric-value">{model.auc_roc.toFixed(2)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">CV Mean</span>
                  <span className="metric-value">{(model.cv_mean * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Accuracy Bar Chart */}
      <div className="accuracy-chart">
        <h3>📊 Accuracy Comparison</h3>
        <div className="chart-bars">
          {models.map((model, idx) => {
            const accuracy = model.accuracy;
            const percent = (accuracy * 100).toFixed(1);
            return (
              <div key={idx} className="chart-bar-row">
                <span className="chart-bar-label">{model.name}</span>
                <div className="chart-bar-track">
                  <div
                    className={`chart-bar-fill ${MODEL_COLORS[idx % MODEL_COLORS.length]}`}
                    style={{ width: `${accuracy * 100}%` }}
                  ></div>
                </div>
                <span className="chart-bar-value">{percent}%</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default ModelComparison;
