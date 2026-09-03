import { useMemo } from 'react';
import RecommendationCard from './RecommendationCard';
import CoreDimensions from './CoreDimensions';
import LocalShapChart from './LocalShapChart';
import AcademicPerformanceProfile from './AcademicPerformanceProfile';
import '../styles/ResultsDashboard.css';

function ResultsDashboard({ results = {}, onReset }) {
  // Destructure with safe fallback defaults
  const {
    aq_score = 0,
    aq_category = 'Medium',
    confidence = 0,
    core_scores = {},
    weak_dimensions = [],
    behavioral_pattern = '',
    recommendations = [],
    local_shap = [],
    academic_profile = null,
  } = results;

  // 1. Sanitize top-level numbers safely to prevent NaN
  const numericScore = useMemo(() => {
    const parsed = Number(aq_score);
    return isNaN(parsed) ? 0 : parsed;
  }, [aq_score]);
  
  // Auto-detects if confidence is a decimal (0.70) or percentage (70)
  const numericConfidence = useMemo(() => {
    const raw = Number(confidence);
    if (isNaN(raw)) return 0;
    return raw <= 1 ? raw * 100 : raw;
  }, [confidence]);

  // Category styling normalized for CSS classes
  const categoryClass = useMemo(() => {
    const cat = String(aq_category || 'Medium').toLowerCase();
    if (cat.includes('high')) return 'high';
    if (cat.includes('low')) return 'low';
    return 'medium';
  }, [aq_category]);

  // Gauge color based on classification
  const gaugeColor = useMemo(() => {
    switch (categoryClass) {
      case 'high': return '#10b981'; // Green
      case 'low': return '#ef4444';  // Red
      default: return '#f59e0b';     // Orange/Yellow
    }
  }, [categoryClass]);

  
  const maxScore = useMemo(() => {
    return numericScore <= 5 ? 5 : 50;
  }, [numericScore]);

  
  const scorePercent = useMemo(() => {
    if (maxScore === 0) return 0;
    const percent = Math.round((numericScore / maxScore) * 100);
    return Math.max(0, Math.min(percent, 100));
  }, [numericScore, maxScore]);

  // Thread-safe CSS Conic Gradient string builder
  const gaugeStyle = useMemo(() => {
    const degrees = scorePercent * 3.6;
    return {
      background: `conic-gradient(
        ${gaugeColor} 0deg,
        ${gaugeColor} ${degrees}deg,
        rgba(0, 0, 0, 0.06) ${degrees}deg,
        rgba(0, 0, 0, 0.06) 360deg
      )`,
      mask: 'radial-gradient(farthest-side, transparent calc(100% - 12px), #fff calc(100% - 11px))',
      WebkitMask: 'radial-gradient(farthest-side, transparent calc(100% - 12px), #fff calc(100% - 11px))',
    };
  }, [gaugeColor, scorePercent]);

  // Download JSON payload utility
  const handleDownloadJSON = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aq-analysis-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // System Print utility
  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="results-dashboard">
      {/* AQ Overview */}
      <div className="aq-overview card-elevated">
        <div className="aq-gauge">
          <div 
            className="aq-gauge-ring" 
            style={{ 
              width: '100%', 
              height: '100%', 
              borderRadius: '50%',
              ...gaugeStyle 
            }}
          ></div>
          <div className="aq-gauge-inner">
            <div className="aq-score-value" style={{ color: gaugeColor }}>
              {numericScore.toFixed(2)}
            </div>
            <div className="aq-score-label">AQ Score</div>
          </div>
        </div>

        <div className="aq-details">
          <div>
            <h2>Your AQ Analysis</h2>
            <span className={`aq-category-badge ${categoryClass}`}>
              {categoryClass === 'high' && '🟢'}
              {categoryClass === 'medium' && '🟡'}
              {categoryClass === 'low' && '🔴'}
              {' '}{aq_category}
            </span>
          </div>

          <div className="confidence-section">
            <span className="confidence-label">
              Model Confidence: <span className="confidence-value">{numericConfidence.toFixed(1)}%</span>
            </span>
            <div className="confidence-bar-track">
              <div
                className="confidence-bar-fill"
                style={{ width: `${Math.min(numericConfidence, 100)}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Local SHAP — per-response explanation */}
      {Array.isArray(local_shap) && local_shap.length > 0 && (
        <LocalShapChart data={local_shap} predictedCategory={aq_category} />
      )}

      {/* CORE Dimensions Grid */}
      {core_scores && Object.keys(core_scores).length > 0 && (
        <CoreDimensions scores={core_scores} />
      )}

      {/* Academic Performance Profile */}
      {academic_profile && (
        <AcademicPerformanceProfile academic_profile={academic_profile} />
      )}

      {/* Weak Dimensions Summary Section */}
      {Array.isArray(weak_dimensions) && weak_dimensions.length > 0 && (
        <div className="weak-section">
          <div className="section-header">
            <h2>⚠️ Areas for Improvement</h2>
            <p>These dimensions need your attention for growth</p>
          </div>
          <div className="weak-cards">
            {weak_dimensions.map((dim, idx) => {
              
              const current = Number(dim?.score);
              const sanitizedCurrent = isNaN(current) ? 0 : current;
              
              const target = Number(dim?.target_score);
              const sanitizedTarget = isNaN(target) ? 0 : target;
              
               
              const gap = sanitizedTarget - sanitizedCurrent > 0 ? sanitizedTarget - sanitizedCurrent : 0;

              return (
                <div key={idx} className="weak-card" style={{ animationDelay: `${idx * 0.1}s` }}>
                  <div className="weak-card-header">
                    <span className="weak-dimension-name">{dim?.dimension || 'Unknown'}</span>
                    <span className={`severity-badge ${(dim?.severity || 'medium').toLowerCase()}`}>
                      {dim?.severity || 'Moderate'}
                    </span>
                  </div>
                  <div className="weak-scores">
                    <div className="weak-score-item">
                      <span className="weak-score-label">Current</span>
                      <span className="weak-score-value current">{sanitizedCurrent.toFixed(2)}</span>
                    </div>
                    <div className="weak-score-item">
                      <span className="weak-score-label">Target</span>
                      <span className="weak-score-value target">{sanitizedTarget.toFixed(2)}</span>
                    </div>
                    <div className="weak-score-item">
                      <span className="weak-score-label">Gap</span>
                      <span className="weak-score-value improvement">+{gap.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Behavioral Pattern Card */}
      {behavioral_pattern && (
        <div className="pattern-section">
          <div className="section-header">
            <h2> Behavioral Pattern</h2>
            <p>Your resilience and personality interpretation</p>
          </div>
          <div className="pattern-card">
            <p className="pattern-text">{behavioral_pattern}</p>
          </div>
        </div>
      )}

      {/* Actionable Recommendations Grid */}
      {Array.isArray(recommendations) && recommendations.length > 0 && (
        <div className="recommendations-section">
          <div className="section-header">
            <h2>Personalized Recommendations</h2>
            <p>Actionable steps to improve your Adversity Quotient</p>
          </div>
          <div className="recommendations-grid">
            {recommendations.map((rec, idx) => (
              <RecommendationCard
                key={rec?.id || idx}
                recommendation={rec}
              />
            ))}
          </div>
        </div>
      )}

      {/* Actionable Export Triggers */}
      <div className="export-section">
        <button className="btn btn-secondary" onClick={handlePrint}>
          Print Results
        </button>
        <button className="btn btn-secondary" onClick={handleDownloadJSON}>
          Download JSON
        </button>
      </div>

      
      <div className="reset-section">
        <button className="btn btn-primary btn-lg" onClick={onReset}>
          🔄 Take Another Assessment
        </button>
      </div>
    </div>
  );
}

export default ResultsDashboard;
