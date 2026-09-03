import { useMemo, useState, useEffect, useCallback } from 'react';
import LoadingSpinner from './LoadingSpinner';
import { getFeatureImportance } from '../api/service';
import '../styles/FeatureImportance.css';

const QUESTION_LABELS = {
  Q1: 'Influence on academic outcomes (Control)',
  Q2: 'Recovery from disappointment (Ownership)',
  Q3: 'Motivation without visible results (Reach)',
  Q4: "Failures don't define ability (Endurance)",
  Q5: 'Control under pressure (Control)',
  Q6: 'Learning from mistakes (Ownership)',
  Q7: 'Problems are temporary (Reach)',
  Q8: 'Confidence across subjects (Endurance)',
  Q9: 'Taking responsibility (Control)',
  Q10: 'Finding ways to overcome (Endurance)',
};

const INSIGHTS = {
  Q1: 'Believing in your ability to influence outcomes is the foundation of personal control.',
  Q2: 'Quick recovery from setbacks signals strong emotional resilience.',
  Q3: 'Sustained motivation without immediate feedback shows deep intrinsic drive.',
  Q4: 'Separating identity from failure enables a growth mindset.',
  Q5: 'Reflective learning from mistakes strongly influences AQ prediction.',
  Q6: 'Maintaining control under pressure demonstrates mature coping strategies.',
  Q7: 'Viewing problems as temporary prevents learned helplessness.',
  Q8: 'Compartmentalizing setbacks prevents negative spiraling across domains.',
  Q9: 'Taking ownership of improvement drives adaptive behavior.',
  Q10: 'Self-efficacy in overcoming challenges is a core resilience indicator.',
};


function SourceBadge({ sourceType }) {
  const isShap = sourceType === 'shap';
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 9px',
      background: isShap
        ? 'linear-gradient(135deg, #10b981, #059669)'
        : 'linear-gradient(135deg, #f59e0b, #d97706)',
      color: '#fff',
      borderRadius: '10px',
      fontSize: '0.7rem',
      fontWeight: 700,
      letterSpacing: '0.04em',
      verticalAlign: 'middle',
      marginLeft: '6px',
    }}>
      {isShap ? '✦ True SHAP' : '⚠ Gini (fallback)'}
    </span>
  );
}

function FeatureImportance() {
  const [data, setData]           = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError]         = useState(null);
  const [selectedModel, setSelectedModel] = useState(''); 

  const fetchData = useCallback(async (modelName) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await getFeatureImportance(modelName || null);
      setData(result);
      
      
      if (!modelName && result?.source_model) {
        setSelectedModel(result.source_model);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);


  useEffect(() => { fetchData(''); }, [fetchData]);

  const features = data?.features ?? [];

  const sortedFeatures = useMemo(() =>
    [...features].sort((a, b) => (b.importance || 0) - (a.importance || 0)),
  [features]);

  const maxImportance = useMemo(() =>
    Math.max(...sortedFeatures.map(f => f.importance || 0), 0.001),
  [sortedFeatures]);


  const handleModelChange = (e) => {
    const chosen = e.target.value;
    setSelectedModel(chosen);
    fetchData(chosen);
  };


  if (isLoading) {
    return <LoadingSpinner message="Computing feature importance..." />;
  }
  if (error) {
    return (
      <div className="feature-importance">
        <div className="section-header">
          <h2>Global Feature Importance</h2>
        </div>
        <div className="no-data-message">
          <div className="empty-state-icon">⚠️</div>
          <h3>Could not load data</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }
  if (!data || !data.features || data.features.length === 0) {
    return (
      <div className="feature-importance">
        <div className="section-header">
          <h2>Global Feature Importance</h2>
          <p>Understanding which factors most influence AQ prediction</p>
        </div>
        <div className="no-data-message">
          <div className="empty-state-icon">🔍</div>
          <h3>No Feature Importance Data Available</h3>
          <p>Start the backend server to load feature importance data.</p>
        </div>
      </div>
    );
  }

  const { source_model, source_type, available_models = [] } = data;

  const formatImportance = (val) => {
    if (val === 0) return '0.0000';
    if (val < 0.0001) return val.toExponential(2);
    if (val < 0.01)   return val.toFixed(6);
    return val.toFixed(4);
  };

  return (
    <div className="feature-importance">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <div className="section-header">
        <h2>Global Feature Importance</h2>
        <p>
          Mean |SHAP| across all {source_type === 'shap' ? 'training samples' : 'tree splits'} —
          which questions drive AQ predictions the most
          
        </p>
      </div>

    
      {available_models.length > 1 && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          marginBottom: '20px',
          padding: '14px 18px',
          background: 'var(--card-bg, #e5eaf0)',
          borderRadius: '12px',
          border: '1px solid var(--border-color, rgba(190, 17, 17, 0.08))',
        }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
           View model:
          </span>
          <select
            value={selectedModel}
            onChange={handleModelChange}
            style={{
              flex: 1,
              padding: '8px 12px',
              borderRadius: '8px',
              border: '1px solid var(--border-color, rgba(255,255,255,0.15))',
              background: 'var(--input-bg, #ffffff)',
              color: 'var(--text-primary, #050505)',
              fontSize: '0.9rem',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            {available_models.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          {source_model && (
            <span style={{
              fontSize: '0.78rem',
              color: 'var(--text-secondary)',
              whiteSpace: 'nowrap',
            }}>
              {source_type === 'shap' ? 'True mean |SHAP|' : 'Gini importance'} from
              <strong style={{ marginLeft: '4px' }}>{source_model}</strong>
            </span>
          )}
        </div>
      )}

      {/* ── Chart ───────────────────────────────────────────────────── */}
      <div className="importance-chart card">
        <h3>
          Feature Rankings — Training Data
          <span style={{
            fontSize: '0.78rem',
            fontWeight: 400,
            color: 'var(--text-secondary)',
            marginLeft: '8px',
          }}>
            ({source_model})
          </span>
        </h3>

        <div className="feature-rows">
          {sortedFeatures.map((feature, idx) => {
            const featureId  = feature.feature || feature.name || `Q${idx + 1}`;
            const label      = QUESTION_LABELS[featureId] || featureId;
            const importance = feature.importance || 0;
            const barWidth   = (importance / maxImportance) * 100;
            const rank       = feature.rank || idx + 1;
            const isTop3     = rank <= 3;

            let barClass = 'mid';
            if (rank <= 3)  barClass = 'top';
            else if (rank >= 8) barClass = 'low';

            return (
              <div key={featureId} className="feature-row">
                <div className="feature-row-header">
                  <span className={`rank-badge ${isTop3 ? 'top-3' : 'regular'}`}>
                    {rank}
                  </span>
                  <div className="feature-name">
                    <span className="feature-question-id">{featureId}</span>
                    <span className="feature-description"> — {label}</span>
                  </div>
                  <span className="feature-importance-value">
                    {formatImportance(importance)}
                  </span>
                </div>

                <div className="feature-bar-track">
                  <div
                    className={`feature-bar-fill ${barClass}`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>

                {isTop3 && INSIGHTS[featureId] && (
                  <div className="insight-card">
                    <strong> Insight:</strong> {INSIGHTS[featureId]}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="chart-legend">
          <div className="legend-item">
            <span className="legend-dot high" />
            High Impact
          </div>
          <div className="legend-item">
            <span className="legend-dot medium" />
            Medium Impact
          </div>
          <div className="legend-item">
            <span className="legend-dot low" />
            Low Impact
          </div>
        </div>
      </div>
    </div>
  );
}

export default FeatureImportance;
