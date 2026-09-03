import { useMemo } from 'react';
import '../styles/LocalShapChart.css';

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

function LocalShapChart({ data, predictedCategory }) {
  const maxAbs = useMemo(
    () => Math.max(...(data || []).map(d => Math.abs(d.shap_value)), 0.001),
    [data]
  );

  if (!data || data.length === 0) return null;

  const catClass = (predictedCategory || 'medium').toLowerCase();

  return (
    <div className="local-shap-card">
      {/* ── Header ── */}
      <div className="local-shap-header">
        <div className="local-shap-title-row">
          <span className="local-shap-icon"></span>
          <h3 className="local-shap-title">Your Personal SHAP Explanation</h3>
        </div>
        <p className="local-shap-subtitle">
          How each of your answers influenced the&nbsp;
          <span className={`shap-cat-badge ${catClass}`}>{predictedCategory} AQ</span>
          &nbsp;prediction
        </p>
      </div>

      {/* ── Legend ── */}
      <div className="shap-legend">
        <span className="shap-legend-item">
          <span className="shap-dot pos" />
          Pushes <strong>toward</strong> {predictedCategory}
        </span>
        <span className="shap-legend-item">
          <span className="shap-dot neg" />
          Pushes <strong>away from</strong> {predictedCategory}
        </span>
      </div>

      {/* ── Axis labels ── */}
      <div className="shap-axis-row">
        <span className="shap-axis-label left">← Away</span>
        <span className="shap-axis-label center">0</span>
        <span className="shap-axis-label right">Toward →</span>
      </div>

      {/* ── Bars ── */}
      <div className="shap-rows">
        {data.map((item, idx) => {
          const isPos = item.shap_value >= 0;
          const barPct = (Math.abs(item.shap_value) / maxAbs) * 44; // max 44% each side

          return (
            <div
              key={item.feature}
              className="shap-row"
              style={{ animationDelay: `${idx * 0.04}s` }}
            >
              {/* Feature label */}
              <div className="shap-feat-label">
                <span className="shap-feat-id">{item.feature}</span>
                <span className="shap-feat-desc">{QUESTION_LABELS[item.feature] || ''}</span>
              </div>

              {/* Bidirectional bar */}
              <div className="shap-bar-track">
                <div className="shap-bar-center-line" />
                <div
                  className={`shap-bar-fill ${isPos ? 'pos' : 'neg'}`}
                  style={{
                    width: `${barPct}%`,
                    ...(isPos ? { left: '50%' } : { right: '50%' }),
                  }}
                />
              </div>

              {/* Value + direction */}
              <div className="shap-val-col">
                <span className={`shap-val ${isPos ? 'pos' : 'neg'}`}>
                  {item.shap_value > 0 ? '+' : ''}
                  {item.shap_value.toFixed(4)}
                </span>
                <span className={`shap-dir-label ${isPos ? 'pos' : 'neg'}`}>
                  {isPos ? '↑' : '↓'} {item.direction}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Footer note ── */}
      <div className="shap-footer">
        Signed SHAP values from the <strong>{predictedCategory}</strong> class · best model
      </div>
    </div>
  );
}

export default LocalShapChart;
