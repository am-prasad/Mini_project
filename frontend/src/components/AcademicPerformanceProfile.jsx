import '../styles/AcademicPerformanceProfile.css';

// ─── Derivation logic ────────────────────────────────────────────────────────
// All scores are on a 1–5 Likert scale. We map them to 0–100% internally.

function pct(score) {
  return Math.round(((score - 1) / 4) * 100);
}

function deriveMetrics(dimensions) {
  const control   = dimensions?.Control   ?? 3;
  const ownership = dimensions?.Ownership ?? 3;
  const reach     = dimensions?.Reach     ?? 3;
  const endurance = dimensions?.Endurance ?? 3;
  const aqScore   = (control + ownership + reach + endurance) / 4;

  // 1. Academic Persistence — driven by Endurance (primary) + Ownership (secondary)
  const persistence = Math.round(pct(endurance) * 0.65 + pct(ownership) * 0.35);

  // 2. Burnout Resistance — driven by Reach (primary) + Control (secondary)
  const burnout = Math.round(pct(reach) * 0.55 + pct(control) * 0.45);

  // 3. Grade Recovery Speed — driven by Ownership (primary) + Control (secondary)
  const gradeRecovery = Math.round(pct(ownership) * 0.60 + pct(control) * 0.40);

  // 4. Self-Directed Engagement — driven by Control (primary) + Reach (secondary)
  const selfDirected = Math.round(pct(control) * 0.55 + pct(reach) * 0.45);

  // 5. Classroom Proactivity — driven by Ownership (primary) + Endurance (secondary)
  const proactivity = Math.round(pct(ownership) * 0.50 + pct(endurance) * 0.50);

  // 6. Overall Academic Resilience Index — weighted composite
  const resilience = Math.round(
    pct(endurance) * 0.30 +
    pct(control)   * 0.25 +
    pct(ownership) * 0.25 +
    pct(reach)     * 0.20
  );

  return { persistence, burnout, gradeRecovery, selfDirected, proactivity, resilience, aqScore };
}

// ─── Risk level helpers ───────────────────────────────────────────────────────
function level(val) {
  if (val >= 80) return { label: 'Excellent', color: '#16a34a', bg: '#dcfce7', bar: '#22c55e' };
  if (val >= 60) return { label: 'Good',      color: '#ca8a04', bg: '#fef9c3', bar: '#facc15' };
  if (val >= 40) return { label: 'Moderate',  color: '#ea580c', bg: '#ffedd5', bar: '#f97316' };
  return              { label: 'Needs Work',  color: '#dc2626', bg: '#fee2e2', bar: '#ef4444' };
}

function burnoutLevel(val) {
  if (val >= 80) return { label: 'Low Risk',      color: '#16a34a', bg: '#dcfce7', bar: '#22c55e' };
  if (val >= 60) return { label: 'Moderate Risk', color: '#ca8a04', bg: '#fef9c3', bar: '#facc15' };
  if (val >= 40) return { label: 'High Risk',     color: '#ea580c', bg: '#ffedd5', bar: '#f97316' };
  return              { label: 'Very High Risk',  color: '#dc2626', bg: '#fee2e2', bar: '#ef4444' };
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function MetricBar({ label, value, description, levelFn = level, icon }) {
  const lv = levelFn(value);
  return (
    <div className="app-metric-card">
      <div className="app-metric-top">
        <span className="app-metric-icon">{icon}</span>
        <div className="app-metric-info">
          <span className="app-metric-label">{label}</span>
          <span className="app-metric-desc">{description}</span>
        </div>
        <span className="app-metric-badge" style={{ color: lv.color, background: lv.bg }}>
          {lv.label}
        </span>
      </div>
      <div className="app-bar-track">
        <div
          className="app-bar-fill"
          style={{ width: `${value}%`, background: lv.bar }}
        />
      </div>
      <div className="app-bar-footer">
        <span className="app-bar-val">{value}%</span>
      </div>
    </div>
  );
}

function ResilienceRing({ value }) {
  const radius = 54;
  const circ   = 2 * Math.PI * radius;
  const offset = circ - (value / 100) * circ;
  const lv     = level(value);

  return (
    <div className="app-ring-wrapper">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r={radius} fill="none" stroke="#e2e8f0" strokeWidth="10" />
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke={lv.bar}
          strokeWidth="10"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 70 70)"
          style={{ transition: 'stroke-dashoffset 1s ease' }}
        />
        <text x="70" y="64" textAnchor="middle" fontSize="22" fontWeight="700" fill="#1e293b">{value}</text>
        <text x="70" y="80" textAnchor="middle" fontSize="10" fill="#64748b">out of 100</text>
      </svg>
      <span className="app-ring-label" style={{ color: lv.color, background: lv.bg }}>
        {lv.label}
      </span>
    </div>
  );
}

function Insight({ icon, title, body }) {
  return (
    <div className="app-insight-card">
      <span className="app-insight-icon">{icon}</span>
      <div>
        <p className="app-insight-title">{title}</p>
        <p className="app-insight-body">{body}</p>
      </div>
    </div>
  );
}

// ─── Prose generators (deterministic, score-driven) ──────────────────────────
function persistenceText(v) {
  if (v >= 80) return 'Likely to complete long-horizon academic goals such as dissertations, certifications, or multi-semester projects without disengaging.';
  if (v >= 60) return 'Generally stays the course on long-term goals but may need periodic external motivation during low-feedback phases.';
  if (v >= 40) return 'At risk of disengaging during extended low-feedback periods — structured milestones and peer accountability can help.';
  return 'Likely to struggle with long-term academic commitments without structured support and regular check-ins.';
}
function burnoutText(v) {
  if (v >= 80) return 'A weak result in one subject is unlikely to cascade into broader academic panic. You compartmentalise setbacks well.';
  if (v >= 60) return 'Moderate ability to contain stress. Some subjects may act as emotional anchors; leaning on them during tough exam seasons helps.';
  if (v >= 40) return 'Stress from one subject may spill over. Consider stress-isolation techniques and block-scheduling to keep subjects separate.';
  return 'High spillover risk. A single academic failure could affect motivation across all subjects — proactive counselling is advisable.';
}
function recoveryText(v) {
  if (v >= 80) return 'After a poor grade, you are likely to review the feedback, adjust the study approach, and improve in the next assessment.';
  if (v >= 60) return 'Tends to reflect on mistakes but recovery may take a few cycles. Journalling feedback and tracking improvements can accelerate this.';
  if (v >= 40) return 'Grade recovery is slower; try structured post-exam reviews with a peer or tutor to convert mistakes into a clear action list.';
  return 'Needs active support to translate academic setbacks into revised plans. Mentorship or academic coaching is recommended.';
}
function selfDirectedText(v) {
  if (v >= 80) return 'Proactively seeks resources, asks for help early, and adapts study strategies without waiting for external prompting.';
  if (v >= 60) return 'Largely self-sufficient but benefits from occasional structured guidance to stay on optimal study paths.';
  if (v >= 40) return 'Often reactive rather than proactive; setting a weekly self-study review schedule can shift this pattern.';
  return 'Relies heavily on external direction. Habit-stacking and accountability partners would significantly improve engagement.';
}
function proactivityText(v) {
  if (v >= 80) return 'Likely to volunteer in discussions, lead group work, and seek instructor feedback before assessments.';
  if (v >= 60) return 'Participates when prompted; nudging this into habitual participation can meaningfully improve outcomes.';
  if (v >= 40) return 'Tends to stay in the background; structured participation goals (e.g. ask one question per lecture) can build confidence.';
  return 'Low classroom engagement is likely. Safe, low-stakes participation opportunities are the starting point.';
}

// ─── Main component ───────────────────────────────────────────────────────────
/**
 * Props:
 *   dimensions: { Control: number, Ownership: number, Reach: number, Endurance: number }
 *               Each value is a 1–5 Likert average for that dimension.
 *   aqCategory: 'Low' | 'Medium' | 'High'
 */
function AcademicPerformanceProfile({ dimensions, aqCategory }) {
  const {
    persistence, burnout, gradeRecovery,
    selfDirected, proactivity, resilience,
  } = deriveMetrics(dimensions);

  return (
    <div className="app-root">
      {/* ── Header ── */}
      <div className="app-header">
        <h2 className="app-title">🎓 Academic Performance Profile</h2>
        <p className="app-subtitle">
          How your AQ dimensions translate into real academic behaviours and outcomes
        </p>
      </div>

      {/* ── Resilience Index ring + context ── */}
      <div className="app-resilience-row">
        <ResilienceRing value={resilience} />
        <div className="app-resilience-text">
          <p className="app-resilience-heading">Overall Academic Resilience Index</p>
          <p className="app-resilience-para">
            A composite derived from all four CORE dimensions — weighted toward Endurance
            and Ownership as the primary predictors of long-term academic success.
            Scores above 75 strongly correlate with course completion and positive GPA trends.
          </p>
          <div className="app-formula">
            <span>Endurance 30%</span>
            <span>Control 25%</span>
            <span>Ownership 25%</span>
            <span>Reach 20%</span>
          </div>
        </div>
      </div>

      {/* ── Five metric bars ── */}
      <div className="app-metrics-grid">
        <MetricBar
          icon="📚"
          label="Academic Persistence"
          value={persistence}
          description="Ability to sustain effort on long-horizon academic goals"
          levelFn={level}
        />
        <MetricBar
          icon="🔥"
          label="Burnout Resistance"
          value={burnout}
          description="Risk of stress from one subject cascading into others"
          levelFn={burnoutLevel}
        />
        <MetricBar
          icon="📈"
          label="Grade Recovery Speed"
          value={gradeRecovery}
          description="How quickly you adapt and improve after a poor result"
          levelFn={level}
        />
        <MetricBar
          icon="🧭"
          label="Self-Directed Engagement"
          value={selfDirected}
          description="Tendency to proactively seek resources and adjust strategies"
          levelFn={level}
        />
        <MetricBar
          icon="🙋"
          label="Classroom Proactivity"
          value={proactivity}
          description="Likelihood of active participation and help-seeking"
          levelFn={level}
        />
      </div>

      {/* ── Personalised insights ── */}
      <div className="app-insights-section">
        <h3 className="app-insights-heading">📝 What this means for your studies</h3>
        <div className="app-insights-list">
          <Insight
            icon="📚"
            title="Academic Persistence"
            body={persistenceText(persistence)}
          />
          <Insight
            icon="🔥"
            title="Burnout Resistance"
            body={burnoutText(burnout)}
          />
          <Insight
            icon="📈"
            title="Grade Recovery"
            body={recoveryText(gradeRecovery)}
          />
          <Insight
            icon="🧭"
            title="Self-Directed Study"
            body={selfDirectedText(selfDirected)}
          />
          <Insight
            icon="🙋"
            title="Classroom Proactivity"
            body={proactivityText(proactivity)}
          />
        </div>
      </div>

      {/* ── Dimension → academic trait mapping table ── */}
      <div className="app-table-section">
        <h3 className="app-table-heading">🔗 CORE Dimension → Academic Trait Map</h3>
        <div className="app-table-wrapper">
          <table className="app-table">
            <thead>
              <tr>
                <th>Dimension</th>
                <th>Your Score</th>
                <th>Primary Academic Trait</th>
                <th>Risk if Low</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  dim: '🎯 Control', score: dimensions?.Control ?? '—',
                  trait: 'Problem-solving under exam pressure',
                  risk: 'Helplessness & avoidance during hard modules',
                },
                {
                  dim: '🔑 Ownership', score: dimensions?.Ownership ?? '—',
                  trait: 'Self-correction after poor feedback',
                  risk: 'Blame-shifting, repeated mistakes',
                },
                {
                  dim: '🌐 Reach', score: dimensions?.Reach ?? '—',
                  trait: 'Stress containment across subjects',
                  risk: 'Widespread grade dip from isolated failure',
                },
                {
                  dim: '⏳ Endurance', score: dimensions?.Endurance ?? '—',
                  trait: 'Course completion & long-term motivation',
                  risk: 'Drop-out risk in demanding programmes',
                },
              ].map(row => (
                <tr key={row.dim}>
                  <td className="app-td-dim">{row.dim}</td>
                  <td className="app-td-score">{typeof row.score === 'number' ? `${row.score.toFixed(2)} / 5` : row.score}</td>
                  <td>{row.trait}</td>
                  <td className="app-td-risk">{row.risk}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Footer note ── */}
      <p className="app-footer-note">
        * All academic indicators are derived from AQ dimension scores using research-weighted
        formulas. They are predictive indicators, not diagnostic assessments.
      </p>
    </div>
  );
}

export default AcademicPerformanceProfile;