const DIMENSION_COLORS = {
  Control: { bg: 'rgba(99, 102, 241, 0.12)', color: '#6366f1' },
  Ownership: { bg: 'rgba(20, 184, 166, 0.12)', color: '#14b8a6' },
  Reach: { bg: 'rgba(245, 158, 11, 0.12)', color: '#f59e0b' },
  Endurance: { bg: 'rgba(16, 185, 129, 0.12)', color: '#10b981' },
};

const PRIORITY_STYLES = {
  Critical: 'high',
  High: 'high',
  Moderate: 'medium',
  Medium: 'medium',
  Low: 'low',
  Growth: 'growth',
  high: 'high',
  moderate: 'medium',
  medium: 'medium',
  low: 'low',
  growth: 'growth',
};

function RecommendationCard({ recommendation }) {
  const {
    // Backend sends PascalCase keys: Dimension, Priority, Suggestion, Actions
    Dimension: dimension = 'General',
    Priority: priority = 'Medium',
    Suggestion: suggestion = '',
    Actions: action_items = [],
  } = recommendation;

  const dimStyle = DIMENSION_COLORS[dimension] || DIMENSION_COLORS.Control;
  const priorityClass = PRIORITY_STYLES[priority] || 'medium';

  return (
    <div className="recommendation-card">
      <div className="recommendation-header">
        <span
          className="rec-dimension-badge"
          style={{ background: dimStyle.bg, color: dimStyle.color }}
        >
          {dimension}
        </span>
        <span className={`priority-badge ${priorityClass}`}>
          {priority} Priority
        </span>
      </div>

      <div className="recommendation-body">
        <p className="recommendation-suggestion">{suggestion}</p>

        {action_items && action_items.length > 0 && (
          <ul className="action-items">
            {action_items.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        )}
      </div>


    </div>
  );
}

export default RecommendationCard;
