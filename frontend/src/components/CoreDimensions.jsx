import React from 'react';

const DIMENSION_INFO = {
  Control: {
    icon: '🎯',
    color: '#6366f1',
    questions: ['Q1', 'Q5', 'Q9'],
    description: 'Belief in personal influence and problem-solving ability.',
    idealScore: 5, // Fixed: Max average score is 5
  },
  Ownership: {
    icon: '🔑',
    color: '#14b8a6',
    questions: ['Q2', 'Q6'],
    description: 'Taking responsibility and learning from mistakes.',
    idealScore: 5, // Fixed
  },
  Reach: {
    icon: '🌐',
    color: '#f59e0b',
    questions: ['Q3', 'Q7'],
    description: 'Preventing setbacks from affecting all life areas.',
    idealScore: 5, // Fixed
  },
  Endurance: {
    icon: '⏳',
    color: '#10b981',
    questions: ['Q4', 'Q8', 'Q10'],
    description: 'Belief that problems are temporary and solvable.',
    idealScore: 5, // Fixed
  },
};

function CoreDimensions({ dimensions, scores }) {
  const entries = Object.entries(DIMENSION_INFO);

  return (
    <div className="core-section">
      <div className="section-header">
        <h2>🧩 CORE Dimensions</h2>
        <p>Your Adversity Quotient is built on four key dimensions</p>
      </div>
      <div className="dimensions-grid">
        {entries.map(([name, info]) => {
          // FIXED: Safely grab the score, checking for both Capital and lowercase keys to prevent NaN
          const score = scores ? (scores[name] ?? scores[name.toLowerCase()] ?? 0) : 0;
          const ideal = info.idealScore;
          const percentage = Math.round((score / ideal) * 100);
          
          // Flag as weak if the score is below 3.0 (60%)
          const isWeak = score < 3.0; 
          const dimClass = name.toLowerCase();

          return (
            <div key={name} className={`dimension-card ${dimClass}`}>
              <div className="dimension-card-header">
                <div className="dimension-name">
                  <span className="dimension-icon">{info.icon}</span>
                  <span className="dimension-title">{name}</span>
                </div>
                <span className="dimension-score-badge">
                  {Number(score).toFixed(2)} / {ideal}
                </span>
              </div>

              <p className="dimension-description">{info.description}</p>

              <div className="dimension-questions">
                {info.questions.map(q => (
                  <span key={q} className="dimension-q-tag">{q}</span>
                ))}
              </div>

              <div className="dimension-progress-track">
                <div
                  className={`dimension-progress-fill ${dimClass}`}
                  style={{ width: `${Math.min(percentage, 100)}%` }}
                ></div>
              </div>
              
              {isWeak && (
                <div className="dimension-weak-alert">
                  ⚠️ Needs improvement
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default CoreDimensions;