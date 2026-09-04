import { useState } from 'react';
import '../styles/QuestionnaireForm.css';

const QUESTIONS = [
  { id: 'Q1', text: 'When I face academic difficulties, I believe I can find a way to overcome them.', dimension: 'Endurance' },
  { id: 'Q2', text: 'I take responsibility for improving my situation when things go wrong in my studies.', dimension: 'Ownership' },
  { id: 'Q3', text: 'A setback in one subject does not affect my confidence in other subjects.', dimension: 'Reach' },
  { id: 'Q4', text: 'I believe most academic problems are temporary and can be resolved with effort.', dimension: 'Endurance' },
  { id: 'Q5', text: 'Even under pressure, I feel I have control over how I respond to academic challenges.', dimension: 'Control' },
  { id: 'Q6', text: 'I reflect on my mistakes and use them to perform better next time.', dimension: 'Ownership' },
  { id: 'Q7', text: 'Academic failures do not define my overall ability as a student.', dimension: 'Reach' },
  { id: 'Q8', text: 'I remain motivated even when results are not immediately visible.', dimension: 'Control' },
  { id: 'Q9', text: 'I believe my actions can positively influence my academic outcomes.', dimension: 'Control' },
  { id: 'Q10', text: 'I recover quickly after experiencing disappointment in my studies.', dimension: 'Ownership' },
];


const SCALE_LABELS = {
  1: 'Strongly Disagree',
  2: 'Disagree',
  3: 'Neither Agree nor Disagree',
  4: 'Agree',
  5: 'Strongly Agree',
};

function ConsentModal({ onAccept, onDecline }) {
  const [checked, setChecked] = useState(false);

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      backgroundColor: 'rgba(0,0,0,0.55)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '16px',
    }}>
      <div style={{
        background: '#fff', borderRadius: '16px',
        maxWidth: '560px', width: '100%',
        padding: '36px 32px', boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
        display: 'flex', flexDirection: 'column', gap: '20px',
      }}>
        {/* Icon + Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          
          <h2 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 700, color: '#1e293b' }}>
            Informed Consent
          </h2>
        </div>

        {/* Consent Text */}
        <div style={{
          background: '#f8fafc', border: '1px solid #e2e8f0',
          borderRadius: '10px', padding: '18px 20px',
          fontSize: '0.92rem', lineHeight: '1.7', color: '#374151',
        }}>
          <p style={{ margin: '0 0 10px 0' }}>
            I voluntarily consent to participate in this <strong>Adversity Quotient (AQ) Assessment</strong>.
          </p>
          <p style={{ margin: '0 0 10px 0' }}>
            I understand that my responses will be used solely for <strong>research, analysis, and model training purposes</strong>.
          </p>
          <p style={{ margin: 0 }}>
            All responses will be treated <strong>confidentially</strong>, stored securely, and reported only in
            anonymized or aggregated form. No personally identifiable information will be disclosed without
            my explicit permission.
          </p>
        </div>

        {/* Checkbox */}
        <label style={{
          display: 'flex', alignItems: 'flex-start', gap: '10px',
          cursor: 'pointer', fontSize: '0.9rem', color: '#1e293b',
        }}>
          <input
            type="checkbox"
            checked={checked}
            onChange={e => setChecked(e.target.checked)}
            style={{ marginTop: '3px', accentColor: '#3b82f6', width: '16px', height: '16px', flexShrink: 0 }}
          />
          I have read and understood the above, and I voluntarily agree to participate.
        </label>

        {/* Buttons */}
        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            onClick={onDecline}
            style={{
              flex: 1, padding: '11px 0', borderRadius: '8px',
              border: '1.5px solid #e2e8f0', background: '#fff',
              color: '#64748b', fontWeight: 600, fontSize: '0.92rem',
              cursor: 'pointer',
            }}
          >
            Decline
          </button>
          <button
            onClick={onAccept}
            disabled={!checked}
            style={{
              flex: 2, padding: '11px 0', borderRadius: '8px',
              border: 'none',
              background: checked ? '#3b82f6' : '#cbd5e1',
              color: '#fff', fontWeight: 600, fontSize: '0.92rem',
              cursor: checked ? 'pointer' : 'not-allowed',
              transition: 'background 0.2s',
            }}
          >
            I Agree &amp; Continue
          </button>
        </div>
      </div>
    </div>
  );
}

function QuestionnaireForm({ onSubmit, isLoading }) {
  const [consentGiven, setConsentGiven] = useState(false);
  const [consentDeclined, setConsentDeclined] = useState(false);

  const [answers, setAnswers] = useState(() => {
    const initial = {};
    QUESTIONS.forEach(q => { initial[q.id] = null; });
    return initial;
  });
  const [showValidation, setShowValidation] = useState(false);

  const answeredCount = Object.values(answers).filter(v => v !== null).length;
  const totalQuestions = QUESTIONS.length;
  const completionPercent = Math.round((answeredCount / totalQuestions) * 100);
  const allAnswered = answeredCount === totalQuestions;

  const handleAnswer = (questionId, value) => {
    setAnswers(prev => ({ ...prev, [questionId]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!allAnswered) {
      setShowValidation(true);
      const firstUnanswered = QUESTIONS.find(q => answers[q.id] === null);
      if (firstUnanswered) {
        const el = document.getElementById(`question-${firstUnanswered.id}`);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
      return;
    }
    onSubmit(answers);
  };

  // Declined screen
  if (consentDeclined) {
    return (
      <div style={{
        minHeight: '60vh', display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: '16px',
        padding: '40px 20px', textAlign: 'center',
      }}>
        <div style={{ fontSize: '48px' }}>🔒</div>
        <h2 style={{ margin: 0, color: '#1e293b', fontSize: '1.3rem' }}>Participation Declined</h2>
        <p style={{ color: '#64748b', maxWidth: '400px', lineHeight: '1.6', margin: 0 }}>
          You have chosen not to participate. Your decision is respected. You may close this page.
        </p>
        <button
          onClick={() => { setConsentDeclined(false); setConsentGiven(false); }}
          style={{
            marginTop: '8px', padding: '10px 24px', borderRadius: '8px',
            border: '1.5px solid #e2e8f0', background: '#fff',
            color: '#3b82f6', fontWeight: 600, cursor: 'pointer', fontSize: '0.92rem',
          }}
        >
          Go Back
        </button>
      </div>
    );
  }

  return (
    <>
      {/* Consent modal shown until accepted */}
      {!consentGiven && (
        <ConsentModal
          onAccept={() => setConsentGiven(true)}
          onDecline={() => setConsentDeclined(true)}
        />
      )}

      {/* Original questionnaire UI — unchanged */}
      <div className="questionnaire">
        <div className="questionnaire-header">
          <h2>AQ Assessment Questionnaire</h2>
          <p>Rate each statement based on how much you agree with it in your academic life.</p>
        </div>

        {/* Progress Bar */}
        <div className="progress-section">
          <div className="progress-info">
            <span>{answeredCount}/{totalQuestions} questions answered</span>
            <span className="progress-percentage">{completionPercent}%</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${completionPercent}%` }}
            ></div>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="questions-list">
            {QUESTIONS.map((question, index) => {
              const isAnswered = answers[question.id] !== null;
              const isUnanswered = showValidation && !isAnswered;

              return (
                <div
                  key={question.id}
                  id={`question-${question.id}`}
                  className={`question-card${isAnswered ? ' answered' : ''}${isUnanswered ? ' unanswered' : ''}`}
                >
                  <div className="question-header">
                    <span className="question-number">{index + 1}</span>
                  </div>

                  <p className="question-text">{question.text}</p>

                  <div className="likert-scale" role="radiogroup" aria-label={`Question ${index + 1}`}>
                    {[1, 2, 3, 4, 5].map(value => (
                      <label
                        key={value}
                        className={`likert-option${answers[question.id] === value ? ' selected' : ''}`}
                      >
                        <input
                          type="radio"
                          name={question.id}
                          value={value}
                          checked={answers[question.id] === value}
                          onChange={() => handleAnswer(question.id, value)}
                        />
                        <span className="likert-circle">{value}</span>
                        <span className="likert-label">{SCALE_LABELS[value]}</span>
                      </label>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Submit Section */}
          <div className="submit-section">
            <button
              type="submit"
              className="submit-btn"
              disabled={!allAnswered || isLoading}
            >
              {isLoading ? '⏳ Analyzing...' : '🔍 Get My AQ Analysis'}
            </button>
            {!allAnswered && (
              <p className="submit-hint">
                Please answer all {totalQuestions} questions to continue
              </p>
            )}
          </div>
        </form>
      </div>
    </>
  );
}

export default QuestionnaireForm;