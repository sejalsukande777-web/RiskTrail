import { useState } from 'react';

export default function App() {
  const [txnId, setTxnId] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInvestigate = async (e) => {
    e.preventDefault();
    if (!txnId.trim()) return;

    setLoading(true);
    setResult(null);
    setError(null);

    const attemptedId = txnId.trim();

    try {
      const response = await fetch('http://localhost:8000/api/investigate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transaction_id: attemptedId }),
      });

      const data = await response.json();

      if (!response.ok) {
        if (response.status === 404) {
          setError({
            message: data.error || 'Transaction not found',
            id: data.transaction_id || attemptedId,
          });
        } else if (response.status === 422) {
          setError({
            message: 'Malformed request. Please check the transaction ID.',
            id: attemptedId,
          });
        } else if (response.status === 500) {
          setError({
            message: data.error || 'ML or LLM investigation failed.',
            id: data.transaction_id || attemptedId,
          });
        } else {
          setError({
            message: data.error || 'An unexpected error occurred.',
            id: attemptedId,
          });
        }
      } else {
        setResult(data);
      }
    } catch (err) {
      setError({
        message: 'Unable to reach backend server. Please verify http://localhost:8000 is running.',
        id: attemptedId,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="logo">RiskTrail</h1>
        <p className="tagline">
          Traces the signals and evidence behind transaction risk and explains what to do next.
        </p>
      </header>

      <section className="search-section">
        <form onSubmit={handleInvestigate} className="search-form">
          <input
            type="text"
            className="input-field"
            placeholder="Enter Transaction ID (e.g., TXN_001)"
            value={txnId}
            onChange={(e) => setTxnId(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="submit-btn" disabled={loading || !txnId.trim()}>
            {loading ? 'Investigating...' : 'Investigate'}
          </button>
        </form>
        <p className="hint">Try: TXN_001 to TXN_006, or TXN_999 to test error handling.</p>
      </section>

      {loading && (
        <div className="state-card loading-card">
          <div className="spinner"></div>
          <p>Running ML detection and agent investigation model...</p>
        </div>
      )}

      {error && (
        <div className="state-card error-card">
          <h3>Investigation Failed</h3>
          <p className="error-message">{error.message}</p>
          {error.id && <p className="error-sub">Attempted ID: <strong>{error.id}</strong></p>}
        </div>
      )}

      {result && (
        <main className="result-container">
          <div className="overview-card">
            <div className="metrics">
              <div className="metric-item">
                <span className="metric-label">Transaction ID</span>
                <span className="metric-value">{result.transaction_id}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Risk Score</span>
                <span className="metric-value score-value">{result.risk_score} / 100</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Risk Level</span>
                <span className={`badge badge-${result.risk_level.toLowerCase()}`}>
                  {result.risk_level}
                </span>
              </div>
            </div>

            <div className="section-block">
              <h4>Risk Factors</h4>
              <ul className="factors-list">
                {result.risk_factors.map((factor, index) => (
                  <li key={index}>{factor}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="section-block">
            <h4>Evidence</h4>
            <div className="evidence-grid">
              {result.evidence.map((item, index) => (
                <div key={index} className="evidence-card">
                  <div className="evidence-header">
                    <span className="evidence-type">{item.type.toUpperCase()}</span>
                    <span className="evidence-id">{item.id}</span>
                  </div>
                  <p className="evidence-text">{item.text}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="section-block">
            <h4>AI Investigation</h4>
            <div className="narrative-box">
              <p>{result.investigation}</p>
            </div>
          </div>

          <div className="section-block">
            <h4>Recommended Action</h4>
            <div className="recommendation-box">
              <p>{result.recommendation}</p>
            </div>
          </div>
        </main>
      )}
    </div>
  );
}