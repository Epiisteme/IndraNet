import { ClipboardList, Loader2, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { getApiErrorMessage, getAuditLog, type AuditLogEntry } from "../api/qbasClient";
import { useAuth } from "../hooks/useAuth";

const labels: Record<string, string> = {
  enroll: "Identity enrolled",
  authenticate: "Identity verification",
  authenticate_fhe: "Encrypted comparison",
  revoke: "Enrollment revoked",
};

export function Audit() {
  const { ensureToken } = useAuth();
  const [events, setEvents] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(undefined);
    try {
      setEvents(await getAuditLog(await ensureToken()));
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [ensureToken]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <main className="content-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Accountability</p>
          <h1>Audit trail</h1>
          <p>Review enrollment and verification outcomes without exposing biometric payloads.</p>
        </div>
        <button className="icon-button" onClick={refresh} disabled={loading}>
          {loading ? <Loader2 className="spin" size={18} /> : <RefreshCw size={18} />}
          Refresh
        </button>
      </header>
      <section className="panel">
        <div className="panel-title">
          <ClipboardList size={18} />
          <h2>Recent security events</h2>
        </div>
        {error ? (
          <div className="result-badge danger">{error}</div>
        ) : events.length ? (
          <div className="audit-list">
            {events.map((event) => (
              <article key={event.id} className="audit-card">
                <div>
                  <strong>{labels[event.event_type] ?? event.event_type.replace(/_/g, " ")}</strong>
                  <span>{event.user_id ?? "System event"}</span>
                </div>
                <p>{event.reason}</p>
                <time>{new Date(event.created_at).toLocaleString()}</time>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <strong>No audit events yet</strong>
            <span>Enroll or verify an identity to create the first traceable event.</span>
          </div>
        )}
      </section>
    </main>
  );
}
