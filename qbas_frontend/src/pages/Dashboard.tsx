import { RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { getAuditLog, getHealth, type AuditLogEntry, type FeatureVector, type HealthResult } from "../api/qbasClient";
import { AuthPanel } from "../components/AuthPanel";
import { EnrollForm } from "../components/EnrollForm";
import { HealthStrip } from "../components/HealthStrip";
import { QRNGPanel } from "../components/QRNGPanel";
import { QuantumViz } from "../components/QuantumViz";
import { useAuth } from "../hooks/useAuth";

export function Dashboard() {
  const { ensureToken } = useAuth();
  const [health, setHealth] = useState<HealthResult>();
  const [audit, setAudit] = useState<AuditLogEntry[]>([]);
  const [vector, setVector] = useState<FeatureVector>();
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const [nextHealth, token] = await Promise.all([getHealth(), ensureToken()]);
      setHealth(nextHealth);
      setAudit(await getAuditLog(token));
    } finally {
      setRefreshing(false);
    }
  }, [ensureToken]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <main className="dashboard">
      <div className="toolbar">
        <div>
          <p className="eyebrow">Identity assurance workspace</p>
          <h1>Verification operations overview</h1>
          <p className="subtitle">
            An operator workflow for product demonstrations: enroll, verify, audit, and explain identity assurance
            decisions.
          </p>
        </div>
        <button className="icon-button" onClick={refresh} disabled={refreshing} title="Refresh system state">
          <RefreshCw className={refreshing ? "spin" : undefined} size={18} />
          <span>Refresh</span>
        </button>
      </div>

      <HealthStrip health={health} />
      <div className="work-grid">
        <EnrollForm onVector={setVector} onComplete={refresh} />
        <AuthPanel onVector={setVector} onComplete={refresh} />
        <QuantumViz vector={vector} />
        <div className="side-stack">
          <QRNGPanel />
          <section className="panel audit-panel">
            <div className="panel-title">
              <h2>Recent audit events</h2>
            </div>
            <div className="audit-table">
              {audit.length ? (
                audit.map((entry) => (
                  <div className="audit-row" key={entry.id}>
                    <span>{entry.event_type}</span>
                    <span>{entry.user_id ?? "system"}</span>
                    <span>{entry.reason}</span>
                    <time>{new Date(entry.created_at).toLocaleTimeString()}</time>
                  </div>
                ))
              ) : (
                <div className="empty-state">
                  <strong>No audit events yet</strong>
                  <span>Complete an enrollment or verification to begin the trail.</span>
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
