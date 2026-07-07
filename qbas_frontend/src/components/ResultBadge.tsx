import { CircleAlert, ShieldCheck, ShieldX } from "lucide-react";

import type { AuthResult } from "../api/qbasClient";

interface ResultBadgeProps {
  result?: AuthResult;
  error?: string;
}

export function ResultBadge({ result, error }: ResultBadgeProps) {
  if (error) {
    return (
      <div className="result-badge danger" role="alert">
        <CircleAlert size={20} />
        <div>
          <strong>Verification request failed</strong>
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  const Icon = result.authenticated ? ShieldCheck : ShieldX;
  const publicMessage = result.authenticated ? "Biometric match and token proof both passed." : "Verification not approved.";

  return (
    <div className={`result-badge ${result.authenticated ? "success" : "warning"}`} role="status">
      <Icon size={22} />
      <div>
        <strong>{result.authenticated ? "Identity verified" : "Verification not approved"}</strong>
        <span>
          {result.identity ?? "No identity confirmed"}; {(result.confidence * 100).toFixed(1)}% confidence
        </span>
        <p>{publicMessage}</p>
        <details>
          <summary>Advanced / Technical Details</summary>
          <p>{result.reason}</p>
          <p>
            Decision {result.decision_code}; threshold {(result.threshold * 100).toFixed(0)}%; processing{" "}
            {result.latency_ms?.toFixed(0) ?? "-"} ms.
          </p>
        </details>
      </div>
    </div>
  );
}
