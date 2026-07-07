import { Loader2, ScanFace } from "lucide-react";
import { useState } from "react";

import {
  authenticateIris,
  extractFeatures,
  getApiErrorMessage,
  type AuthResult,
  type FeatureVector,
} from "../api/qbasClient";
import { useAuth } from "../hooks/useAuth";
import { IrisCapture } from "./IrisCapture";
import { ResultBadge } from "./ResultBadge";

interface AuthPanelProps {
  onVector: (vector: FeatureVector) => void;
  onComplete: () => void;
}

export function AuthPanel({ onVector, onComplete }: AuthPanelProps) {
  const { ensureToken } = useAuth();
  const [userId, setUserId] = useState("");
  const [blob, setBlob] = useState<Blob>();
  const [result, setResult] = useState<AuthResult>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [validation, setValidation] = useState<string>();

  const capture = async (nextBlob: Blob) => {
    setBlob(nextBlob);
    setResult(undefined);
    setError(undefined);
    setLoading(true);

    try {
      onVector(await extractFeatures(nextBlob));
    } catch (err) {
      setBlob(undefined);
      setError(getApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const submit = async () => {
    if (!userId.trim()) {
      setValidation("Enter the identity being verified.");
      return;
    }
    if (!blob) {
      setValidation("Capture or upload an approved image before verification.");
      return;
    }

    setValidation(undefined);
    setLoading(true);
    setError(undefined);

    try {
      const nextResult = await authenticateIris(blob, await ensureToken(), userId.trim());
      setResult(nextResult);
      onComplete();
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel flow-panel">
      <div className="panel-title">
        <ScanFace size={18} />
        <div>
          <h2>Verify an identity</h2>
          <p className="help-text">Compare a fresh iris sample with a protected enrolled template.</p>
        </div>
      </div>

      <IrisCapture onCapture={capture} disabled={loading} />

      <label className="field-label">
        Claimed identity
        <span>Required for a clear one-to-one demo verification decision.</span>
        <div className="form-row">
          <input
            value={userId}
            onChange={(event) => setUserId(event.target.value)}
            placeholder="e.g. employee-1042"
            aria-label="Claimed identity"
          />
          <button className="icon-button primary" onClick={submit} disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} /> : <ScanFace size={18} />}
            <span>Verify identity</span>
          </button>
        </div>
      </label>

      {validation ? <p className="inline-error" role="alert">{validation}</p> : null}
      <ResultBadge result={result} error={error} />
    </section>
  );
}
