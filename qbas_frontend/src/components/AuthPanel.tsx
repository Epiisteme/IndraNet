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
  const [leftBlob, setLeftBlob] = useState<Blob>();
  const [rightBlob, setRightBlob] = useState<Blob>();
  const [result, setResult] = useState<AuthResult>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [validation, setValidation] = useState<string>();

  const capture = async (eye: "left" | "right", nextBlob: Blob) => {
    if (eye === "left") setLeftBlob(nextBlob);
    else setRightBlob(nextBlob);
    setResult(undefined);
    setError(undefined);
    setLoading(true);

    try {
      onVector(await extractFeatures(nextBlob));
    } catch (err) {
      if (eye === "left") setLeftBlob(undefined);
      else setRightBlob(undefined);
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
    if (!leftBlob || !rightBlob) {
      setValidation("Capture or upload approved left and right eye images before verification.");
      return;
    }

    setValidation(undefined);
    setLoading(true);
    setError(undefined);

    try {
      const nextResult = await authenticateIris(leftBlob, rightBlob, await ensureToken(), userId.trim());
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

      <div className="dual-capture-grid">
        <IrisCapture label="Left eye iris" onCapture={(nextBlob) => capture("left", nextBlob)} disabled={loading} />
        <IrisCapture label="Right eye iris" onCapture={(nextBlob) => capture("right", nextBlob)} disabled={loading} />
      </div>

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
