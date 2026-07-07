import { Fingerprint, Loader2, Save } from "lucide-react";
import { useState } from "react";

import {
  enrollIris,
  extractFeatures,
  getApiErrorMessage,
  type EnrollResult,
  type FeatureVector,
} from "../api/qbasClient";
import { IrisCapture } from "./IrisCapture";

interface EnrollFormProps {
  onVector: (vector: FeatureVector) => void;
  onComplete: () => void;
}

export function EnrollForm({ onVector, onComplete }: EnrollFormProps) {
  const [userId, setUserId] = useState("");
  const [blob, setBlob] = useState<Blob>();
  const [result, setResult] = useState<EnrollResult>();
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
      setValidation("Enter an identity reference before enrollment.");
      return;
    }
    if (!blob) {
      setValidation("Capture or upload an approved image before enrollment.");
      return;
    }

    setValidation(undefined);
    setLoading(true);
    setError(undefined);

    try {
      const enrolled = await enrollIris(blob, userId.trim());
      setResult(enrolled);
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
        <Fingerprint size={18} />
        <div>
          <h2>Enroll an identity</h2>
          <p className="help-text">Create a protected biometric reference for future verification.</p>
        </div>
      </div>

      <IrisCapture onCapture={capture} disabled={loading} />

      <label className="field-label">
        Identity reference
        <span>Use an existing employee, citizen, or case identifier; never a secret.</span>
        <div className="form-row">
          <input
            value={userId}
            onChange={(event) => setUserId(event.target.value)}
            placeholder="e.g. employee-1042"
            aria-label="Identity reference"
          />
          <button className="icon-button primary" onClick={submit} disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} /> : <Save size={18} />}
            <span>Complete enrollment</span>
          </button>
        </div>
      </label>

      {validation ? <p className="inline-error" role="alert">{validation}</p> : null}

      {result ? (
        <div className="compact-result success-state" role="status">
          <strong>Identity enrolled successfully</strong>
          <span>{result.user_id} is ready for verification. Enrollment was recorded in the audit trail.</span>
          <details>
            <summary>Advanced / Technical Details</summary>
            <p>
              {result.feature_dim} derived features; {result.qrng_entropy} bits of generated entropy. Raw iris imagery is
              not returned by the API.
            </p>
          </details>
        </div>
      ) : null}

      {error ? <div className="result-badge danger" role="alert">{error}</div> : null}
    </section>
  );
}
