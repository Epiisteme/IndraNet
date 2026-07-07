import { Dices, Loader2 } from "lucide-react";
import { useState } from "react";

import { generateQRNG, type EntropyMetadataResult } from "../api/qbasClient";
import { useAuth } from "../hooks/useAuth";

export function QRNGPanel() {
  const { ensureToken } = useAuth();
  const [result, setResult] = useState<EntropyMetadataResult>();
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    try {
      const token = await ensureToken();
      setResult(await generateQRNG(token, 256));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel qrng-panel">
      <div className="panel-title">
        <Dices size={18} />
        <h2>QRNG</h2>
      </div>
      <button className="icon-button" onClick={run} disabled={loading}>
        {loading ? <Loader2 className="spin" size={18} /> : <Dices size={18} />}
        <span>Generate</span>
      </button>
      {result ? (
        <dl className="kv-list">
          <div>
            <dt>Entropy</dt>
            <dd>{result.min_entropy_lb} bits</dd>
          </div>
          <div>
            <dt>Salt</dt>
            <dd>{result.salt_hex.slice(0, 24)}...</dd>
          </div>
          <div>
            <dt>SHA3</dt>
            <dd>{result.sha3_256.slice(0, 24)}...</dd>
          </div>
        </dl>
      ) : null}
    </section>
  );
}
