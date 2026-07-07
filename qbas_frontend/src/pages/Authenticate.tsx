import { useState } from "react";

import type { FeatureVector } from "../api/qbasClient";
import { AuthPanel } from "../components/AuthPanel";
import { QuantumViz } from "../components/QuantumViz";

export function Authenticate() {
  const [vector, setVector] = useState<FeatureVector>();

  return (
    <main className="content-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Step 2 of 4</p>
          <h1>Verify identity</h1>
          <p>Capture a fresh sample and receive a clear approval decision with supporting rationale.</p>
        </div>
      </header>
      <div className="page-grid">
        <AuthPanel onVector={setVector} onComplete={() => undefined} />
        <QuantumViz vector={vector} />
      </div>
    </main>
  );
}
