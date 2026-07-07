import { useState } from "react";

import type { FeatureVector } from "../api/qbasClient";
import { EnrollForm } from "../components/EnrollForm";
import { QuantumViz } from "../components/QuantumViz";

export function Enroll() {
  const [vector, setVector] = useState<FeatureVector>();

  return (
    <main className="content-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Step 1 of 4</p>
          <h1>Enroll identity</h1>
          <p>Register a protected biometric reference and create an auditable enrollment event.</p>
        </div>
      </header>
      <div className="page-grid">
        <EnrollForm onVector={setVector} onComplete={() => undefined} />
        <QuantumViz vector={vector} />
      </div>
    </main>
  );
}
