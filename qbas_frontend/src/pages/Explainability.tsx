import { Gauge, KeyRound, LockKeyhole, ScanEye } from "lucide-react";

const items = [
  {
    icon: ScanEye,
    title: "Biometric proof",
    text: "A fresh iris image is transformed into derived measurements. The service compares those measurements rather than returning raw image data.",
  },
  {
    icon: KeyRound,
    title: "Protected iris template",
    text: "Enrollment stores encrypted derived features and a biometric binding token so verification can combine biometric similarity with a secondary proof check.",
  },
  {
    icon: Gauge,
    title: "Confidence score",
    text: "Confidence expresses the model's match strength from 0-100%. Approval requires meeting the configured threshold and claimed-identity checks.",
  },
  {
    icon: LockKeyhole,
    title: "Encrypted verification",
    text: "The experimental CKKS path can compute encrypted similarity scores. Final interpretation requires trusted client-side decryption.",
  },
];

export function Explainability() {
  return (
    <main className="content-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Decision transparency</p>
          <h1>How verification decisions work</h1>
          <p>Plain-language guidance for operators, reviewers, and client demonstrations.</p>
        </div>
      </header>
      <div className="explain-grid">
        {items.map(({ icon: Icon, title, text }) => (
          <section className="panel explain-card" key={title}>
            <Icon size={22} />
            <h2>{title}</h2>
            <p>{text}</p>
          </section>
        ))}
      </div>
      <section className="panel callout">
        <strong>Demo assurance boundary</strong>
        <p>
          IndraNet demonstrates the end-to-end identity assurance workflow. Production deployment requires calibrated
          models, independent security review, key management, access controls, retention policy, liveness detection,
          and biometric/privacy compliance.
        </p>
      </section>
    </main>
  );
}
