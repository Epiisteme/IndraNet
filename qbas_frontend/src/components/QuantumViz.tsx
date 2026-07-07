import { Activity } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { FeatureVector } from "../api/qbasClient";

interface QuantumVizProps {
  vector?: FeatureVector;
}

export function QuantumViz({ vector }: QuantumVizProps) {
  const rows =
    vector?.features.map((value, index) => ({
      qubit: `q${index}`,
      expectation: Number(value.toFixed(5)),
    })) ?? [];

  return (
    <section className="panel viz-panel">
      <div className="panel-title">
        <Activity size={18} />
        <div>
          <h2>Decision signal</h2>
          <p className="help-text">Derived measurements used by the matching service, not a raw iris image.</p>
        </div>
      </div>
      <div className="circuit-strip" aria-hidden="true">
        {Array.from({ length: vector?.dim ?? 8 }, (_, index) => (
          <div className="wire" key={index}>
            <span>q{index}</span>
            <i>H</i>
            <b>QFT</b>
            <i>Z</i>
          </div>
        ))}
      </div>
      <div className="chart-frame">
        {rows.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 12, right: 16, left: -18, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="qubit" />
              <YAxis domain={[-1, 1]} />
              <Tooltip formatter={(value) => Number(value).toFixed(5)} />
              <Bar dataKey="expectation" fill="#0f766e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty-state">
            <strong>No sample analyzed yet</strong>
            <span>Capture or upload an iris image to view derived decision signals.</span>
          </div>
        )}
      </div>
    </section>
  );
}
