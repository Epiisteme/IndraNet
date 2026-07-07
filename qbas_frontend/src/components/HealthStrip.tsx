import { Cpu, Database, RadioTower, Waves } from "lucide-react";

import type { HealthResult } from "../api/qbasClient";

interface HealthStripProps {
  health?: HealthResult;
}

export function HealthStrip({ health }: HealthStripProps) {
  return (
    <div className="health-strip">
      <div>
        <RadioTower size={18} />
        <span>{health ? "Service available" : "Checking service..."}</span>
      </div>
      <div>
        <Cpu size={18} />
        <span>{health?.qsvm_ready ? "Matcher ready" : "Awaiting enrollment"}</span>
      </div>
      <div>
        <Waves size={18} />
        <span>{health?.ckks_ready ? "Encrypted matching ready" : "Encrypted matching optional"}</span>
      </div>
      <div>
        <Database size={18} />
        <span>{health?.enrolled_templates ?? 0} active identities</span>
      </div>
    </div>
  );
}
