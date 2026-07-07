import { useEffect, useState } from "react";

import { getHealth, type HealthResult } from "../api/qbasClient";
import { runtimeConfig } from "../config/runtime";

const configuredEnvironment = runtimeConfig.environment;

export function EnvironmentStatus() {
  const [health, setHealth] = useState<HealthResult>();

  useEffect(() => {
    getHealth().then(setHealth).catch(() => undefined);
  }, []);

  const environment = (health?.environment ?? configuredEnvironment).toLowerCase();
  const label = environment === "production" ? "Production" : environment === "demo" ? "Demo" : "Development";

  return (
    <span className={`environment-badge ${environment}`} aria-label={`Environment: ${label}`}>
      {label}
    </span>
  );
}
