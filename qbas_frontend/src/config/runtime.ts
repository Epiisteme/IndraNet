export type ConsoleEnvironment = "development" | "demo" | "production";

export interface FrontendRuntimeConfig {
  environment: ConsoleEnvironment;
  apiBase: string;
  demoApiKey?: string;
  demoApiKeyEnabled: boolean;
  demoTokenIssuerEnabled: boolean;
}

const normalizeEnvironment = (value?: string): ConsoleEnvironment =>
  value?.toLowerCase() === "production" ? "production" : value?.toLowerCase() === "demo" ? "demo" : "development";

export const validateRuntimeConfig = (config: FrontendRuntimeConfig): FrontendRuntimeConfig => {
  if (config.environment === "production" && config.demoApiKey) {
    throw new Error("Unsafe frontend configuration: production bundles must not contain VITE_QBAS_API_KEY.");
  }
  if (config.environment === "production" && config.demoTokenIssuerEnabled) {
    throw new Error("Unsafe frontend configuration: the demo token issuer must be disabled in production.");
  }
  if (config.demoApiKey && !config.demoApiKeyEnabled) {
    throw new Error("VITE_QBAS_API_KEY is set but VITE_QBAS_ENABLE_DEMO_API_KEY is not explicitly enabled.");
  }
  return config;
};

export const runtimeConfig = validateRuntimeConfig({
  environment: normalizeEnvironment(import.meta.env.VITE_QBAS_ENVIRONMENT),
  apiBase: import.meta.env.VITE_API_URL ?? "/api/v1",
  demoApiKey: import.meta.env.VITE_QBAS_API_KEY,
  demoApiKeyEnabled: import.meta.env.VITE_QBAS_ENABLE_DEMO_API_KEY === "true",
  demoTokenIssuerEnabled: import.meta.env.VITE_QBAS_ENABLE_DEMO_TOKEN_ISSUER !== "false",
});
