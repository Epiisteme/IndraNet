/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  readonly VITE_QBAS_ENVIRONMENT?: "development" | "demo" | "production";
  readonly VITE_QBAS_API_KEY?: string;
  readonly VITE_QBAS_ENABLE_DEMO_API_KEY?: string;
  readonly VITE_QBAS_ENABLE_DEMO_TOKEN_ISSUER?: string;
}
