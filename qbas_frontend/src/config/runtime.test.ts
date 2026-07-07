import { describe, expect, it } from "vitest";
import { validateRuntimeConfig, type FrontendRuntimeConfig } from "./runtime";

const config = (overrides: Partial<FrontendRuntimeConfig> = {}): FrontendRuntimeConfig => ({
  environment: "development",
  apiBase: "/api/v1",
  demoApiKeyEnabled: false,
  demoTokenIssuerEnabled: true,
  ...overrides,
});

describe("frontend runtime security", () => {
  it("rejects a static API key in production", () => {
    expect(() => validateRuntimeConfig(config({
      environment: "production",
      demoApiKey: "bundled-secret",
      demoApiKeyEnabled: true,
      demoTokenIssuerEnabled: false,
    }))).toThrow(/must not contain VITE_QBAS_API_KEY/);
  });

  it("requires explicit opt-in for a demo API key", () => {
    expect(() => validateRuntimeConfig(config({ demoApiKey: "demo-key" }))).toThrow(/explicitly enabled/);
  });

  it("accepts production with external session authentication", () => {
    expect(validateRuntimeConfig(config({
      environment: "production",
      demoTokenIssuerEnabled: false,
    })).environment).toBe("production");
  });
});
