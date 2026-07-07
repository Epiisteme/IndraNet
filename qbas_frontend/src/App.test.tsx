import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { EnrollForm } from "./components/EnrollForm";
import { AuthPanel } from "./components/AuthPanel";
import * as api from "./api/qbasClient";

vi.mock("./api/qbasClient", async (importOriginal) => {
  const actual = await importOriginal<typeof api>();
  return {
    ...actual,
    getHealth: vi.fn().mockResolvedValue({ environment: "demo", demo_mode: true }),
    getAuditLog: vi.fn().mockResolvedValue([]),
    issueToken: vi.fn().mockResolvedValue("test-token")
  };
});

const routerFuture = { v7_relativeSplatPath: true, v7_startTransition: true };

const renderApp = (path = "/") =>
  render(
    <MemoryRouter initialEntries={[path]} future={routerFuture}>
      <App />
    </MemoryRouter>
  );

describe("operator console", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders the overview with an unobtrusive environment badge", async () => {
    renderApp();
    expect(screen.getByRole("heading", { name: /verification operations overview/i })).toBeTruthy();
    expect(await screen.findByText("Demo", { selector: ".environment-badge" })).toBeTruthy();
    expect(screen.queryByRole("note")).toBeNull();
  });

  it("navigates through the workflow", async () => {
    renderApp();
    expect(await screen.findByText("Demo", { selector: ".environment-badge" })).toBeTruthy();
    fireEvent.click(screen.getByRole("link", { name: /enroll identity/i }));
    expect(screen.getByRole("heading", { name: "Enroll identity" })).toBeTruthy();
  });

  it("validates enrollment inputs", () => {
    render(<EnrollForm onVector={vi.fn()} onComplete={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /complete enrollment/i }));
    expect(screen.getByRole("alert").textContent).toMatch(/identity reference/i);
  });

  it("validates verification inputs", () => {
    render(
      <MemoryRouter future={routerFuture}>
        <AuthPanel onVector={vi.fn()} onComplete={vi.fn()} />
      </MemoryRouter>
    );
    fireEvent.click(screen.getByRole("button", { name: /verify identity/i }));
    expect(screen.getByRole("alert").textContent).toMatch(/identity being verified/i);
  });

  it("shows understandable API failures", async () => {
    vi.mocked(api.getAuditLog).mockRejectedValueOnce(new Error("Service maintenance is in progress."));
    renderApp("/audit");
    expect((await screen.findByRole("alert")).textContent).toContain("Service maintenance is in progress.");
  });

  it("shows the empty audit trail state", async () => {
    renderApp("/audit");
    expect(await screen.findByText(/no audit events yet/i)).toBeTruthy();
  });

  it("renders explainability without crashing", async () => {
    renderApp("/explainability");
    expect(await screen.findByText("Demo", { selector: ".environment-badge" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /how verification decisions work/i })).toBeTruthy();
    expect(screen.getByText(/secondary proof check/i)).toBeTruthy();
  });
});
