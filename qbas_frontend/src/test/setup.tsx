import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(cleanup);
vi.mock("react-webcam", () => ({ default: () => <div data-testid="camera-preview" /> }));
