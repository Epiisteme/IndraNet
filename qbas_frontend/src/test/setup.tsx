import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { forwardRef } from "react";

afterEach(cleanup);
vi.mock("react-webcam", () => ({ default: forwardRef(() => <div data-testid="camera-preview" />) }));
