import axios, { AxiosInstance } from "axios";
import { runtimeConfig } from "../config/runtime";

export interface FeatureVector {
  features: number[];
  dim: number;
  circuit: string;
  latency_ms?: number;
}

export interface EnrollResult {
  user_id: string;
  qbt_token: string;
  qrng_entropy: number;
  feature_dim: number;
  left_feature_dim?: number | null;
  right_feature_dim?: number | null;
  fused_feature_dim?: number | null;
  fusion_strategy?: string | null;
  enrolled_at: string;
}

export interface AuthResult {
  authenticated: boolean;
  identity: string | null;
  confidence: number;
  reason: string;
  latency_ms?: number;
  left_confidence?: number | null;
  right_confidence?: number | null;
  fused_confidence?: number | null;
  score_fusion_confidence?: number | null;
  fusion_strategy?: string | null;
  threshold: number;
  decision_code: string;
}

export interface HealthResult {
  status: string;
  database_connected: boolean;
  quantum_device: string;
  n_qubits: number;
  enrolled_templates: number;
  qsvm_ready: boolean;
  ckks_ready: boolean;
  version: string;
  environment: string;
  demo_mode: boolean;
}

export interface EntropyMetadataResult {
  salt_hex: string;
  sha3_256: string;
  min_entropy_lb: number;
  n_bits: number;
}

export interface AuditLogEntry {
  id: number;
  event_type: string;
  user_id: string | null;
  authenticated: boolean | null;
  confidence: number | null;
  latency_ms: number | null;
  reason: string;
  created_at: string;
}

export const createQBASClient = (token?: string): AxiosInstance => {
  const client = axios.create({ baseURL: runtimeConfig.apiBase, timeout: 60000, withCredentials: true });
  if (token) client.defaults.headers.common.Authorization = `Bearer ${token}`;
  if (runtimeConfig.demoApiKeyEnabled && runtimeConfig.demoApiKey) {
    client.defaults.headers.common["X-API-Key"] = runtimeConfig.demoApiKey;
  }
  return client;
};

export const getHealth = async (): Promise<HealthResult> => {
  const { data } = await createQBASClient().get<HealthResult>("/health");
  return data;
};

export const issueToken = async (role: "admin" | "operator" = "admin"): Promise<string> => {
  const form = new FormData();
  form.append("user_id", "console");
  form.append("role", role);
  const { data } = await createQBASClient().post<{ access_token: string }>("/auth/token", form);
  return data.access_token;
};

export const enrollIris = async (leftBlob: Blob, rightBlob: Blob, userId: string): Promise<EnrollResult> => {
  const form = new FormData();
  form.append("left_file", leftBlob, "left-iris.jpg");
  form.append("right_file", rightBlob, "right-iris.jpg");
  form.append("user_id", userId);
  const { data } = await createQBASClient().post<EnrollResult>("/enroll", form);
  return data;
};

export const extractFeatures = async (imageBlob: Blob): Promise<FeatureVector> => {
  const form = new FormData();
  form.append("file", imageBlob, "iris.jpg");
  const { data } = await createQBASClient().post<FeatureVector>("/extract-features", form);
  return data;
};

export const authenticateIris = async (
  leftBlob: Blob,
  rightBlob: Blob,
  token?: string,
  userId?: string
): Promise<AuthResult> => {
  const form = new FormData();
  form.append("left_file", leftBlob, "left-iris.jpg");
  form.append("right_file", rightBlob, "right-iris.jpg");
  if (userId) form.append("user_id", userId);
  const { data } = await createQBASClient(token).post<AuthResult>("/authenticate", form);
  return data;
};

export const generateQRNG = async (token?: string, nBits = 256): Promise<EntropyMetadataResult> => {
  const { data } = await createQBASClient(token).post<EntropyMetadataResult>(`/qrng/generate?n_bits=${nBits}`);
  return data;
};

export const getAuditLog = async (token?: string): Promise<AuditLogEntry[]> => {
  const { data } = await createQBASClient(token).get<AuditLogEntry[]>("/admin/audit-log?limit=20");
  return data;
};

export const getApiErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const message = error.response?.data?.error?.message;
    const requestId = error.response?.data?.error?.request_id;
    if (message) return requestId ? `${message} Reference: ${requestId.slice(0, 8)}.` : message;
    if (error.response?.status === 429) return "Too many requests were submitted. Wait a moment and try again.";
    if (error.response?.status === 401) return "The service could not authorize this request. Ask an administrator to check access configuration.";
    if (!error.response) return "The service could not be reached. Confirm that the backend is running.";
  }
  return error instanceof Error ? error.message : "The request could not be completed.";
};
