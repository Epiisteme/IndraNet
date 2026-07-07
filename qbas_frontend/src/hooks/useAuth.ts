import { useCallback, useState } from "react";

import { issueToken } from "../api/qbasClient";
import { useAuthStore } from "../store/authStore";
import { runtimeConfig } from "../config/runtime";

export const useAuth = () => {
  const token = useAuthStore((state) => state.token);
  const setToken = useAuthStore((state) => state.setToken);
  const [loading, setLoading] = useState(false);

  const ensureToken = useCallback(async () => {
    if (token) return token;
    if (!runtimeConfig.demoTokenIssuerEnabled) {
      throw new Error(
        "No operator session is available. Sign in through the deployment identity provider and retry."
      );
    }
    setLoading(true);
    try {
      const nextToken = await issueToken("admin");
      setToken(nextToken);
      return nextToken;
    } finally {
      setLoading(false);
    }
  }, [setToken, token]);

  return { token, ensureToken, loading };
};
