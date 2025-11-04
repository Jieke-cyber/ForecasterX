import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { jwtDecode } from "jwt-decode";
import api from "../lib/api.js";
import { useNavigate } from "react-router-dom";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  const logout = () => {
    localStorage.removeItem("token");
    setToken(null);
    setUser(null);
    navigate("/login");
  };

  const login = (newToken) => {
    localStorage.setItem("token", newToken);
    setToken(newToken);
    try {
      const payload = jwtDecode(newToken);
      setUser({ email: payload?.email });
    } catch {
      setUser(null);
    }
    navigate("/");
  };

  // logout automatico alla scadenza
  useEffect(() => {
    if (!token) return;
    try {
      const { exp } = jwtDecode(token) || {};
      if (!exp) return;
      const ms = exp * 1000 - Date.now();
      if (ms <= 0) return logout();
      const t = setTimeout(logout, ms);
      return () => clearTimeout(t);
    } catch {
      logout();
    }
  }, [token]);

  // intercetta 401 -> logout
  useEffect(() => {
    const id = api.interceptors.response.use(
      (r) => r,
      (err) => {
        if (err?.response?.status === 401) logout();
        return Promise.reject(err);
      }
    );
    return () => api.interceptors.response.eject(id);
  }, []);

  const value = useMemo(() => ({ token, user, login, logout }), [token, user]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};
