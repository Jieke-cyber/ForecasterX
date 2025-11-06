import React, { useState } from "react";
import api from "../lib/api.js";
import { useAuth } from "../context/AuthContext.jsx";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function extractErrorMessage(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail[0]?.message || detail[0]?.msg || "Dati non validi.";
  }
  return "Login fallito.";
}

export default function Login() {
  const { loginWithToken } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e) => {
    e.preventDefault();
    setErr(null);

    const emailNorm = email.trim().toLowerCase();
    if (!EMAIL_RE.test(emailNorm)) {
      setErr("Formato email non valido. Usa es. 123@example.com");
      return;
    }
    if (!password) {
      setErr("Inserisci la password.");
      return;
    }

    setLoading(true);
    try {
      const res = await api.post("/auth/login", { email: emailNorm, password });
      const tk =
        res.data?.access_token ??
        res.data?.token ??
        (typeof res.data === "string" ? res.data : null);

      if (!tk) throw new Error("Token mancante nella risposta");
      loginWithToken(tk);
    } catch (e) {
      setErr(extractErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="center-screen" style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
      {/* 👇 disattiva validazione nativa del browser */}
      <form noValidate onSubmit={onSubmit} style={{ width: 320, display: "grid", gap: 12, border: "1px solid #ddd", padding: 16, borderRadius: 12 }}>
        <h1 style={{ margin: 0 }}>Accedi</h1>

        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {err && <small style={{ color: "crimson" }}>{err}</small>}

        <button disabled={loading}>{loading ? "…" : "Login"}</button>

        <a href="/register">Non hai un account? Registrati</a>
      </form>
    </div>
  );
}
