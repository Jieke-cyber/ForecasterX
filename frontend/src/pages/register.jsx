import React, { useState } from "react";
import api from "../lib/api.js";
import { useAuth } from "../context/AuthContext.jsx";
import { useNavigate } from "react-router-dom";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function extractErrorMessage(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail[0]?.message || detail[0]?.msg || "Dati non validi.";
  }
  return "Registrazione fallita.";
}

export default function Register() {
  const { loginWithToken } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState(null);
  const [ok, setOk] = useState(false);
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
      // BACKEND: JSON { email, password }
      const res = await api.post("/auth/register", { email: emailNorm, password });

      // Auto-login se /auth/register ritorna un token
      const tk =
        res.data?.access_token ??
        res.data?.token ??
        (typeof res.data === "string" ? res.data : null);

      if (tk) {
        loginWithToken(tk);
        return;
      }

      setOk(true); // altrimenti mostra conferma e vai al login
    } catch (e) {
      setErr(extractErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  if (ok) {
    return (
      <div className="center-screen" style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
        <div style={{ width: 320, display: "grid", gap: 12, border: "1px solid #ddd", padding: 16, borderRadius: 12 }}>
          <h1 style={{ margin: 0 }}>Registrazione riuscita 🎉</h1>
          <button onClick={() => navigate("/login")}>Vai al login</button>
        </div>
      </div>
    );
  }

  return (
    <div className="center-screen" style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
      {/* 👇 disattiva la validazione nativa del browser */}
      <form noValidate onSubmit={onSubmit} style={{ width: 320, display: "grid", gap: 12, border: "1px solid #ddd", padding: 16, borderRadius: 12 }}>
        <h1 style={{ margin: 0 }}>Registrati</h1>

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

        <button disabled={loading}>{loading ? "…" : "Crea account"}</button>
      </form>
    </div>
  );
}
