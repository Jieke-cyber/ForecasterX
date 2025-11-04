import React, { useState } from "react";
import api from "../lib/api.js";
import { useAuth } from "../context/AuthContext.jsx";

export default function Login() {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setErr(null);
    try {
      const { data } = await api.post("/auth/login", { email, password });
      // compat: data.access_token oppure data.token
      login(data?.access_token ?? data?.token ?? data);
    } catch (e) {
      setErr(e?.response?.data?.detail ?? "Login fallito");
    } finally { setLoading(false); }
  };

  return (
    <div style={{minHeight:"100vh", display:"grid", placeItems:"center", padding:24}}>
      <form onSubmit={onSubmit} style={{width:320, display:"grid", gap:12, border:"1px solid #ddd", padding:16, borderRadius:12}}>
        <h1 style={{margin:0}}>Accedi</h1>
        <input placeholder="Email" value={email} onChange={(e)=>setEmail(e.target.value)} />
        <input placeholder="Password" type="password" value={password} onChange={(e)=>setPassword(e.target.value)} />
        {err && <small style={{color:"crimson"}}>{err}</small>}
        <button disabled={loading}>{loading ? "…" : "Login"}</button>
        <a href="/register">Non hai un account? Registrati</a>
      </form>
    </div>
  );
}
