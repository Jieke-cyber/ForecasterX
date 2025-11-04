import React, { useState } from "react";
import api from "../lib/api.js";

export default function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [ok, setOk] = useState(false);
  const [err, setErr] = useState(null);

  const onSubmit = async (e) => {
    e.preventDefault(); setErr(null);
    try {
      await api.post("/auth/register", { email, password });
      setOk(true);
    } catch (e) {
      setErr(e?.response?.data?.detail ?? "Registrazione fallita");
    }
  };

  if (ok) {
    return (
      <div style={{minHeight:"100vh", display:"grid", placeItems:"center", padding:24}}>
        <div style={{width:320, display:"grid", gap:12, border:"1px solid #ddd", padding:16, borderRadius:12}}>
          <h1 style={{margin:0}}>Registrazione riuscita 🎉</h1>
          <a href="/login">Vai al login</a>
        </div>
      </div>
    );
  }

  return (
    <div style={{minHeight:"100vh", display:"grid", placeItems:"center", padding:24}}>
      <form onSubmit={onSubmit} style={{width:320, display:"grid", gap:12, border:"1px solid #ddd", padding:16, borderRadius:12}}>
        <h1 style={{margin:0}}>Registrati</h1>
        <input placeholder="Email" value={email} onChange={(e)=>setEmail(e.target.value)} />
        <input placeholder="Password" type="password" value={password} onChange={(e)=>setPassword(e.target.value)} />
        {err && <small style={{color:"crimson"}}>{err}</small>}
        <button>Crea account</button>
      </form>
    </div>
  );
}
