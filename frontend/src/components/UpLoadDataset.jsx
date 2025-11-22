import React, { useState } from "react";
import api from "../lib/api.js";

export default function UpLoadDataset({ onDone, onCancel }) {
  const [file, setFile] = useState(null);
  const [msg, setMsg] = useState(null);
  const [loading, setLoading] = useState(false);
  const btn = { padding: "6px 10px", border: "1px solid #222", borderRadius: 8, background: "#fff", cursor: "pointer" };

  const onSubmit = async (e) => {
    e.preventDefault();
    setMsg(null);
    if (!file) { setMsg("Seleziona un file .csv"); return; }
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setMsg("Il file deve essere .csv"); return;
    }


    const form = new FormData();
    form.append("file", file);

    setLoading(true);
    try {
      await api.post("/datasets/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMsg("Caricato con successo.");
      onDone?.();
    } catch (e) {
      setMsg(e?.response?.data?.detail ?? "Upload fallito");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={onSubmit} style={{ display:"grid", gap:12, border:"1px solid #eee", padding:12, borderRadius:8 }}>
      <strong>Carica dataset (.csv)(Assicurare che il dataset abbia come la prima e la seconda colonna nominato "ds" e "value" rispettivamente)</strong>
      <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      {msg && <small style={{ color: msg.includes("successo") ? "green" : "crimson" }}>{msg}</small>}
      <div style={{ display:"flex", gap:8 }}>
        <button style={btn} type="submit" disabled={loading}>{loading ? "Carico…" : "Carica"}</button>
        {onCancel && <button style={btn} type="button" onClick={onCancel}>Annulla</button>}
      </div>
    </form>
  );
}
