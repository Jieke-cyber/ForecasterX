import React, { useEffect, useState } from "react";
import api from "../lib/api.js";
import { useAuth } from "../context/AuthContext.jsx";
import JobsTable from "../components/JobsTable.jsx";


export default function Jobs() {
  const { user } = useAuth();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState(null);

  const load = async () => {
    setLoading(true);
    try {
      const { data } = await api.get("/jobs");
      setItems(Array.isArray(data) ? data : []);
    } catch (e) {
      setMsg(e?.response?.data?.detail ?? "Errore nel caricamento");
    } finally { setLoading(false); }
  };

  const asMsg = (e, fallback) => {
  const d = e?.response?.data;
  const detail = d?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) return detail[0]?.message || detail[0]?.msg || fallback;
  if (typeof d === "string") return d;
  if (typeof detail === "object") return JSON.stringify(detail);
  if (typeof d === "object") return JSON.stringify(d);
  return fallback;
};
  useEffect(() => { load(); }, []);


const onUpdate = async (id) => {
  if (!id) return setMsg("ID mancante.");
  setMsg(null);
  try {
    await api.get(`/jobs/${id}/status`); // o /impute
    setMsg("Stato aggiornato");
    load();
  } catch (e) {
    setMsg(asMsg(e, "Errore aggiornamento stato"));
  }
};

const onDelete = async (id) => {
  if (!id) return setMsg("ID mancante.");
  setMsg(null);
  try {
    await api.post(`/train/${id}/delete`); // o /impute
    setMsg("Eliminazione completata");
    load();
  } catch (e) {
    setMsg(asMsg(e, "Errore eliminazione"));
  }
};


   return (
    <div style={{ padding: 16 }}>
      <header style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <h1>Jobs</h1>
        <span style={{ fontSize: 12, color: "#666" }}>{user?.email}</span>
      </header>

      {msg && <div style={{ marginTop: 12, border: "1px solid #eee", padding: 8, borderRadius: 8 }}>{msg}</div>}

      {loading
        ? <p>Caricamento…</p>
        : <JobsTable items={items} onUpdate={onUpdate} onDelete={onDelete} />}
    </div>
  );
}
