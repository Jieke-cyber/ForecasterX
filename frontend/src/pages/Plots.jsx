import React, { useEffect, useState } from "react";
import api from "../lib/api";
import { useAuth } from "../context/AuthContext.jsx";
import PlotsTable from "../components/PlotsTable.jsx";

export default function Plots() {
  const { user } = useAuth();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState(null);

  const load = async () => {
    setLoading(true);
    try {
      const { data } = await api.get("/plots");
      setItems(Array.isArray(data) ? data : []);
    } catch (e) {
      setMsg("Errore nel caricamento");
    } finally {
      setLoading(false);
    }
  };

  const asMsg = (e, fallback) => {
  const d = e?.response?.data;
  const detail = d?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) return detail[0]?.message || detail[0]?.msg || fallback;
  if (typeof d === "string") return d;
  if (typeof detail === "object") return JSON.stringify(detail);
  if (typeof d === "object") return JSON.stringify(d);
  return fallback;}

  useEffect(() => { load(); }, []);

  const onPlot = (id) => {
  const GRAFANA = import.meta.env.VITE_GRAFANA_BASE;
  const UID     = import.meta.env.VITE_GRAFANA_DASH_UID;
  const SLUG    = import.meta.env.VITE_GRAFANA_DASH_SLUG;
  const API     = import.meta.env.VITE_API_BASE;

  const csvUrl = `${API}/public/plots/forecast/${encodeURIComponent(id)}/csv`;
  const qs = new URLSearchParams({ "var-csv_url": csvUrl, from: "now-7d", to: "now" });

  window.open(`${GRAFANA}/d/${UID}/${SLUG}?${qs.toString()}`, "_blank", "noopener,noreferrer");
};

  const onDelete = async (id) => {
  if (!id) return setMsg("ID mancante.");
  setMsg(null);
  try {
    await api.post(`/plots/${id}/delete`);
    setMsg("Eliminazione completata");
    load();
  } catch (e) {
    setMsg(asMsg(e, "Errore eliminazione"));
  }
};


  return (
    <div style={{ padding: 16 }}>
      <header style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <h1>Plots</h1>
        <span style={{ fontSize: 12, color: "#666" }}>{user?.email}</span>
      </header>

      {msg && <div style={{ marginTop: 12, border: "1px solid #eee", padding: 8, borderRadius: 8 }}>{msg}</div>}

      {loading
        ? <p>Caricamento…</p>
        : <PlotsTable items={items} onPlot={onPlot} onDelete={onDelete} />}
    </div>
  );
}
