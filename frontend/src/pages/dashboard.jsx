import React, { useEffect, useState } from "react";
import api from "../lib/api.js";
import { useAuth } from "../context/AuthContext.jsx";
import DatasetsTable from "../components/DatasetsTable.jsx";
import UploadDataset from "../components/UploadDataset.jsx";

export default function Dashboard() {
  const { logout, user } = useAuth();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState(null);
  const [showUpload, setShowUpload] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const { data } = await api.get("/datasets");
      setItems(Array.isArray(data) ? data : []);
    } catch (e) {
      setMsg(e?.response?.data?.detail ?? "Errore nel caricamento");
    } finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const onClean = async (id) => {
    setMsg(null);
    try { await api.post(`/datasets/${id}/clean-outliers`); setMsg("Pulizia outlier completata"); load(); }
    catch (e) { setMsg(e?.response?.data?.detail ?? "Errore pulizia outlier"); }
  };

  const onImpute = async (id) => {
    setMsg(null);
    try { await api.post(`/datasets/${id}/impute-linear`); setMsg("Imputazione completata"); load(); }
    catch (e) { setMsg(e?.response?.data?.detail ?? "Errore imputazione"); }
  };

  return (
    <div style={{ padding:16 }}>
      <header style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <h1>Dashboard</h1>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <button onClick={() => setShowUpload(v => !v)}>
            {showUpload ? "Chiudi upload" : "Carica dataset"}
          </button>
          <span style={{ fontSize:12, color:"#666" }}>{user?.email}</span>
          <button onClick={logout}>Logout</button>
        </div>
      </header>

      {showUpload && (
        <div style={{ marginTop:12 }}>
          <UploadDataset onDone={() => { setShowUpload(false); load(); }} onCancel={() => setShowUpload(false)} />
        </div>
      )}

      {msg && <div style={{ marginTop:12, border:"1px solid #eee", padding:8, borderRadius:8 }}>{msg}</div>}

      {loading
        ? <p>Caricamento…</p>
        : <DatasetsTable items={items} onClean={onClean} onImpute={onImpute} />}
    </div>
  );
}
