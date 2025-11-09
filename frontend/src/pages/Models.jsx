// src/pages/Models.jsx
import React, { useState } from "react";
import api from "../lib/api";
import ModelsTable from "../components/ModelsTable.jsx";
import DatasetPickerModal from "../components/DatasetPickerModal.jsx";

const wrap = { padding: 16 };
const msgBox = { marginTop: 12, border: "1px solid #eee", padding: 8, borderRadius: 8 };

export default function Models() {
  const [msg, setMsg] = useState(null);
  const [picking, setPicking] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [loadingDs, setLoadingDs] = useState(false);
  const [status, setStatus] = useState(null);   // 'PENDING' | 'RUNNING' | 'SUCCESS' | 'FAILURE' | null
    const [runId, setRunId]   = useState(null);

  const items = [
    {
      key: "autots",
      name: "AutoTS",
      description: "Auto addestramento e previsione automatica.",
      actions: [{ key: "auto-train", label: "Predizione (auto-train)" }],
    },
  ];

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

  const openPicker = async () => {
    try {
      setMsg(null);
      setLoadingDs(true);
      const { data } = await api.get("/datasets");
      setDatasets(Array.isArray(data) ? data : []);
      setPicking(true);
    } catch (e) {
      setMsg(asMsg(e, "Errore nel caricamento dei dataset"));
    } finally {
      setLoadingDs(false);
    }
  };

  const onAction = (modelKey, actionKey) => {
    if (modelKey === "autots" && actionKey === "auto-train") {
      openPicker();
    }
  };

  // Lancia /train e fa un polling breve su /train/{id} per aprire Grafana al termine
  const onPickConfirm = async ({ datasetId, horizon }) => {
  setPicking(false);

  // messaggio iniziale "pinnato"
  setStatus('PENDING');
  setMsg('Auto-addestramento e predizione in esecuzione…');
  setRunId(null);

  try {
    // 1) avvia job
    const { data } = await api.post("/train", {
      dataset_id: datasetId,
      horizon: horizon ?? 30,
    });
    const rid = data?.job_id;
    if (!rid) throw new Error("Risposta senza job_id");
    setRunId(rid);

    // 2) polling su /train/{runId}
    const pollOnce = async () => {
      const r = await api.get(`/train/${rid}`);
      return r.data; // { status, plot_id, error }
    };

    const deadline = Date.now() + 4 * 60 * 10000;
    while (Date.now() < deadline) {
      const st = await pollOnce();

      // aggiorna banner "pinnato" in base allo stato corrente
      if (st.status === 'PENDING' || st.status === 'RUNNING') {
        setStatus(st.status);
        setMsg(`Job ${rid}: auto-addestramento e predizione in esecuzione…`);
      }

      if (st.status === 'FAILURE') {
        setStatus('FAILURE');
        setMsg(`Job ${rid}: processo fallito. ${st.error ? `Dettagli: ${st.error}` : ''}`);
        return;
      }

      if (st.status === 'SUCCESS' && st.plot_id) {
        setStatus('SUCCESS');
        setMsg(`Job ${rid}: processo terminato con successo. Plot: ${st.plot_id}`);
        // opzionale: apri Grafana qui se vuoi
        // const GRAFANA = import.meta.env.VITE_GRAFANA_BASE;
        // const UID     = import.meta.env.VITE_GRAFANA_DASH_UID;
        // const SLUG    = import.meta.env.VITE_GRAFANA_DASH_SLUG;
        // const API     = import.meta.env.VITE_API_BASE;
        // const csvUrl  = `${API}/public/plots/forecast/${st.plot_id}/csv`;
        // window.open(`${GRAFANA}/d/${UID}/${SLUG}?var-csv_url=${encodeURIComponent(csvUrl)}`, "_blank");
        return;
      }

      await new Promise(r => setTimeout(r, 2500));
    }

    // timeout: lasciamo il banner spiegando lo stato
    setStatus('FAILURE');
    setMsg(`Job ${rid}: timeout in attesa del risultato. Riprova a verificare tra poco.`);
  } catch (e) {
    setStatus('FAILURE');
    const err = asMsg(e, "Errore durante l’avvio dell’auto-train");
    setMsg(err);
  }
};


  return (
    <div style={wrap}>
      <h1>Modelli</h1>
     {msg && (
  <div
    style={{
      ...msgBox,
      borderColor:
        status === 'FAILURE' ? '#e00'
        : status === 'SUCCESS' ? '#0a0'
        : '#eee',
      background:
        status === 'FAILURE' ? '#ffecec'
        : status === 'SUCCESS' ? '#f0fff0'
        : '#fafafa',
    }}
  >
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span>
        {(status === 'PENDING' || status === 'RUNNING') ? '⏳ ' : ''}
        {status === 'SUCCESS' ? '✅ ' : ''}
        {status === 'FAILURE' ? '❌ ' : ''}
        {msg}
      </span>
      {!(status === 'PENDING' || status === 'RUNNING') && (
        <button onClick={() => { setMsg(null); setStatus(null); setRunId(null); }} style={{ marginLeft: 8 }}>
          ✕
        </button>
      )}
    </div>
    {runId && (status === 'PENDING' || status === 'RUNNING') && (
      <div style={{ marginTop: 6, fontSize: 12, color: "#666" }}>
        Job ID: {runId}
      </div>
    )}
  </div>
)}

      <ModelsTable items={items} onAction={onAction} />
      <DatasetPickerModal
        open={picking}
        loading={loadingDs}
        datasets={datasets}
        onClose={() => setPicking(false)}
        onConfirm={onPickConfirm}
      />
    </div>
  );
}
