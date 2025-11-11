import React, { useEffect, useState } from "react";
import api, { listModels } from "../lib/api";
import ModelsTable from "../components/ModelsTable.jsx";
import DatasetPickerModal from "../components/DatasetPickerModal.jsx";
import { llamaZeroShotSave, llamaFinetune, llamaPredictFT, llamaPredictFTSave } from "../lib/api";

const wrap = { padding: 16 };
const msgBox = { marginTop: 12, border: "1px solid #eee", padding: 8, borderRadius: 8 };

export default function Models() {
  const [msg, setMsg] = useState(null);
  const [status, setStatus] = useState(null);
  const [runId, setRunId] = useState(null);

  const [models, setModels] = useState([]);
  const [items, setItems] = useState([]);

  // dataset picker
  const [picking, setPicking] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [loadingDs, setLoadingDs] = useState(false);
  const [pendingAction, setPendingAction] = useState(null); // { modelKey, actionKey, modelId? }
  const [modalMode, setModalMode] = useState("zeroShotSave"); // "zeroShotSave" | "fineTune"

  // 1) carica /models all’avvio
  useEffect(() => {
    (async () => {
      try {
        const { data } = await listModels();
        const arr = Array.isArray(data) ? data : [];
        setModels(arr);
      } catch (e) {
        console.warn("GET /models failed:", e?.response?.status, e?.response?.data);
        setModels([]);
      }
    })();
  }, []);

  // 2) costruisci righe tabella
  useEffect(() => {
    const rows = [];
    rows.push({
      key: "autots",
      name: "AutoTS",
      description: "Addestramento e previsione automatica.",
      actions: [{ key: "auto-train", label: "Predizione (auto-train)" }],
    });

    const norm = (s) => (s || "").toString().trim().toLowerCase();
    const other = [];

    for (const m of models) {
      const kind = norm(m.kind);
      const base = norm(m.base_model);
      const id = m.id;

      if (kind === "foundation" && base === "lag-llama") {
        rows.push({
          key: `fm:${id}`,
          name: "Lag-Llama (Foundation)",
          description: "Modello foundation per zero-shot o come base del fine-tuning.",
          actions: [
            { key: "zz-save",  label: "Zero-shot → salva CSV" },
            { key: "ft-train", label: "Fine-tuning" },
          ],
          _model: m,
        });
        continue;
      }

      if (kind === "fine_tuned" && base === "lag-llama") {
        rows.push({
          key: `ft:${id}`,
          name: m.name || `Lag-Llama FT (${String(id).slice(0, 8)})`,
          description: "Modello fine-tunato su dataset specifici.",
          actions: [
            { key: "ft-preview", label: "Predici (preview)" },
            { key: "ft-save",    label: "Predici → salva CSV" },
          ],
          _model: m,
        });
        continue;
      }

      other.push(m);
    }

    for (const m of other) {
      rows.push({
        key: `other:${m.id}`,
        name: m.name || `Modello (${String(m.id).slice(0, 8)})`,
        description: `kind=${m.kind ?? "?"} base_model=${m.base_model ?? "?"}`,
        actions: [],
        _model: m,
      });
    }

    setItems(rows);
  }, [models]);

  // helpers
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
    const row = items.find((r) => r.key === modelKey);
    setPendingAction({ modelKey, actionKey, modelId: row?._model?.id });
    // scegli campi del modal
    if (actionKey === "ft-train") setModalMode("fineTune");
    else setModalMode("zeroShotSave");
    openPicker();
  };

  const pollRun = async (rid) => {
    const deadline = Date.now() + 4 * 60 * 1000;
    while (Date.now() < deadline) {
      const { data: st } = await api.get(`/train/${rid}`);
      if (st.status === "PENDING" || st.status === "RUNNING") {
        setStatus(st.status);
        setMsg(`Job ${rid}: in esecuzione…`);
        await new Promise((r) => setTimeout(r, 2500));
        continue;
      }
      if (st.status === "FAILURE") {
        setStatus("FAILURE");
        setMsg(`Job ${rid}: fallito. ${st.error ? `Dettagli: ${st.error}` : ""}`);
        return;
      }
      if (st.status === "SUCCESS") {
        setStatus("SUCCESS");
        setMsg(`Job ${rid}: completato. Plot: ${st.plot_id}`);
        return;
      }
    }
    setStatus("FAILURE");
    setMsg(`Job ${rid}: timeout in attesa del risultato.`);
  };

  const onPickConfirm = async (payload) => {
    setPicking(false);
    const { datasetId, horizon, context_len, epochs } = payload || {};

    const H = Number.isFinite(+horizon) ? parseInt(horizon, 10) : 30;
    const C = Number.isFinite(+context_len) ? parseInt(context_len, 10) : 512;
    const E = Number.isFinite(+epochs) ? parseInt(epochs, 10) : 1;

    try {
      setRunId(null);
      setStatus("PENDING");

      const a = pendingAction;
      if (!a) {
        setStatus("FAILURE");
        setMsg("Azione non riconosciuta.");
        return;
      }

      // AutoTS
      if (a.modelKey === "autots" && a.actionKey === "auto-train") {
        setMsg("AutoTS in esecuzione…");
        const { data } = await api.post("/train", { dataset_id: datasetId, horizon: H });
        const rid = data?.job_id;
        if (!rid) throw new Error("Risposta senza job_id");
        setRunId(rid);
        await pollRun(rid);
        return;
      }

      // Lag-Llama FOUNDATION
      if (a.modelKey.startsWith("fm:")) {
        if (a.actionKey === "zz-save") {
          setMsg("Zero-shot → salvataggio CSV…");
          const { data } = await llamaZeroShotSave({ dataset_id: datasetId, horizon: H, context_len: C });
          const rid = data?.run_id;
          if (!rid) throw new Error("Risposta senza run_id");
          setRunId(rid);
          await pollRun(rid);
          return;
        }
        if (a.actionKey === "ft-train") {
          setMsg("Fine-tuning in esecuzione…");
          const { data } = await llamaFinetune({ dataset_id: datasetId, epochs: E });
          setStatus("SUCCESS");
          setMsg(`Fine-tuning completato. Nuovo modello: ${data?.model_id || "?"}`);
          // opzionale: ricarica lista modelli
          // const r = await listModels(); setModels(Array.isArray(r.data) ? r.data : []);
          return;
        }
      }

      // Lag-Llama FINE-TUNED
      if (a.modelKey.startsWith("ft:")) {
        const id = a.modelId;
        if (a.actionKey === "ft-preview") {
          setMsg("Predizione FT (anteprima)...");
          const { data } = await llamaPredictFT(id, { dataset_id: datasetId, horizon: H, context_len: C });
          setStatus("SUCCESS");
          setMsg(`Anteprima FT: [${(data?.head || []).slice(0, 5).map((v) => Number(v).toFixed(3)).join(", ")}]…`);
          return;
        }
        if (a.actionKey === "ft-save") {
          setMsg("Predizione FT → salvataggio CSV…");
          const { data } = await llamaPredictFTSave(id, { dataset_id: datasetId, horizon: H, context_len: C });
          const rid = data?.run_id;
          if (!rid) throw new Error("Risposta senza run_id");
          setRunId(rid);
          await pollRun(rid);
          return;
        }
      }

      setStatus("FAILURE");
      setMsg("Azione non riconosciuta.");
    } catch (e) {
      setStatus("FAILURE");
      setMsg(asMsg(e, "Errore nell'azione richiesta"));
    } finally {
      setPendingAction(null);
    }
  };

  return (
    <div style={wrap}>
      <h1>Modelli</h1>

      {msg && (
        <div
          style={{
            ...msgBox,
            borderColor: status === "FAILURE" ? "#e00" : status === "SUCCESS" ? "#0a0" : "#eee",
            background: status === "FAILURE" ? "#ffecec" : status === "SUCCESS" ? "#f0fff0" : "#fafafa",
          }}
        >
          {(status === "PENDING" || status === "RUNNING") ? "⏳ " : ""}
          {status === "SUCCESS" ? "✅ " : ""}
          {status === "FAILURE" ? "❌ " : ""}
          {msg}
          {runId && <div style={{ fontSize: 12, color: "#666" }}>Job ID: {runId}</div>}
        </div>
      )}

      <ModelsTable items={items} onAction={onAction} />

      <DatasetPickerModal
        open={picking}
        mode={modalMode}
        loading={loadingDs}
        datasets={datasets}
        onClose={() => setPicking(false)}
        onConfirm={onPickConfirm}
      />
    </div>
  );
}
