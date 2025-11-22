import React, { useEffect, useState } from "react";
import api, {listModels, pypotsPredictSave} from "../lib/api";
import ModelsTable from "../components/ModelsTable.jsx";
import DatasetPickerModal from "../components/DatasetPickerModal.jsx";
import { llamaZeroShotSave, llamaFinetune, llamaPredictFTSave } from "../lib/api";

const wrap = { padding: 16 };
const msgBox = { marginTop: 12, border: "1px solid #eee", padding: 8, borderRadius: 8 };

export default function Models() {
  const [msg, setMsg] = useState(null);
  const [status, setStatus] = useState(null);
  const [runId, setRunId] = useState(null);

  const [models, setModels] = useState([]);
  const [items, setItems] = useState([]);

  const [picking, setPicking] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [loadingDs, setLoadingDs] = useState(false);
  const [pendingAction, setPendingAction] = useState(null);
  const [okText, setOkText] = useState("Conferma");
  const PYPOTS_DESCRIPTIONS = {
  pattern1_TimesNet: "Vendite annuali di un prodotto in crescita con picchi fissi. (Trend lineare + stagionalità annuale)",
  pattern2_TimesNet: "Traffico a un sito. (Stagionalità: settimanale + annuale)",
  pattern3_TimesNet: "Fatturato con chiusure contabili mensili e picchi ricorrenti in alcuni mesi dell’anno. (Stagionalità mensile + annuale)",
  pattern4_TimesNet: "Crescita utenti di una piattaforma digitale con forte espansione nel tempo e ciclicità stagionale. (Trend esponenziale + stagionalità annuale)",
  pattern5_TimesNet: "Serie di tipo finanziario dove prevale la volatilità e non c’è una stagionalità stabile. (Rumore rosso / alta volatilità)",
};

  useEffect(() => {
    (async () => {
      try {
        const { data } = await listModels();
        setModels(Array.isArray(data) ? data : []);
      } catch {
        setModels([]);
      }
    })();
  }, []);

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
            { key: "ft-save", label: "Predici → salva CSV" },
          ],
          _model: m,
        });
        continue;
      }

      if (kind === "pypots") {
        const base = m.base_model ?? id;
        const modelName = m.name || `PyPOTS (${base})`;

        const extraDesc = PYPOTS_DESCRIPTIONS[modelName];

        const desc =
          extraDesc
            ? `${extraDesc} (L=${m.params_json?.L ?? "?"}, H=${m.params_json?.H ?? "?"})`
            : `(${base}) – L=${m.params_json?.L ?? "?"}, H=${m.params_json?.H ?? "?"}`;

        rows.push({
          key: `pyp:${id}`,
          name: m.name || `PyPOTS (${base})`,
          description: desc,
          actions: [
            { key: "pyp-save", label: "PyPOTS → salva CSV" },
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

    if (actionKey === "ft-train") {
      setOkText("Avvia fine-tuning");
    } else if (actionKey === "pyp-save") {
      setOkText("Avvia PyPOTS");
    } else {
      setOkText("Avvia & salva CSV");
    }

    openPicker();
  };

  const pollRun = async (rid) => {
    const deadline = Date.now() + 12 * 60 * 10000;
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

    const onPickConfirm = async ({datasetIds, horizon, context_len, epochs, modelName}) => {
        setPicking(false);

        const H = Number.isFinite(Number(horizon)) ? parseInt(horizon, 10) : 30;
        const C = Number.isFinite(Number(context_len)) ? parseInt(context_len, 10) : 64;
        const E = Number.isFinite(Number(epochs)) ? parseInt(epochs, 10) : 5;
        const M = modelName || null;

        const a = pendingAction;
        const actionKey = a?.actionKey;

        const datasetId = Array.isArray(datasetIds) && datasetIds.length === 1
            ? datasetIds[0]
            : null;

        const isTrainingAction = actionKey === "auto-train" || actionKey === "ft-train";
        const isPredictionAction = actionKey === "zz-save" || actionKey === "ft-save" || actionKey === "pyp-save";

        try {
            setRunId(null);
            setStatus("PENDING");

            if (!a) {
                setStatus("FAILURE");
                setMsg("Azione non riconosciuta.");
                return;
            }


            if (isPredictionAction && datasetIds.length !== 1) {
                setStatus("FAILURE");
                setMsg(`Per la previsione devi selezionare ESATTAMENTE un dataset.`);
                return;
            }

            if (isTrainingAction && datasetIds.length === 0) {
                setStatus("FAILURE");
                setMsg(`Per l'addestramento devi selezionare almeno un dataset.`);
                return;
            }

            if (a.modelKey === "autots" && isTrainingAction) {

                setMsg(`Auto-addestramento multiserie su ${datasetIds.length} serie in esecuzione…`);

                const {data} = await api.post("/train", {
                    dataset_ids: datasetIds,
                    horizon: H,
                    model_name: M,
                });

                const rid = data?.job_id;
                if (!rid) throw new Error("Risposta senza job_id / run_id");
                setRunId(rid);
                await pollRun(rid);
                return;
            }

            if (a.modelKey.startsWith("fm:") && actionKey === "ft-train" && isTrainingAction) {
                const id = a.modelId;
                setMsg(`Avvio del job di fine-tuning su ${datasetIds.length} serie...`);

                const finetunePayload = {
                    dataset_ids: datasetIds,
                    horizon: H,
                    context_len: C,
                    epochs: E,
                    lr: 1e-4,
                    aug_prob: 0.1,
                };

                const {data} = await llamaFinetune(id, finetunePayload);

                const rid = data?.job_id;
                if (!rid) {
                    throw new Error("Risposta dall'API non valida: job_id mancante.");
                }

                setRunId(rid);
                await pollRun(rid);

                try {
                    const {data: reload} = await listModels();
                    setModels(Array.isArray(reload) ? reload : []);
                } catch {
                }

                return;
            }

            if (a.modelKey.startsWith("fm:")) {
                const id = a.modelId;
                if (a.actionKey === "zz-save" && isPredictionAction) {
                    setMsg("Zero-shot → salvataggio CSV…");
                    const {data} = await llamaZeroShotSave({
                        dataset_id: datasetId,
                        horizon: H, context_len: C
                    });

                    const n = data?.rows ? data.rows : "?";
                    const plotId = data?.plot_id ? data.plot_id : "?";

                    setStatus("SUCCESS");
                    setMsg(`Predizione Zero-shot completata. Plot: ${data?.plot_id ? data.plot_id : '?'}`);
                    return;
                }

            }

            if (a.modelKey.startsWith("ft:")) {
                const id = a.modelId;
                if (a.actionKey === "ft-save") {
                    setMsg("Predizione FT → salvataggio CSV…");
                    const { data } = await llamaPredictFTSave(id, {
                        dataset_id: datasetId,
                        horizon: H,
                        context_len: C,
                    });
                    setStatus("SUCCESS");
                    setMsg(`Predizione avviata. Il file CSV verrà salvato nello storage.`);

                    return;
                }
            }

            if (a.modelKey.startsWith("pyp:")) {
                const id = a.modelId;

                if (a.actionKey === "pyp-save" && isPredictionAction) {
                    setMsg("Predizione PyPOTS → salvataggio CSV…");
                    const {data} = await pypotsPredictSave(id, {
                        dataset_id: datasetId,
                        horizon: H,
                    });

                    const n = Array.isArray(data?.rows) ? data.rows.length : "?";
                    setStatus("SUCCESS");
                    setMsg(`Predizione PyPOTS completata. Righe restituite: ${Array.isArray(data?.rows) ? data.rows.length : '?'}`);
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
        loading={loadingDs}
        datasets={datasets}
        onClose={() => setPicking(false)}
        onConfirm={onPickConfirm}
        okText={okText}
        actionKey={pendingAction?.actionKey}
      />
    </div>
  );
}
