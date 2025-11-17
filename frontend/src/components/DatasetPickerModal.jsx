import React, { useMemo, useState } from "react";

const overlay = { position: "fixed", inset: 0, background: "rgba(0,0,0,.25)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 20 };
const card    = { background: "#fff", borderRadius: 12, padding: 16, width: 720, maxHeight: "80vh", overflow: "auto", boxShadow: "0 10px 30px rgba(0,0,0,.2)" };
const th      = { textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #eee", fontWeight: 600 };
const td      = { padding: "8px 6px", borderBottom: "1px solid #f5f5f5" };

function NumberField({ label, value, onChange, min = 1, step, width = 140 }) {
  const parse = (v) => {
    if (v === "" || v === null || v === undefined) return "";
    const n = step && String(step).includes(".") ? parseFloat(v) : parseInt(v, 10);
    return Number.isFinite(n) ? n : "";
  };
  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontSize: 12, color: "#444" }}>{label}</label>
      <input
        type="number"
        min={min}
        step={step}
        value={value}
        onChange={(e) => onChange(parse(e.target.value))}
        style={{ marginLeft: 8, width }}
      />
    </div>
  );
}

/**
 * actionKey decide i campi:
 *  - "zz-save" → horizon + context_len
 *  - "ft-train" → epochs (solo)
 *  - "ft-save" → horizon + context_len
 */
export default function DatasetPickerModal({
  open,
  loading,
  datasets = [],
  onClose,
  onConfirm,
  okText,
  actionKey = "zz-save",
}) {
  const [selected, setSelected] = useState(null);
  const [horizon, setHorizon] = useState(30);
  const [contextLen, setContextLen] = useState(64);
  const [epochs, setEpochs] = useState(5);

  const show = useMemo(() => {
    if (actionKey === "ft-train") return { hor: false, ctx: false, ep: true };
    if (actionKey === "zz-save")return { hor: true, ctx: true, ep: false };
    if (actionKey === "pyp-save")return { hor: true, ctx: false, ep: false };
    return { hor: true, ctx: false, ep: false } // zz-save & ft-save
  }, [actionKey]);

  const title = useMemo(() => {
    if (actionKey === "ft-train") return "Seleziona dataset (Fine-tuning)";
    if (actionKey === "ft-save")  return "Seleziona dataset (FT → salva CSV)";
    return "Seleziona dataset (Zero-shot → salva CSV)";
  }, [actionKey]);

  const fmt = (iso) => (iso ? new Date(iso).toLocaleString() : "-");
  if (!open) return null;

  const pInt = (v, def) => (Number.isFinite(Number(v)) ? parseInt(v, 10) : def);

  const handleConfirm = () => {
    if (!selected) return;
    if (actionKey === "ft-train") {
      onConfirm?.({ datasetId: selected, epochs: Math.max(1, pInt(epochs, 5)) });
    } else {
      const H = pInt(horizon, 30);
      const C = Math.max(H, pInt(contextLen, 64)); // difesa
      onConfirm?.({ datasetId: selected, horizon: H, context_len: C });
    }
  };

  return (
    <div style={overlay} onClick={onClose}>
      <div style={card} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>{title}</h3>

        {loading ? (
          <p>Caricamento…</p>
        ) : (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0,1fr))", gap: 12, marginBottom: 12 }}>
              {show.hor && <NumberField label="Horizon" value={horizon} onChange={setHorizon} min={1} />}
              {show.ctx && <NumberField label="Context length" value={contextLen} onChange={setContextLen} min={horizon || 1} />}
              {show.ep  && <NumberField label="Epochs" value={epochs} onChange={setEpochs} min={1} />}
            </div>

            <div style={{ border: "1px solid #eee", borderRadius: 8, overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th style={th}></th>
                    <th style={th}>Nome</th>
                    <th style={th}>Creato</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((d) => (
                    <tr key={d.id}>
                      <td style={td} width="40">
                        <input type="radio" name="ds" checked={selected === d.id} onChange={() => setSelected(d.id)} />
                      </td>
                      <td style={td}>{d.name ?? d.path ?? d.id}</td>
                      <td style={td}>{fmt(d.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 12 }}>
              <button onClick={onClose}>Annulla</button>
              <button
                onClick={handleConfirm}
                disabled={!selected}
                style={{ padding: "6px 12px", border: "1px solid #222", borderRadius: 8, background: "#fff", cursor: "pointer" }}
              >
                {okText ?? "Conferma"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
