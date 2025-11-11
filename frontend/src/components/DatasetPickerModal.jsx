import React, { useMemo, useState } from "react";

const overlay = { position: "fixed", inset: 0, background: "rgba(0,0,0,.25)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 20 };
const card = { background: "#fff", borderRadius: 12, padding: 16, width: 720, maxHeight: "80vh", overflow: "auto", boxShadow: "0 10px 30px rgba(0,0,0,.2)" };
const th = { textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #eee", fontWeight: 600 };
const td = { padding: "8px 6px", borderBottom: "1px solid #f5f5f5" };

function NumberField({ label, value, onChange, min = 1 }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontSize: 12, color: "#444" }}>{label}</label>
      <input
        type="number"
        min={min}
        value={value}
        onChange={(e) => {
          const v = e.target.value;
          if (v === "") return onChange("");
          const n = parseInt(v, 10);
          onChange(Number.isFinite(n) ? n : 0);
        }}
        style={{ marginLeft: 8, width: 120 }}
      />
    </div>
  );
}

/**
 * mode:
 *  - "zeroShotSave" -> chiede horizon + context_len
 *  - "fineTune"     -> chiede epochs
 */
export default function DatasetPickerModal({
  open,
  mode = "zeroShotSave",
  loading,
  datasets = [],
  onClose,
  onConfirm,
  okText,
}) {
  const [selected, setSelected] = useState(null);
  const [horizon, setHorizon] = useState(30);
  const [contextLen, setContextLen] = useState(512);
  const [epochs, setEpochs] = useState(1);

  const title = useMemo(
    () => (mode === "fineTune" ? "Seleziona dataset (Fine-tuning)" : "Seleziona dataset (Zero-shot → salva CSV)"),
    [mode]
  );

  const fmt = (iso) => (iso ? new Date(iso).toLocaleString() : "-");
  if (!open) return null;

  const handleConfirm = () => {
    if (!selected) return;
    if (mode === "fineTune") {
      const E = Number.isFinite(+epochs) ? parseInt(epochs, 10) : 1;
      onConfirm?.({ datasetId: selected, epochs: Math.max(1, E) });
    } else {
      const H = Number.isFinite(+horizon) ? parseInt(horizon, 10) : 1;
      const C0 = Number.isFinite(+contextLen) ? parseInt(contextLen, 10) : H;
      const C = Math.max(H, C0);
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
            {mode === "zeroShotSave" && (
              <>
                <NumberField label="Orizzonte (giorni)" value={horizon} onChange={setHorizon} min={1} />
                <NumberField label="Context length (giorni)" value={contextLen} onChange={setContextLen} min={horizon || 1} />
              </>
            )}

            {mode === "fineTune" && (
              <NumberField label="Epochs" value={epochs} onChange={setEpochs} min={1} />
            )}

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
