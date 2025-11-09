// src/components/DatasetPickerModal.jsx
import React, { useState } from "react";

const overlay = { position: "fixed", inset: 0, background: "rgba(0,0,0,.25)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 20 };
const card = { background: "#fff", borderRadius: 12, padding: 16, width: 720, maxHeight: "80vh", overflow: "auto", boxShadow: "0 10px 30px rgba(0,0,0,.2)" };
const th = { textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #eee", fontWeight: 600 };
const td = { padding: "8px 6px", borderBottom: "1px solid #f5f5f5" };

export default function DatasetPickerModal({ open, loading, datasets = [], onClose, onConfirm }) {
  const [selected, setSelected] = useState(null);
  const [horizon, setHorizon] = useState(30);

  if (!open) return null;
  const fmt = (iso) => (iso ? new Date(iso).toLocaleString() : "-");

  return (
    <div style={overlay} onClick={onClose}>
      <div style={card} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>Seleziona dataset</h3>
        {loading ? (
          <p>Caricamento…</p>
        ) : (
          <>
            <div style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 12, color: "#444" }}>Orizzonte (giorni)</label>
              <input
                type="number"
                min={1}
                value={horizon}
                onChange={e => setHorizon(parseInt(e.target.value || "0", 10))}
                style={{ marginLeft: 8, width: 100 }}
              />
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
                  {datasets.map(d => (
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
                onClick={() => selected && onConfirm?.({ datasetId: selected, horizon })}
                style={{ padding: "6px 12px", border: "1px solid #222", borderRadius: 8, background: "#fff", cursor: "pointer" }}
                disabled={!selected}
              >
                Avvia auto-train
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
