import React from "react";

const th = { textAlign: "left", padding: "10px 8px", borderBottom: "1px solid #eee", fontWeight: 600 };
const td = { padding: "10px 8px", borderBottom: "1px solid #f3f3f3" };
const btn = { padding: "6px 10px", border: "1px solid #222", borderRadius: 8, background: "#fff", cursor: "pointer" };

export default function DatasetsTable({ items = [], onPlot, onClean, onImpute, onDelete }) {
  if (!items.length) {
    return (
      <div style={{ marginTop: 12, border: "1px solid #eee", padding: 12, borderRadius: 8 }}>
        Nessun dataset
      </div>
    );
  }

  const fmt = (iso) => (iso ? new Date(iso).toLocaleString() : "-");

  return (
    <div style={{ marginTop: 16 }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={th}>Nome</th>
            <th style={th}>Creato</th>
            <th style={th}>Azioni</th>
          </tr>
        </thead>
        <tbody>
          {items.map((r) => (
            <tr key={r.id ?? r.path /* id usato solo come key, non mostrato */}>
              <td style={td}>{r.name ?? "-"}</td>
              <td style={td}>{fmt(r.created_at)}</td>
              <td style={td}>
                <div style={{display: "flex", gap: 8}}>
                  <button style={btn} onClick={() => onPlot?.(r.id)}>Plottaggio</button>
                  <button style={btn} onClick={() => onClean?.(r.id)}>Pulizia outlier</button>
                  <button style={btn} onClick={() => onImpute?.(r.id)}>Imputazione</button>
                  <button style={btn} onClick={() => onDelete?.(r.id)}>Elimina</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
