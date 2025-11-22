import React from "react";

const th = { textAlign: "left", padding: "10px 8px", borderBottom: "1px solid #eee", fontWeight: 600 };
const td = { padding: "10px 8px", borderBottom: "1px solid #f3f3f3" };
const btn = { padding: "6px 10px", border: "1px solid #222", borderRadius: 8, background: "#fff", cursor: "pointer" };

export default function ModelsTable({ items = [], onAction }) {
  if (!items.length) {
    return <div style={{ marginTop: 12, border: "1px solid #eee", padding: 12, borderRadius: 8 }}>Nessun modello</div>;
  }
  return (
    <div style={{ marginTop: 16 }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={th}>Modello</th>
            <th style={th}>Descrizione</th>
            <th style={th}>Azioni</th>
          </tr>
        </thead>
        <tbody>
          {items.map(m => (
            <tr key={m.key}>
              <td style={td} width="220">{m.name}</td>
              <td style={td}>{m.description}</td>
              <td style={td} width="320">
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {m.actions?.map(a => (
                    <button key={a.key} style={btn} onClick={() => onAction?.(m.key, a.key)}>
                      {a.label}
                    </button>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
