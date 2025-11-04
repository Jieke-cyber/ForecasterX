import React from "react";

export default function DatasetsTable({ items, onClean, onImpute }) {
  return (
    <div style={{overflowX:"auto", border:"1px solid #eee", borderRadius:12}}>
      <table style={{minWidth:600, width:"100%", borderCollapse:"collapse"}}>
        <thead>
          <tr>
            <th style={{textAlign:"left", padding:8}}>ID</th>
            <th style={{textAlign:"left", padding:8}}>Nome</th>
            <th style={{textAlign:"left", padding:8}}>Creato</th>
            <th style={{textAlign:"left", padding:8}}>Azioni</th>
          </tr>
        </thead>
        <tbody>
          {items.map(d => (
            <tr key={d.id} style={{borderTop:"1px solid #eee"}}>
              <td style={{padding:8}}>{d.id}</td>
              <td style={{padding:8}}>{d.name ?? "—"}</td>
              <td style={{padding:8}}>{d.created_at ? new Date(d.created_at).toLocaleString() : "—"}</td>
              <td style={{padding:8}}>
                <button onClick={()=>onClean(d.id)}>Clean outliers</button>
                <button onClick={()=>onImpute(d.id)} style={{marginLeft:8}}>Impute</button>
              </td>
            </tr>
          ))}
          {items.length === 0 && (
            <tr><td style={{padding:8}} colSpan="4">Nessun dataset</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
