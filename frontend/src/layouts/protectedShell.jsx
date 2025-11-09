// src/layouts/ProtectedShell.jsx
import React from "react";
import SidebarNav from "../components/SidebarNav.jsx";

export default function ProtectedShell({ children }) {
  const pages = [
    { path: "/datasets", label: "Datasets" },
    { path: "/plots", label: "Plots" },
      { path: "/models", label: "Models" },
  ];

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>

      {/* ✅ BARRA LATERALE FISSA */}
      <aside
        style={{
          width: "260px",
          background: "#ffffff",
          borderRight: "1px solid #e5e7eb",
          padding: "16px 0",
          flexShrink: 0,
        }}
      >
        <SidebarNav items={pages} />
      </aside>

      {/* ✅ CONTENUTO */}
      <main
        style={{
          flex: 1,
          background: "#f8fafc",
          padding: "24px",
        }}
      >
        {children}
      </main>

    </div>
  );
}
