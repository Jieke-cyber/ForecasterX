// src/components/SidebarNav.jsx
import React from "react";
import { NavLink, useLocation } from "react-router-dom";

export default function SidebarNav({ items = [] }) {
  const location = useLocation();

  return (
    <nav style={{ padding: "0 16px" }}>
      <div
        style={{
          marginBottom: "12px",
          paddingBottom: "8px",
          borderBottom: "1px solid #ddd",
          fontWeight: "600",
          fontSize: "14px",
        }}
      >
        Menu
      </div>

      {items.map((item) => {
        const active = location.pathname.startsWith(item.path);

        return (
          <NavLink
            key={item.path}
            to={item.path}
            style={{
              display: "block",
              padding: "8px 12px",
              borderRadius: "6px",
              marginBottom: "6px",
              textDecoration: "none",
              color: active ? "white" : "#374151",
              background: active ? "#111827" : "transparent",
            }}
          >
            {item.label}
          </NavLink>
        );
      })}

      <div
        style={{
          marginTop: "16px",
          paddingTop: "8px",
          borderTop: "1px solid #ddd",
          fontSize: "12px",
          color: "#9ca3af",
        }}
      >
        v1.0
      </div>
    </nav>
  );
}
