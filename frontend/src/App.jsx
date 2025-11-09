// src/App.jsx
import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import ProtectedRoute from "./routes/ProtectedRoute.jsx";
import ProtectedShell from "./layouts/ProtectedShell.jsx";
import Datasets from "./pages/Datasets.jsx";
import Plots from "./pages/Plots.jsx";
import Login from "./pages/Login.jsx";
import Register from "./pages/Register.jsx";

export default function App() {
  return (
    <Routes>
      <Route
        path="/datasets"
        element={
          <ProtectedRoute>
            <ProtectedShell>
              <Datasets />
            </ProtectedShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/plots"
        element={
          <ProtectedRoute>
            <ProtectedShell>
              <Plots />
            </ProtectedShell>
          </ProtectedRoute>
        }
      />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="*" element={<Navigate to="/datasets" replace />} />
    </Routes>
  );
}
