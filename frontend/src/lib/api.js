console.log("API base:", import.meta.env.VITE_API_BASE);

import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || "http://localhost:8000",
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export default api;

export const listModels = () => api.get("/models");

export const llamaZeroShotSave = (body) => api.post("/lag-llama/predict/save", body);
export const llamaFinetune     = (id ,body) => api.post(`/lag-llama/${id}/finetune`, body);

export const llamaPredictFTSave = (id, body) => api.post(`/lag-llama-ft/${id}/predict/save`, body);

export const pypotsPredictSave = (modelId, payload) =>
  api.post(`/pypots/${modelId}/forecast-csv`, payload);
