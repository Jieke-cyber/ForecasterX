console.log("API base:", import.meta.env.VITE_API_BASE);

import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export default api;

export const listModels        = () => api.get("/models"); // backend: ritorna rows tabella models

// Lag-Llama foundation
export const llamaZeroShot     = (body)      => api.post("/lag-llama/predict", body);
export const llamaZeroShotSave = (body)      => api.post("/lag-llama/predict/save", body);
export const llamaFinetune     = (body)      => api.post("/lag-llama/finetune", body);

// Fine-tuned
export const llamaPredictFT       = (id, body) => api.post(`/models/${id}/predict`, body);
export const llamaPredictFTSave   = (id, body) => api.post(`/models/${id}/predict/save`, body);