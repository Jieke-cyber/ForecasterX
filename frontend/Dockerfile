# frontend/Dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

# Valori di default per l'ambiente Docker (puoi cambiarli da compose)
ARG VITE_API_BASE=http://backend:8000
ARG VITE_GRAFANA_BASE=http://localhost:3001
ARG VITE_GRAFANA_DASH_UID=ef3yl2hyoiscgb
ARG VITE_GRAFANA_DASH_SLUG=forecast-viewer

ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_GRAFANA_BASE=$VITE_GRAFANA_BASE
ENV VITE_GRAFANA_DASH_UID=$VITE_GRAFANA_DASH_UID
ENV VITE_GRAFANA_DASH_SLUG=$VITE_GRAFANA_DASH_SLUG

# Genera .env.production per la build (senza creare file nel repo)
RUN printf "VITE_API_BASE=%s\nVITE_GRAFANA_BASE=%s\nVITE_GRAFANA_DASH_UID=%s\nVITE_GRAFANA_DASH_SLUG=%s\n" \
  "$VITE_API_BASE" "$VITE_GRAFANA_BASE" "$VITE_GRAFANA_DASH_UID" "$VITE_GRAFANA_DASH_SLUG" \
  > .env.production

RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY <<'NGINX' /etc/nginx/conf.d/default.conf
server {
  listen 80;
  server_name _;
  root /usr/share/nginx/html;
  index index.html;
  location / { try_files $uri $uri/ /index.html; }
}
NGINX
EXPOSE 80
