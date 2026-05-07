# --- Stage 1: build ---
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- Stage 2: serve ---
FROM nginx:1.27-alpine AS runner

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

# BusyBox wget --spider just sends a HEAD-like request, no body download.
# More portable across nginx:alpine variants than `wget -qO-`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget --spider -q -T 3 http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
