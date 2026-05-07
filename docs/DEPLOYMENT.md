# FurnaceFlow — Deployment & Access Guide

ตารางสรุป port และวิธีเข้าใช้งาน Frontend / Backend / Database
ทั้งโหมดทดสอบบนเครื่อง dev (`local`) และโหมดขึ้น production (`deploy`).

---

## 1. Quick reference — what runs where

### A. Local test mode  (`docker compose --env-file .env.local --profile local up -d`)

| Service | Container | Host port | Internal port | URL |
|---|---|---|---|---|
| **Frontend** (Nginx + React build) | `furnaceflow-frontend` | `${FRONTEND_PORT:-80}` | `80` | http://localhost/ |
| **Backend** (FastAPI / uvicorn) | `furnaceflow-backend`  | *(not published)* | `8000` | reachable only via Nginx → http://localhost/api/... |
| **Database** (Postgres 16-alpine) | `furnaceflow-db`       | `${POSTGRES_HOST_PORT:-5432}` | `5432` | `postgresql://postgres:postgres@localhost:5432/furnaceflow` |

### B. Deploy mode  (`docker compose --env-file .env.deploy up -d`)

| Service | Container | Host port | Internal port | URL |
|---|---|---|---|---|
| **Frontend** (Nginx + React build) | `furnaceflow-frontend` | `${FRONTEND_PORT:-80}` | `80` | http://`<deploy-host-ip>`/ |
| **Backend** (FastAPI / uvicorn) | `furnaceflow-backend`  | *(not published)* | `8000` | reachable only via Nginx → http://`<deploy-host-ip>`/api/... |
| **Database** (existing on-prem Postgres) | *external — not in compose* | `5432` on **192.168.125.10** | — | `postgresql://postgres:PGP@ss!@192.168.125.10:5432/furnaceflow` |

> **หมายเหตุ:** Backend (port 8000) ไม่ได้ publish ออกมาที่ host เพื่อความปลอดภัย — ทุก request จากเบราว์เซอร์จะวิ่งผ่าน Nginx ที่ frontend แล้วถูก proxy ไปที่ `backend:8000` ภายใน docker network.

---

## 2. Frontend (FE)

### Local
- เปิดเบราว์เซอร์ที่ **http://localhost/**
- ถ้า port 80 ชนกับบริการอื่นบนเครื่อง ให้แก้ `FRONTEND_PORT=8080` (หรือเลขอื่น) ใน `.env.local` แล้วเข้า http://localhost:8080/
- หน้า: Dashboard, Production Planning, Operator Execution, Reports, Settings (5 tabs)

### Deploy
- เปิดเบราว์เซอร์ที่ **http://`<deploy-host-ip>`/** (เช่น http://192.168.125.20/ ถ้า deploy server อยู่ที่ IP นั้น)
- ต้องอัปเดต `CORS_ORIGINS` ใน `.env.deploy` ให้ตรงกับ origin ที่ผู้ใช้เปิดจริง ๆ (เช่น `http://192.168.125.20`) ไม่งั้น browser จะถูก block

---

## 3. Backend (BE)

Backend ไม่ publish port 8000 ออกมาตรง ๆ — เข้าผ่าน Nginx proxy เท่านั้น.

| Want to do | Local | Deploy |
|---|---|---|
| Call an API endpoint | http://localhost/api/settings/ | http://`<deploy-host-ip>`/api/settings/ |
| Health check | `docker compose exec backend curl -s http://localhost:8000/health` | same |
| ดู Swagger / OpenAPI docs | ดู §3.1 ด้านล่าง | ดู §3.1 ด้านล่าง |

### 3.1 เปิด Swagger UI (`/docs`) ชั่วคราวเพื่อดู API

Nginx ตอนนี้ proxy เฉพาะ path `/api/...` เท่านั้น ไม่ได้ proxy `/docs` หรือ `/openapi.json` — เพื่อความเรียบง่ายและปลอดภัย. ถ้าต้องการเปิดดู Swagger UI ให้ทำหนึ่งใน 2 วิธี:

**วิธี A — publish backend port ชั่วคราว (เร็วสุด, dev/staging)**

แก้ `docker-compose.yml` ส่วน `backend:` :
```yaml
    # ลบ expose แล้วเปิดเป็น ports
    ports:
      - "8000:8000"
```
แล้ว `docker compose up -d`. หลังจากนั้นเข้าได้ที่:
- http://localhost:8000/docs (local)
- http://`<deploy-host-ip>`:8000/docs (deploy)

**ก่อนขึ้น production จริง ให้เปลี่ยนกลับเป็น `expose: ["8000"]`** เพื่อไม่ให้ FastAPI โดน internet โดยตรง.

**วิธี B — ใช้ docker exec (ไม่ต้องเปิด port)**
```bash
docker compose exec backend curl -s http://localhost:8000/openapi.json | jq .
```

---

## 4. Database (DB)

### A. Local mode — Postgres ใน compose

| Field | Value |
|---|---|
| Host (จากเครื่อง dev) | `localhost` |
| Host (จาก backend container) | `db` |
| Port | `5432` (ปรับด้วย `POSTGRES_HOST_PORT`) |
| User | `postgres` |
| Password | `postgres` |
| Database | `furnaceflow` |
| Connection string | `postgresql+psycopg2://postgres:postgres@localhost:5432/furnaceflow` |

วิธีเข้า DB:

```bash
# ผ่าน psql ใน container
docker compose exec db psql -U postgres -d furnaceflow

# ผ่าน psql จากเครื่อง dev (ต้องมี postgresql-client ติดตั้ง)
psql -h localhost -p 5432 -U postgres -d furnaceflow

# ผ่าน GUI tools (DBeaver / TablePlus / pgAdmin local)
#   Host: localhost · Port: 5432 · User: postgres · Password: postgres · DB: furnaceflow
```

ข้อมูลคงอยู่ใน docker volume ชื่อ `pgdata`. ถ้าต้องการ wipe ทั้งหมด:
```bash
docker compose --env-file .env.local --profile local down -v
```

### B. Deploy mode — on-prem Postgres ที่มีอยู่แล้ว

| Field | Value |
|---|---|
| Host | `192.168.125.10` |
| Port | `5432` |
| User | `postgres` |
| Password | `PGP@ss!`  *(ใน DATABASE_URL ต้อง URL-encode เป็น `PGP%40ss%21`)* |
| Database | `furnaceflow` (ต้องสร้างเองครั้งแรก — ดูด้านล่าง) |

**pgAdmin4 (web UI ของ DB ที่เครือข่ายเดียวกัน):**
- URL: http://192.168.125.10/pgadmin4
- User: `prd@fec-corp.com`
- Password: `eng123456`

**สร้าง database ครั้งแรก** (ทำผ่าน pgAdmin4 หรือ psql):
```sql
CREATE DATABASE furnaceflow OWNER postgres;
```
> Backend จะสร้าง *table* ทั้งหมดให้อัตโนมัติตอน startup (`SQLModel.metadata.create_all`) แต่จะไม่สร้าง *database* — ต้องมีอยู่แล้วก่อน start backend.

วิธีเข้า DB จากเครื่องอื่นในเครือข่าย:
```bash
psql -h 192.168.125.10 -p 5432 -U postgres -d furnaceflow
# กรอก password: PGP@ss!
```

---

## 5. การเปลี่ยน port (ถ้า port 80 / 5432 ชนกัน)

แก้ใน `.env.local` หรือ `.env.deploy`:

```env
FRONTEND_PORT=8080         # เข้าผ่าน http://localhost:8080/
POSTGRES_HOST_PORT=5433    # ใช้ตอนเครื่องมี Postgres ตัวอื่นจอง 5432 อยู่
```

`FRONTEND_PORT` ใช้ได้ทั้ง 2 โหมด · `POSTGRES_HOST_PORT` ใช้เฉพาะ local (deploy ใช้ DB ภายนอก).

---

## 6. Verify ว่าทุกอย่างขึ้นถูกต้อง

```bash
# ดู status ของทุก container
docker compose ps

# ดู log สดของ backend
docker compose logs -f backend
# คาดว่าจะเห็น: "Application startup complete."

# ทดสอบเรียก API ผ่าน Nginx proxy
curl http://localhost/api/settings/
# → ควรได้ JSON list ~50 rows ที่ถูก seed ตอน startup

# ตรวจ schema ใน DB (local)
docker compose exec db psql -U postgres -d furnaceflow -c "\dt"
# ควรเห็น tables: plans, batches, energy_logs, settings, plant_load_profile

# ตรวจ schema ใน DB (deploy) — ผ่าน pgAdmin4 ที่ http://192.168.125.10/pgadmin4
```

---

## 7. ปิด / รีสตาร์ท / ลบ stack

> ทุกคำสั่งต้องใส่ `--env-file` ให้ตรงโหมด ไม่งั้น guard ใน `backend/main.py` จะ block.

### A. Local mode

```bash
# หยุดและลบ container (ข้อมูล DB ใน volume pgdata ยังอยู่ครบ)  ← ปกติใช้อันนี้
docker compose --env-file .env.local --profile local down

# แค่ pause container ไว้ (เร็วกว่า down → up เพราะไม่ต้อง recreate)
docker compose --env-file .env.local --profile local stop
docker compose --env-file .env.local --profile local start    # กลับมาเปิด

# ปิด + ลบ volume pgdata ด้วย (⚠️ ลบข้อมูลทั้ง DB ใน compose ทิ้ง)
docker compose --env-file .env.local --profile local down -v

# ดู logs ขณะรัน
docker compose --env-file .env.local --profile local logs -f backend
docker compose --env-file .env.local --profile local logs -f frontend
```

### B. Deploy mode

```bash
# หยุดและลบ container (DB on-prem ที่ 192.168.125.10 ไม่ถูกแตะต้อง)
docker compose --env-file .env.deploy down

# แค่หยุดไว้
docker compose --env-file .env.deploy stop
docker compose --env-file .env.deploy start

# ⚠️ deploy mode ไม่มี profile local จึงไม่ต้องใส่ --profile
# ⚠️ ห้ามใช้ -v เพราะไม่มี volume ของเรา (DB อยู่นอก compose) แต่จะลบ network ด้วย ก็ไม่กระทบอะไร
```

### C. คำสั่งเฉพาะ service เดียว

```bash
# รีสตาร์ทแค่ backend (เช่นหลังแก้โค้ดแล้ว rebuild)
docker compose --env-file .env.local --profile local up -d --build --force-recreate backend

# หยุดแค่ frontend
docker compose --env-file .env.local --profile local stop frontend
```

### Quick reference

| ต้องการ | คำสั่ง |
|---|---|
| ปิด app ทั้งหมด (เก็บข้อมูล DB) | `docker compose --env-file <env> [--profile local] down` |
| pause ชั่วคราว | `... stop` |
| เปิดกลับมา | `... up -d` หรือ `... start` |
| ปิด + ล้าง DB ทิ้ง (เฉพาะ local) | `... --profile local down -v` |
| ดู status | `docker compose ps` |

---

## 8. สรุปสั้น — ใช้ port อะไรบ้าง

| Port | Local | Deploy |
|---|---|---|
| **80** | Frontend (เปิดเบราว์เซอร์) | Frontend (เปิดเบราว์เซอร์) |
| **8000** | Backend — *internal only* (ไม่เปิด) | Backend — *internal only* (ไม่เปิด) |
| **5432** | Postgres ใน compose (เปิดที่ host) | ใช้ DB ที่ `192.168.125.10:5432` |
