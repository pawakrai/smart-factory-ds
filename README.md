## FurnaceFlow (scheduling + RL/GA) – Dev Notes

### โครงสร้างหลัก
- `src/` : โค้ดวิจัย/โมเดล GA/RL เดิม (`app_v5.py` ฯลฯ)
- `backend/` : FastAPI service สำหรับคำนวณตารางหลอม
  - `backend/service_core.py` : wrapper เรียก HGA
  - `backend/service_main.py` : FastAPI app (endpoint `/api/schedule`)
- `web/` : สร้างสำหรับ Next.js frontend (ยังเป็น skeleton)

### รัน backend (dev)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # ใช้ --break-system-packages ถ้าจำเป็นใน macOS managed Python
PYTHONPATH=. uvicorn backend.service_main:app --reload --port 8000
# docs: http://127.0.0.1:8000/docs
```

ตัวอย่าง curl:
```bash
curl -X POST http://127.0.0.1:8000/api/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "num_batches": 10,
    "if_cfg": {"use_furnace_a": true, "use_furnace_b": true},
    "mh": {
      "max_capacity": {"A": 400, "B": 250},
      "initial_level": {"A": 400, "B": 200},
      "consumption_rate": {"A": 3.5, "B": 2.5}
    },
    "solar": {"windows": [[720, 780]], "discount_factor": 0.5},
    "ga": {"pop_size": 50, "n_gen": 100, "seed": 42}
  }'
```

### Frontend (web/)
- Next.js + TS skeleton พร้อม `package.json`, `pages/index.tsx`, `components/Dashboard.tsx`
- ตั้ง env `NEXT_PUBLIC_API_BASE` ให้ชี้ backend เช่น `http://localhost:8000`
- รัน dev:
```bash
cd web
npm install
npm run dev   # http://localhost:3000
```
start service
uvicorn src.service_main:app --reload --port 8000