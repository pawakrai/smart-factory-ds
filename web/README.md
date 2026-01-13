### Frontend (Next.js) Skeleton

- Folder `web/` เตรียมไว้สำหรับ Next.js + TypeScript deploy ที่ Vercel
- ยังไม่ได้ `npm install` ให้คุณรันเองในเครื่อง/CI
- ชี้ API ผ่าน env `NEXT_PUBLIC_API_BASE` (เช่น `http://localhost:8000`)

ขั้นตอนเริ่มต้น (บนเครื่องคุณ):

```bash
cd web
npm install       # หรือติดตั้งด้วย pnpm/yarn ก็ได้
npm run dev       # เริ่มที่ http://localhost:3000
```

ไฟล์สำคัญ:
- `package.json`          : scripts/dep พื้นฐาน Next 14 + TS
- `next.config.js`        : next config
- `tsconfig.json`         : TS config
- `pages/index.tsx`       : หน้า Dashboard ตัวอย่าง (เรียก API ยังเป็น mock)
- `pages/_app.tsx`        : wrapper หลัก
- `components/Dashboard.tsx` : โครง UI ดึงคอนฟิกพื้นฐาน + ปุ่ม Recalculate
- `lib/api.ts`            : helper เรียก backend (`/api/schedule`)

การชี้ API:
- ตั้ง env: `NEXT_PUBLIC_API_BASE=http://localhost:8000` (หรือ URL ของ backend ที่ deploy แล้ว)




