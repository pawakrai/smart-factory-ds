# CLAUDE.md — Manufacturing Energy & Production Dashboard
## Sharp Aluminum Melting Factory · Smart Factory DS

This file is the authoritative specification for building the Manufacturing Energy & Production Dashboard MVP. Claude Code must refer to this document at the start of every session before making any changes.

---

## 1. Project Overview & Goals

**Project Name:** FurnaceFlow Dashboard
**Brand:** Sharp
**Domain:** Aluminum Melting Factory — Induction & M&H Furnace Operations

### Goal
Build a full-stack web dashboard that surfaces real-time energy data, production schedules (from a Genetic Algorithm), and operator execution logs for an aluminum melting factory. The system bridges existing research ML code (GA + DQN/RL in `src/`) with a production-grade web UI.

### MVP Scope (5 Tabs)
1. **Dashboard** — Factory status overview, energy bar chart (Simulated vs Actual), furnace status indicators with 3D background overlay
2. **Production Planning** — Input form → GA schedule → Interactive Gantt chart + Metal Level line chart
3. **Operator Execution** — Power profile guideline chart (RL output), batch data entry form, batch completion validation
4. **Reports & Logs** — Historical batch table with CSV export
5. **Master Settings** — Config key/value editor for furnace parameters

---

## 2. Tech Stack

### Frontend (`frontend/`)
| Layer | Choice | Reason |
|---|---|---|
| Bundler | Vite + React 18 | Fast HMR, modern tooling |
| Styling | Tailwind CSS v3 | Utility-first, industrial theming |
| Components | shadcn/ui | Accessible, customizable base components |
| Charts | Recharts | React-native, flexible, Gantt-compatible |
| State | Zustand | Lightweight global state |
| Data fetching | TanStack Query (React Query v5) | Cache + async management |
| Forms | React Hook Form + Zod | Type-safe validation |
| HTTP | Axios | Interceptors for auth/error handling |
| Router | React Router v6 | Tab-based SPA routing |

### Backend (`backend/`)
| Layer | Choice |
|---|---|
| Framework | FastAPI (Python 3.11+) |
| ORM | SQLModel (SQLAlchemy + Pydantic) |
| Database | Supabase (PostgreSQL) |
| GA Integration | Wrapper around `src/ga/ga.py` and `src/normal_ga.py` |
| RL Integration | Wrapper around `src/agents/` + `models/*.pth` |
| Server | Uvicorn |

---

## 3. Brand & Design System

### Brand Identity — Sharp
The UI mood mirrors the **HMI:A reference dashboard** (clean card grid, status dots, mini charts per machine, bar chart + donut chart) but adapted with the Sharp brand color palette:

| Token | Hex | Usage |
|---|---|---|
| `brand-red` | `#E3000F` | Primary buttons, active nav underline, KPI highlights, alert badges |
| `brand-red-dark` | `#B0000C` | Hover states on red elements |
| `brand-gray` | `#3D3D3D` | Secondary brand color (from logo dark "EC") |
| `bg-base` | `#09090B` (zinc-950) | App background |
| `bg-card` | `#18181B` (zinc-900) | Card / panel backgrounds |
| `bg-elevated` | `#27272A` (zinc-800) | Inputs, table rows, secondary cards |
| `border` | `#3F3F46` (zinc-700) | Card borders, dividers |
| `text-primary` | `#FAFAFA` | Headings, primary text |
| `text-muted` | `#A1A1AA` (zinc-400) | Labels, secondary info |
| `status-green` | `#22C55E` | Online / running states |
| `status-amber` | `#F59E0B` | Warning / standby states |
| `status-red` | `#E3000F` | Error / fault states (same as brand-red) |
| `chart-sim` | `#E3000F` | Simulated data series |
| `chart-actual` | `#3B82F6` | Actual data series |
| `chart-grid` | `#3F3F46` | Chart grid lines |

### Design Rules
- **Background:** zinc-950 (`#09090B`) for the page, zinc-900 (`#18181B`) for all cards
- **Cards:** 1px `zinc-700` border, `rounded-xl`, subtle `shadow-lg`
- **Active nav tab:** red bottom border `border-b-2 border-brand-red`, text white
- **Inactive nav tabs:** zinc-400 text, no border
- **Primary buttons:** `bg-brand-red hover:bg-brand-red-dark text-white`
- **Status dots:** 8px solid circle — green (running), red (fault), amber (warning), zinc-500 (offline)
- **Fonts:** `Inter` for all UI text · `JetBrains Mono` for KPI numbers and timestamps
- **Chart style:** dark background, brand-red for simulated series, blue for actual series, zinc-700 grid
- **Spacing:** generous padding inside cards (p-5 or p-6), 4-column grid on desktop

### Tailwind Config Extension
```js
// tailwind.config.js
theme: {
  extend: {
    colors: {
      brand: {
        red: '#E3000F',
        'red-dark': '#B0000C',
        gray: '#3D3D3D',
      },
      bg: {
        base: '#09090B',
        card: '#18181B',
        elevated: '#27272A',
      },
    },
    fontFamily: {
      sans: ['Inter', 'sans-serif'],
      mono: ['JetBrains Mono', 'monospace'],
    },
  },
},
```

---

## 4. Database Schema (PostgreSQL via Supabase)

### Table: `plans`
| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `UUID` | PK, default `gen_random_uuid()` | |
| `target_batches` | `INTEGER` | NOT NULL | Number of batches planned |
| `consumption_rate` | `FLOAT` | NOT NULL | kWh/batch target |
| `shift_start` | `TIMESTAMPTZ` | NOT NULL | Shift start datetime |
| `status` | `TEXT` | NOT NULL, default `'draft'` | `draft` \| `active` \| `completed` |
| `created_at` | `TIMESTAMPTZ` | default `now()` | |

### Table: `batches`
| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `UUID` | PK, default `gen_random_uuid()` | |
| `plan_id` | `UUID` | FK → `plans.id` ON DELETE CASCADE | |
| `batch_number` | `INTEGER` | NOT NULL | Sequence within plan (1-based) |
| `expected_start` | `TIMESTAMPTZ` | | From GA schedule output |
| `actual_start` | `TIMESTAMPTZ` | | Entered by operator |
| `ingot_kg` | `FLOAT` | | Input material weight |
| `fe_kg` | `FLOAT` | | Iron additive weight |
| `si_kg` | `FLOAT` | | Silicon additive weight |
| `scrap_kg` | `FLOAT` | | Scrap material weight |
| `status` | `TEXT` | NOT NULL, default `'pending'` | `pending` \| `in_progress` \| `completed` |
| `created_at` | `TIMESTAMPTZ` | default `now()` | |

### Table: `energy_logs`
| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `UUID` | PK, default `gen_random_uuid()` | |
| `batch_id` | `UUID` | FK → `batches.id`, nullable | |
| `timestamp` | `TIMESTAMPTZ` | NOT NULL | Reading time |
| `sim_kw` | `FLOAT` | | Simulated power (kW) |
| `actual_kw` | `FLOAT` | | Actual measured power (kW) |

### Table: `settings`
| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | `UUID` | PK, default `gen_random_uuid()` | |
| `config_key` | `TEXT` | UNIQUE, NOT NULL | e.g. `furnace_capacity_kg` |
| `config_value` | `TEXT` | NOT NULL | JSON-serialized value |
| `description` | `TEXT` | | Human-readable label |
| `updated_at` | `TIMESTAMPTZ` | default `now()` | |

### Entity Relationships
```
plans 1 ──────< batches
batches 1 ────< energy_logs
```

### Default Settings Seed Data
```sql
INSERT INTO settings (config_key, config_value, description) VALUES
  ('induction_furnace_capacity_kg', '1000', 'Induction furnace max capacity (kg)'),
  ('mh_furnace_capacity_kg', '2000', 'M&H furnace max capacity (kg)'),
  ('peak_hours_start', '09:00', 'Peak energy tariff start time'),
  ('peak_hours_end', '22:00', 'Peak energy tariff end time'),
  ('target_energy_per_batch_kwh', '450', 'Target energy consumption per batch (kWh)'),
  ('shift_duration_hours', '8', 'Default shift duration in hours');
```

---

## 5. Folder Structure

```
smart-factory-ds/
├── CLAUDE.md                        ← This file (authoritative spec)
├── README.md
├── requirements.txt                 ← Root Python deps (FastAPI, torch, etc.)
├── .env                             ← Local env vars (gitignored)
├── .env.example                     ← Template (committed)
│
├── src/                             ← Existing research code (DO NOT MODIFY)
│   ├── ga/                          ← ga.py, operators.py, ga_real_value.py
│   ├── training/                    ← DQN training scripts
│   ├── agents/                      ← agent.py, agent2.py, agent3.py
│   ├── environment/                 ← aluminum_melting_env_7.py … env_11.py
│   └── visualization/               ← Plot utilities
│
├── models/                          ← Trained DQN .pth checkpoints (gitignored)
│
├── backend/                         ← FastAPI application
│   ├── main.py                      ← App entrypoint, CORS, router registration
│   ├── config.py                    ← Pydantic Settings (reads .env)
│   ├── database.py                  ← SQLModel engine, session dependency
│   ├── models/                      ← SQLModel ORM table definitions
│   │   ├── __init__.py
│   │   ├── plan.py                  ← Plan table
│   │   ├── batch.py                 ← Batch table
│   │   ├── energy_log.py            ← EnergyLog table
│   │   └── setting.py               ← Setting table
│   ├── schemas/                     ← Pydantic request/response DTOs
│   │   ├── plan.py
│   │   ├── batch.py
│   │   └── energy_log.py
│   ├── routers/                     ← FastAPI APIRouter (one per resource)
│   │   ├── plans.py                 ← CRUD + GA trigger
│   │   ├── batches.py               ← CRUD + completion endpoint
│   │   ├── energy.py                ← Energy log queries
│   │   ├── schedule.py              ← GA schedule generation endpoint
│   │   └── settings.py              ← Config CRUD
│   └── services/                    ← Business logic layer
│       ├── ga_service.py            ← Wraps src/ga/ga.py & src/normal_ga.py
│       └── rl_service.py            ← Wraps src/agents/ + models/*.pth
│
└── frontend/                        ← Vite + React 18 + TypeScript app
    ├── index.html
    ├── vite.config.ts
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── tsconfig.json
    ├── package.json
    └── src/
        ├── main.tsx                 ← React DOM root
        ├── App.tsx                  ← Router + TopNav shell
        ├── api/                     ← Axios API layer
        │   ├── client.ts            ← Base axios instance (baseURL, interceptors)
        │   ├── plans.ts
        │   ├── batches.ts
        │   ├── energy.ts
        │   └── settings.ts
        ├── components/
        │   ├── layout/
        │   │   ├── TopNav.tsx       ← 5-tab nav bar, Sharp logo, active red underline
        │   │   └── PageWrapper.tsx  ← Consistent page padding wrapper
        │   ├── charts/
        │   │   ├── EnergyBarChart.tsx      ← Sim vs Actual bar chart
        │   │   ├── GanttChart.tsx          ← Production schedule Gantt
        │   │   ├── MetalLevelChart.tsx     ← Metal level line chart
        │   │   └── PowerProfileChart.tsx  ← RL power profile (read-only)
        │   ├── dashboard/
        │   │   ├── FurnaceStatusCard.tsx   ← Status dot + KPI tile
        │   │   └── TrendSummaryWidget.tsx  ← Mini line + delta badge
        │   └── ui/                  ← shadcn/ui component re-exports
        ├── pages/
        │   ├── DashboardPage.tsx
        │   ├── ProductionPlanningPage.tsx
        │   ├── OperatorExecutionPage.tsx
        │   ├── ReportsPage.tsx
        │   └── SettingsPage.tsx
        ├── store/
        │   └── appStore.ts          ← Zustand: active plan, shift state
        ├── hooks/
        │   ├── usePlans.ts          ← React Query hooks for plans
        │   ├── useBatches.ts
        │   └── useEnergyLogs.ts
        └── types/
            └── index.ts             ← TypeScript interfaces mirroring DB schema
```

---

## 6. Step-by-Step Build Plan (Phase 1 → Phase 5)

### Phase 1 — Project Initialization & Infrastructure
**Goal:** Both servers run, DB connected, theme applied.
- [ ] Scaffold `frontend/` with `npm create vite@latest frontend -- --template react-ts`
- [ ] Install: `tailwindcss`, `shadcn/ui`, `recharts`, `zustand`, `@tanstack/react-query`, `react-hook-form`, `zod`, `axios`, `react-router-dom`
- [ ] Apply Tailwind custom theme (Sharp Red + zinc palette + Inter/JetBrains Mono fonts)
- [ ] Initialize shadcn/ui (`npx shadcn-ui@latest init`) with zinc base color
- [ ] Scaffold `TopNav.tsx` with 5 tabs + React Router `<Routes>`
- [ ] Refactor `backend/` to new folder structure (models, schemas, routers, services)
- [ ] Add Supabase `DATABASE_URL` to `.env`, implement `database.py` SQLModel session
- [ ] Run `SQLModel.metadata.create_all(engine)` to create all 4 tables
- [ ] Seed `settings` table with default values
- [ ] Verify: backend on `:8000/docs`, frontend on `:5173` — both start clean

### Phase 2 — Dashboard Tab
**Goal:** Real-looking overview with mock data.
- [ ] Build `FurnaceStatusCard` (status dot, furnace name, temperature KPI, power KPI)
- [ ] Build `EnergyBarChart` (Recharts `BarChart` — hourly Sim vs Actual, red + blue bars)
- [ ] Build `TrendSummaryWidget` (mini Recharts `LineChart` + `+/-X%` delta badge)
- [ ] Wire `GET /api/energy-logs` → seed DB with 24h of mock data
- [ ] Dashboard layout: 4-col header stats row + 2-col chart area + furnace card grid (matches HMI reference density)
- [ ] Donut chart for furnace capacity utilization (optional, mirrors reference image)

### Phase 3 — Production Planning Tab
**Goal:** GA integration working end-to-end.
- [ ] Build `PlanForm` with React Hook Form + Zod (target_batches, consumption_rate, shift_start datetime picker)
- [ ] `POST /api/plans` → persist → trigger `ga_service.py` → return batch schedule
- [ ] Build `GanttChart` component using Recharts custom `BarChart` with time axis
- [ ] Build `MetalLevelChart` (Recharts `LineChart` showing metal kg over batch sequence)
- [ ] Wire `ga_service.py` to call `src/normal_ga.py` or `src/ga/ga.py`

### Phase 4 — Operator Execution & Reports
**Goal:** Full operator loop — plan → execute → log.
- [ ] Build `PowerProfileChart` (read-only line chart from RL model or mock kW profile)
- [ ] Build `BatchEntryForm` (actual_start, ingot_kg, fe_kg, si_kg, scrap_kg) with Zod validation
- [ ] `PATCH /api/batches/{id}` → update fields + set `status: completed`
- [ ] Build `ReportsTable` with TanStack Table (sort, filter, pagination)
- [ ] Add CSV export: `StreamingResponse` from FastAPI or client-side `json2csv`

### Phase 5 — Master Settings, Polish & Hardening
**Goal:** Production-ready MVP.
- [ ] Build `SettingsEditor` (inline-editable key-value table with `PUT /api/settings/{key}`)
- [ ] Add loading skeletons (shadcn `Skeleton`) for all charts and tables
- [ ] Add toast notifications (shadcn `Toaster`) for form success/error states
- [ ] Add React `ErrorBoundary` for chart components
- [ ] Create `.env.example` with all required variables
- [ ] Final theme pass: ensure every component matches brand spec exactly

---

## 7. Development Commands

```bash
# Backend (from project root)
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev        # http://localhost:5173

# Run GA manually (for testing)
python -m src.ga.ga

# Run RL agent inference
python src/training/run_trained_agent.py
```

---

## 8. Key Integration Points

| Frontend Action | Backend Endpoint | Service Called |
|---|---|---|
| Create plan + run GA | `POST /api/plans` | `ga_service.py` → `src/normal_ga.py` |
| Get Gantt data | `GET /api/plans/{id}/batches` | DB query (batch list with expected_start) |
| Submit batch completion | `PATCH /api/batches/{id}` | DB update → optional energy log |
| Get RL power profile | `GET /api/batches/{id}/power-profile` | `rl_service.py` → DQN `.pth` model |
| Energy bar chart data | `GET /api/energy-logs?shift_start=...` | DB query with time filter |
| Read/write settings | `GET/PUT /api/settings` | DB CRUD |

---

## 9. Environment Variables

### `.env` (gitignored)
```
# Supabase / PostgreSQL
DATABASE_URL=postgresql+asyncpg://postgres:[password]@[host]:5432/postgres
SUPABASE_URL=https://[project-ref].supabase.co
SUPABASE_KEY=[anon-public-key]

# App
CORS_ORIGINS=http://localhost:5173
ENV=development
```

### `.env.example` (committed)
```
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/dbname
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-anon-key
CORS_ORIGINS=http://localhost:5173
ENV=development
```

---

## 10. Important Constraints

- **`src/` is read-only** — Never modify existing GA/RL research code. Only wrap it via `backend/services/`.
- **Supabase is the only DB** — No SQLite fallback. Always use `DATABASE_URL` from env.
- **Dark theme is non-negotiable** — Every component must use the zinc + Sharp Red palette. No light mode.
- **Brand color is `#E3000F`** — Never substitute with orange, pink, or other reds.
- **`models/*.pth` files are gitignored** — Document where to download them in README if needed.
