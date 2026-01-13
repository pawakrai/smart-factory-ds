import { useState } from "react";
import { fetchSchedule, ScheduleRequest } from "../lib/api";

type ScheduleItem = {
  batch_id: number;
  furnace: string;
  start_min: number;
  end_min: number;
};

export default function Dashboard() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const [form, setForm] = useState<ScheduleRequest>({
    num_batches: 10,
    if_cfg: { use_furnace_a: true, use_furnace_b: true },
    mh: {
      max_capacity: { A: 400, B: 250 },
      initial_level: { A: 400, B: 200 },
      consumption_rate: { A: 3.5, B: 2.5 },
    },
    solar: { windows: [[720, 780]], discount_factor: 0.5 },
    ga: { pop_size: 30, n_gen: 50, seed: 42 },
  });

  async function onRun() {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchSchedule(form);
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const schedule: ScheduleItem[] = result?.schedule || [];

  return (
    <div style={{ display: "flex", gap: 24 }}>
      <div style={{ minWidth: 280 }}>
        <h3>Config</h3>
        <div>
          <label>Num batches</label>
          <input
            type="number"
            value={form.num_batches ?? 10}
            onChange={(e) =>
              setForm({ ...form, num_batches: Number(e.target.value) })
            }
          />
        </div>
        <div>
          <label>
            <input
              type="checkbox"
              checked={form.if_cfg?.use_furnace_a ?? true}
              onChange={(e) =>
                setForm({
                  ...form,
                  if_cfg: { ...form.if_cfg, use_furnace_a: e.target.checked },
                })
              }
            />
            Furnace A
          </label>
        </div>
        <div>
          <label>
            <input
              type="checkbox"
              checked={form.if_cfg?.use_furnace_b ?? true}
              onChange={(e) =>
                setForm({
                  ...form,
                  if_cfg: { ...form.if_cfg, use_furnace_b: e.target.checked },
                })
              }
            />
            Furnace B
          </label>
        </div>
        <div>
          <label>Solar window (start, end min)</label>
          <input
            type="text"
            value={form.solar?.windows?.[0]?.[0] ?? 720}
            onChange={(e) =>
              setForm({
                ...form,
                solar: {
                  ...form.solar,
                  windows: [
                    [
                      Number(e.target.value),
                      form.solar?.windows?.[0]?.[1] ?? 780,
                    ],
                  ],
                },
              })
            }
          />
          <input
            type="text"
            value={form.solar?.windows?.[0]?.[1] ?? 780}
            onChange={(e) =>
              setForm({
                ...form,
                solar: {
                  ...form.solar,
                  windows: [
                    [
                      form.solar?.windows?.[0]?.[0] ?? 720,
                      Number(e.target.value),
                    ],
                  ],
                },
              })
            }
          />
        </div>
        <button onClick={onRun} disabled={loading}>
          {loading ? "Running..." : "Recalculate"}
        </button>
        {error && <div style={{ color: "red" }}>{error}</div>}
      </div>

      <div style={{ flex: 1 }}>
        <h3>Result</h3>
        {result ? (
          <>
            <div>
              <strong>Total Cost:</strong>{" "}
              {result.objectives?.total_cost?.toFixed?.(2)}
            </div>
            <div>
              <strong>Energy Cost (kWh eq):</strong>{" "}
              {result.objectives?.energy_cost_kwh_equiv?.toFixed?.(2)}
            </div>
            <div>
              <strong>Makespan (min):</strong>{" "}
              {result.objectives?.makespan_min?.toFixed?.(2)}
            </div>
            <h4>Schedule</h4>
            <ul>
              {schedule.map((s) => (
                <li key={s.batch_id}>
                  Batch {s.batch_id} – Furnace {s.furnace} – {s.start_min} →
                  {s.end_min} min
                </li>
              ))}
            </ul>
          </>
        ) : (
          <div>No result yet</div>
        )}
      </div>
    </div>
  );
}



