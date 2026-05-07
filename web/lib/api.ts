export type ScheduleRequest = {
  num_batches?: number;
  if_cfg?: { use_furnace_a?: boolean; use_furnace_b?: boolean };
  mh?: {
    max_capacity?: { A?: number; B?: number };
    initial_level?: { A?: number; B?: number };
    consumption_rate?: { A?: number; B?: number };
  };
  solar?: { windows?: [number, number][]; discount_factor?: number };
  ga?: { pop_size?: number; n_gen?: number; seed?: number };
};

export async function fetchSchedule(body: ScheduleRequest) {
  const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const res = await fetch(`${base}/api/schedule`, {
    method: "POST",
    headers: { "Content-Type": "application/json", accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`API error ${res.status}: ${msg}`);
  }
  return res.json();
}



