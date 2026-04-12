import client from "./client";

export interface PlantLoadSummary {
  active: boolean;
  profile: {
    id: string;
    name: string;
    created_at: string;
    uploaded_filename: string | null;
    entry_count: number;
    spike_count: number;
    spikes: { start_tod: string; end_tod: string; extra_kw: number; probability: number }[];
    hourly_summary: { hour: number; time: string; avg_load_kw: number }[];
  } | null;
}

export interface TouUploadResult {
  updated_keys: string[];
  count: number;
}

export interface PlantLoadUploadResult {
  id: string;
  name: string;
  entry_count: number;
  spike_count: number;
  is_active: boolean;
}

// ── TOU Rate ──────────────────────────────────────────────────────────────────

export const uploadsApi = {
  // Download blank TOU template
  downloadTouTemplate(): void {
    window.open("/api/uploads/tou-template", "_blank");
  },

  // Export current TOU settings as Excel
  downloadTouExport(): void {
    window.open("/api/uploads/tou-rates/export", "_blank");
  },

  // Upload TOU Excel → returns updated keys
  uploadTouRates(file: File): Promise<TouUploadResult> {
    const form = new FormData();
    form.append("file", file);
    return client
      .post<TouUploadResult>("/uploads/tou-rates", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },

  // ── Plant Load ─────────────────────────────────────────────────────────────

  // Download blank plant load template
  downloadPlantLoadTemplate(): void {
    window.open("/api/uploads/plant-load-template", "_blank");
  },

  // Export active plant load profile as Excel
  downloadPlantLoadExport(): void {
    window.open("/api/uploads/plant-load/export", "_blank");
  },

  // Get active plant load profile metadata (for UI display)
  getActivePlantLoad(): Promise<PlantLoadSummary> {
    return client.get<PlantLoadSummary>("/uploads/plant-load/active").then((r) => r.data);
  },

  // Upload plant load Excel → replace active profile
  uploadPlantLoad(file: File): Promise<PlantLoadUploadResult> {
    const form = new FormData();
    form.append("file", file);
    return client
      .post<PlantLoadUploadResult>("/uploads/plant-load", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },
};
