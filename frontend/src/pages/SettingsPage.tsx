import { useRef, useState } from "react";
import {
  Settings2, Save, RotateCcw, Check, AlertCircle,
  Flame, Container, Zap, Cpu, Clock, ChevronDown, ChevronRight,
  Upload, Download, FileSpreadsheet, Activity, Info,
} from "lucide-react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import PageWrapper from "@/components/layout/PageWrapper";
import { Skeleton } from "@/components/ui/Skeleton";
import { useSettings, useSettingDefaults, useUpdateSetting } from "@/hooks/useSettings";
import { useToast } from "@/store/toastStore";
import { uploadsApi, type PlantLoadSummary } from "@/api/uploads";
import type { Setting } from "@/types";

// ── Thai tooltips per setting key ─────────────────────────────────────────
const SETTING_TOOLTIPS: Record<string, { what: string; recommend: string; effect: string }> = {
  // IF Furnace
  if_visual_kw_max: {
    what: "กำลังไฟสูงสุดที่แสดงในกราฟพลังงาน IF (ไม่กระทบการคำนวณจริง)",
    recommend: "ตั้งให้เท่ากับ if_power_option_high_kw (ปกติ 500 kW)",
    effect: "เพิ่ม → กราฟดูแบนลง | ลด → กราฟดูสูงขึ้นเกินจริง",
  },
  if_power_option_low_kw: {
    what: "ตัวเลือกกำลังไฟต่ำสุดของเตาหลอม IF ที่ GA เลือกได้",
    recommend: "450 kW (ตามสเปคเตา)",
    effect: "เพิ่ม → GA มีแนวโน้มเลือกพลังงานสูงขึ้น ประหยัดเวลา แต่ต้นทุนสูง",
  },
  if_power_option_mid_kw: {
    what: "ตัวเลือกกำลังไฟกลางของเตาหลอม IF",
    recommend: "475 kW",
    effect: "เพิ่ม → หลอมเร็วขึ้น พลังงานต่อ batch สูงขึ้น",
  },
  if_power_option_high_kw: {
    what: "ตัวเลือกกำลังไฟสูงสุดของเตาหลอม IF",
    recommend: "500 kW (ห้ามเกินพิกัดเตา)",
    effect: "เพิ่ม → หลอมเร็วที่สุด แต่ค่า demand charge สูง ระวัง peak kW",
  },
  if_batch_output_kg: {
    what: "น้ำหนักอะลูมิเนียมที่ได้จากการหลอม 1 batch (kg) — กำหนดพื้นที่ว่างใน MH ที่ต้องมีก่อนเทได้",
    recommend: "500 kg (ตามพิกัดเตา)",
    effect: "เพิ่ม → แต่ละ batch ต้องการพื้นที่ว่างใน MH มากขึ้น GA ต้องรอให้ MH drain ก่อนเท ส่งผลให้ schedule ตึงขึ้น | ลด → เทได้ง่ายขึ้น ลด JIT waiting time",
  },
  if_efficiency_factor_a: {
    what: "ตัวคูณประสิทธิภาพพลังงานของเตา IF-A (< 1 = ประหยัดกว่า, > 1 = สิ้นเปลืองกว่า)",
    recommend: "0.99 (IF-A ประหยัดกว่า IF-B เล็กน้อย)",
    effect: "เพิ่ม → GA หลีกเลี่ยงการใช้ IF-A มากขึ้น | ลด → GA เลือก IF-A บ่อยขึ้น",
  },
  if_efficiency_factor_b: {
    what: "ตัวคูณประสิทธิภาพพลังงานของเตา IF-B",
    recommend: "1.03",
    effect: "เพิ่ม → IF-B ดูสิ้นเปลืองขึ้น GA เลือกน้อยลง | ลด → GA เลือก IF-B มากขึ้น",
  },
  cold_start_gap_threshold_min: {
    what: "ระยะเวลาว่างขั้นต่ำ (นาที) ที่ต้องผ่านไปก่อนจะถือว่าเตานั้นเป็น cold start",
    recommend: "180 นาที (3 ชั่วโมง)",
    effect: "เพิ่ม → เตาต้องหยุดนานกว่าเดิมถึงจะถือว่า cold start → cold start เกิดน้อยลง | ลด → แม้เว้นว่างไม่นานก็ถือว่า cold start → cold start เกิดบ่อยขึ้น",
  },
  cold_start_extra_duration_min: {
    what: "เวลาหลอมที่เพิ่มขึ้นเมื่อเตาเริ่มจาก cold start (นาที)",
    recommend: "8 นาที",
    effect: "เพิ่ม → schedule ที่มี cold start ใช้เวลานานขึ้น | ลด → ลด penalty ของ cold start",
  },
  cold_start_extra_energy_kwh: {
    what: "พลังงานพิเศษที่ต้องใช้เพิ่มเมื่อ cold start (kWh)",
    recommend: "30 kWh",
    effect: "เพิ่ม → ต้นทุนพลังงาน cold start สูงขึ้น GA หลีกเลี่ยงมากขึ้น",
  },
  post_pour_downtime_min: {
    what: "เวลา downtime หลังเทน้ำอะลูมิเนียม ก่อนเตากลับมาใช้งานได้ (นาที)",
    recommend: "10 นาที",
    effect: "เพิ่ม → เตาว่างนานขึ้น throughput ลดลง | ลด → เตาพร้อมเร็วขึ้น ระวังความปลอดภัย",
  },
  // M&H Furnace
  mh_a_capacity_kg: {
    what: "ความจุสูงสุดของเตาพักอุ่น M&H A (kg)",
    recommend: "400 kg (ตามสเปคจริง ห้ามเกิน)",
    effect: "เพิ่ม → GA เทน้ำได้มากขึ้นต่อครั้งโดยไม่ overflow | ลด → overflow penalty เกิดง่ายขึ้น",
  },
  mh_a_initial_level_kg: {
    what: "ปริมาณน้ำอะลูมิเนียมในเตา M&H A ที่จุดเริ่มต้น shift (kg)",
    recommend: "ตามระดับจริงหน้างาน ก่อนเริ่ม shift",
    effect: "เพิ่ม → เตามีน้ำพอนานขึ้นช่วงต้น shift | ลด → อาจต้องเทน้ำเร็วขึ้น",
  },
  mh_a_consumption_rate_kg_per_min: {
    what: "อัตราการใช้น้ำอะลูมิเนียมของสายการผลิต M&H A (kg/นาที) — นี่คือค่า default หาก Plan ไม่ได้ระบุค่าเอง (Plan สามารถ override ได้รายแผน)",
    recommend: "2.20 kg/min (วัดจากข้อมูลจริง)",
    effect: "เพิ่ม → เตาเหลือน้ำเร็วขึ้น GA ต้องเทบ่อยขึ้น | ลด → น้ำอยู่นานขึ้น",
  },
  mh_a_min_operational_level_kg: {
    what: "ระดับน้ำต่ำสุดที่ยอมให้ M&H A ทำงานต่อได้อย่างปลอดภัย (kg)",
    recommend: "200 kg",
    effect: "เพิ่ม → GA ต้องเทน้ำถี่ขึ้น ลด low-level risk | ลด → ยอมให้น้ำต่ำกว่าเดิม เสี่ยงสายการผลิตหยุด",
  },
  mh_b_capacity_kg: {
    what: "ความจุสูงสุดของเตาพักอุ่น M&H B (kg)",
    recommend: "250 kg",
    effect: "เพิ่ม → รับน้ำได้มากขึ้นต่อครั้ง | ลด → overflow เกิดง่ายขึ้น",
  },
  mh_b_initial_level_kg: {
    what: "ปริมาณน้ำเริ่มต้นในเตา M&H B ก่อน shift เริ่ม (kg)",
    recommend: "ตามระดับจริงหน้างาน",
    effect: "เพิ่ม → เตา B มีน้ำสำรองช่วงต้น shift มากขึ้น",
  },
  mh_b_consumption_rate_kg_per_min: {
    what: "อัตราการใช้น้ำอะลูมิเนียมของสายการผลิต M&H B (kg/นาที) — นี่คือค่า default หาก Plan ไม่ได้ระบุค่าเอง (Plan สามารถ override ได้รายแผน)",
    recommend: "2.30 kg/min",
    effect: "เพิ่ม → น้ำหมดเร็วขึ้น GA ต้องวางแผนเทบ่อยขึ้น",
  },
  mh_b_min_operational_level_kg: {
    what: "ระดับน้ำต่ำสุดที่ยอมให้ M&H B ทำงานได้ (kg)",
    recommend: "125 kg",
    effect: "เพิ่ม → GA ให้ความสำคัญกับการเติมน้ำ B มากขึ้น | ลด → ยอมให้วิ่งต่ำกว่าเดิม",
  },
  mh_empty_penalty_per_min: {
    what: "โทษต่อนาทีที่เตา M&H ว่างจนไม่มีน้ำเลย (Baht/นาที)",
    recommend: "150 (ค่าสูงเพื่อบังคับให้ GA หลีกเลี่ยงเตาว่าง)",
    effect: "เพิ่ม → GA กลัวเตาว่างมากขึ้น เร่งเทน้ำก่อนหมด | ลด → GA ยอมให้เตาว่างนานขึ้น",
  },
  mh_low_level_minute_penalty: {
    what: "โทษต่อนาทีที่ระดับน้ำต่ำกว่า min operational level (Baht/นาที)",
    recommend: "40–80 (เพิ่มถ้า GA ยังปล่อยให้น้ำต่ำบ่อย)",
    effect: "เพิ่ม → GA หลีกเลี่ยงช่วงน้ำต่ำมากขึ้น schedule เทน้ำเร็วขึ้น | ลด → ยอมให้น้ำต่ำนานขึ้น",
  },
  mh_low_level_penalty_rate: {
    what: "อัตราโทษพื้นฐาน (ต่อนาที) ที่ใช้คำนวณ low_level_shape_penalty ใน simulation — ทำงานร่วมกับ nonlinear factor เป็น rate × (1 + factor × deficit²)",
    recommend: "200 (ปรับร่วมกับ mh_low_level_nonlinear_factor)",
    effect: "เพิ่ม → low_level_shape_penalty component สูงขึ้นทั้ง shift GA รักษาระดับน้ำให้สูงกว่า min มากขึ้น | ลด → penalty รวมต่ำลง GA ยอมให้น้ำใกล้ min ได้บ่อยขึ้น",
  },
  mh_low_level_nonlinear_factor: {
    what: "ตัวคูณความรุนแรงของ penalty เมื่อน้ำต่ำกว่า min — formula: rate × (1 + factor × deficit_ratio²) คือยิ่งต่ำมากยิ่ง penalty เพิ่มแบบ quadratic",
    recommend: "3.0",
    effect: "เพิ่ม → penalty พุ่งสูงอย่างรวดเร็วเมื่อน้ำใกล้หมด GA กลัวน้ำวิกฤตมากขึ้น | ลด → penalty ใกล้เส้นตรงมากขึ้น ลงโทษสม่ำเสมอกว่า",
  },
  mh_max_empty_min_allow: {
    what: "จำนวนนาทีสะสมสูงสุดที่ยอมให้เตา M&H ว่างได้ก่อนถือว่าละเมิด constraint",
    recommend: "60–120 นาที (ยิ่งน้อยยิ่งเข้มงวด)",
    effect: "ลด → GA ถูกบังคับให้เติมน้ำก่อนเตาว่าง เพิ่ม → ยอมให้เตาว่างนานขึ้น",
  },
  mh_max_low_level_min_allow: {
    what: "จำนวนนาทีสะสมสูงสุดที่ยอมให้น้ำต่ำกว่า min ได้ก่อนถือว่าละเมิด constraint",
    recommend: "120–240 นาที",
    effect: "ลด → GA ต้องรักษาระดับน้ำเข้มงวดขึ้น | เพิ่ม → ผ่อนปรนให้ GA มีอิสระวางแผนพลังงานมากขึ้น",
  },
  // Energy & Tariff
  tou_onpeak_baht_per_kwh: {
    what: "ราคาพลังงานช่วง On-Peak ตามอัตรา TOU (บาท/kWh ไม่รวม FT)",
    recommend: "4.1839 บาท/kWh (อัตรา TOU 22-33kV ปัจจุบัน)",
    effect: "เพิ่ม → GA หลีกเลี่ยงการหลอมช่วง peak มากขึ้น | ลด → GA ไม่กลัวหลอมช่วง peak",
  },
  tou_offpeak_baht_per_kwh: {
    what: "ราคาพลังงานช่วง Off-Peak (บาท/kWh ไม่รวม FT)",
    recommend: "2.6037 บาท/kWh",
    effect: "เพิ่ม → ส่วนต่าง peak/off-peak แคบลง GA ไม่ shift load มากนัก",
  },
  ft_baht_per_kwh: {
    what: "ค่า Fuel Tariff (FT) ที่บวกเพิ่มทุก kWh ทั้งช่วง peak และ off-peak (บาท/kWh)",
    recommend: "0.0972 บาท/kWh (ปรับตาม EGAT ทุกรอบ 4 เดือน)",
    effect: "เพิ่ม → ต้นทุนพลังงานรวมสูงขึ้น GA ประหยัดพลังงานมากขึ้น",
  },
  demand_charge_baht_per_kw_month: {
    what: "ค่า demand charge รายเดือน (บาท/kW/เดือน) คิดจาก peak kW ใน billing period",
    recommend: "132.93 บาท/kW/เดือน (อัตรา TOU 22-33kV)",
    effect: "เพิ่ม → GA หลีกเลี่ยง peak kW สูงมากขึ้น ป้องกัน demand spike",
  },
  contract_demand_kw: {
    what: "ขีดจำกัด demand ตามสัญญาไฟฟ้าของโรงงาน (kW)",
    recommend: "1600 kW (ตามสัญญา กฟภ./กฟน.)",
    effect: "เพิ่ม → GA มีอิสระใช้ไฟสูงขึ้น | ลด → GA ระวัง peak มากขึ้น อาจชะลอ batch",
  },
  peak_hours_start: {
    what: "เวลาเริ่มช่วง On-Peak (รูปแบบ HH:MM) ตามอัตรา TOU",
    recommend: "09:00 (ตามประกาศ กฟภ./กฟน.)",
    effect: "เลื่อนเช้าขึ้น → GA หลีกเลี่ยง peak นานขึ้น | เลื่อนสายขึ้น → ช่วง peak สั้นลง",
  },
  peak_hours_end: {
    what: "เวลาสิ้นสุดช่วง On-Peak (รูปแบบ HH:MM)",
    recommend: "22:00",
    effect: "เลื่อนสายขึ้น → GA มีแรงจูงใจ shift load หลังเที่ยงคืนมากขึ้น",
  },
  solar_window_start: {
    what: "เวลาเริ่มช่วงที่ solar เจเนอเรตและลด effective price (HH:MM)",
    recommend: "12:00",
    effect: "เลื่อนเช้าขึ้น → GA เลือกหลอมช่วงนี้เร็วขึ้น",
  },
  solar_window_end: {
    what: "เวลาสิ้นสุดช่วง solar window (HH:MM)",
    recommend: "13:00",
    effect: "ขยาย → GA ได้ประโยชน์จาก solar นานขึ้น",
  },
  solar_price_factor: {
    what: "ตัวคูณราคาไฟฟ้าช่วง solar (0.35 = ราคาเหลือ 35% ของปกติ)",
    recommend: "0.35 (ขึ้นกับขนาดแผง solar จริง)",
    effect: "ลด → GA ชอบหลอมช่วง solar มากขึ้น | เพิ่ม → ส่วนลด solar น้อยลง GA ไม่ prefer ช่วงนี้",
  },
  // GA Optimization
  ga_pop_size: {
    what: "จำนวน solution ใน population ของ Genetic Algorithm",
    recommend: "80–150 (เพิ่มถ้ามีเวลาคำนวณมาก)",
    effect: "เพิ่ม → GA search กว้างขึ้น solution ดีขึ้น แต่ใช้เวลานานขึ้น | ลด → เร็วขึ้น แต่อาจติด local optimum",
  },
  ga_n_generations: {
    what: "จำนวน generation สูงสุดที่ GA วิ่ง",
    recommend: "100–200",
    effect: "เพิ่ม → ผลดีขึ้นแต่ช้าลง | ลด → เร็วขึ้นแต่ solution อาจยังไม่ converge",
  },
  ga_early_stop_patience: {
    what: "จำนวน generation ที่ไม่มีการพัฒนาก่อน GA หยุดเร็ว",
    recommend: "20 generations",
    effect: "เพิ่ม → GA อดทนรอนานขึ้นก่อน stop | ลด → GA หยุดเร็วถ้าไม่มีการพัฒนา ประหยัดเวลา",
  },
  ga_random_seed: {
    what: "ค่า seed ของ random number generator เพื่อให้ผลลัพธ์ reproducible",
    recommend: "42 (หรือค่าใดก็ได้ที่คงที่)",
    effect: "เปลี่ยน → ได้ผลลัพธ์ที่แตกต่างออกไปจาก random initialization ที่ต่างกัน",
  },
  ga_obj_weight_empty_penalty: {
    what: "น้ำหนักของ penalty เมื่อเตา M&H ว่างในการคำนวณ objective ของ GA",
    recommend: "1.10–2.00 (เพิ่มถ้า GA ยังปล่อยให้เตาว่างบ่อย)",
    effect: "เพิ่ม → GA กลัวเตาว่างมากขึ้น วาง schedule เติมน้ำถี่ขึ้น | ลด → GA โฟกัสด้านพลังงานมากขึ้น",
  },
  ga_obj_weight_low_level_min: {
    what: "น้ำหนักของ penalty เมื่อน้ำต่ำกว่าขั้นต่ำ (low_level_min component)",
    recommend: "0.80–2.00 (ค่านี้มีผลมากที่สุดต่อการป้องกันน้ำต่ำ)",
    effect: "เพิ่ม → GA รักษาระดับน้ำเหนือ min อย่างจริงจัง | ลด → GA ยอมให้น้ำต่ำชั่วคราว",
  },
  ga_obj_weight_low_level_shape: {
    what: "น้ำหนักของ penalty รูปแบบน้ำต่ำทั้ง shift (shape component — ลงโทษช่วงยาวที่น้ำต่ำ)",
    recommend: "0.90–2.00",
    effect: "เพิ่ม → GA หลีกเลี่ยงช่วงยาวที่น้ำต่ำกว่า min | ลด → GA ยอมให้น้ำต่ำเป็นช่วงสั้น ๆ ได้",
  },
  // Shift Configuration
  shift_duration_hours: {
    what: "ความยาวของ shift การผลิต (ชั่วโมง)",
    recommend: "8 ชั่วโมง (1 shift มาตรฐาน)",
    effect: "เพิ่ม → GA มีเวลาวางแผนมากขึ้น ใส่ batch ได้มากขึ้น | ลด → GA ต้องอัดแน่น",
  },
  shift_start_hhmm: {
    what: "เวลาเริ่ม shift ค่าเริ่มต้น (HH:MM) ใช้เมื่อสร้าง plan ใหม่โดยไม่ระบุเวลา",
    recommend: "08:00",
    effect: "เปลี่ยน → เวลาอ้างอิงของกราฟและ schedule ทั้งหมดเลื่อนตาม",
  },
  target_batches_default: {
    what: "จำนวน batch เป้าหมายค่าเริ่มต้นเมื่อสร้าง plan ใหม่",
    recommend: "8–12 batch ต่อ shift",
    effect: "เพิ่ม → GA ต้องยัด batch มากขึ้น อาจกระทบเวลาว่างระหว่าง batch | ลด → schedule หลวมขึ้น",
  },
};

// ── InfoTooltip component ──────────────────────────────────────────────────
function InfoTooltip({ configKey }: { configKey: string }) {
  const [visible, setVisible] = useState(false);
  const info = SETTING_TOOLTIPS[configKey];
  if (!info) return null;

  return (
    <div className="relative inline-flex items-center">
      <button
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        onFocus={() => setVisible(true)}
        onBlur={() => setVisible(false)}
        className="flex items-center justify-center w-4 h-4 rounded-full text-zinc-600 hover:text-zinc-300 transition-colors focus:outline-none"
        tabIndex={0}
        aria-label="ข้อมูลเพิ่มเติม"
      >
        <Info size={12} />
      </button>

      {visible && (
        <div className="absolute left-5 top-1/2 -translate-y-1/2 z-50 w-72 bg-zinc-900 border border-zinc-700 rounded-xl shadow-xl p-3.5 pointer-events-none">
          {/* arrow */}
          <div className="absolute -left-1.5 top-1/2 -translate-y-1/2 w-2.5 h-2.5 bg-zinc-900 border-l border-b border-zinc-700 rotate-45" />
          <div className="space-y-2">
            <p className="text-[11px] text-zinc-200 leading-relaxed">{info.what}</p>
            <div className="border-t border-zinc-700/60 pt-2 space-y-1">
              <p className="text-[10px] text-zinc-400">
                <span className="text-zinc-500 font-medium">ค่าแนะนำ: </span>{info.recommend}
              </p>
              <p className="text-[10px] text-zinc-400">
                <span className="text-zinc-500 font-medium">ผลการปรับ: </span>{info.effect}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Settings groups ────────────────────────────────────────────────────────
const GROUPS: { label: string; icon: React.ReactNode; prefixes: string[] }[] = [
  {
    label: "IF Furnace",
    icon: <Flame size={14} />,
    prefixes: ["if_", "cold_start_", "post_pour_"],
  },
  {
    label: "M&H Furnace",
    icon: <Container size={14} />,
    prefixes: ["mh_"],
  },
  {
    label: "Energy & Tariff",
    icon: <Zap size={14} />,
    prefixes: ["tou_", "ft_", "demand_", "contract_", "peak_hours_", "solar_"],
  },
  {
    label: "GA Optimization",
    icon: <Cpu size={14} />,
    prefixes: ["ga_"],
  },
  {
    label: "Shift Configuration",
    icon: <Clock size={14} />,
    prefixes: ["shift_", "target_batches_default"],
  },
];

function matchesGroup(key: string, prefixes: string[]): boolean {
  return prefixes.some((p) => key.startsWith(p));
}

function groupSettings(settings: Setting[]): { group: typeof GROUPS[0]; items: Setting[] }[] {
  const assigned = new Set<string>();
  const result = GROUPS.map((group) => {
    const items = settings.filter((s) => {
      if (assigned.has(s.config_key)) return false;
      if (matchesGroup(s.config_key, group.prefixes)) {
        assigned.add(s.config_key);
        return true;
      }
      return false;
    });
    return { group, items };
  });
  // Any remaining ungrouped settings
  const ungrouped = settings.filter((s) => !assigned.has(s.config_key));
  if (ungrouped.length > 0) {
    result.push({
      group: { label: "Other", icon: <Settings2 size={14} />, prefixes: [] },
      items: ungrouped,
    });
  }
  return result;
}

// ── SettingRow ─────────────────────────────────────────────────────────────
function SettingRow({ setting, codeDefault }: { setting: Setting; codeDefault?: string }) {
  const [value, setValue] = useState(setting.config_value);
  const [saved, setSaved] = useState(false);
  const isDirty = value !== setting.config_value;
  const isModifiedFromDefault = codeDefault !== undefined && setting.config_value !== codeDefault;
  const updateSetting = useUpdateSetting();
  const toast = useToast();

  async function handleSave() {
    try {
      await updateSetting.mutateAsync({ key: setting.config_key, value });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      toast.success("Saved", setting.config_key);
    } catch {
      toast.error("Save failed", `Could not update ${setting.config_key}`);
    }
  }

  async function handleResetToDefault() {
    if (!codeDefault) return;
    setValue(codeDefault);
    try {
      await updateSetting.mutateAsync({ key: setting.config_key, value: codeDefault });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      toast.success("Reset to default", setting.config_key);
    } catch {
      toast.error("Reset failed", `Could not reset ${setting.config_key}`);
    }
  }

  return (
    <div className="flex items-center justify-between px-5 py-3.5 gap-4">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <p className="text-xs font-mono text-[var(--text-secondary)] leading-tight">{setting.config_key}</p>
          <InfoTooltip configKey={setting.config_key} />
          {isModifiedFromDefault && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20 font-medium">
              แก้ไขแล้ว
            </span>
          )}
        </div>
        {setting.description && (
          <p className="text-[11px] text-zinc-500 mt-0.5">{setting.description}</p>
        )}
        {codeDefault !== undefined && (
          <p className="text-[10px] text-zinc-600 mt-0.5 font-mono">
            default: {codeDefault}
            {isModifiedFromDefault && (
              <button
                onClick={handleResetToDefault}
                className="ml-1.5 text-zinc-500 hover:text-zinc-300 underline underline-offset-2 transition-colors"
              >
                reset
              </button>
            )}
          </p>
        )}
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <input
          type="text"
          value={value}
          onChange={(e) => { setValue(e.target.value); setSaved(false); }}
          onKeyDown={(e) => e.key === "Enter" && isDirty && handleSave()}
          className={`w-36 bg-bg-elevated border rounded-lg px-3 py-1.5 text-sm font-mono text-[var(--text-primary)] text-right focus:outline-none transition-colors
            ${isDirty ? "border-brand-red/60 focus:border-brand-red" : "border-[var(--border-color)] focus:border-zinc-500"}`}
        />
        {isDirty && (
          <button
            onClick={() => setValue(setting.config_value)}
            title="Discard unsaved changes"
            className="text-zinc-600 hover:text-[var(--text-secondary)] transition-colors"
          >
            <RotateCcw size={13} />
          </button>
        )}
        {saved ? (
          <span className="flex items-center gap-1 text-xs text-green-400 w-16">
            <Check size={13} /> Saved
          </span>
        ) : (
          <button
            onClick={handleSave}
            disabled={!isDirty || updateSetting.isPending}
            className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg w-16 justify-center transition-colors
              ${isDirty ? "bg-brand-red hover:bg-brand-red-dark text-white" : "text-zinc-700 cursor-default"}`}
          >
            <Save size={12} />
            Save
          </button>
        )}
      </div>
    </div>
  );
}

function SettingRowSkeleton() {
  return (
    <div className="flex items-center justify-between px-5 py-3.5 gap-4">
      <div className="flex-1 space-y-1.5">
        <Skeleton className="h-3 w-56" />
        <Skeleton className="h-2.5 w-40" />
      </div>
      <Skeleton className="h-8 w-36 rounded-lg" />
    </div>
  );
}

// ── Collapsible settings group card ──────────────────────────────────────
function SettingsGroupCard({
  label,
  icon,
  items,
  defaults,
  defaultOpen = true,
}: {
  label: string;
  icon: React.ReactNode;
  items: Setting[];
  defaults: Record<string, string>;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const modifiedCount = items.filter(
    (s) => defaults[s.config_key] !== undefined && s.config_value !== defaults[s.config_key]
  ).length;

  return (
    <div className="bg-bg-card border border-[var(--border-color)] rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-4 border-b border-[var(--border-color)] hover:bg-[var(--bg-elevated)]/50 transition-colors"
      >
        <div className="flex items-center gap-2 text-brand-red">
          {icon}
          <span className="text-sm font-semibold text-[var(--text-primary)]">{label}</span>
          <span className="text-[10px] text-zinc-500 font-normal">({items.length})</span>
          {modifiedCount > 0 && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20 font-medium">
              {modifiedCount} แก้ไข
            </span>
          )}
        </div>
        {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
      </button>

      {open && (
        <div className="divide-y divide-[var(--border-color)]">
          {items.map((s) => (
            <SettingRow key={s.config_key} setting={s} codeDefault={defaults[s.config_key]} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── TOU Rate Upload Panel ──────────────────────────────────────────────────
function TouUploadPanel() {
  const toast = useToast();
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadsApi.uploadTouRates(file);
      toast.success("TOU rates updated", `${result.count} settings replaced`);
      qc.invalidateQueries({ queryKey: ["settings"] });
    } catch {
      toast.error("Upload failed", "Check file format and try again");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="bg-bg-card border border-[var(--border-color)] rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-5 py-4 border-b border-[var(--border-color)]">
        <Zap size={14} className="text-brand-red" />
        <span className="text-sm font-semibold text-[var(--text-primary)]">TOU Rate Schedule</span>
        <span className="ml-auto text-[10px] font-mono text-zinc-500">TOU_22-33kV_4.2.2</span>
      </div>

      <div className="px-5 py-4 space-y-4">
        {/* Description */}
        <p className="text-xs text-zinc-500">
          Upload an Excel file to replace the on-peak / off-peak rates and demand charges immediately.
          Download the template first to see the expected format.
        </p>

        {/* Action buttons */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => uploadsApi.downloadTouTemplate()}
            className="flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg border border-[var(--border-color)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-zinc-500 transition-colors"
          >
            <FileSpreadsheet size={13} />
            Download Template
          </button>

          <button
            onClick={() => uploadsApi.downloadTouExport()}
            className="flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg border border-[var(--border-color)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-zinc-500 transition-colors"
          >
            <Download size={13} />
            Export Current TOU
          </button>

          <label
            className={`flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg cursor-pointer transition-colors
              ${uploading
                ? "bg-zinc-700 text-zinc-500 cursor-not-allowed"
                : "bg-brand-red hover:bg-brand-red-dark text-white"}`}
          >
            <Upload size={13} />
            {uploading ? "Uploading…" : "Upload Excel"}
            <input
              ref={fileRef}
              type="file"
              accept=".xlsx"
              className="hidden"
              disabled={uploading}
              onChange={handleFile}
            />
          </label>
        </div>

        {/* Format hint */}
        <div className="bg-bg-elevated rounded-lg px-4 py-3 text-[11px] font-mono text-zinc-500 space-y-0.5">
          <p className="text-zinc-400 font-semibold mb-1">Expected sheets</p>
          <p>TOU_Rates · Demand_Charges · Instructions</p>
          <p className="mt-1 text-zinc-600">Columns: day_type | period_type | time_start | time_end | rate_baht_per_kwh | ft_baht_per_kwh</p>
        </div>
      </div>
    </div>
  );
}

// ── Plant Load Upload Panel ────────────────────────────────────────────────
function PlantLoadPanel() {
  const toast = useToast();
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  const { data: plantLoad, isLoading } = useQuery<PlantLoadSummary>({
    queryKey: ["plant-load-active"],
    queryFn: uploadsApi.getActivePlantLoad,
    staleTime: 30_000,
  });

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadsApi.uploadPlantLoad(file);
      toast.success("Plant load updated", `${result.entry_count} minute entries loaded`);
      qc.invalidateQueries({ queryKey: ["plant-load-active"] });
    } catch {
      toast.error("Upload failed", "Check file format and try again");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  const profile = plantLoad?.profile;
  const hourly = profile?.hourly_summary ?? [];

  // Mini bar chart: max load for scaling
  const maxKw = hourly.length > 0 ? Math.max(...hourly.map((h) => h.avg_load_kw)) : 1000;

  return (
    <div className="bg-bg-card border border-[var(--border-color)] rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-5 py-4 border-b border-[var(--border-color)]">
        <Activity size={14} className="text-brand-red" />
        <span className="text-sm font-semibold text-[var(--text-primary)]">Plant Load Profile</span>
        {profile ? (
          <span className="ml-2 text-[10px] font-medium px-2 py-0.5 rounded-full bg-green-500/10 text-green-400 border border-green-500/20">
            Active
          </span>
        ) : (
          <span className="ml-2 text-[10px] font-medium px-2 py-0.5 rounded-full bg-zinc-700/50 text-zinc-500 border border-zinc-700">
            Default
          </span>
        )}
        <span className="ml-auto text-[10px] font-mono text-zinc-500">1440 min/day</span>
      </div>

      <div className="px-5 py-4 space-y-4">
        {/* Profile meta */}
        {isLoading ? (
          <div className="space-y-1.5">
            <Skeleton className="h-3 w-48" />
            <Skeleton className="h-3 w-32" />
          </div>
        ) : profile ? (
          <div className="text-xs space-y-0.5">
            <p className="text-[var(--text-primary)] font-medium">{profile.name}</p>
            <p className="text-zinc-500 font-mono">
              {profile.entry_count} entries · {profile.spike_count} spike event{profile.spike_count !== 1 ? "s" : ""} · uploaded {new Date(profile.created_at).toLocaleDateString()}
            </p>
          </div>
        ) : (
          <p className="text-xs text-zinc-500">No profile uploaded — using built-in default step profile.</p>
        )}

        {/* Mini bar chart: hourly load */}
        {hourly.length > 0 && (
          <div>
            <p className="text-[10px] text-zinc-500 mb-1.5">Hourly average load (kW)</p>
            <div className="flex items-end gap-px h-12">
              {hourly.map((h) => {
                const heightPct = maxKw > 0 ? (h.avg_load_kw / maxKw) * 100 : 0;
                const isPeak = h.hour >= 9 && h.hour < 22;
                return (
                  <div
                    key={h.hour}
                    className="flex-1 relative group"
                    style={{ height: "100%" }}
                  >
                    <div
                      className={`absolute bottom-0 w-full rounded-sm transition-all
                        ${isPeak ? "bg-brand-red/70" : "bg-zinc-600/70"}`}
                      style={{ height: `${heightPct}%` }}
                    />
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:flex flex-col items-center z-10 pointer-events-none">
                      <div className="bg-zinc-900 border border-zinc-700 rounded px-1.5 py-0.5 text-[9px] font-mono text-zinc-200 whitespace-nowrap">
                        {h.time} · {h.avg_load_kw} kW
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="flex justify-between text-[9px] font-mono text-zinc-600 mt-1">
              <span>00:00</span>
              <span className="text-brand-red/60">09:00 ── on-peak ── 22:00</span>
              <span>24:00</span>
            </div>
          </div>
        )}

        {/* Action buttons */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => uploadsApi.downloadPlantLoadTemplate()}
            className="flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg border border-[var(--border-color)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-zinc-500 transition-colors"
          >
            <FileSpreadsheet size={13} />
            Download Template
          </button>

          <button
            onClick={() => uploadsApi.downloadPlantLoadExport()}
            className="flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg border border-[var(--border-color)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-zinc-500 transition-colors"
          >
            <Download size={13} />
            Export Current Profile
          </button>

          <label
            className={`flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg cursor-pointer transition-colors
              ${uploading
                ? "bg-zinc-700 text-zinc-500 cursor-not-allowed"
                : "bg-brand-red hover:bg-brand-red-dark text-white"}`}
          >
            <Upload size={13} />
            {uploading ? "Uploading…" : "Upload Excel"}
            <input
              ref={fileRef}
              type="file"
              accept=".xlsx"
              className="hidden"
              disabled={uploading}
              onChange={handleFile}
            />
          </label>
        </div>

        {/* Format hint */}
        <div className="bg-bg-elevated rounded-lg px-4 py-3 text-[11px] font-mono text-zinc-500 space-y-0.5">
          <p className="text-zinc-400 font-semibold mb-1">Expected sheets</p>
          <p>Plant_Load · Spike_Events · Instructions</p>
          <p className="mt-1 text-zinc-600">Columns: minute (0–1439) | time (HH:MM) | load_kw</p>
        </div>
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function SettingsPage() {
  const { data: settings, isLoading, isError } = useSettings();
  const { data: defaults = {} } = useSettingDefaults();

  const grouped = settings ? groupSettings(settings) : [];

  return (
    <PageWrapper
      title="Master Settings"
      subtitle="Furnace parameters, energy tariff, GA optimization, and shift configuration"
    >
      <div className="space-y-4">
        {isLoading ? (
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl overflow-hidden">
            <div className="px-5 py-4 border-b border-[var(--border-color)] flex items-center gap-2">
              <Settings2 size={16} className="text-brand-red" />
              <span className="text-sm font-semibold text-[var(--text-primary)]">Loading…</span>
            </div>
            <div className="divide-y divide-[var(--border-color)]">
              {Array.from({ length: 8 }).map((_, i) => <SettingRowSkeleton key={i} />)}
            </div>
          </div>
        ) : isError ? (
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl px-5 py-12 flex flex-col items-center gap-2 text-zinc-500">
            <AlertCircle size={20} className="text-brand-red/60" />
            <p className="text-sm">Failed to load settings</p>
            <p className="text-xs">Check that the backend is running</p>
          </div>
        ) : settings?.length === 0 ? (
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl px-5 py-12 text-center text-xs text-zinc-600">
            No settings found — check that the database is seeded
          </div>
        ) : (
          grouped.map(({ group, items }) =>
            items.length === 0 ? null : (
              <SettingsGroupCard
                key={group.label}
                label={group.label}
                icon={group.icon}
                items={items}
                defaults={defaults}
                defaultOpen={true}
              />
            )
          )
        )}

        {/* TOU Rate Upload */}
        <TouUploadPanel />

        {/* Plant Load Upload */}
        <PlantLoadPanel />

        {/* System info */}
        <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wide mb-4">
            System Info
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Backend", value: "FastAPI 0.115", status: "ok" },
              { label: "Database", value: "Supabase / PostgreSQL", status: "ok" },
              { label: "GA Engine", value: "src/app_v9.py", status: "ok" },
              { label: "RL Model", value: "DQN (mock fallback)", status: "warn" },
            ].map((info) => (
              <div key={info.label} className="bg-bg-elevated rounded-lg px-3 py-2.5">
                <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">{info.label}</p>
                <p className="text-xs text-[var(--text-primary)] font-mono truncate">{info.value}</p>
                <span
                  className="inline-block mt-1 text-[10px] font-medium"
                  style={{ color: info.status === "ok" ? "#22C55E" : "#F59E0B" }}
                >
                  {info.status === "ok" ? "● Online" : "● Warning"}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[10px] text-zinc-600 font-mono">
            PUT /api/settings/{"{"}<span className="text-zinc-500">key</span>{"}"} · values stored as text · press Enter or click Save per row
          </p>
        </div>
      </div>
    </PageWrapper>
  );
}
