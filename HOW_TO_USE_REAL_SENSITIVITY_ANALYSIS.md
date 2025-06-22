# วิธีการใช้ Real Training สำหรับ Sensitivity Analysis

## ภาพรวม

ไฟล์ `experiments/sensitivity_analysis_rl.py` ได้รับการปรับปรุงให้สามารถใช้การ training จริงแทนการจำลอง (simulation) เพื่อให้ได้ผลลัพธ์ที่แม่นยำจริง

## ความแตกต่างระหว่าง Simulation และ Real Training

### Simulation Mode (เดิม)
- **ข้อดี**: รวดเร็ว (2-5 นาที)
- **ข้อเสีย**: ผลลัพธ์ไม่ใช่ค่าจริง เป็นเพียงการประมาณ
- **การใช้งาน**: เหมาะสำหรับการทดสอบและพัฒนา script

### Real Training Mode (ใหม่)
- **ข้อดี**: ได้ผลลัพธ์จริงจากการ training
- **ข้อเสีย**: ใช้เวลานาน (15 นาที - 8 ชั่วโมง)
- **การใช้งาน**: เหมาะสำหรับการวิจัยจริง

## วิธีการใช้งาน

### 1. การใช้งานแบบง่าย (Recommended)

```python
from experiments.sensitivity_analysis_rl import RLSensitivityAnalyzer

# สร้าง analyzer สำหรับ real training
analyzer = RLSensitivityAnalyzer(
    output_dir="results/my_real_analysis",
    use_real_training=True  # สำคัญ!
)

# ทดสอบ parameter เดียวก่อน (แนะนำสำหรับครั้งแรก)
results = analyzer.run_single_parameter_analysis("learning_rate")
```

### 2. การใช้งานแบบเต็มรูปแบบ

```python
# ระวัง: จะใช้เวลา 3-8 ชั่วโมง!
analyzer = RLSensitivityAnalyzer(
    output_dir="results/complete_real_analysis",
    use_real_training=True
)

analyzer.run_complete_sensitivity_analysis()
```

### 3. การกำหนด Parameter Range แบบกำหนดเอง

```python
analyzer = RLSensitivityAnalyzer(
    output_dir="results/custom_analysis",
    use_real_training=True
)

# กำหนดค่าที่ต้องการทดสอบ (น้อยลงเพื่อประหยัดเวลา)
analyzer.param_ranges = {
    "learning_rate": [0.0005, 0.001],  # ทดสอบแค่ 2 ค่า
    "gamma": [0.95, 0.99],
    "batch_size": [64, 128],
}

analyzer.run_complete_sensitivity_analysis()
```

## การรัน Example Script

```bash
cd experiments
python example_real_sensitivity_analysis.py
```

จากนั้นเลือกตัวเลือกที่ต้องการ:
1. **Single parameter analysis** - เริ่มต้นด้วยตัวเลือกนี้
2. **Complete analysis** - สำหรับการวิจัยจริง (ใช้เวลานาน)
3. **Fast simulation** - เพื่อเปรียบเทียบกับ real training
4. **Custom ranges** - กำหนดค่าทดสอบเอง

## เวลาที่ใช้ (ประมาณการ)

| การวิเคราะห์ | จำนวน Training Runs | เวลาที่ใช้ |
|------------|---------------------|-----------|
| Single parameter (4 values) | 4 runs | 15-30 นาที |
| 2 parameters (4 values each) | 8 runs | 30-60 นาที |
| Complete analysis (6 params) | 24 runs | 3-8 ชั่วโมง |
| Custom (3 params, 2 values) | 6 runs | 30-60 นาที |

## Parameters ที่สามารถทดสอบได้

```python
param_ranges = {
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
    "gamma": [0.90, 0.95, 0.99, 0.995],
    "epsilon_decay": [0.995, 0.999, 0.9995, 0.9999],
    "batch_size": [32, 64, 128, 256],
    "hidden_size": [64, 128, 256, 512],
    "target_update_freq": [100, 500, 1000, 2000],
}
```

## ผลลัพธ์ที่ได้รับ

Real training จะให้ metrics ดังนี้:
- **final_reward**: ค่า reward เฉลี่ยจากการ training จริง
- **convergence_episodes**: episodes ที่ใช้จนกว่าจะ converge
- **training_stability**: ความเสถียรของการ training (standard deviation)
- **final_accuracy**: ความแม่นยำ (normalized จาก reward)
- **training_time**: เวลาที่ใช้ในการ training (นาที)
- **all_rewards**: รายละเอียด rewards ทุก episode
- **all_losses**: รายละเอียด losses ทุก episode

## Tips สำหรับการใช้งาน

1. **เริ่มต้นด้วย single parameter**: ทดสอบ 1 parameter ก่อนเพื่อดูเวลาที่ใช้
2. **ลด episodes**: ใช้ 100-200 episodes แค่พอเพื่อประหยัดเวลา
3. **ใช้ Custom ranges**: กำหนดค่าที่สนใจเท่านั้น
4. **รัน overnight**: การวิเคราะห์แบบเต็มควรรันค้างคืน

## การเปรียบเทียบผลลัพธ์

```python
# รัน simulation แบบเร็วเพื่อเปรียบเทียบ
sim_analyzer = RLSensitivityAnalyzer(use_real_training=False)
sim_results = sim_analyzer.run_single_parameter_analysis("learning_rate")

# รัน real training
real_analyzer = RLSensitivityAnalyzer(use_real_training=True)
real_results = real_analyzer.run_single_parameter_analysis("learning_rate")

# เปรียบเทียบผลลัพธ์
print("Simulation vs Real Training Results:")
for sim, real in zip(sim_results, real_results):
    print(f"LR {sim['value']}: Sim={sim['final_reward']:.1f}, Real={real['final_reward']:.1f}")
```

## การตรวจสอบความถูกต้อง

เมื่อใช้ real training คุณจะได้:
- ค่า metrics ที่แท้จริงจากการ training
- การมี variance ตามธรรมชาติของการ training
- ความสัมพันธ์ที่แท้จริงระหว่าง parameters กับ performance

ซึ่งจะแตกต่างจาก simulation ที่เป็นเพียงการประมาณแบบ deterministic