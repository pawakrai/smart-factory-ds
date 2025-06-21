import torch
from collections import OrderedDict

# ระบุตำแหน่งของไฟล์ checkpoint
# (ต้องแน่ใจว่า path ถูกต้องจากตำแหน่งที่รันสคริปต์นี้)
checkpoint_path = "models/dqn_final_model_10.pth"

print(f"Inspecting checkpoint file: {checkpoint_path}\n")

try:
    # โหลด state dictionary จากไฟล์
    # map_location='cpu' ช่วยให้โหลดโมเดลได้แม้ว่าเครื่องจะไม่มี GPU
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    print("✅ Checkpoint loaded successfully.")
    print("-" * 50)
    print("Model Architecture and Parameter Shapes:")
    print("-" * 50)

    # วนลูปเพื่อแสดงข้อมูลของแต่ละ layer
    for param_name, param_tensor in state_dict.items():
        print(f"Layer: {param_name}")
        print(f"  - Shape: {param_tensor.shape}")

    print("\n" + "-" * 50)
    print("Example Parameter Values (first 2 values of some tensors):")
    print("-" * 50)

    # แสดงตัวอย่างค่าพารามิเตอร์ของบาง layer
    for param_name, param_tensor in list(state_dict.items())[:4]:  # แสดงแค่ 4 layers แรก
        print(f"Layer: {param_name}")
        # แปลง tensor เป็น numpy array เพื่อให้แสดงผลง่ายขึ้น
        param_numpy = param_tensor.numpy().flatten()  # ทำให้เป็น array 1 มิติ

        # แสดงแค่ 2 ค่าแรกเพื่อไม่ให้ผลลัพธ์ยาวเกินไป
        if len(param_numpy) > 2:
            print(f"  - Example values: {param_numpy[:2]} ...")
        else:
            print(f"  - Values: {param_numpy}")


except FileNotFoundError:
    print(f"❌ Error: Checkpoint file not found at '{checkpoint_path}'")
except Exception as e:
    print(f"❌ An error occurred: {e}")
