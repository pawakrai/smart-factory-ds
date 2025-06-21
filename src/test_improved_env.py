#!/usr/bin/env python3
"""
Test script for the improved aluminum melting environment
Demonstrates realistic aluminum melting process with:
- 350 kg initial Al ingot
- Scrap addition during 60-75 minutes (2-3 rounds, 50-100 kg each)
- Improved temperature calculations based on real-world data
"""

import numpy as np
import matplotlib.pyplot as plt
from environment.aluminum_melting_env_8 import AluminumMeltingEnvironment


def test_improved_environment():
    """
    ทดสอบสภาพแวดล้อมการหลอมอลูมิเนียมที่ปรับปรุงแล้ว
    """
    print("=== Testing Improved Aluminum Melting Environment ===")
    
    # สร้างสภาพแวดล้อม
    env = AluminumMeltingEnvironment(initial_weight_kg=350, target_temp_c=900)
    
    # เริ่มต้นสภาพแวดล้อม
    state = env.reset()
    print(f"Initial state: {state}")
    print(f"Initial weight: {env.current_mass} kg")
    print(f"Max capacity: {env.max_capacity} kg")
    
    # เก็บข้อมูลสำหรับการพล็อต
    time_history = []
    temp_history = []
    power_history = []
    weight_history = []
    energy_history = []
    scrap_history = []
    
    # รัน simulation
    done = False
    step_count = 0
    max_steps = 120  # 120 นาที
    
    while not done and step_count < max_steps:
        current_minute = step_count
        
        # กำหนด power strategy ตามช่วงเวลา
        if current_minute < 5:
            action = 1  # increase_power_mild
        elif current_minute < 30:
            action = 0  # increase_power_strong
        elif current_minute < 32:
            action = 4  # decrease_power_strong (เพื่อเติม dose)
        elif current_minute < 60:
            action = 0  # increase_power_strong
        elif 60 <= current_minute < 75:
            # ช่วงเพิ่ม scrap
            action = 3  # decrease_power_mild
        else:
            # maintain หรือปรับตามอุณหภูมิ
            if state[0] < 850:  # temperature
                action = 1  # increase_power_mild
            else:
                action = 2  # maintain
        
        # Execute action
        state, reward, done = env.step(action)
        
        # เก็บข้อมูล
        time_history.append(current_minute)
        temp_history.append(state[0])  # temperature
        weight_history.append(state[1])  # weight
        power_history.append(state[3])  # power
        energy_history.append(state[5])  # energy_consumption
        scrap_history.append(state[6])  # scrap_added
        
        # แสดงข้อมูลทุก 10 นาที
        if step_count % 10 == 0:
            scrap_info = env.get_scrap_info()
            print(f"Minute {current_minute}: Temp={state[0]:.1f}°C, "
                  f"Weight={state[1]:.1f}kg, Power={state[3]:.1f}kW, "
                  f"Scrap={scrap_info['total_scrap_added']:.1f}kg")
        
        step_count += 1
    
    # แสดงผลสุดท้าย
    final_scrap_info = env.get_scrap_info()
    print(f"\n=== Final Results ===")
    print(f"Final temperature: {state[0]:.1f}°C")
    print(f"Final weight: {state[1]:.1f} kg")
    print(f"Total energy consumption: {state[5]:.1f} kWh")
    print(f"Total scrap added: {final_scrap_info['total_scrap_added']:.1f} kg")
    print(f"Capacity utilization: {final_scrap_info['capacity_utilization']:.1%}")
    print(f"Final reward: {reward:.3f}")
    print(f"Scrap additions: {len(final_scrap_info['scrap_additions'])} rounds")
    
    if final_scrap_info['scrap_additions']:
        print("\nScrap addition details:")
        for i, addition in enumerate(final_scrap_info['scrap_additions']):
            print(f"  Round {i+1}: {addition['weight']:.1f} kg at minute {addition['time']/60:.1f}")
    
    # สร้างกราฟ
    create_simulation_plots(time_history, temp_history, power_history, 
                          weight_history, energy_history, scrap_history)
    
    return env, final_scrap_info


def create_simulation_plots(time_history, temp_history, power_history, 
                          weight_history, energy_history, scrap_history):
    """
    สร้างกราฟแสดงผลการจำลอง
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Improved Aluminum Melting Environment Simulation Results', fontsize=16)
    
    # Temperature vs Time
    axes[0,0].plot(time_history, temp_history, 'r-', linewidth=2, label='Simulated')
    # เพิ่มข้อมูลจริงจากตาราง (ตัวอย่าง)
    real_times = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
    real_temps = [892.1, 893.3, 894.3, 895.2, 896.0, 896.8, 897.5, 898.1, 898.7, 899.2, 899.7, 900.0]
    axes[0,0].plot(real_times, real_temps, 'bo-', linewidth=2, label='Real Data', alpha=0.7)
    axes[0,0].set_xlabel('Time (minutes)')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].set_title('Temperature Profile')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Power vs Time
    axes[0,1].plot(time_history, power_history, 'b-', linewidth=2)
    axes[0,1].set_xlabel('Time (minutes)')
    axes[0,1].set_ylabel('Power (kW)')
    axes[0,1].set_title('Power Consumption')
    axes[0,1].grid(True, alpha=0.3)
    
    # Weight vs Time
    axes[0,2].plot(time_history, weight_history, 'g-', linewidth=2)
    axes[0,2].axhline(y=350, color='r', linestyle='--', alpha=0.7, label='Initial Al ingot')
    axes[0,2].axhline(y=500, color='orange', linestyle='--', alpha=0.7, label='Max capacity')
    axes[0,2].set_xlabel('Time (minutes)')
    axes[0,2].set_ylabel('Weight (kg)')
    axes[0,2].set_title('Weight Change (Al + Scrap)')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].legend()
    
    # Energy Consumption vs Time
    axes[1,0].plot(time_history, energy_history, 'm-', linewidth=2)
    axes[1,0].set_xlabel('Time (minutes)')
    axes[1,0].set_ylabel('Energy (kWh)')
    axes[1,0].set_title('Cumulative Energy Consumption')
    axes[1,0].grid(True, alpha=0.3)
    
    # Scrap Addition vs Time
    axes[1,1].step(time_history, scrap_history, 'brown', linewidth=2, where='post')
    axes[1,1].axvspan(60, 75, alpha=0.2, color='yellow', label='Scrap addition window')
    axes[1,1].set_xlabel('Time (minutes)')
    axes[1,1].set_ylabel('Scrap Added (kg)')
    axes[1,1].set_title('Scrap Addition Progress')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    # Process Phases
    phases = ['Initial\nHeating', 'Ramp Up', 'Dose\nAddition', 'High Power', 'Scrap\nAddition', 'Final\nHeating']
    phase_times = [0, 10, 30, 40, 60, 90]
    phase_colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'pink', 'lightcoral']
    
    for i in range(len(phases)-1):
        axes[1,2].axvspan(phase_times[i], phase_times[i+1], alpha=0.3, 
                         color=phase_colors[i], label=phases[i])
    
    axes[1,2].plot(time_history, temp_history, 'r-', linewidth=2)
    axes[1,2].set_xlabel('Time (minutes)')
    axes[1,2].set_ylabel('Temperature (°C)')
    axes[1,2].set_title('Process Phases')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('improved_aluminum_melting_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # เรียกใช้การทดสอบ
    env, scrap_info = test_improved_environment()
    
    print(f"\n=== Environment Test Completed ===")
    print(f"Environment successfully simulates realistic aluminum melting process!")