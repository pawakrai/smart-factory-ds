import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

def plot_schedule(schedule):
    """
    schedule: (ls, le, ms, me, ps, pe, furnace, batch_id)
    เวลาเริ่มที่ 9:00 -> SHIFT_START = 540
    1 slot = 30 นาที
    """
    import matplotlib.patches as mpatches

    SHIFT_START = 9*60
    furnace_y = {0:10, 1:30}
    height = 8

    fig, ax = plt.subplots(figsize=(12,6))

    for (ls, le, ms, me, ps, pe, f, b_id) in schedule:
        # แปลง slot->นาที + shift 9:00
        load_start_t = SHIFT_START + ls*30
        load_dur = (le - ls)*30

        melt_start_t = SHIFT_START + ms*30
        melt_dur = (me - ms)*30

        pour_start_t = SHIFT_START + ps*30
        pour_dur = (pe - ps)*30

        # Loading (เทาอ่อน)
        ax.broken_barh(
            [(load_start_t, load_dur)],
            (furnace_y[f], height),
            facecolors="lightgray",
            alpha=0.4,
            edgecolor="black"
        )
        # Melting (สีน้ำเงิน)
        ax.broken_barh(
            [(melt_start_t, melt_dur)],
            (furnace_y[f], height),
            facecolors="tab:blue",
            alpha=1.0,
            edgecolor="black"
        )
        # Pouring (เทาเข้ม)
        ax.broken_barh(
            [(pour_start_t, pour_dur)],
            (furnace_y[f], height),
            facecolors="dimgray",
            alpha=0.8,
            edgecolor="black"
        )

        # ใส่ Label (batch) ตรงกลาง Melting (หรือจะใส่ที่ Pour ก็ได้)
        ax.text(
            melt_start_t + melt_dur/2,
            furnace_y[f] + height/2,
            f"{b_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=10
        )

    # แกน X: 9:00 -> 9:00 + 24 ชม = 540 -> 1980
    ax.set_xlim(540, 540+1440)
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Furnace")

    # สร้าง tick ชั่วโมง
    import numpy as np
    xticks = np.arange(540, 540+1441, 60)
    ax.set_xticks(xticks)
    xlabels = []
    for x in xticks:
        hr = (x // 60) % 24
        mn = x % 60
        xlabels.append(f"{hr:02d}:{mn:02d}")
    ax.set_xticklabels(xlabels)

    # Y-axis
    ax.set_yticks([furnace_y[0]+height/2, furnace_y[1]+height/2])
    ax.set_yticklabels(["Furnace A", "Furnace B"])

    ax.set_title("Melting Schedule with 3 phases (Loading, Melting, Pouring)")
    ax.grid(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="lightgray", alpha=0.4, edgecolor="black", label="Loading"),
        Patch(facecolor="tab:blue", alpha=1.0, edgecolor="black", label="Melting"),
        Patch(facecolor="dimgray", alpha=0.8, edgecolor="black", label="Pouring"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()
