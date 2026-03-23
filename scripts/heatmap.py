# heatmap_from_report.py
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- input/output ----
report_path = Path(
    "/Users/aya.elmir/gameth_llm_tmp_netys/results/k1_5_groups/k1_20_trials_5_rounds/analysis_report_2-3-4-5-6_players.txt"
)
out_path = report_path.with_name("perm_sw_heatmap_n2_to_n6.png")

# ---- parse table rows from section (g) ----
# Expected row format:
# 2 Mixtral 9 9 100.0% FD
pattern = re.compile(
    r"^\s*(\d+)\s+([A-Za-z0-9\- ]+?)\s+(\d+)\s+(\d+)\s+([0-9.]+)%\s+(FD|PC)\s*$",
    re.M,
)

rows = []
for m in pattern.finditer(report_path.read_text()):
    n = int(m.group(1))
    family = m.group(2).strip()
    perm_sw_pct = float(m.group(5))
    rows.append((family, n, perm_sw_pct))

if not rows:
    raise RuntimeError("No rows parsed from report file.")

# ---- build matrix: rows=families, cols=n=2..6 ----
families = sorted({r[0] for r in rows})
n_values = [2, 3, 4, 5, 6]

fam_to_i = {f: i for i, f in enumerate(families)}
n_to_j = {n: j for j, n in enumerate(n_values)}

mat = np.full((len(families), len(n_values)), np.nan)
for family, n, pct in rows:
    if n in n_to_j:
        mat[fam_to_i[family], n_to_j[n]] = pct

# ---- plot heatmap ----
fig, ax = plt.subplots(figsize=(9, 4))
# cmap = plt.cm.viridis.copy()

cmap = LinearSegmentedColormap.from_list(
    "soft_blue",
    ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd"]
)
cmap.set_bad("#e6e6e6")  # gray for missing combinations

im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=100)

ax.set_xticks(range(len(n_values)))
ax.set_xticklabels([str(n) for n in n_values])
ax.set_yticks(range(len(families)))
ax.set_yticklabels(families)

ax.set_xlabel("Number of players (n)")
ax.set_ylabel("Model families")
ax.set_title("perm_sw% across model families and player count")

# Optional cell labels
for i in range(len(families)):
    for j in range(len(n_values)):
        v = mat[i, j]
        if np.isnan(v):
            ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="black")
        else:
            txt_color = "white" if v >= 55 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=txt_color)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("perm_sw%")

plt.tight_layout()
fig.savefig(out_path, dpi=300)
print(f"Saved: {out_path}")
