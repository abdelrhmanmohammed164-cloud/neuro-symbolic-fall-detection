"""plot_symbolic_v3.py: figures for reports/symbolic_v3_precision from results.json (run after evaluate_symbolic_v3.py)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
OUT = os.path.join(ROOT, "reports", "symbolic_v3_precision"); FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
r = json.load(open(os.path.join(OUT, "results.json")))
BLUE, ORANGE, GREY = "#4c78a8", "#f58518", "#9aa5b1"

# precision-recall trade-off over tau_h (inner CV, selection data only)
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for lab, col, name in (("A", GREY, "A original V3"), ("chosen", BLUE, "E (each fold's chosen variant)")):
    pts = r["pr_tradeoff_inner_cv"][lab]
    ax.plot([p["Recall"] for p in pts], [p["Precision"] for p in pts], "o-", color=col, ms=4, lw=1.5, label=name)
    for p in pts[::2]:
        ax.annotate(f"{p['tau_h']:.2f}", (p["Recall"], p["Precision"]), fontsize=7, color=col, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Recall (inner CV)"); ax.set_ylabel("Precision (inner CV)"); ax.set_xlim(0, 0.7); ax.set_ylim(0, 1.05)
ax.set_title("Precision-recall over tau_h (labels = tau_h in m)", fontsize=10); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "pr_tradeoff.png"), dpi=140); plt.close()

# confusion matrices on the test folds
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
for a, n in zip(axs, ("A", "E (selected)")):
    m = r["test"][n]; M = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    a.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{M[i, j]}", ha="center", va="center", color="white" if M[i, j] > M.max() * .6 else "black")
    a.set_xticks([0, 1], ["pred no fall", "pred fall"]); a.set_yticks([0, 1], ["no fall", "fall"])
    a.set_title(f"{'A original V3' if n == 'A' else 'E improved V3'}\nP {m['Precision']:.3f}  R {m['Recall']:.3f}  F1 {m['F1']:.3f}", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "confusion_test.png"), dpi=140); plt.close()

# false positives of A by neck height, and how many the chosen variant removes (inner CV)
import pandas as pd
fp = pd.read_csv(os.path.join(OUT, "false_positives_A_inner_cv.csv"))
order = ["<0.3", "0.3-0.5", "0.5-0.8", ">=0.8"]
tot = [int((fp.height_bin == b).sum()) for b in order]; rem = [int(((fp.height_bin == b) & fp.removed_by_chosen).sum()) for b in order]
fig, ax = plt.subplots(figsize=(6.2, 3.4)); x = np.arange(4)
ax.bar(x, tot, color=GREY, label="false-alarm frames of A"); ax.bar(x, rem, color=BLUE, label="removed by the chosen variant")
for i in range(4):
    ax.text(i, tot[i] + 2, f"{rem[i]}/{tot[i]}", ha="center", fontsize=8)
ax.set_xticks(x, [f"neck {b} m" for b in order]); ax.set_ylabel("frames (inner CV)"); ax.legend(fontsize=8)
ax.set_title("Where V3's false alarms happen and which ones the improvement removes", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "false_positives.png"), dpi=140); plt.close()

# stress tests
kinds = ["none", "noise 3 cm", "noise 6 cm", "20% frames missing", "height scale x0.9", "height scale x1.1", "floor offset +5 cm", "floor offset -5 cm"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for a, met in zip(axs, ("Precision", "F1")):
    for i, (n, col) in enumerate((("A", GREY), ("E (selected)", BLUE))):
        a.barh(np.arange(len(kinds)) + i * 0.38, [r["stress_test"][f"{n}|{k}"][met] for k in kinds], height=0.36, color=col, label=n.replace("E (selected)", "E improved").replace("A", "A original"))
    a.set_yticks(np.arange(len(kinds)) + 0.19, kinds, fontsize=8); a.invert_yaxis(); a.set_xlim(0, 1); a.set_title(f"Test {met} under input changes", fontsize=10)
axs[0].legend(fontsize=8, loc="lower right"); plt.tight_layout(); plt.savefig(os.path.join(FIG, "stress_tests.png"), dpi=140); plt.close()
print("figures in", FIG)
