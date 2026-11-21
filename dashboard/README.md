# Dashboard

`dashboard.html` is a self-contained page (data embedded, no internet, no server). Open it in Chrome, Edge,
Firefox or Safari.

| Tab | What it shows |
|---|---|
| Overview | Neural ensemble (out-of-fold, threshold 0.5) and improved V3 on the URFD test folds, plus the neural ensemble on GMDCSA-24 |
| Video explorer | Any of the 70 videos frame by frame: neural probability, ground truth and metric neck height; CSV export of the selected video |
| Model comparison | Neural threshold explorer (post-hoc, for exploration only) and the V3 results per fold |
| Generalization | FUKinect-Fall external test of V3 and the split audit warnings |

After any result changes, rebuild it from the project files:

```bash
python scripts/build_dashboard.py      # template.html + reports/ + processed_data/kinect_height -> dashboard.html
```

The dashboard only displays saved results. It does not run the models.
