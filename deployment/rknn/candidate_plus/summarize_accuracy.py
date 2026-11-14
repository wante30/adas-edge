import json
from pathlib import Path

path = Path(__file__).resolve().parent / "accuracy.json"
data = json.loads(path.read_text(encoding="utf-8"))
summary = data["summary"]
keys = [key for key in summary if isinstance(summary[key], dict) and "logits_mae" in summary[key]]
for key in keys:
    item = summary[key]
    print("==", key)
    print("  logits_mae", item["logits_mae"]["mean"], "max", item["logits_max_abs"]["mean"])
    if "prob_mae" in item:
        print("  prob_mae", item["prob_mae"]["mean"])
    for name, value in item.get("classes", {}).items():
        print(
            "  %-14s iou=%s f1=%s disagree=%s"
            % (
                name,
                value["iou"]["mean"] if value["iou"] else None,
                value["f1"]["mean"] if value["f1"] else None,
                value["disagreement_fraction"]["mean"],
            )
        )
print("color_order", summary["color_order_check"]["winner"])
print("samples", summary["samples"])
