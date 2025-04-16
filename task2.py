import json
import pandas as pd
from collections import defaultdict

# Load your model's prediction results
with open('pred.json', 'r') as f:
    predictions = json.load(f)

# Group by image_id
image_predictions = defaultdict(list)
for pred in predictions:
    image_predictions[pred['image_id']].append(pred)

# Extract numbers by sorting bbox by x-axis
image_numbers = {}
for image_id, preds in image_predictions.items():
    sorted_preds = sorted(
        preds, key=lambda x: x['bbox'][0])
    number_str = ''.join(str(p['category_id'] - 1) for p in sorted_preds)
    image_numbers[image_id] = number_str


all_image_ids = list(range(1, 13069))

full_results = []
for image_id in sorted(all_image_ids):
    if image_id in image_numbers:
        number = image_numbers[image_id]
    else:
        number = "-1"
    full_results.append({"image_id": image_id, "pred_label": number})


df = pd.DataFrame(full_results)
df.to_csv("pred.csv", index=False)
