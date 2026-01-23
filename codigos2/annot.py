import json
import copy

# --- (1) Define your input COCO annotation JSONs here ---
# E.g., the "limpiotrain.json", "limpiovalid.json", "limpiotest.json"
json_paths = [
    "_annotations.coco.limpiotrain.json",
    "_annotations.coco.limpiovalid.json",
    "_annotations.coco.limpiotest.json"
]

# --- (2) Load all datasets into memory ---
datasets = []
for path in json_paths:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        datasets.append(data)

# --- (3) Prepare the final merged dict ---
#     We'll copy categories, licenses, info from the first dataset
merged_coco = {
    "info": datasets[0].get("info", {}),
    "licenses": datasets[0].get("licenses", []),
    "categories": datasets[0].get("categories", []),
    "images": [],
    "annotations": []
}

# Counters to offset IDs (so that different JSONs do not overlap)
image_id_offset = 0
annotation_id_offset = 0

# --- (4) Merge images + annotations from each dataset, adjusting IDs ---
for dataset in datasets:
    # If "images" or "annotations" is missing in any dataset, handle safely
    images = dataset.get("images", [])
    annotations = dataset.get("annotations", [])

    # 4A. Copy over images, applying an offset to each image's "id"
    for img in images:
        new_img = copy.deepcopy(img)
        new_img["id"] = img["id"] + image_id_offset
        merged_coco["images"].append(new_img)

    # 4B. Copy over annotations, adjusting "id" and "image_id" by offsets
    for ann in annotations:
        new_ann = copy.deepcopy(ann)
        new_ann["id"] = ann["id"] + annotation_id_offset
        new_ann["image_id"] = ann["image_id"] + image_id_offset
        merged_coco["annotations"].append(new_ann)

    # 4C. Increase offsets after we've processed all images & annotations in this dataset
    image_id_offset += len(images)
    annotation_id_offset += len(annotations)

# --- (5) Write out the final merged COCO file ---
with open("merged_annotations.json", "w", encoding='utf-8') as f:
    json.dump(merged_coco, f, ensure_ascii=False, indent=2)

print("Merging complete! Results in merged_annotations.json")
