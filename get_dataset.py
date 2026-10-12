from datasets import load_dataset
import matplotlib.pyplot as plt

ds = load_dataset("chehablab/NIHChestXR", split="train")
sample = ds[314]
img = sample["image"]

plt.imshow(img, cmap="gray")
plt.title(f"Patient {sample['patient_id']} | Scan {sample['scan_id']} | Age {sample['age']} | Sex {sample['sex']} | Labels {sample['labels']}")
plt.axis("off")
plt.show()
