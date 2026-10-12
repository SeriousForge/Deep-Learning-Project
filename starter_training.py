import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

ds = load_dataset("chehablab/NIHChestXR", split="train")
ds = ds.select(range(3000))          # small test run; remove later

# patient-level split
pids = ds["patient_id"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(range(len(pids)), groups=pids))

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def prep(batch):
    x = torch.stack([tf(im.convert("RGB")) for im in batch["image"]])
    y = torch.zeros(len(x), 14)
    for i, labs in enumerate(batch["labels"]):
        for l in labs:
            if l > 0:               # index 0 = "No Finding": leave all zeros
                y[i, l - 1] = 1
    return {"x": x, "y": y}

def collate(items):
    return {"x": torch.stack([i["x"] for i in items]),
            "y": torch.stack([i["y"] for i in items])}

train_ds = ds.select(train_idx).with_transform(prep)
test_ds = ds.select(test_idx).with_transform(prep)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=64, collate_fn=collate, num_workers=0)

model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, 14)
model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

start = time.time()
for epoch in range(3):
    model.train()
    for i, b in enumerate(train_dl):
        opt.zero_grad()
        loss = loss_fn(model(b["x"].to(device)), b["y"].to(device))
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(f"epoch {epoch} batch {i} loss {loss.item():.4f} ({time.time()-start:.0f}s)", flush=True)

model.eval()
probs, ys = [], []
with torch.no_grad():
    for b in test_dl:
        probs.append(torch.sigmoid(model(b["x"].to(device))).cpu())
        ys.append(b["y"])
probs, ys = torch.cat(probs).numpy(), torch.cat(ys).numpy()
for k in range(14):
    if ys[:, k].min() != ys[:, k].max():     # AUC needs both classes
        print("label", k + 1, "AUC", round(roc_auc_score(ys[:, k], probs[:, k]), 3))