# Deep-Learning-Project
Deep-Learning-Project: Chest X-ray Classification

Multi-label classification of 14 thoracic diseases on NIH ChestX-ray14, using DenseNet-121.

Programs
File	What it does
get_dataset.py	Downloads the dataset from Hugging Face (chehablab/NIHChestXR, 112,120 images, resized copy of NIH ChestX-ray14) and plots one sample X-ray with its patient, scan, age, sex and label numbers.
starter_training.py	Pilot training run on 3,000 images. Splits by patient, fine-tunes a pretrained DenseNet-121 with 14 sigmoid outputs for 3 epochs, then prints per-label AUC on the held-out patients.
Steps taken
Installed Python 3.12 and the packages below (py -m pip install ...).
Upgraded NumPy, pyarrow and datasets to fix a NumPy/pyarrow version error.
Ran get_dataset.py. The data is cached at C:\Users\<user>\.cache\huggingface (about 2.5 GB).
Ran starter_training.py on the CPU, which was too slow.
Installed the CUDA build of PyTorch (py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126) and confirmed the GPU (RTX 4070 Laptop) with torch.cuda.is_available().
Re-ran the pilot on the GPU (about 72 seconds). Loss fell from about 0.65 to about 0.12 and AUC was printed for 14 labels. These numbers are from a tiny pilot, so they are not results.
Label mapping

Dataset indices: 0 = No Finding, 1 to 14 = Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia. The training script drops index 0 and treats "No Finding" as all zeros.

Imports

Install:

py -m pip install --upgrade numpy pyarrow datasets matplotlib scikit-learn
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
Import	Package	Used for
datasets.load_dataset	datasets	Downloading and loading the Hugging Face dataset
matplotlib.pyplot	matplotlib	Plotting a sample X-ray (get_dataset.py)
torch, torch.nn	torch	Tensors, model, loss, optimizer, GPU
torch.utils.data.DataLoader	torch	Batching and shuffling
torchvision.models, torchvision.transforms	torchvision	Pretrained DenseNet-121, image resize and normalization
sklearn.model_selection.GroupShuffleSplit	scikit-learn	Patient-level train/test split
sklearn.metrics.roc_auc_score	scikit-learn	Per-label AUC
time	standard library	Elapsed-time prints
Notes
Splits are by patient_id to avoid leakage. This copy has no official train/test lists, so the split is our own and is not directly comparable to published results.
Next: remove ds.select(range(3000)), add a patient-level validation split, and run the pretraining and loss comparisons on the full data.