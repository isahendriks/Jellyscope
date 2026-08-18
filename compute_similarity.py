"""
Build two similarity indexes on the jellyscope_crops dataset so you can do
"find similar images" in the FiftyOne App:

    - img_sim_clip:   generic visual similarity via CLIP (FiftyOne zoo model)
    - img_sim_custom: domain-specific similarity via your own trained ViT
                       classifier's backbone features

Run this on the machine/server that has both GPU access and a working
connection to the network drive (same credential requirements as
load_jellyscope_dataset.py -- run it from a session that can actually reach
\\uw.lu.se\research, e.g. the RDP session, not a bare SSH one).

Install:
    pip install fiftyone-brain

Run:
    python compute_similarity.py
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

# ---- CONFIG -------------------------------------------------------------

DATASET_NAME = "jellyscope_crops"

CUSTOM_CKPT_PATH = r"C:\Users\Admin\Documents\GitHub\Jellyscope\ImagePipeline\ClassClassification\Models\vit_classifier_F1_0.8697_acc_0.9293.pth"
NUM_CLASSES = 14  # matches classifier.3.weight shape in the checkpoint

BATCH_SIZE = 256
NUM_WORKERS = 32 # parallel image-loading workers; tune for your machine

# ---------------------------------------------------------------------------


# ---- Custom ViT (same architecture as Pipeline_development/ClassClassification/Train_ViT.py) ----
class ViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = models.vit_b_16(weights=weights)
        vit_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

        self.resize = transforms.Resize((224, 224))

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.size_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(vit_features + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def embed(self, img):
        # Same as forward(), through the ViT backbone only -- this is the
        # 768-dim visual embedding used for similarity search. It skips the
        # size-fusion/classification head, since crop size isn't a property
        # of visual similarity (it's already queryable separately via the
        # region_size_mm2 field).
        img = self.stem(img)
        img = self.resize(img)
        img = (img - self.imagenet_mean) / self.imagenet_std
        return self.vit(img)  # [batch, 768]


class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        diff = abs(w - h)
        if w < h:
            left = diff // 2
            right = diff - left
            padding = (left, 0, right, 0)
        else:
            top = diff // 2
            bottom = diff - top
            padding = (0, top, 0, bottom)
        return TF.pad(img, padding, fill=0)


custom_preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    PadToSquare(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class CropDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img = Image.open(path).convert("RGB")
        return idx, custom_preprocess(img)


def collate(batch):
    idxs, tensors = zip(*batch)
    return list(idxs), torch.stack(tensors)


def compute_custom_embeddings(filepaths, device):
    model = ViT(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CUSTOM_CKPT_PATH, map_location=device))
    model.to(device).eval()

    loader = DataLoader(
        CropDataset(filepaths),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
    )

    embeddings = np.zeros((len(filepaths), 768), dtype=np.float32)
    n_done = 0

    with torch.no_grad():
        for idxs, imgs in loader:
            imgs = imgs.to(device)
            feats = model.embed(imgs).cpu().numpy()
            for i, idx in enumerate(idxs):
                embeddings[idx] = feats[i]
            n_done += len(idxs)
            print(f"Custom ViT embeddings: {n_done}/{len(filepaths)}", end="\r")

    print()
    return embeddings


def update_custom_index(dataset, all_ids, id_to_filepath, device, brain_key="img_sim_custom"):
    if brain_key in dataset.list_brain_runs():
        results = dataset.load_brain_results(brain_key)
        indexed_ids = set(results.sample_ids)
        new_ids = [i for i in all_ids if i not in indexed_ids]

        if not new_ids:
            print(f"{brain_key}: already up to date ({len(indexed_ids)} samples)")
            return

        print(f"{brain_key}: {len(new_ids)} new samples to add (already have {len(indexed_ids)})")
        new_filepaths = [id_to_filepath[i] for i in new_ids]
        embeddings = compute_custom_embeddings(new_filepaths, device)
        results.add_to_index(embeddings, sample_ids=new_ids)
        results.save()
        print(f"{brain_key}: added {len(new_ids)} samples")
    else:
        print(f"{brain_key}: building from scratch for {len(all_ids)} samples...")
        filepaths = [id_to_filepath[i] for i in all_ids]
        embeddings = compute_custom_embeddings(filepaths, device)
        fob.compute_similarity(dataset, embeddings=embeddings, brain_key=brain_key)
        print(f"{brain_key}: built")


def update_clip_index(dataset, all_ids, brain_key="img_sim_clip"):
    if brain_key in dataset.list_brain_runs():
        results = dataset.load_brain_results(brain_key)
        indexed_ids = set(results.sample_ids)
        new_ids = [i for i in all_ids if i not in indexed_ids]

        if not new_ids:
            print(f"{brain_key}: already up to date ({len(indexed_ids)} samples)")
            return

        print(f"{brain_key}: {len(new_ids)} new samples to add (already have {len(indexed_ids)})")
        new_view = dataset.select(new_ids)
        model = foz.load_zoo_model("clip-vit-base32-torch")
        embeddings = new_view.compute_embeddings(model)
        results.add_to_index(embeddings, sample_ids=new_ids)
        results.save()
        print(f"{brain_key}: added {len(new_ids)} samples")
    else:
        print(f"{brain_key}: building from scratch for {len(all_ids)} samples...")
        fob.compute_similarity(dataset, model="clip-vit-base32-torch", brain_key=brain_key)
        print(f"{brain_key}: built")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = fo.load_dataset(DATASET_NAME)
    all_ids = dataset.values("id")
    all_filepaths = dataset.values("filepath")
    id_to_filepath = dict(zip(all_ids, all_filepaths))
    print(f"Dataset has {len(all_ids)} samples")

    update_custom_index(dataset, all_ids, id_to_filepath, device)
    update_clip_index(dataset, all_ids)

    print("Done. Both 'img_sim_custom' and 'img_sim_clip' are up to date in the App.")


if __name__ == "__main__":
    main()
