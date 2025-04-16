import os
import json
import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CocoDetection, ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from PIL import Image

from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
FREEZE_BACK_BONE_EPOCHS = 5


class CustomTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(root) if f.endswith(
                ('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]


def collate_fn(batch):
    return tuple(zip(*batch))


def convert_coco_to_frcnn_format(target_list):
    boxes = []
    labels = []
    for obj in target_list:
        bbox = obj["bbox"]
        x, y, w, h = bbox
        boxes.append([x, y, x + w, y + h])
        labels.append(obj["category_id"])
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
    }


def main():
    train_transform = T.Compose([T.ToTensor()])
    test_transform = T.Compose([T.ToTensor()])

    train_dataset = CocoDetection(
        root="./nycu-hw2-data/train/",
        annFile="./nycu-hw2-data/train.json",
        transform=train_transform)
    val_dataset = CocoDetection(
        root="./nycu-hw2-data/valid/",
        annFile="./nycu-hw2-data/valid.json",
        transform=train_transform)
    test_dataset = CustomTestDataset(
        root="./nycu-hw2-data/test",
        transform=test_transform)

    trainloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn)
    valloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn)
    testloader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn_v2(weights="COCO_V1")
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=0, min_lr=1e-10)

    mAP = 0
    frozen = True

    for epoch in range(EPOCHS):
        losses = []

        # Freeze backbone for fine tuning
        if frozen and epoch >= FREEZE_BACK_BONE_EPOCHS:
            for param in model.backbone.parameters():
                param.requires_grad = True
            frozen = False

        # Train
        model.train()
        for images, targets in trainloader:
            images = [img.to(DEVICE) for img in images]
            targets = [
                {k: v.to(DEVICE) for k, v in convert_coco_to_frcnn_format(t).items()}
                for t in targets
            ]

            optimizer.zero_grad()  # 1. Zero out old gradients
            # 2. Forward pass 3. Compute loss
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()  # 4. Backward pass
            optimizer.step()  # 5. Update weights

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)  # 6. Update learning rate

        # Validation
        model.eval()
        map_metric = MeanAveragePrecision(iou_type="bbox")
        with torch.no_grad():
            for images, targets in valloader:
                images = [img.to(DEVICE) for img in images]
                outputs = model(images)

                outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
                targets = [
                    {k: v.cpu() for k, v in convert_coco_to_frcnn_format(t).items()}
                    for t in targets
                ]

                map_metric.update(outputs, targets)

        map_results = map_metric.compute()
        current_mAP = map_results['map']

        print(
            f"Epoch: {epoch + 1} | Loss: {avg_loss:.4f} | Validation mAP@[.5:.95]: {current_mAP * 100:.2f}%")

        if mAP < current_mAP:
            mAP = current_mAP
            torch.save(model.state_dict(), "rcnn.pth")

    model.load_state_dict(torch.load("rcnn.pth", weights_only=True))

    # Test
    model.eval()
    results = []
    with torch.no_grad():
        for images, paths in testloader:
            images = [x.to(DEVICE) for x in images]
            outputs = model(images)

            for path, output in zip(paths, outputs):
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    if score > 0.5:
                        img_name = os.path.basename(path)
                        clean_name = os.path.splitext(img_name)[0]

                        # change into coco format
                        x_min, y_min, x_max, y_max = box
                        width = x_max - x_min
                        height = y_max - y_min
                        bbox = [x_min, y_min, width, height]

                        results.append({
                            "image_id": int(clean_name),
                            "bbox": [float(x) for x in bbox],
                            "score": float(score),
                            "category_id": int(label)
                        })

    with open("pred.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Predictions saved to pred.json")


if __name__ == '__main__':
    main()
