import torch
from utils import *
from torchvision import models
from tqdm import tqdm
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

DATA_DIR = '/srv/submission/stenosis/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Load data
    batch_size = 16
    train_dataset = CustomImageDataset(DATA_DIR)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    # Load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 50  # Define the number of epochs
    save_every = 32   # Save checkpoint every certain amount of iterations

    for epoch in range(num_epochs):
        print("### Epoch ", epoch + 1, "###")
        model.train()
        running_loss = 0.0
        iteration = 0
        for imgs, targets in tqdm(data_loader):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            # Put your training logic here

            print(f"{[img.shape for img in imgs] = }")
            print(f"{[type(target) for target in targets] = }")
            for name, loss_val in loss_dict.items():
                print(f"{name:<20}{loss_val:.3f}")

            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            iteration += 1

            if iteration % save_every == 0:
                iteration_loss = running_loss / save_every
                save_checkpoint(model, optimizer, iteration, iteration_loss, filename=f"checkpoints/checkpoint_batch{batch_size}_epoch{epoch+1}_iter{iteration}.pth")
                running_loss = 0

    if iteration % save_every != 0:  # Check if there were remaining iterations after the last save
        iteration_loss = running_loss / (iteration % save_every)
        save_checkpoint(model, optimizer, iteration, iteration_loss, filename=f"checkpoints/checkpoint_batch{batch_size}_epoch{epoch+1}_final.pth")


def eval(): 
    # load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    load_checkpoint(model, optimizer, filename="checkpoints/checkpoint_iter30")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load data
    train_dataset = CustomImageDataset(DATA_DIR)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    all_outs = []
    for imgs, _ in data_loader:  # We only need imgs for inference
        imgs = list(img.to(device) for img in imgs)

        with torch.no_grad():
            outputs = model(imgs)

        # Process the outputs as needed
        for i, output in enumerate(outputs):
            print(f"Image {i}:")
            print(f"  Boxes: {output['boxes']}")
            print(f"  Labels: {output['labels']}")
            print(f"  Scores: {output['scores']}")
            print(f"  Masks: {output['masks']}")
            output['masks'] = output['masks'].squeeze(1)

            output['masks'][output['masks'] > 0.5] = 1
            output['masks'][output['masks'] <= 0.5] = 0
            topk_scores, topk_indices = torch.topk(output['scores'], k=5, largest=True)

            all_outs.append((imgs[i], {
                    'boxes': output['boxes'][topk_indices],
                    'labels': output['labels'][topk_indices],
                    'scores': output['scores'][topk_indices],
                    'masks': output['masks'][topk_indices]
                }))
        break

    plot([all_outs[0]])


def main(): 
    train()


if __name__ == '__main__':
    main()