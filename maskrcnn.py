import torch
from utils import *
from torchvision import models
from tqdm import tqdm
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

DATA_DIR = 'datasets/stenosis/'


def train():
    # Load data
    train_dataset = CustomImageDataset(DATA_DIR, split='train')
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    # Load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 1  # Define the number of epochs
    save_every = 10   # Save checkpoint every 10 iterations

    for epoch in range(num_epochs):
        print("### Epoch ", epoch + 1, "###")
        model.train()
        running_loss = 0.0
        iteration = 0
        for imgs, targets in tqdm(data_loader):
            if imgs is None or targets is None:
                continue
            loss_dict = model(imgs, targets)
            # Put your training logic here

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
                save_checkpoint(model, optimizer, iteration, iteration_loss, filename=f"checkpoints/checkpoint_iter{iteration}")
                running_loss = 0

    if iteration % save_every != 0:  # Check if there were remaining iterations after the last save
        iteration_loss = running_loss / (iteration % save_every)
        save_checkpoint(model, optimizer, iteration, iteration_loss, filename="checkpoints/final_model.pth")


def eval(checkpoint_file, conf=0.8, k=None, num_to_plot=1, to_plot=False): 

    # load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    load_checkpoint(model, optimizer, filename=checkpoint_file)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load data
    train_dataset = CustomImageDataset(DATA_DIR, 'val')
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    all_outs = []
    all_tp, all_fp, all_fn = 0, 0, 0

    for imgs, targets in tqdm(data_loader):  
        imgs = list(img.to(device) for img in imgs)
        print()

        with torch.no_grad():
            # We only need imgs for inference
            outputs = model(imgs)

        # Process the outputs as needed
        for i, output in enumerate(outputs):
    
            output['masks'] = output['masks'].squeeze(1)

            output['masks'][output['masks'] > conf] = 1
            output['masks'][output['masks'] <= conf] = 0

            if k is None: 
                all_outs.append((imgs[i], {
                    'boxes': output['boxes'],
                    'labels': output['labels'],
                    'scores': output['scores'],
                    'masks': output['masks']
                }))
                all_masks = output['masks']
            else: 
                topk_scores, topk_indices = torch.topk(output['scores'], k=k, largest=True)

                all_outs.append((imgs[i], {
                        'boxes': output['boxes'][topk_indices],
                        'labels': output['labels'][topk_indices],
                        'scores': output['scores'][topk_indices],
                        'masks': output['masks'][topk_indices]
                    }))
                all_masks = output['masks'][topk_indices]
            
            # Get one single mask by doing pixel wise 'or' operation on all masks
            master_mask = torch.any(all_masks.bool(), dim=0)
            target_mask = torch.any(targets[i]['masks'].bool(), dim=0)
            tp, fp, fn = get_metrics(master_mask, target_mask)
            all_tp += tp
            all_fp += fp
            all_fn += fn
    
    print(f"TP: {all_tp}, FP: {all_fp}, FN: {all_fn}")
    f1 = compute_f1(all_tp, all_fp, all_fn)
    print(f"F1 Score: {f1}")
        
    if to_plot: 
        plot([all_outs[:num_to_plot]])


def main(): 
    #train()
    eval("checkpoints/final_model.pth", conf=0.9, num_to_plot=5, k=3)


if __name__ == '__main__':
    main()