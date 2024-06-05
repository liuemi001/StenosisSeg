import torch
from utils import *
from torchvision import models
from tqdm import tqdm
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

DATA_DIR = '/srv/submission/stenosis/'
SYNTAX_DIR = '/srv/submission/syntax/'
DANILOV_DIR = '/srv/danilov/dataset/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(semi_super=False, other_dataset=False):
    # Load data
    batch_size = 4
    train_dataset = CustomImageDataset(DATA_DIR, split='train')
    concat_dataset = None
    if semi_super:
        semi_super_path = 'checkpoints/checkpoint_batch4_epoch50_iter250.pth'
        pseudo_eval_dataset = PseudoSyntaxDataset(SYNTAX_DIR, semi_super_path, split='train')
        plot([pseudo_eval_dataset[1]])
        concat_dataset = torch.utils.data.ConcatDataset([train_dataset, pseudo_eval_dataset])
        print("Concat Successful")
    
    if other_dataset:
        checkpoint = 'checkpoints/checkpoint_batch4_epoch50_iter250.pth'
        danilov_dataset = DanilovDataset(DANILOV_DIR, checkpoint, split='train')
        plot([danilov_dataset[1]])
        concat_dataset = torch.utils.data.ConcatDataset([concat_dataset, danilov_dataset])
        print("Danilov Successful")

    if not semi_super and not other_dataset:
        concat_dataset = train_dataset

    data_loader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    # Load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 30  # Define the number of epochs
    save_every = 250   # Save checkpoint every certain amount of iterations

    start_epoch = 0 # Check this!
    start_iteration = 0
    checkpoint_path = 'checkpoints/checkpoint_batch4_epoch3_iter750_pseudo_other.pth' # Modify this to '' if no checkpoint
    if checkpoint_path:
        model, optimizer, start_epoch, start_iteration = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, num_epochs):
        print("### Epoch ", epoch + 1, "###")
        model.train()
        running_loss = 0.0
        iteration = 0
        if epoch == start_epoch:
            iteration = start_iteration
        for imgs, targets in tqdm(data_loader):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
                filename = f"checkpoints/checkpoint_batch{batch_size}_epoch{epoch+1}_iter{iteration}.pth"
                if semi_super:
                    filename = filename.replace('.pth', '_pseudo.pth')
                if other_dataset:
                    filename = filename.replace('.pth', '_other.pth')
                save_checkpoint(model, optimizer, epoch, iteration, iteration_loss, filename=filename)
                running_loss = 0

        if iteration % save_every != 0:  # Check if there were remaining iterations after the last save
            iteration_loss = running_loss / (iteration % save_every)
            filename = f"checkpoints/checkpoint_batch{batch_size}_epoch{epoch+1}.pth"
            if semi_super:
                filename = filename.replace('.pth', '_pseudo.pth')
            if other_dataset:
                filename = filename.replace('.pth', '_other.pth')
            filename = filename.replace('.pth', '_final.pth')
            save_checkpoint(model, optimizer, epoch, iteration, iteration_loss, filename=filename)


def eval(checkpoint_file, split='val', conf=0.8, k=None, num_to_plot=1, to_plot=False):
    # load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    load_checkpoint(model, optimizer, filename=checkpoint_file)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)

    # load data
    train_dataset = CustomImageDataset(DATA_DIR, split)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    all_outs = []
    all_tp, all_fp, all_fn = 0, 0, 0

    for imgs, targets in tqdm(data_loader):
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
                if k >= len(output['scores']):
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
    train(semi_super=True, other_dataset=True)
    # eval("checkpoints/final_model.pth", split='val', conf=0.8, k=None)


if __name__ == '__main__':
    main()
