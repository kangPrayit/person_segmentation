import torch
from tqdm import tqdm


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for imgs, masks in tqdm(loader):
            imgs = imgs.to(device)
            masks = masks.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(imgs))
            preds = (preds > 0.5).float()
            num_correct += (preds == masks).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * masks).sum()) / ((preds + masks).sum() + 1e-7)
    print(
        f"Got {num_correct}/{num_pixels} with pixel accuracy {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader) * 100:.2f}")
    model.train()