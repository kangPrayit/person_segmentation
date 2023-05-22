import torch
from tqdm import tqdm


def compute_dice_score(y_pred, y_true):
    intersection = (y_pred * y_true).sum()
    union = (y_pred + y_true).sum()
    score = 2 * intersection / (union + 1e-7)
    return score


def compute_iou_score(y_pred, y_true):
    intersection = (y_pred * y_true).sum()
    union = (y_pred + y_true).sum()
    score = intersection / (union + 1e-7)
    return score


def compute_precision(y_pred, y_true):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    score = TP / (TP + FP)
    return score


def compute_recall(y_pred, y_true):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    score = TP / (TP + FN)
    return score


def compute_accuracy(y_pred, y_true):
    score = (y_pred == y_true).sum()
    return score

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    accuracy_score = 0
    precision_score = 0
    recall_score = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    model.eval()

    with torch.no_grad():
        for img, mask in tqdm(loader):
            img = img.to(device)
            mask = mask.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(img))
            preds = (preds > 0.5).float()
            # num_correct += (preds == mask).sum()
            num_pixels += torch.numel(preds)
            # dice_score += (2 * (preds * mask).sum()) / (
            #     (preds + mask).sum() + 1e-7
            # )
            accuracy_score += compute_accuracy(y_pred=preds, y_true=mask)
            precision_score += compute_precision(y_pred=preds, y_true=mask)
            recall_score += compute_recall(y_pred=preds, y_true=mask)
            dice_score += compute_dice_score(y_pred=preds, y_true=mask)
            iou_score += compute_iou_score(y_pred=preds, y_true=mask)

    print(
        # f"Got {num_correct}/{num_pixels} with pixel accuracy {num_correct / num_pixels:.2f}"
        f"Got {accuracy_score}/{num_pixels} with pixel accuracy {accuracy_score / num_pixels:.2f}"
    )
    # score = {
    #     'accuracy_score':
    # }
    accuracy_score = accuracy_score/num_pixels
    precision_score = precision_score/num_pixels
    recall_score = recall_score/num_pixels
    dice_score = dice_score/len(loader)
    iou_score = iou_score/len(loader)
    print(f"Accuracy score: {accuracy_score:.2f}")
    print(f"Precision score: {precision_score:.2f}")
    print(f"Recall score: {recall_score:.2f}")
    print(f"Dice score: {dice_score:.2f}")
    print(f"IoU score: {iou_score:.2f}")
    model.train()

    return accuracy_score, precision_score, recall_score, dice_score, iou_score
