import torch.cuda
from tqdm import tqdm
from metrics import *

def train_fn(data_loader, model, optimizer, loss_fn, device):
    model.train()

    accuracy_score = 0
    precision_score = 0
    recall_score = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    loop = tqdm(data_loader)
    for batch_idx, (image, mask) in enumerate(loop):
        image = image.to(device)
        mask = mask.float().unsqueeze(1).to(device)
        model = model.to(device)

        # Forward
        predictions = model(image)
        loss = loss_fn(predictions, mask)

        # backward
        model.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy_score += compute_accuracy(y_pred=predictions, y_true=mask)
        precision_score += compute_precision(y_pred=predictions, y_true=mask)
        recall_score += compute_recall(y_pred=predictions, y_true=mask)
        dice_score += compute_dice_score(y_pred=predictions, y_true=mask)
        iou_score += compute_iou_score(y_pred=predictions, y_true=mask)

        loop.set_postfix(loss=loss.item())

    accuracy_score = accuracy_score / num_pixels
    precision_score = precision_score / num_pixels
    recall_score = recall_score / num_pixels
    dice_score = dice_score / len(data_loader)
    iou_score = iou_score / len(data_loader)
    score_train = {
        'loss': loss,
        'accuracy_score': accuracy_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'dice_score': dice_score,
        'iou_score': iou_score,
    }
    return score_train

def evaluate_fn(data_loader, model, loss_fn, device):
    model.eval()
    accuracy_score = 0
    precision_score = 0
    recall_score = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    with torch.no_grad():
        for img, mask in tqdm(data_loader):
            img = img.to(device)
            mask = mask.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(img))
            preds = (preds > 0.5).float()
            loss = loss_fn(preds, mask)

            accuracy_score += compute_accuracy(y_pred=preds, y_true=mask)
            precision_score += compute_precision(y_pred=preds, y_true=mask)
            recall_score += compute_recall(y_pred=preds, y_true=mask)
            dice_score += compute_dice_score(y_pred=preds, y_true=mask)
            iou_score += compute_iou_score(y_pred=preds, y_true=mask)

    accuracy_score = accuracy_score / num_pixels
    precision_score = precision_score / num_pixels
    recall_score = recall_score / num_pixels
    dice_score = dice_score / len(data_loader)
    iou_score = iou_score / len(data_loader)
    score_eval = {
        'loss': loss,
        'accuracy_score': accuracy_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'dice_score': dice_score,
        'iou_score': iou_score,
    }
    return score_eval