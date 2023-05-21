import torch.cuda
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_fn(data_loader, model, optimizer, loss_fn):
    model.train()
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

        loop.set_postfix(loss=loss.item())
