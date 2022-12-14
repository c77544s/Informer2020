import torch
import main_informer
from exp.exp_informer import Exp_Informer

args = main_informer.parse_args();
exp = Exp_Informer(args)
model = exp.model
model.load_state_dict(torch.load('model.pt'))
model.eval()

preds = []

for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    pred, true = exp._process_one_batch(
        pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
    preds.append(pred.detach().cpu().numpy())

preds = np.array(preds)
preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

print(model)

