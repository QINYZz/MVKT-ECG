import torch
import torch.nn as nn

def multi_label_KL_loss(logits_S, logits_T, temperature, num_classes):
    logits_S = logits_S.sigmoid().unsqueeze(2)
    logits_T = logits_T.sigmoid().unsqueeze(2)
    
    logits_S = torch.cat([logits_S, 1-logits_S], dim=2)
    logits_T = torch.cat([logits_T, 1-logits_T], dim=2)
    ans = 0
    for i in range(num_classes):
        logits_S_i = logits_S[:, i, :]
        logits_T_i = logits_T[:, i, :]
        ans += nn.KLDivLoss()(torch.log(logits_S_i / temperature + 1e-8), logits_T_i / temperature + 1e-8)
    return ans
    
def multi_label_KL_loss_v2(logits_S, logits_T, temperature):
    logits_S = logits_S.sigmoid() / temperature
    logits_T = logits_T.sigmoid() / temperature
    
    loss = -logits_T * torch.log(logits_T) + logits_T * torch.log(logits_S) - (1-logits_T) * torch.log(1-logits_T) + (1-logits_T) * torch.log(1-logits_S)
    return torch.sum(loss) / logits_S.shape[0]
    