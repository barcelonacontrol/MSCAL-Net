import torch
import torch.nn as nn
import torch.nn.functional as F
num_classes = 2
def mlwr(s_pred, s_f, t_pred, t_f):
    batch_o, c_o, w_o, h_o, d_o = t_pred.shape
    batch_f, c_f, w_f, h_f, d_f = t_f.shape

 #   teacher_o = t_pred.reshape(batch_o, c_o, -1)
    teacher_f = t_f.reshape(batch_f, c_f, -1)
    #   stu_f = s_f.reshape(batch_f, c_f, -1)

    index = torch.argmax(t_pred, dim=1, keepdim=True)
    prototype_bank = torch.zeros(batch_f, num_classes, c_f).cuda()
    for ba in range(batch_f):
        for n_class in range(num_classes):
            mask_temp = (index[ba] == n_class).float()
            top_fea = t_f[ba] * mask_temp
            prototype_bank[ba, n_class] = top_fea.sum(-1).sum(-1).sum(-1) / (mask_temp.sum() + 1e-6)

    prototype_bank = F.normalize(prototype_bank, dim=-1)
    # mask_t = torch.zeros_like(s_pred).cuda()
    mask_t = torch.zeros_like(t_pred).cuda()
    for ba in range(batch_o):
        for n_class in range(num_classes):
            class_prototype = prototype_bank[ba, n_class]
            mask_t[ba, n_class] = F.cosine_similarity(teacher_f[ba],
                                                      class_prototype.unsqueeze(1),
                                                      dim=0).view(w_f, h_f, d_f)

    weight_pixel_t = (1 - nn.MSELoss(reduction='none')(mask_t, t_pred)).mean(1)
    return weight_pixel_t
    # consistency_criterion = nn.CrossEntropyLoss(reduction='none')
    # loss_t = consistency_criterion(s_pred, torch.argmax(t_pred, dim=1).detach())
    # loss_consist = (loss_t * weight_pixel_t.detach()).sum() / (mask.sum() + 1e-6)