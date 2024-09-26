import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from mlwr import mlwr
from networks.discriminator import FC3DDiscriminator_first, FC3DDiscriminatorNIH_first, Discriminator_DYBAC3D, \
    Discriminator_first_DYBAC3D, FC3DDiscriminator_new, FC3DDiscriminator_new2
from networks.ResNet34 import Resnet34
from test import test_calculate_metric,test_calculate_metric_memory
from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.process import pack, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default='LA', help='Name of Experiment')
parser.add_argument('--root_path', type=str,default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='UAMT_LUD2_SASS_mlwr_Ablation1_FC3DDiscriminator_new2', help='model_name')
parser.add_argument('--max_iterations', type=int,default=13000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--tau_p', type=float,  default=0.7, help='maximum epoch number to train')
parser.add_argument('--tau_n', type=float,  default=0.3, help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4, help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--trainnum', type=int,  default=80, help='random seed')
parser.add_argument('--testnum', type=int,  default=20, help='random seed')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--labelnum', type=int,  default=16, help='random seed')
parser.add_argument('--seed', type=int,  default=3407, help='random seed')
parser.add_argument('--consistency_weight', type=float,  default=0.1, help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5, help='balance factor to control supervised and consistency loss')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--coefficient', type=float,  default=0.99, help='coefficient')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--with_cons', type=str, default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--Ent_th', type=float,  default=0.75, help='')
args = parser.parse_args()

def loss_fn(candidates, prototype):
    x = F.normalize(candidates, dim=0, p=2).permute(1, 0).unsqueeze(0)
    y = F.normalize(prototype, dim=0, p=2).permute(1, 0).unsqueeze(0)
    loss = torch.cdist(x, y, p=2.0).mean()
    return loss
snapshot_path = "../model/" + args.dataset + "/" + args.exp + \
    "_{}labels_beta_{}/".format(
        args.labelnum, args.beta)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
def entropy_map(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    return y1

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen
num_classes = 2
patch_size = (112, 112, 80)
T = 0.1
Good_student = 0
def compute_uxi_loss(predicta, predictb, represent_a, percent=20):
    batch_size, num_class, h, w, d = predicta.shape
    logits_u_a, label_u_a = torch.max(predicta, dim=1)
    logits_u_a, label_u_b = torch.max(predictb, dim=1)
    target = label_u_a | label_u_b
        # drop pixels with high entropy from a
    prob_a = predicta
    entropy_a = -torch.sum(prob_a * torch.log(prob_a + 1e-10), dim=1)
    thresh_a = np.percentile(entropy_a.detach().cpu().numpy().flatten(), percent)
    thresh_mask_a = entropy_a.ge(thresh_a).bool()
    prob_b = predictb
    entropy_b = -torch.sum(prob_b * torch.log(prob_b + 1e-10), dim=1)
    thresh_b = np.percentile(entropy_b.detach().cpu().numpy().flatten(), percent)
    thresh_mask_b = entropy_b.ge(thresh_b).bool()
    thresh_mask = torch.logical_and(thresh_mask_a, thresh_mask_b)
    target[thresh_mask] = 2
    target_clone = torch.clone(target.view(-1))
    represent_a = represent_a.permute(1, 0, 2, 3, 4)
    represent_a = represent_a.contiguous().view(represent_a.size(0), -1)
    prototype_f = represent_a[:, target_clone == 1].mean(dim=1)
    prototype_b = represent_a[:, target_clone == 0].mean(dim=1)
    forground_candidate = represent_a[:, (target_clone == 2) & (label_u_a.view(-1) == 1)]
    background_candidate = represent_a[:, (target_clone == 2) & (label_u_a.view(-1) == 0)]
    num_samples = forground_candidate.size(1) // 100
    selected_indices_f = torch.randperm(forground_candidate.size(1))[:num_samples]
    selected_indices_b = torch.randperm(background_candidate.size(1))[:num_samples]
    contrastive_loss_f = loss_fn(forground_candidate[:, selected_indices_f], prototype_f.unsqueeze(dim=1))
    contrastive_loss_b = loss_fn(background_candidate[:, selected_indices_b], prototype_b.unsqueeze(dim=1))
    contrastive_loss_c = loss_fn(prototype_f.unsqueeze(dim=1), prototype_b.unsqueeze(dim=1))
    con_loss = contrastive_loss_f + contrastive_loss_b + contrastive_loss_c
    weight = batch_size * h * w * d / torch.sum(target != 2)
    loss_a = weight * F.cross_entropy(predicta, target, ignore_index=2)
    loss_b = weight * F.cross_entropy(predictb, target, ignore_index=2)
    return loss_a, loss_b, con_loss

def gather_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c-1):
        temp_line = vec[:,i,:].unsqueeze(1)  # b 1 c
        star_index = i+1
        rep_num = c-star_index
        repeat_line = temp_line.repeat(1, rep_num,1)
        two_patch = vec[:,star_index:,:]
        temp_cat = torch.cat((repeat_line,two_patch),dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result,dim=1)
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    def update_ema_variables(ema_model, model1, model2, alpha, r1, r2, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        x = r1 / (r1 + r2)
        y = r2 / (r1 + r2)
        for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
            ema_param.data.mul_(alpha).add_(x*(1 - alpha), param1.data).add_(y*(1-alpha), param2.data)
    def update_ema_variables3(ema_model, model1, model2, alpha, coefficient, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
            ema_param.data.mul_(alpha).add_(coefficient*(1 - alpha), param1.data).add_((1-coefficient)*(1-alpha), param2.data)
    def update_ema_variables2(model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    def create_model(name='vnet', ema=False):
        # Network definition
        if name == 'vnet':
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if name == 'resnet34':
            net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    ema_model = create_model(name='vnet', ema=True)
    model_vnet = create_model(name='vnet', ema=False)
    model_resnet = create_model(name='vnet', ema=False)
    train_data_path = args.root_path + "/" + args.dataset
    db_train = pack(base_dir=train_data_path,
                    num=args.trainnum,
                    split='train',  # train/val sp
                    common_transform=transforms.Compose([
                        RandomCrop(patch_size),
                    ]),
                    sp_transform=transforms.Compose([
                        ToTensor(),
                    ]))

    labelnum = args.labelnum    # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.trainnum))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    D = FC3DDiscriminator_new2(num_classes=num_classes)
    D = D.cuda()
    Dopt = optim.Adam(D.parameters(), lr=args.D_lr, betas=(0.9, 0.99))
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model_vnet.train()
    model_resnet.train()
    vnet_optimizer = optim.SGD(model_vnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    resnet_optimizer = optim.SGD(model_resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    best_evl = [0, 0, 100000, 100000]
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            Dtarget = torch.tensor([1, 1, 0, 0]).cuda()
            model_vnet.train()
            model_resnet.train()
            D.eval()
            volume_batch, volume_label = sampled_batch['image'], sampled_batch['label']
            volume_batch, volume_label = volume_batch.cuda(), volume_label.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            noise2 = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            noisy_ema_inputs = volume_batch + noise2
            noisy_stu_inputs = noise2 + volume_batch
            v_outputs, v_rep = model_vnet(volume_batch)
            r_outputs, r_rep = model_resnet(volume_batch)
            v_outputs_soft = F.softmax(v_outputs, dim=1)
            r_outputs_soft = F.softmax(r_outputs, dim=1)
            v_loss_seg = F.cross_entropy(v_outputs[:labeled_bs], volume_label[:labeled_bs])
            v_loss_seg_dice = losses.dice_loss(v_outputs_soft[:labeled_bs, 1, :, :, :], volume_label[:labeled_bs] == 1)
            r_loss_seg = F.cross_entropy(r_outputs[:labeled_bs], volume_label[:labeled_bs])
            r_loss_seg_dice = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], volume_label[:labeled_bs] == 1)
            v_supervised_loss = (v_loss_seg + v_loss_seg_dice)
            r_supervised_loss = (r_loss_seg + r_loss_seg_dice)
            T = 8
            r_outputs_clone = r_outputs.clone().detach()
            Doutputs = D(v_outputs[labeled_bs:], r_outputs_clone[labeled_bs:], volume_batch[labeled_bs:])
            loss_adv = F.cross_entropy(Doutputs, torch.tensor([1, 1, 0, 0]).cuda().long())
            _, _, d, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            with torch.no_grad():
                ema_output, ema_rep = ema_model(ema_inputs)
                ema_output_soft = F.softmax(ema_output, dim=1)
                teacher_output, teacher_rep = ema_model(unlabeled_volume_batch)
                teacher_output_soft = F.softmax(teacher_output, dim=1)
            preds = torch.zeros([stride * T, 2, d, w, h]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride *(i + 1)], _ = ema_model(ema_inputs)
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, d, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            weight_pixel_t = mlwr(v_outputs_soft[labeled_bs:], v_rep[labeled_bs:], teacher_output_soft, teacher_rep)
            consistency_dist = consistency_criterion(v_outputs_soft[labeled_bs:, :, :, :, :], ema_output)
            b, c, w, h, d = consistency_dist.shape
            consistency_dist = 2 * torch.sum(mask* weight_pixel_t.mean().detach() * consistency_dist)/(2*torch.sum(mask)+1e-16)
            loss_u_v, loss_u_t, con_loss = compute_uxi_loss(v_outputs[labeled_bs:, :, :, :, :], teacher_output_soft, v_rep[labeled_bs:])
            consistency_loss = con_loss + consistency_dist + loss_adv
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = r_supervised_loss + v_supervised_loss + consistency_weight * consistency_loss + loss_u_v + loss_u_t
            writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)
            if (torch.any(torch.isnan(loss)) or torch.any(torch.isnan(loss))):
                print('nan find')
            vnet_optimizer.zero_grad()
            resnet_optimizer.zero_grad()
            loss.backward()
            vnet_optimizer.step()
            resnet_optimizer.step()
            model_vnet.eval()
            model_resnet.eval()
            D.train()
            with torch.no_grad():
                v_outputs, v_rep = model_vnet(volume_batch)
                r_outputs, r_rep = model_resnet(volume_batch)
            Doutputs = D(v_outputs, r_outputs, volume_batch)
            # D want to classify unlabel data and label data rightly.
            D_loss = F.cross_entropy(Doutputs, torch.tensor([1, 1, 0, 0, 1, 1, 0, 0]).cuda().long())
            update_ema_variables3(ema_model, model_vnet, model_resnet, args.ema_decay, args.coefficient, iter_num)
            Dopt.zero_grad()
            D_loss.backward()
            Dopt.step()
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/v_loss_seg', v_loss_seg, iter_num)
            writer.add_scalar('loss/v_loss_seg_dice', v_loss_seg_dice, iter_num)
            writer.add_scalar('loss/v_supervised_loss', v_supervised_loss, iter_num)
            writer.add_scalar('loss/r_loss_seg', r_loss_seg, iter_num)
            writer.add_scalar('loss/r_loss_seg_dice', r_loss_seg_dice, iter_num)
            writer.add_scalar('loss/r_supervised_loss', r_supervised_loss, iter_num)
            writer.add_scalar('train/Good_student', Good_student, iter_num)
            logging.info(
                'iteration ： %d v_supervised_loss : %f v_loss_seg : %f v_loss_seg_dice : %f  r_supervised_loss : %f r_loss_seg : %f r_loss_seg_dice : %f Good_student: %f' %
                (iter_num,
                 v_supervised_loss.item(), v_loss_seg.item(), v_loss_seg_dice.item(),
                 r_supervised_loss.item(), r_loss_seg.item(), r_loss_seg_dice.item(), Good_student))
            # change lr
            if iter_num % 2500 == 0 and iter_num != 0:
                lr_ = lr_ * 0.1
                for param_group in vnet_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in resnet_optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 200 == 0:

            # if iter_num % 1 == 0 :
                save_test_path = snapshot_path + "test/"
                if not os.path.exists(save_test_path):
                    os.makedirs(save_test_path)
                vnet_save_mode_path = os.path.join(snapshot_path, 'vnet_iter_' + str(iter_num) + '.pth')
                torch.save(model_vnet.state_dict(), vnet_save_mode_path)
                logging.info("save vnet_model to {}".format(vnet_save_mode_path))
                vnet_metric = test_calculate_metric(vnet_save_mode_path, train_data_path, save_test_path, args)  # 6000
                t_save_mode_path = os.path.join(snapshot_path, 'teacher_vnet_iter_' + str(iter_num) + '.pth')
                torch.save(ema_model.state_dict(), t_save_mode_path)
                logging.info("save t_model to {}".format(t_save_mode_path))
                t_metric = test_calculate_metric(t_save_mode_path, train_data_path, save_test_path, args)
                for x in range(2):
                    if best_evl[x] < vnet_metric[x]: best_evl[x] = vnet_metric[x]
                    if best_evl[x] < t_metric[x]: best_evl[x] = t_metric[x]
                    if best_evl[x+2] > vnet_metric[x+2]: best_evl[x+2] = vnet_metric[x+2]
                    if best_evl[x + 2] > t_metric[x + 2]: best_evl[x + 2] = t_metric[x + 2]
                logging.info(
                    'iteration %d : vnet_Dice: %f, vnet_JA: %f, vnet_95HD: %f, vnet_ASD: %f' % (iter_num, vnet_metric[0], vnet_metric[1], vnet_metric[2], vnet_metric[3]))
                logging.info(
                    'iteration %d : t_Dice: %f, t_JA: %f, t_95HD: %f, t_ASD: %f' % (iter_num, t_metric[0], t_metric[1], t_metric[2], t_metric[3]))
                logging.info('iteration %d : best_Dice: %f, best_JA: %f, best_95HD: %f, best_ASD: %f' % (iter_num, best_evl[0], best_evl[1], best_evl[2], best_evl[3]))
            if iter_num > max_iterations:
                break
            time1 = time.time()
            iter_num = iter_num + 1
        if iter_num > max_iterations:
            iterator.close()
            break
    writer.close()
