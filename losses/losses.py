import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def calculate_triplet_loss(x_real, teacher_gen_image, student_gen_image, studentD, triplet_margin=0.25):
    a = torch.mean(torch.abs(studentD.cut_forward(x_real) - studentD.cut_forward(teacher_gen_image)), dim=(1, 2, 3))
    b = torch.mean(torch.abs(studentD.cut_forward(x_real) - studentD.cut_forward(student_gen_image)), dim=(1, 2, 3))
    tmp_max = torch.max(a, b).detach()
    func = a - b + triplet_margin * tmp_max
    triplet_loss = F.relu(func).mean()
    return triplet_loss


def calculate_perceptual_loss(teacher_gen_image, student_gen_image, feature_network):
    tmp = torch.abs(feature_network(teacher_gen_image) - feature_network(student_gen_image))
    perc_loss = torch.pow(torch.mean(tmp, dim=(1, 2, 3)), 2).mean()
    return perc_loss


def compute_gka_loss(X, Y):
    X_ = X.transpose(1, 3).flatten(end_dim=2)
    Y_ = Y.transpose(1, 3).flatten(end_dim=2)
    assert X_.shape[0] == Y_.shape[0], \
        f'X_ and Y_ must have the same shape on dim 0, but got {X_.shape[0]} for X_ and {Y_.shape[0]} for Y_.'
    X_vec = X_.T @ X_
    Y_vec = Y_.T @ Y_
    XY = Y_.T @ X_
    res = (XY ** 2).sum() / ((X_vec ** 2).sum() * (Y_vec ** 2).sum()) ** 0.5
    return res


class WKD:
    def __init__(self, wkd_level=3, wkd_basis='haar'):
        self.xfm = DWTForward(J=wkd_level, mode='zero', wave=wkd_basis).cuda()
        self.wkd_steps=wkd_level

    def get_wavelet_loss(self, student, teacher):
        student_l, student_h = self.xfm(student)
        teacher_l, teacher_h = self.xfm(teacher)
        loss = 0.0
        for index in range(len(student_h)):
            loss+= torch.nn.functional.l1_loss(teacher_h[index], student_h[index])
        return loss

    def get_wkd_gka_loss(self, student, teacher):
        student_l, student_h = self.xfm(student)
        teacher_l, teacher_h = self.xfm(teacher)
        loss = 0.0
        for index in range(self.wkd_steps):
            loss -= compute_gka_loss(student_h[index].flatten(start_dim=2, end_dim=3),
                                     teacher_h[index].flatten(start_dim=2, end_dim=3))
        return loss

    def get_wkd_gka_loss_v2(self, student, teacher, dwt_out_minsize=8):
        student_l, student_h = self.xfm(student)
        teacher_l, teacher_h = self.xfm(teacher)
        loss = 0.0
        for index in range(self.wkd_steps):
            if student_h[index].shape[-1] >= dwt_out_minsize:
                loss -= compute_gka_loss(student_h[index].flatten(start_dim=2, end_dim=3),
                                         teacher_h[index].flatten(start_dim=2, end_dim=3))
        return loss
