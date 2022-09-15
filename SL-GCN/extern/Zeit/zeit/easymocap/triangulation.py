import numpy as np


def projectN3(kpts3d, Pall):
    # kpts3d: (N, 3)
    nViews = len(Pall)
    kp3d = np.concatenate([kpts3d[..., :3], np.ones_like(kpts3d)[..., :1]], axis=-1)
    kp2ds = []
    for nv in range(nViews):
        kp2d = kp3d @ Pall[nv].T
        depth = kp2d[..., 2:]
        depth[np.abs(depth)<1e-5] = 1e-5
        kp2d[..., :2] /= kp2d[..., 2:]
        kp2ds.append(kp2d)
    kp2ds = np.stack(kp2ds)
    if kpts3d.shape[-1] == 4:
        kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)
    return kp2ds


def batch_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    if keypoints_pre is None:
        valid_joint = np.where(v > 1)[0]
    else:
        valid_joint = np.where(v > 0)[0]
    if len(valid_joint) < 1:
        result = np.zeros((keypoints_.shape[1], 4))
        return result
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    # ATTN: use middle point
    result[:, :3] = X[:, :3].mean(axis=0, keepdims=True)
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result

eps = 0.01
def triangulate(keypoints_use, Puse):
    out = batch_triangulate(keypoints_use, Puse)
    # compute reprojection error
    kpts_repro = projectN3(out, Puse)
    square_diff = (keypoints_use[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = np.repeat(out[None, :, -1:], len(Puse), 0)
    kpts_repro = np.concatenate((kpts_repro, conf), axis=2)
    return out, kpts_repro