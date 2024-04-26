from test_ivae import test_ivae_tx
import numpy as np
import torch
import numpy.random as random


def seed(seed_num=None):
    if seed_num is None:
        seed_num = 123
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.random.manual_seed(seed_num)

def data_generator_np(size=2000, obs_size=1200, dim_x=5, dim_s=10):
    seed()
    w1_xl = np.random.normal(1, 1, size=[dim_x, int(dim_s / 2)])
    w0_xl = np.random.normal(1, 1, size=[dim_x, int(dim_s / 2)])
    w_ls = np.random.normal(1, 1, size=[int(dim_s / 2), dim_s])
    w_ly = np.random.normal(1, 1, size=[int(dim_s / 2), 1])
    w_xy = np.random.normal(1, 1, size=[dim_x, 1])

    # data
    t = np.random.binomial(1, 0.6, size=[size, 1])
    x = np.random.normal(0, 1, size=[size, dim_x])
    e_l, e_s, e_y = np.random.normal(0, 1, size=[size, int(dim_s / 2)]),\
        np.random.normal(0, 1, size=[size, dim_s]), np.random.normal(0, 1, size=[size, 1])
    l1 = np.matmul(x, w1_xl) + 3 + e_l
    l0 = np.matmul(x, w0_xl) + e_l
    s1 = np.matmul(l1, w_ls) + e_s
    s0 = np.matmul(l0, w_ls) + e_s
    y1 = np.matmul(l1, w_ly) + np.matmul(x, w_xy) + e_y
    y0 = np.matmul(l0, w_ly) + np.matmul(x, w_xy) + e_y

    s = np.where(t==1, s1, s0)
    y = np.where(t==1, y1, y0)
    if obs_size >= size:
        return
    xo, xe = x[:obs_size, :], x[obs_size:, :]
    so, se = s[:obs_size, :], s[obs_size:, :]
    yo, ye = y[:obs_size, :], y[obs_size:, :]
    to, te = t[:obs_size, :], t[obs_size:, :]
    Obs = xo, to, so, yo
    Exp = xe, te, se, ye
    tau_real = np.mean(y1[obs_size:, :]-y0[obs_size:, :])
    print(tau_real)

    return Obs, Exp, tau_real


def data_generator_np_wo(size=2000, obs_size=1200, dim_x=10, dim_so=2, dim_sl=2, dim_proxy=2, wo=1, seednum=0):
    # wo=0, no lack
    # wo=1, lack so
    # wo=0, lack sl
    seed(seednum)
    w1_x2so = np.random.normal(1, 1, size=[dim_x, int(dim_so)])
    w0_x2so = np.random.normal(1, 1, size=[dim_x, int(dim_so)])
    w1_x2sl = np.random.normal(1, 1, size=[dim_x, int(dim_sl)])
    w0_x2sl = np.random.normal(1, 1, size=[dim_x, int(dim_sl)])

    w_sl2p =  np.random.normal(1, 1, size=[dim_sl, int(dim_proxy)])

    w_s2y = np.random.normal(1, 1, size=[int((dim_so + dim_sl)), 1])
    w_xy = np.random.normal(1, 1, size=[dim_x, 1])

    # data
    x0 = np.random.normal(0, 1, size=[size, int(dim_x-3)])
    x1 = np.random.normal(1, 1, size=[size, int(3)])
    x = np.concatenate((x0,x1),axis=1)

    t_obs = []
    for i in range(obs_size):
        p = 1/ (1+ np.exp(-np.mean(x[i,:])))
        # print(p)
        t = np.random.binomial(1, p=p)
        t_obs.append(t)
    t_obs = np.array(t_obs)
    t_exp = np.random.binomial(1, 0.6, size=[int(size-obs_size), 1])
    t = np.concatenate((t_obs[:,None], t_exp), axis=0)

    e_so, e_sl, e_p, e_y = np.random.normal(0, 1, size=[size, dim_so]),\
        np.random.normal(0, 1, size=[size, dim_sl]),\
        np.random.normal(0, 1, size=[size, dim_proxy]), np.random.normal(0, 1, size=[size, 1])

    so1 = np.matmul(x, w1_x2so+1) + e_so
    so0 = np.matmul(x, w0_x2so-1) + e_so
    sl1 = np.matmul(x, w1_x2sl+1) + e_sl
    sl0 = np.matmul(x, w0_x2sl-1) + e_sl

    so = np.where(t==1, so1, so0)
    sl = np.where(t==1, sl1, sl0)
    surrogate1 = np.concatenate((so1,sl1),axis=1)
    surrogate0 = np.concatenate((so0,sl0),axis=1)

    p = np.matmul(sl, w_sl2p) + e_sl

    y1 = np.matmul(surrogate1, w_s2y) + np.matmul(x, w_xy) + e_y
    y0 = np.matmul(surrogate0, w_s2y) + np.matmul(x, w_xy) + e_y

    if wo == 0:
        s = np.concatenate((so,p),axis=1)
    if wo == 1:
        s = p
    if wo == 2:
        s = so


    # s = np.where(t==1, so1, so0)
    y = np.where(t==1, y1, y0)
    if obs_size >= size:
        return
    xo, xe = x[:obs_size, :], x[obs_size:, :]
    so, se = s[:obs_size, :], s[obs_size:, :]
    yo, ye = y[:obs_size, :], y[obs_size:, :]
    to, te = t[:obs_size, :], t[obs_size:, :]
    Obs = xo, to, so, yo
    Exp = xe, te, se, ye
    tau_real = np.mean(y1[obs_size:, :]-y0[obs_size:, :])
    print(tau_real)

    return Obs, Exp, tau_real


if __name__ == '__main__':
    # demo to run our method
    wo = 0
    if wo == 0:
        ld = 4
    else:
        ld = 2
    print(wo)
    tau_ests = []
    tau_reals = []
    errors = []

    for i in range(5):
        data = data_generator_np_wo(wo=wo, seednum=i)
        Obs, Exp, tau_real = data
        print(tau_real)
        tau_ivae, tau_real, E_y1, E_y0, ivae = test_ivae_tx(data=data, print_log=True, early_stop=True,
                                                            is_rct=True, treated=1, control=0, lr=1e-4, max_epoch=1000,seed_num=i,
                                                            latent_dims=ld)
        tau_ests.append(tau_ivae)
        tau_reals.append(tau_real)
        errors.append(np.abs(tau_ivae-tau_real))

    print('wo=' + str(wo))
    print(tau_ests)
    print(tau_reals)
    print(np.mean(errors), np.std(errors))
    perror = np.array(errors)/np.array(tau_reals)
    print(np.mean(perror), np.std(perror))

