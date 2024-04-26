import sys
sys.path.append('../..')
import torch
import numpy as np
import random
from laser import IVAE_tx_wrapper
from ate_estimator import ipw_estimator


# data simulated for showing that
#      our method is robust for large noisy proxy


def seed(seed_num=None):
    if seed_num is None:
        seed_num = 123
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.random.manual_seed(seed_num)


def normalization(train, test):
    col = train.shape[1]
    for i in range(col):
        mu, std = np.mean(train[:, i]), np.std(train[:, i])
        if std == 0 or np.isnan(std):
            continue
        train[:, i] = (train[:, i] - mu) / std
        test[:, i] = (test[:, i] - mu) / std
    return train, test


def test_ivae_tx(seed_num=1, data=None, is_rct=True, latent_dims=5, n_layers=3, hidden_dim=200, lr=1e-3, beta=1, recon_theta=1,
                 normalization=True, base_eopch=100, early_stop=False, print_log=False, batch_size=100, max_epoch=int(300),
                 treatment_dim=1, treated=0.7, control=0.97):
    seed(seed_num=seed_num)
    if torch.cuda.is_available():
        # if print_log:
        #     print('using cuda')
        cuda = True
        # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        # if print_log:
        #     print('using cpu')
        torch.set_default_dtype(torch.double)

    Obs, Exp, tau_real = data

    # numpy to tensor
    xo, to, so, yo = Obs
    xe, te, se, ye = Exp
    xo, xe, so, yo, te, se, ye, to = torch.tensor(xo), torch.tensor(xe), torch.tensor(so), torch.tensor(yo), torch.tensor(
        te), torch.tensor(se), torch.tensor(ye), torch.tensor(to)

    # mu = torch.mean(so, dim=0)
    # std = torch.std(so, dim=0)
    # se, so = (se-mu)/std, (so-mu)/std

    Obs = xo, to, so, yo
    Exp = xe, te, se, ye
    data = Obs, Exp, tau_real

    losses, ivae = IVAE_tx_wrapper(data=data, batch_size=batch_size, max_epoch=max_epoch, n_layers=n_layers, hidden_dim=hidden_dim,
                                learn_rate=lr, weight_decay=1e-4, base_eopch=base_eopch,
                                activation='lrelu', slope=.1, inference_dim=latent_dims, optm='Adam',
                                anneal=False, print_log=print_log, is_rct=is_rct, cuda=cuda, normalization=normalization,
                                early_stop=early_stop, beta=beta, theta=recon_theta,
                                treatment_dim=treatment_dim,treated=treated, control=control)
    # if print_log:
    #     print('loss from {} to {}'.format(losses[0], losses[-1]))

    ye = ivae.test(covariate=xe.cuda(), s=se.cuda(), treatment=te.cuda())
    y1_index = (te[:,0] == treated).unsqueeze(1)
    y0_index = (te[:,0] == control).unsqueeze(1)
    # estimate ate
    if is_rct:
        E_y1 = torch.mean(ye[y1_index])
        E_y0 = torch.mean(ye[y0_index])
        tau_ivae = E_y1 - E_y0
        E_y1 = E_y1.cpu().detach().numpy()
        E_y0 = E_y0.cpu().detach().numpy()
        tau_ivae = tau_ivae.cpu().detach().numpy()
    else:
        tau_ivae = ipw_estimator(x=xe.cpu().detach().numpy(), t=te.cpu().detach().numpy(), y=ye.cpu().detach().numpy())

    if print_log and is_rct:
        # print('real tau={}'.format(tau_real))
        print('E[y_1]={}, E[y_0]={}, ivae tau={}'.format(E_y1, E_y0, tau_ivae))
    # print(tau_ivae, tau_real, E_y1, E_y0, ivae)
    return tau_ivae, tau_real, E_y1, E_y0, ivae
