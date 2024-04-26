import sys

sys.path.append('..')

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, random_split
from ate_estimator import ipw_estimator

import numpy as np
import torch
from torch import optim

from model_utils import Normal, MLP, weights_init

# avoid grad exploding
grad_clip = 0.5


class iVAE_tx(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, prior=None, decoder=None, encoder=None, y_recon=None,
                 t_recon=None, t_temperature=1e-2,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu', anneal=False,
                 treatment_dim=1):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal
        self.t_temperature = t_temperature
        self.treatment_dim = treatment_dim

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        if y_recon is None:
            self.y_recon_dist = Normal(device=device)
        else:
            self.y_recon_dist = y_recon

        # if t_recon is None:
        #     self.t_recon_dist = RelaxedBernoulli(device=device)
        #     # self.t_recon_dist = Bernoulli(device=device)
        # else:
        #     self.t_recon_dist = t_recon
        #
        # self.x_recon_dist = Normal(device=device)

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim + treatment_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)
        # encoder params
        self.g = MLP(data_dim + aux_dim +treatment_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim + aux_dim +treatment_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)
        # reconstruct long-term y
        self.meany = MLP(aux_dim + latent_dim, 1, hidden_dim, n_layers, activation=activation, slope=slope,
                         device=device)
        self.logvy = .01 * torch.ones(1).to(device)
        # reconstruct treatment t, in fact pro_t have to be input sigmoid to get true probability
        self.pro_t = MLP(aux_dim + latent_dim, 1, hidden_dim, n_layers, activation=activation, slope=slope,
                         device=device)
        self.sigmoid = nn.Sigmoid()

        # balanced x
        self.ph_x = MLP(aux_dim, hidden_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                         device=device)

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]

        # normalizing y param
        self.pre_process_y = False
        self.y_mu = None
        self.y_std = None

    def encoder_params(self, s, covariate, treatment):
        s_covariate_treatment = torch.cat((s, covariate, treatment), 1)
        g = self.g(s_covariate_treatment)
        logv = self.logv(s_covariate_treatment)
        return g, logv.exp()

    def decoder_params(self, s):
        f = self.f(s)
        return f, self.decoder_var

    def prior_params(self, covariate, treatment):
        covariate_treatment = torch.cat((covariate, treatment), 1)
        logl = self.logl(covariate_treatment)
        return self.prior_mean, logl.exp()

    # def sl2x(self, s_latent):
    #     meanx = self.meanx(s_latent)
    #     return meanx, self.logvy

    def sl2y(self, s_latent, covariate):
        meany = self.meany(torch.cat((s_latent, covariate), dim=1))
        return meany, self.logvy

    # def sl2t(self, s_latent, covariate):
    #     t = self.sigmoid(self.pro_t(torch.cat((s_latent, covariate), dim=1)))
    #     return t

    def forward(self, s, covariate, treatment):
        treatment = 1. * treatment
        # encoder
        prior_params = self.prior_params(covariate, treatment)
        encoder_params = self.encoder_params(s, covariate, treatment)
        # s_latent sample from latent variable
        s_latent = self.encoder_dist.sample(*encoder_params)
        # decoder
        decoder_params = self.decoder_params(s_latent)

        # recon_x
        # x_params = self.sl2x(s_latent)
        # x_hat = self.x_recon_dist.sample(*x_params)

        # auxiliary distribution
        y_params = self.sl2y(s_latent, covariate=covariate)
        y_hat = self.y_recon_dist.sample(*y_params)
        # t_params = self.sl2t(s_latent, covariate=covariate)
        # t_hat = self.t_recon_dist.sample(t_params, temperature=self.t_temperature)
        return decoder_params, encoder_params, s_latent, prior_params, y_params, y_hat

    def test(self, s, covariate, treatment):
        treatment = 1. * treatment
        encoder_params = self.encoder_params(s, covariate, treatment)
        sl_mean, _ = encoder_params
        meany, variance = self.sl2y(sl_mean, covariate=covariate)
        if self.pre_process_y:
            meany = meany * self.y_std + self.y_mu
        # print(meany)
        return meany

    def elbo(self, s, decoder_params, g, v, s_latent, prior_params, theta=1):
        log_ps_sl = self.decoder_dist.log_pdf(s, *decoder_params)
        log_qsl_scovariate = self.encoder_dist.log_pdf(s_latent, g, v)
        log_psl_covariate = self.prior_dist.log_pdf(s_latent, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = s_latent.size(0)
            log_qsl_tmp = self.encoder_dist.log_pdf(s_latent.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                    v.view(1, M, self.latent_dim), reduce=False)
            log_qsl = torch.logsumexp(log_qsl_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qsl_i = (torch.logsumexp(log_qsl_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_ps_sl - b * (log_qsl_scovariate - log_qsl) - c * (log_qsl - log_qsl_i) - d * (
                    log_qsl_i - log_psl_covariate)).mean(), s_latent

        else:
            return (log_ps_sl * theta + log_psl_covariate - log_qsl_scovariate).mean(), s_latent

    def log_qy_xsl(self, y, y_params, o_index):
        mu, var = y_params
        mu = mu[o_index].unsqueeze(dim=1)
        y_params = mu, var
        y = y[o_index].unsqueeze(dim=1)
        log_qy_xsl_mean = self.y_recon_dist.log_pdf(y, *y_params).mean()
        return log_qy_xsl_mean

    def loss_tatol(self, covariate, treatment, s, o_index, y, cuda=True, beta=1, theta=1.):
        decoder_params, (g, v), s_latent, prior_params, y_params, y_hat= self.forward(s=s,covariate=covariate,
                                                                                      treatment=treatment)

        elbo, sl_est = self.elbo(s=s, decoder_params=decoder_params, g=g, v=v, s_latent=s_latent,
                                 prior_params=prior_params, theta=theta)
        l_yo = self.log_qy_xsl(y=y, y_params=y_params, o_index=o_index) * beta

        loss = elbo.mul(-1) + l_yo.mul(-1)
        return loss, elbo.mul(-1), l_yo.mul(-1)

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False


def IVAE_tx_wrapper(data, batch_size=256, max_epoch=2000, n_layers=3, hidden_dim=200, learn_rate=1e-3, weight_decay=1e-4,
                 activation='lrelu', slope=.1, inference_dim=None, optm='Adam', min_lr=1e-6, base_eopch=200,
                 anneal=False, print_log=True, is_rct=True, cuda=True, normalization=True, beta=1, theta=1,
                 early_stop=True, early_stop_epoch=100, valid_rate=0.2,
                 treatment_dim=1,treated=0.7, control=0.97):
    device = torch.device('cuda' if cuda else 'cpu')
    if print_log:
        print('training on {}'.format(torch.cuda.get_device_name(device) if cuda else 'cpu'))

    # load data
    Obs, Exp, tau_real = data
    xo, to, so, yo = Obs
    xe, te, se, ye = Exp

    y1_index = (te[:,0] == treated).unsqueeze(1)
    y0_index = (te[:,0] == control).unsqueeze(1)

    if normalization:
        # pre_process
        yo_mu = torch.mean(yo)
        yo_std = torch.std(yo)
        yo = (yo - yo_mu) / yo_std

    # o group first, e group second
    o_indicator = torch.cat((1. * torch.ones_like(yo), 0. * torch.zeros_like(ye)), dim=0)
    x = torch.cat((xo, xe), dim=0)
    t = torch.cat((1. * to, 1. * te), dim=0)
    s = torch.cat((so, se), dim=0)
    y = torch.cat((yo, 0. * torch.zeros_like(ye)), dim=0)
    if torch.cuda.is_available() and cuda:
        xe, se, te = xe.cuda(), se.cuda(), te.cuda()
        x, t, s, y, o_indicator = x.cuda(), t.cuda(), s.cuda(), y.cuda(), o_indicator.cuda()

    data_dim = s.shape[1]
    aux_dim = x.shape[1]
    N = x.shape[0]
    if inference_dim is not None:
        latent_dim = inference_dim
    else:
        latent_dim = 10

    # if print_log:
    #     print('Creating shuffled dataset..')
    dataset = TensorDataset(x, t, s, y, o_indicator)
    if early_stop:
        N = x.shape[0]
        train, valid = random_split(dataset, [N - int(valid_rate * N), int(valid_rate * N)])
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=True)
        for x,t,s,y,o in valid_loader:
            x_valid, t_valid, s_valid, y_valid, o_indicator_valid = x,t,s,y,o
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # define model and optimizer
    # if print_log:
    #     print('Defining model and optimizer..')
    model = iVAE_tx(latent_dim, data_dim, aux_dim, activation=activation, device=device,
                 n_layers=n_layers, hidden_dim=hidden_dim, slope=slope, anneal=anneal,
                 treatment_dim=treatment_dim)
    if normalization:
        model.y_mu, model.y_std, model.pre_process_y = yo_mu, yo_std, True

    if torch.cuda.is_available() and cuda:
        model = model.cuda()

    if optm == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    else:  # default: sgd
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=3, verbose=print_log)

    # training loop
    # if print_log:
    #     print("Training..")
    losses, elbos, l_yos, l_tes, l_inds = [], [], [], [], []
    global_valid_loss, valid_epoch = np.inf, 0

    iter = 0  # 为了计数

    for it in range(max_epoch):
        iter = iter
        for iter, (x, t, s, y, o_indicator) in enumerate(train_loader, iter):
            o_index = o_indicator == 1
            e_index = o_indicator == 0

            # print(s)

            if anneal:
                model.anneal(N, max_epoch, it)
            optimizer.zero_grad()
            loss, elbo, l_yo = model.loss_tatol(covariate=x, treatment=t, s=s, o_index=o_index, y=y,
                                                cuda=True, beta=beta, theta=theta)
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            optimizer.step()

            iter += 1

            with torch.no_grad():
                if print_log and iter % 1 == 0:
                    # xe_test = torch.cat((xe, 1. * te),dim=1)
                    ye = model.test(covariate=xe, treatment=1. * te, s=se)
                    # ye = ye * yo_std + yo_mu
                    if is_rct:
                        Ey1 = torch.mean(ye[y1_index])
                        Ey0 = torch.mean(ye[y0_index])
                        ate = Ey1 - Ey0
                        print('naive:epoch={%d}, iter={%d}, loss={%.4f}, ate={%.4f}, real_ate={%.4f}' % (it, iter, loss, ate, tau_real))
                    else:
                        x, t, y = xe.cpu().detach().numpy(), te.cpu().detach().numpy(), ye.cpu().detach().numpy()
                        ate = ipw_estimator(x=x, t=t, y=y)
                        print('ipw:epoch={}, iter={}, loss={}, ate={}'.format(it, len(losses), losses[-1], ate))
        with torch.no_grad():
            # valid loss
            if early_stop:
                o_index_valid = o_indicator_valid == 1
                valid_loss, elbo_, l_yo_ = model.loss_tatol(covariate=x_valid, s=s_valid, treatment=t_valid,
                                                                         o_index=o_index_valid, y=y_valid, cuda=True)
                if valid_loss < global_valid_loss:
                    if print_log:
                        print('update valid loss from {} to {}'.format(global_valid_loss, valid_loss))
                    global_valid_loss = valid_loss
                    model_dist = model.state_dict()
                    valid_epoch = 0
                else:
                    valid_epoch += 1
                    if valid_epoch >= early_stop_epoch:
                        break

        if it > base_eopch:
            scheduler.step(valid_loss)
        # print('current lr={}'.format(optimizer.param_groups[-1]['lr']))
        if optimizer.param_groups[-1]['lr'] < min_lr:
            break
    if early_stop:
        model.load_state_dict(model_dist)
    return losses, model
