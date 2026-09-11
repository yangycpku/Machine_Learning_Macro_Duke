"""
filename: train_nn.py
@authors: Jonathan Payne, Adam Rebei, Yucheng Yang

This file contains the economic environment parent class and functions
for solving the Hagedorn et al. (2017) model with aggregate shocks using deep learning.
"""

## Import python packages
import os
import sys
import numpy as np
from numpy import linalg as LA
import scipy.io
import torch.nn.functional as F
import torch

## Import user packages
from env import Environment

class Interpolate2DFunction:
    def __init__(self, n_x, n_y, Surp, device):
        self.device = device
        # Ensure x and y grid tensors are properly defined
        self.x_grid = torch.linspace(1/n_x/2, 1-1/n_x/2, n_x, dtype=torch.float32, device=device)
        self.y_grid = torch.linspace(1/n_y/2, 1-1/n_y/2, n_y, dtype=torch.float32, device=device)
        # Convert Surp to a tensor and reshape it to [1, 1, H, W] for grid_sample
        self.Surp_tensor = torch.tensor(Surp, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    def interpolate(self, x_input, y_input):
        # Ensure inputs are 1D tensors and on the correct device
        x_input = x_input.flatten().to(self.device)
        y_input = y_input.flatten().to(self.device)
        
        # Normalize x_input and y_input to [-1, 1]
        x_normalized = 2 * (x_input - self.x_grid[0]) / (self.x_grid[-1] - self.x_grid[0]) - 1
        y_normalized = 2 * (y_input - self.y_grid[0]) / (self.y_grid[-1] - self.y_grid[0]) - 1
        
        # Combine normalized x and y to form the grid
        grid = torch.stack([x_normalized, y_normalized], dim=1).unsqueeze(0).unsqueeze(0)
        
        # Perform interpolation
        interpolated_values = F.grid_sample(self.Surp_tensor, grid, align_corners=True)
        
        return interpolated_values.squeeze()

class Train_NN(Environment):
    """Subclass for solving the system using deep learning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        n_x = self.nx
        n_y = self.ny

        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=(n_x * n_y),
            scramble=True
        )

        self.types_x = torch.linspace(1 / n_x / 2, 1 - 1 / n_x / 2, n_x)
        self.types_y = torch.linspace(1 / n_y / 2, 1 - 1 / n_y / 2, n_y)
        self.unif_x = torch.ones(self.types_x.shape[0])
        self.unif_y = torch.ones(self.types_y.shape[0])


        _, _, _, self.entry_cost = self.create_gm_no_free_entry_refined(np.ones(n_y))



    def solve_steady_state(self):
        for s in ['S', 'L', 'H', 'D']:
            print("\nStarting DSS Fixed Point Algorithm: kappa = " + str(self.kappa) + ", s = " + s)
            self.create_gm_free_entry_refined(s)
        self.gm_high_low_diff = torch.tensor(np.load("gm_high_delta.npy") - np.load("gm_low_delta.npy"), dtype=torch.float32)
        self.gm_dis_low_diff = torch.tensor(np.load("gm_dis_delta.npy") - np.load("gm_low_delta.npy"), dtype=torch.float32)
        self.gm_dis_high_diff = torch.tensor(np.load("gm_dis_delta.npy") - np.load("gm_high_delta.npy"), dtype=torch.float32)
        
        self.gm_high = torch.tensor(np.load("gm_high_delta.npy"), dtype=torch.float32)
        self.gm_low = torch.tensor(np.load("gm_low_delta.npy"), dtype=torch.float32)
        self.gm_dis = torch.tensor(np.load("gm_dis_delta.npy"), dtype=torch.float32)
        self.gm_ss = torch.tensor(np.load("gm_ss.npy"), dtype=torch.float32)
        
    


    def create_gm_no_free_entry_refined(
        self,
        g_f,
        s='S',
        init_alphas=None,
        init_wages=None,
        init_gm_density=None,
        debug_outer_iter=None,
        print_every=None,
        max_main_iter=4000
    ):  # Values of s: 'L', 'H', 'D', 'S'
        ## Solve the fixed pt steady state
        xi = self.xi
        rho = self.rho
        kappa = self.kappa
        beta = self.beta
        n_x = self.nx
        n_y = self.ny
        nu = self.nu
        b = self.b
        g_w = self.gw_np
        z = self.z_0
        tau = float(self.tau.item())
        tau_w = float(self.tau_w.item())

        ## determining delta and varsigma_z
        if s == 'S':
            delta = ((self.delta_L + self.delta_H) / 2).detach().cpu().numpy()
            varsigma_z = 0.5 * (float(self.varsigma_s[0].item()) + float(self.varsigma_s[1].item()))
        elif s == 'L':
            delta = self.delta_L.detach().cpu().numpy()
            varsigma_z = float(self.varsigma_s[0].item())
        elif s == 'H':
            delta = self.delta_H.detach().cpu().numpy()
            varsigma_z = float(self.varsigma_s[1].item())
        elif s == 'D':
            delta = self.delta_Dis.detach().cpu().numpy()
            varsigma_z = float(self.varsigma_s[2].item())
        else:
            raise ValueError("Invalid value for s.")

        tol = 1e-9

        types_x = np.linspace(1 / n_x / 2, 1 - 1 / n_x / 2, n_x)
        types_y = np.linspace(1 / n_y / 2, 1 - 1 / n_y / 2, n_y)

        # Initial values
        if init_alphas is None:
            alphas = np.random.rand(n_x, n_y)
        else:
            alphas = init_alphas.copy()

        if init_wages is None:
            wages = np.random.rand(n_x, n_y)
        else:
            wages = init_wages.copy()

        if init_gm_density is None:
            gm_density = 1e-1 * np.random.rand(n_x, n_y) * g_f.reshape(1, -1)
        else:
            gm_density = init_gm_density.copy()

        ## calculate marginal densities from the joint density (g_m), as well as aggregate
        ## unemployment and vacancies
        def marginals_U_V(g_m):

            # aggregate unemployment
            g_e = 1 / n_y * np.sum(g_m, axis=1)
            g_u = g_w - g_e
            U = 1 / n_x * np.sum(g_u, axis=0)

            # aggregate vacant firms
            g_p = 1 / n_x * np.sum(g_m, axis=0)
            g_v = g_f - g_p
            V = 1 / n_y * np.sum(g_v, axis=0)

            return g_e, g_u, U, g_p, g_v, V

        # Computing the payoffs for all the types combinations
        payoffs = np.empty([n_x, n_y])
        for i in range(n_x):
            x = types_x[i]
            for j in range(n_y):
                y = types_y[j]
                payoffs[i, j] = z * self.calc_f(x, y)

        keep_iterating = True
        iter_num = 0

        # Main loop
        while keep_iterating:
            # Updating the matched density g_m.
            e = sys.float_info.max

            gm_prev_init = gm_density.copy()
            gm_prev = gm_density   # keep original behavior

            while e > tol:
                g_e, g_u, U, g_p, g_v, V = marginals_U_V(gm_prev)
                for i in range(n_x):
                    for j in range(n_y):
                        gm_density[i, j] = (
                            1 / (delta[i, j] + varsigma_z)
                            * self.m(U, V) / (U * V)
                            * alphas[i, j]
                            * g_u[i]
                            * g_v[j]
                        )
                e = np.linalg.norm(gm_prev - gm_density)
                gm_prev = gm_density

            gm_density = 0.99 * gm_prev_init + 0.01 * gm_density
            if np.isnan(gm_density).any():
                raise RuntimeError(
                    f"gm_density became NaN in create_gm_no_free_entry_refined, "
                    f"s={s}, outer_iter={debug_outer_iter}, inner_main_iter={iter_num}"
                )

            g_e, g_u, U, g_p, g_v, V = marginals_U_V(gm_density)

            # Updating the V_u equation
            new_psy = U * V / self.m(U, V)
            eta_u = rho + delta + varsigma_z * (1 - tau_w)
            V_u = (
                (b * n_y * new_psy) + np.dot((alphas * wages) / eta_u, g_v)
            ) / (
                n_y * new_psy * rho + rho * np.dot(alphas / eta_u, g_v)
            )

            V_e = (wages + (delta + varsigma_z * (1 - tau_w)) * V_u.reshape((n_x, 1))) / eta_u

            # Updating the V_v equation
            eta_v = rho + delta + varsigma_z * (1 - tau)
            V_v = np.dot((alphas * (payoffs - wages) / eta_v).T, g_u) / (
                (rho + varsigma_z * (1 - tau)) * n_x * new_psy
                + (rho + varsigma_z * (1 - tau)) * np.dot((alphas / eta_v).T, g_u)
            )
            V_p = np.zeros((n_x, n_y))

            for i in range(n_x):
                for j in range(n_y):
                    V_p[i, j] = (payoffs[i, j] - wages[i, j] + delta[i, j] * V_v[j]) / eta_v[i, j]

            # Calculating the surplus
            surplus = (
                payoffs + varsigma_z * (tau - 1) * V_p - rho * (V_e + V_p)
            ) / (
                delta + beta * varsigma_z * (1 - tau_w)
            )

            # Updating the matching set
            new_alphas = np.zeros([n_x, n_y])
            for i in range(n_x):
                for j in range(n_y):
                    new_alphas[i, j] = 1 / (1 + np.exp(-xi * surplus[i, j]))

            # updating the wage
            new_wages = np.zeros([n_x, n_y])
            for i in range(n_x):
                for j in range(n_y):
                    new_wages[i, j] = (
                        beta * eta_u[i, j] * (payoffs[i, j] - (rho + varsigma_z * (1 - tau)) * V_v[j])
                        + (1 - beta) * eta_v[i, j] * rho * V_u[i]
                    ) / ((1 - beta) * eta_v[i, j] + beta * eta_u[i, j])

            distance = LA.norm(alphas - new_alphas, 'fro')
            distance_wages = LA.norm(wages - new_wages, 'fro')

            iter_num += 1

            if (print_every is not None) and (iter_num % print_every == 0):
                print(
                    f"[no-free-entry debug] s={s} outer_iter={debug_outer_iter} "
                    f"inner_iter={iter_num} dist_a={distance:.6e} dist_w={distance_wages:.6e} "
                    # f"U={U:.6e} V={V:.6e}"
                )

            if iter_num >= max_main_iter:
                raise RuntimeError(
                    f"create_gm_no_free_entry_refined hit max_main_iter for s={s}, "
                    f"outer_iter={debug_outer_iter}. "
                    f"dist_a={distance:.6e}, dist_w={distance_wages:.6e}, "
                    f"U={U:.6e}, V={V:.6e}"
                )

            # Checking if convergence
            if (distance < 1e-9) and (distance_wages < 1e-9):
                keep_iterating = False
            else:
                alphas = new_alphas
                wages = new_wages

        return alphas, surplus, gm_density, np.mean(V_v)



    def create_gm_free_entry_refined(self, s='S', verbose=False):
        xi = self.xi
        rho = self.rho
        kappa = self.kappa
        beta = self.beta
        n_x = self.nx
        n_y = self.ny
        nu = self.nu
        b = self.b
        z = self.z_0
        g_w = self.gw_np
        entry_c = self.entry_cost
        tau = float(self.tau.item())

        print('Entry cost calibration should be ', entry_c)

        if s == 'S':
            delta = ((self.delta_L + self.delta_H) / 2).detach().cpu().numpy()
            varsigma_z = 0.5 * (float(self.varsigma_s[0].item()) + float(self.varsigma_s[1].item()))
        elif s == 'L':
            delta = self.delta_L.detach().cpu().numpy()
            varsigma_z = float(self.varsigma_s[0].item())
        elif s == 'H':
            delta = self.delta_H.detach().cpu().numpy()
            varsigma_z = float(self.varsigma_s[1].item())
        elif s == 'D':
            delta = self.delta_Dis.detach().cpu().numpy()
            varsigma_z = float(self.varsigma_s[2].item())
        else:
            raise ValueError("Invalid value for s.")

        def marginals_U_V(g_m, g_w, g_f):
            g_e = 1 / n_y * np.sum(g_m, axis=1)
            g_u = g_w - g_e
            U = 1 / n_x * np.sum(g_u, axis=0)

            g_p = 1 / n_x * np.sum(g_m, axis=0)
            P = np.mean(g_m)
            g_v = g_f - g_p
            V = 1 / n_y * np.sum(g_v, axis=0)

            return g_e, g_u, U, g_p, g_v, V

        if s == 'S':
            g_f = np.ones(n_y)
            alphas, surplus, gm, _ = self.create_gm_no_free_entry_refined(g_f, s)
        else:
            dis = sys.float_info.max
            max_outer_iter = 5000
            outer_iter = 0

            g_f = (1 + np.random.uniform(-1, 1) * 0.4) * np.ones(n_y)

            prev_alphas = None
            prev_gm = None

            while dis > 1e-9:
                outer_iter += 1

                if outer_iter > max_outer_iter:
                    raise RuntimeError(
                        f"create_gm_free_entry_refined hit max_outer_iter for s={s}. "
                        f"Last dist={dis:.6e}, mean(g_f)={np.mean(g_f):.6e}"
                    )

                alphas, surplus, gm, _ = self.create_gm_no_free_entry_refined(
                    g_f,
                    s,
                    init_alphas=prev_alphas,
                    init_wages=None,
                    init_gm_density=prev_gm,
                    debug_outer_iter=outer_iter,
                    print_every=None,
                    max_main_iter=5000
                )

                prev_alphas = alphas.copy()
                prev_gm = gm.copy()

                g_e, g_u, U, g_p, g_v, V = marginals_U_V(gm, g_w, g_f)

                mean_term = np.mean(alphas * (1 - beta) * surplus * g_u.reshape(-1, 1) / U)
                V_1 = (
                    1 / ((rho + varsigma_z * (1 - tau)) * entry_c) ** (1 / nu)
                    * kappa ** (1 / nu)
                    * U
                    * mean_term ** (1 / nu)
                )

                P = 1 / n_y * np.sum(g_p, axis=0)
                g_f_1 = (V_1 + P) * np.ones(n_y)
                g_f = (g_f + g_f_1) / 2
                dis = np.linalg.norm(g_f - g_f_1)

                if verbose and outer_iter % 5 ==0:
                    print(
                        f"[free-entry debug] s={s} iter={outer_iter} "
                        # f"U={U:.6e} V_old={V:.6e} P={np.mean(gm):.6e} "
                        # f"mean_term={mean_term:.6e} V1={V_1:.6e} "
                        # f"g_f_mean={np.mean(g_f):.6e} dist={dis:.6e}"
                        f"U={U:.6e} dist={dis:.6e}"
                    )

            print('g_f is ', g_f)
            print('Total unemployment U is ', U)
            print('Total vacancy V is ', V)
            print('Total vacancy V1 is ', V_1, '\n')

        if s == 'S':
            np.save("gm_ss", gm)
            np.save("gf_ss", g_f)
            np.save("alpha_ss", alphas)
            np.save("surplus_ss", surplus)

        elif s == 'H':
            np.save("gm_high_delta", gm)
            np.save("gf_high_delta", g_f)
            np.save("alpha_H", alphas)
            np.save("surplus_H", surplus)

        elif s == 'L':
            np.save("gm_low_delta", gm)
            np.save("gf_low_delta", g_f)
            np.save("alpha_L", alphas)
            np.save("surplus_L", surplus)

        elif s == 'D':
            np.save("gm_dis_delta", gm)
            np.save("gf_dis_delta", g_f)
            np.save("alpha_D", alphas)
            np.save("surplus_D", surplus)

        else:
            raise ValueError("Invalid delta for DSS.")


    


    def sample(self, N: int):
        """
        Draw N training samples with (x, y, z, g_m) where:
          - z ~ Categorical(self.prob_agg)
          - g_m is a convex combination of (gm_L, gm_H, gm_D)
            with a dominant weight on the chosen base state (z),
            using ε ~ Beta(α, 1) for the base and splitting the
            remainder across the two non-base states in proportion
            to their probabilities in self.prob_agg.
        Returns S_batch, V_u_batch, V_v_batch with the same shapes as before.
        """
        import torch
        import torch.nn.functional as F

        device = getattr(self, "device", None) or self.gm_low.device
        n_x, n_y = self.nx, self.ny
        L = n_x * n_y

        # --- steady-state surfaces (L, H, D) ---
        gm_L = self.gm_low.to(device)                                  # (n_x, n_y)
        gm_H = (self.gm_low + self.gm_high_low_diff).to(device)         # (n_x, n_y)
        gm_D = (self.gm_low + self.gm_dis_low_diff).to(device)          # (n_x, n_y)  # use LOW anchor consistently
        GM   = torch.stack([gm_L.flatten(), gm_H.flatten(), gm_D.flatten()], dim=0)  # (3, L)

        # --- sample z labels according to target state probabilities ---
        p = self.prob_agg.to(device).flatten()                          # (3,)
        p = p / p.sum()                                                 # ensure normalized
        z = torch.multinomial(p, N, replacement=True).to(device)        # (N,)
        z_col = z.view(-1, 1)                                           # (N,1)

        # --- base-state dominant weight via Beta(alpha,1) ---
        alpha = getattr(self, "alpha_base", 10.0)                       # knob; larger => closer to base
        # sample ε ~ Beta(α,1) via ε = U^(1/α)
        eps = torch.rand(N, 1, device=device) ** (1.0 / float(alpha))   # (N,1) in (0,1)
        one_hot = F.one_hot(z, num_classes=3).to(device=device, dtype=torch.float32)  # (N,3)

        # --- probability-proportional deterministic split of the remainder ---
        p_expand = p.unsqueeze(0).expand(N, 3)                          # (N,3)
        base_p   = (p_expand * one_hot).sum(dim=1, keepdim=True)        # (N,1) = p[z]
        rem      = 1.0 - eps                                            # (N,1)
        # zero out base, then renormalize the two others
        share = p_expand.clone()
        share.scatter_(1, z_col, 0.0)                                   # zero the base col
        denom = (1.0 - base_p).clamp_min(1e-12)                         # avoid /0 if a state has prob 1
        share = share / denom                                           # rows now sum to 1 over the two non-base states

        # --- final convex weights (N,3) ---
        W = eps * one_hot + rem * share                                  # convex, rows sum to 1

        # --- build g_m for each sample: (N,L) = (N,3) @ (3,L) ---
        g_m = W @ GM                                                     # (N, L)

        # --- sample x, y uniformly over types (same as before) ---
        idx_x = self.unif_x.multinomial(N, replacement=True)
        x = self.types_x[idx_x].reshape(N, 1).to(device)
        idx_y = self.unif_y.multinomial(N, replacement=True)
        y = self.types_y[idx_y].reshape(N, 1).to(device)

        # --- pack batches (same shapes as before) ---
        z_feat = z_col.to(torch.float32)                                 # (N,1)
        S_batch  = torch.hstack((x, y, z_feat, g_m))                     # (N, 1+1+1+L)
        V_u_batch = torch.hstack((x, z_feat, g_m))                       # (N, 1+1+L)
        V_v_batch = torch.hstack((y, z_feat, g_m))                       # (N, 1+1+L)

        return S_batch, V_u_batch, V_v_batch



    def get_derivs_1order(self, y_pred, x):
        """ Returns zeroth, first and second derivatives.
            Uses automatic differentation to take all derivatives.
        """
        dy_dx = torch.autograd.grad(y_pred, x,
                            create_graph=True,
                            grad_outputs=torch.ones_like(y_pred))[0]
        return dy_dx ## Return 'automatic' gradient.

    def y_init(self, X, option):
        """
        """
        ## Unpack
        device  = self.device
        n_x     = self.nx 
        n_y     = self.ny
        z_0     = self.z_0
        # z_s     = self.z_s

        ## Set up vectors
        x = X[:, 0].to(device)
        y = X[:, 1].to(device)
        z = X[:, 2].to(device)
        g_m = X[:, 3:(n_x*n_y + 3)].to(device)

        if option == 'S':
            # return z_s[z.int()]* self.f_torch(x, y)
            Surp_dss = np.load('surplus_ss.npy')
            interpolator = Interpolate2DFunction(n_x, n_y, Surp_dss, device)
            return interpolator.interpolate(x, y)
            # data_and_interpolator = DataLoaderAndInterpolator()
            # return data_and_interpolator.interpolator.interpolate(x, y)

        elif option == 'V_u':
            ## Calculate payoffs
            types_x = torch.linspace(1/n_x/2, 1-1/n_x/2, n_x).to(device)
            types_y = torch.linspace(1/n_y/2, 1-1/n_y/2, n_y).to(device)
            payoffs = torch.empty([n_x, n_y]).to(device)
            for i in range(n_x):
                xi = types_x[i]
                for j in range(n_y):
                    yj = types_y[j]
                    payoffs[i, j] = self.f_torch(xi, yj)
            indices_x = (x/(types_x[1] - types_x[0]) - 0.5).long()

            ## Reshape g_m from flattened vector input to (n_x, n_y) array 
            ##   for each of the N data point
            g_m_reshaped_2d = g_m.reshape((len(x), n_x, n_y))

            ## Aggregate vacant firms
            g_p = 1/n_x * torch.sum(g_m_reshaped_2d, axis = 1)
            g_v = g_f - g_p
            z = z.reshape(-1,1)
            
            return torch.sum(z_0*payoffs[indices_x, :]*g_v, axis = 1)

        elif option == 'V_v':
            ## Calculate payoffs
            types_x = torch.linspace(1/n_x/2, 1-1/n_x/2, n_x).to(device)
            types_y = torch.linspace(1/n_y/2, 1-1/n_y/2, n_y).to(device)
            payoffs = torch.empty([n_x, n_y]).to(device)
            for i in range(n_x):
                xi = types_x[i]
                for j in range(n_y):
                    yj = types_y[j]
                    payoffs[i, j] = self.f_torch(xi, yj)
            indices_y = (y/(types_y[1] - types_y[0]) - 0.5).long()

            ## Reshape g_m from flattened vector input to (n_x, n_y) array for 
            ##   each of the N data point
            g_m_reshaped_2d = g_m.reshape((len(x), n_x, n_y))
            # Aggregate unemployment
            g_e = 1/n_y * torch.sum(g_m_reshaped_2d, axis = -1)
            g_u = g_w - g_e

            return torch.sum(payoffs[:, indices_y].T*g_u, axis = 1)
            
    # updated by Yaqi Zeng
    # def initial_guess(self, model, Sampler, optimizer, epochs, option):
    def initial_guess(self, model, optimizer, epochs, option):
        device = self.device
        for epoch in range(1, epochs, 1):
            # updated by Yaqi Zeng
            # X_pretrain_xyzg, X_pretrain_xzg, X_pretrain_yzg  = Sampler.sample(1000)
            X_pretrain_xyzg, X_pretrain_xzg, X_pretrain_yzg  = self.sample(1000)

            # transform into tensor variables
            if option == 'S':
                X_pretrain_tensor = torch.tensor(X_pretrain_xyzg, requires_grad=True, dtype=torch.float32)
            if option == 'V_u':
                X_pretrain_tensor = torch.tensor(X_pretrain_xzg, requires_grad=True, dtype=torch.float32)
            if option == 'V_v':
                X_pretrain_tensor = torch.tensor(X_pretrain_yzg, requires_grad=True, dtype=torch.float32)

        # for epoch in range(1, epochs, 1):
        #     X_pretrain_xyg, X_pretrain_xg, X_pretrain_yg = Sampler.sample(1000)

        #     ## Transform into tensor variables
        #     if option == 'S':
        #         X_pretrain_tensor = torch.tensor(X_pretrain_xyg,
        #                             requires_grad=True, dtype=torch.float32)
        #     if option == 'V_u':
        #         X_pretrain_tensor = torch.tensor(X_pretrain_xg, 
        #                             requires_grad=True, dtype=torch.float32)
        #     if option == 'V_v':
        #         X_pretrain_tensor = torch.tensor(X_pretrain_yg, 
        #                             requires_grad=True, dtype=torch.float32)

            ## Zero the parameter gradients
            optimizer.zero_grad()

            ## Run input through the pinn
            outputs = model(X_pretrain_tensor)

            ## Loss function V
            y_init_vals = self.y_init(X_pretrain_xyzg, option).reshape(1000,1).to(device)
            loss = torch.mean(torch.square(outputs - y_init_vals))

            # ## Loss function V
            # y_init_vals = self.y_init(X_pretrain_xyg, option
            #                 ).reshape(1000,1).to(device)
            # loss        = torch.mean(torch.square(outputs - y_init_vals))

            ## Backward propagation
            loss.backward()

            ## Update model parameters
            optimizer.step()

            ## Turn tensor variable into array
            total_l  = loss.detach().cpu().numpy()

            ## Print loss value in specific epochs
            if epoch%1000 == 0:
                print("Iter %d: Total loss = %.4e" % (epoch, total_l))

            ## Check for convergence
            if total_l < 1e-6:
                print('Converged at epoch %s with training loss %s' 
                        % (epoch, total_l))
                break

    # updated by Yaqi Zeng
    # def training(self, pinn_S, model, Sampler, optimizer, scheduler, 
    #         epochs, save_path, option, save_freq=None, N=256):
    def training(self, pinn_S, model, optimizer, scheduler, 
            epochs, save_path, option, save_freq=None, loss_threshold=1e-5, N=256):
        ## Unpack
        path = self.path
        ## Train
        for epoch in range(1, epochs, 1):
            ## Resample every 75 epochs
            if epoch%2 ==1:
                # updated by Yaqi Zeng
                # X_batch_xyzg, X_batch_xzg, X_batch_yzg = Sampler.sample(N)
                X_batch_xyzg, X_batch_xzg, X_batch_yzg = self.sample(N)
            # X_batch_xyg, X_batch_xg, X_batch_yg = Sampler.sample(N)

            ## S residuals: the pde loss: (pde_oper(S)-0)^2
            if option == 'S':
                residuals, _, _ = self.S_pde_oper(model, X_batch_xyzg)
            if option == 'V_u':
                residuals, _, _ = self.V_u_pde_oper(pinn_S, model, X_batch_xyzg, X_batch_xzg)
            if option == 'V_v':
                residuals, _, _ = self.V_v_pde_oper(pinn_S, model, X_batch_xyzg, X_batch_yzg)

            # if option == 'S':
            #     residuals, _, _ = self.S_pde_oper(model, X_batch_xyg)
            # if option == 'V_u':
            #     residuals, _, _ = self.V_u_pde_oper(pinn_S, model, 
            #                         X_batch_xyg, X_batch_xg)
            # if option == 'V_v':
            #     residuals, _, _ = self.V_v_pde_oper(pinn_S, model, 
            #                         X_batch_xyg, X_batch_yg)
            loss = torch.mean(torch.square(residuals))

            ## Set grad to be zero, backward propogation, and update model
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(pinn_V.parameters(), 1)
            optimizer.step()
            #scheduler.step(loss)
                    
            scheduler.step(loss)

            if epoch%100 == 0:
                fo = open(os.path.join(path,'output_loss_' + str(option) 
                        + '.txt'), "a")
                string = "Iter %d: Loss = %.4e"\
                        % (epoch, loss)
                fo.write( (string+'\n') )
                fo.flush()
            
            if epoch%1000 == 0:
                print("Epoch " + str(epoch) + ", Loss_" + str(option) + ": " 
                    + str(loss.detach().cpu().numpy()))

            ## Checkpoint saving
            if save_freq is not None and epoch % save_freq == 0:
                model_save_name = f"model_S_epoch_{epoch}.pt"
                savepath = os.path.join(save_path, model_save_name)
                
                torch.save({
                    'modelS_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, savepath)

            if (loss<loss_threshold):
                print("Epoch " + str(epoch) + ", Loss: " 
                    + str(loss.detach().cpu().numpy()))
                print("NN Converged.")
                break




    def generate_z_paths(
        self,
        N: int = 50,
        T: int = 50_000,
        dt: float = 0.01,
        init_probs=None,            # length-3 probabilities (p_L, p_H, p_D)
        seed: int | None = None,
        warn_threshold: float = 0.2,
        as_float: bool = True,
        verbose: bool = False,
    ):
        """
        Simulate N×T paths of a 3-state CTMC z_t ∈ {0(L),1(H),2(D)} with at most one jump per step (small q_i*dt).
        Uses explicit rates:
            self.lam_LH, self.lam_LD,
            self.lam_HL, self.lam_HD,
            self.lam_DL, self.lam_DH

        One-step transition (from state i):
            Stay with prob   e^{-q_i dt}
            Jump to j≠i with prob (λ_{i→j}/q_i) * (1 - e^{-q_i dt}), where q_i = Σ_{k≠i} λ_{i→k}.
        """

        # --- basic checks ---
        if N <= 0 or T <= 0:
            raise ValueError("N and T must be positive integers.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if init_probs is None:
            raise ValueError("init_probs must be a length-3 probability vector (p_L, p_H, p_D).")

        import torch

        if seed is not None:
            torch.manual_seed(int(seed))

        device = getattr(self, "device", "cpu")

        # --- gather λ's and q's ---
        lam_LH = torch.as_tensor(self.lam_LH, device=device, dtype=torch.float32)
        lam_LD = torch.as_tensor(self.lam_LD, device=device, dtype=torch.float32)
        lam_HL = torch.as_tensor(self.lam_HL, device=device, dtype=torch.float32)
        lam_HD = torch.as_tensor(self.lam_HD, device=device, dtype=torch.float32)
        lam_DL = torch.as_tensor(self.lam_DL, device=device, dtype=torch.float32)
        lam_DH = torch.as_tensor(self.lam_DH, device=device, dtype=torch.float32)

        # total out-rates q_i
        q = torch.stack([
            lam_LH + lam_LD,   # q_L
            lam_HL + lam_HD,   # q_H
            lam_DL + lam_DH    # q_D
        ])  # shape (3,)

        # warn about potential multi-jump risk in a step
        max_q_dt = (q.max() * dt).item()
        if max_q_dt > warn_threshold:
            print(
                f"[generate_z_paths] WARNING: dt*max(q)={max_q_dt:.3f} > {warn_threshold}. "
                "One-jump-per-step approximation may be inaccurate. Consider smaller dt."
            )

        # --- initial states from a single 3-element probability vector ---
        init_probs = torch.as_tensor(init_probs, device=device, dtype=torch.float32).flatten()
        if init_probs.numel() != 3:
            raise ValueError("init_probs must have exactly 3 elements (p_L, p_H, p_D).")
        if (init_probs < 0).any() or not torch.isclose(init_probs.sum(), torch.tensor(1.0, device=device)):
            raise ValueError("init_probs must be nonnegative and sum to 1.")
        z0 = torch.multinomial(init_probs, num_samples=N, replacement=True)  # (N,), values in {0,1,2}

        # --- precompute per-state jump destination ordering and proportions ---
        # For each current state i, destinations are (j1, j2) with probs proportional to λ_{i→j}
        dest_j1 = torch.tensor([1, 0, 0], device=device, dtype=torch.long)  # from L→H, H→L, D→L
        dest_j2 = torch.tensor([2, 2, 1], device=device, dtype=torch.long)  # from L→D, H→D, D→H

        # λ_{i→j1}, λ_{i→j2}
        lam_to_j1 = torch.stack([lam_LH, lam_HL, lam_DL])
        lam_to_j2 = torch.stack([lam_LD, lam_HD, lam_DH])

        # --- allocate and evolve ---
        z = torch.empty((N, T), device=device, dtype=torch.long)
        z[:, 0] = z0

        for t in range(1, T):
            prev = z[:, t - 1]                     # (N,)
            q_prev = q[prev]                       # (N,)
            p_jump = 1.0 - torch.exp(-q_prev * dt) # (N,)
            u_jump = torch.rand(N, device=device)

            # default: stay
            next_state = prev.clone()

            # those that jump this step
            J = u_jump < p_jump
            if J.any():
                iJ = prev[J]                       # current states for jumpers, (M,)
                qJ = q[iJ]                         # (M,)
                # probabilities to two destinations
                p1 = torch.where(qJ > 0, lam_to_j1[iJ] / qJ, 0.0)
                # p2 = 1 - p1 implicitly; safe when qJ==0 because J implies p_jump>0 ⇒ qJ>0
                u_dest = torch.rand(iJ.shape[0], device=device)
                go_j1 = u_dest < p1

                # pick destinations
                j1 = dest_j1[iJ]
                j2 = dest_j2[iJ]
                next_state[J] = torch.where(go_j1, j1, j2)

            z[:, t] = next_state

            if verbose and (t % max(1, T // 10) == 0):
                shares = torch.stack([(z[:, t] == 0).float().mean(),
                                      (z[:, t] == 1).float().mean(),
                                      (z[:, t] == 2).float().mean()])
                print(f"[generate_z_paths] t={t}/{T-1}  shares(L,H,D)={shares.tolist()}  "
                      f"E[p_jump]={p_jump.mean().item():.4f}")

        return z.to(torch.float32) if as_float else z

    


    def simulate_economy(
        self,
        model_S,
        z_paths,                 # (N, T) ints in {0,1,2}; column 0 is initial time
        dt: float,
        substeps: int = 1,       # micro-steps per macro-step for stability
        g0=None,                 # initial g as (nx*ny,), else uses self.gm_ss
        record_interval: int = 1,
        burn_in_frac: float = 0.2,
        clip_eps: float = 1e-12,
        return_paths: bool = False,
    ):
        """
        Simulate g_t(x,y) forward on given 3-state CTMC paths z_t ∈ {0(L),1(H),2(D)} using trained model_S.
        Alpha is continuous only (no discrete branch): alpha = σ(ξ·S) inside calculate_alphas.
        - Euler update with optional micro-steps (substeps).
        - Applies jump in g at z-switch:
              g_{t+} = (1 - sigma(z_t, z_{t+1})) * g_{t-}
        - Records sparse snapshots and returns ergodic averages after burn-in.

        Returns dict with:
          - 'ergodic_g': (nx*ny,) post–burn-in average over paths & time
          - 'U_avg'    : (K,) average unemployment across paths at each recorded time
          - 'record_idx': list of recorded macro indices
          - 'z_recorded': z aligned with recorded g snapshots (post burn-in)
          - optionally 'g_paths': (N, K, nx*ny), 'U_paths': (N, K)
        """
        import torch

        model_S = model_S.eval()
        device = getattr(self, "device", "cpu")
        nx, ny = self.nx, self.ny
        L = nx * ny

        # --- z paths & basic shapes ---
        if not torch.is_tensor(z_paths):
            z_paths = torch.as_tensor(z_paths, device=device)
        else:
            z_paths = z_paths.to(device)
        assert z_paths.ndim == 2, "z_paths must be (N, T)"
        N, T = z_paths.shape
        assert T >= 1, "z_paths must have at least one time column"

        # sigma matrix (3x3, diagonal should be zero by construction)
        sigma_mat = torch.as_tensor(self.sigma_mat, device=device, dtype=torch.float32)

        # --- initial g (flattened length L), replicated across paths ---
        if g0 is None:
            g0_arr = self.gm_ss
        else:
            g0_arr = g0
        g0_t = torch.as_tensor(g0_arr, device=device, dtype=torch.float32).reshape(-1)
        assert g0_t.numel() == L, "g0 (or self.gm_ss) must flatten to length nx*ny"
        g_t = g0_t.unsqueeze(0).repeat(N, 1)  # (N, L)

        # --- recording plan ---
        record_idx = sorted(set([0] + list(range(0, T, record_interval)) + [T-1]))
        K = len(record_idx)
        if return_paths:
            g_paths = torch.empty((N, K, L), device=device, dtype=torch.float32)
            U_paths = torch.empty((N, K), device=device, dtype=torch.float32)
        U_avg = torch.empty(K, device=device, dtype=torch.float32)

        dt_sub = float(dt) / int(substeps)
        eps_div = 1e-12  # safe division

        # --- helper: record snapshot at macro index t_idx into slot k ---
        @torch.no_grad()
        def _record(k: int, t_idx: int, g_now: torch.Tensor):
            # z feature/dtype: NN usually wants float; drift may need int for indexing
            z_now_long = z_paths[:, t_idx].to(torch.long).view(N)        # (N,)
            z_now_feat = z_now_long.to(torch.float32).view(N, 1)         # (N,1)

            S_now, alphas_now = self.calculate_alphas(model_S, z_now_feat, g_now)  # (N,nx,ny)
            _, g_u, U_now, _, _, _, _ = self.marginals_U_V(g_now, S_now, alphas_now, z_now_feat)  # U_now: (N,1)
            U_now = U_now.view(N)

            if return_paths:
                g_paths[:, k, :] = g_now
                U_paths[:, k] = U_now

            U_avg[k] = U_now.mean()

        # --- record initial state at t=0 ---
        with torch.no_grad():
            _record(k=0, t_idx=0, g_now=g_t.clone())

        # --- main loop over macro steps ---
        with torch.no_grad():
            rec_ptr = 1  # next slot in recorded arrays
            for t in range(T - 1):
                # fixed z for this macro interval (piecewise constant during substeps)
                z_long = z_paths[:, t].to(torch.long).view(N)           # (N,)
                z_feat = z_long.to(torch.float32).view(N, 1)            # (N,1)

                for _ in range(substeps):
                    # 1) surplus & alpha (continuous)
                    S, alphas = self.calculate_alphas(model_S, z_feat, g_t)  # (N,nx,ny)

                    # 2) aggregates (free entry inside)
                    g_e, g_u, U, g_p, g_v, V, g_f = self.marginals_U_V(g_t, S, alphas, z_feat)  # U,V: (N,1)

                    # 3) meetings per unemployed
                    m_total = self.m(U, V)                        # (N,1)
                    M_u = m_total / (U + eps_div)                 # (N,1)

                    # 4) KFE drift using state-specific delta_s[z] (+ varsigma in mu_g)
                    mu = self.mu_g(g_t, M_u, V, alphas, g_e, g_p, g_f, z_long)  # (N, L)

                    # 5) Euler update with clamp
                    g_t = torch.clamp(g_t + dt_sub * mu, min=clip_eps)

                # 6) jump exactly at boundary t -> t+1 if z changes
                z_next_long = z_paths[:, t + 1].to(torch.long).view(N)  # (N,)
                jump_mask = (z_next_long != z_long)
                if jump_mask.any():
                    sigma_jump = sigma_mat[z_long[jump_mask], z_next_long[jump_mask]]  # (n_jump,)
                    g_t[jump_mask, :] = (1.0 - sigma_jump).unsqueeze(1) * g_t[jump_mask, :]
                    g_t = torch.clamp(g_t, min=clip_eps)

                # --- record after finishing macro step to time t+1 (post-jump) ---
                if (t + 1) in record_idx:
                    _record(k=rec_ptr, t_idx=t + 1, g_now=g_t)
                    rec_ptr += 1

        # --- ergodic averages (post burn-in over recorded macro times) ---
        burn_k = int(max(0, min(K - 1, round(burn_in_frac * (K - 1)))))
        tail = slice(burn_k, K)

        if return_paths:
            ergodic_g = g_paths[:, tail, :].mean(dim=(0, 1))  # (L,)
        else:
            # better: return_paths=True when you need ergodic_g precisely
            raise RuntimeError("Set return_paths=True to compute ergodic_g reliably.")
        U_avg_tail = U_avg[tail].clone()

        out = {
            "ergodic_g": ergodic_g,                     # (nx*ny,)
            "U_avg": U_avg_tail,                        # (K - burn_k,)
            "record_idx": record_idx[burn_k:],          # macro indices corresponding to U_avg
            "z_recorded": z_paths[:, record_idx[burn_k:]],  # aligned with returned g_paths
        }
        if return_paths:
            out["g_paths"] = g_paths[:, burn_k:, :]     # (N, K - burn_k, L)
            out["U_paths"] = U_paths[:, burn_k:]        # (N, K - burn_k)
        return out



    

    def expected_error_maps(
        self,
        model_S,
        out,                  # output dict from simulate_economy(..., return_paths=True)
        z_paths,              # (N, T) ints in {0,1,2}
        chunk_paths: int = 64 # number of paths to process per chunk
    ):
        """
        Compute expected squared surplus-PDE residual per (x,y) over post–burn-in recorded snapshots.
        Returns unconditional and by-state maps:
          {
            "E_err":   (nx, ny),
            "E_err_L": (nx, ny), "E_err_H": (nx, ny), "E_err_D": (nx, ny),
            "counts":  {"all": int, "L": int, "H": int, "D": int},
          }
        """
        import torch

        device = getattr(self, "device", "cpu")
        nx, ny = self.nx, self.ny
        L = nx * ny

        # --- pull recorded snapshots (post–burn-in) ---
        g_paths = out["g_paths"]          # (N, K, L)
        record_idx = out["record_idx"]    # list of length K
        assert g_paths.ndim == 3 and g_paths.shape[2] == L, "g_paths must be (N, K, nx*ny)"
        N, K, _ = g_paths.shape

        # --- z_paths to torch on correct device ---
        if not torch.is_tensor(z_paths):
            z_paths = torch.as_tensor(z_paths, device=device)
        else:
            z_paths = z_paths.to(device)
        assert z_paths.shape[0] == N, "z_paths and g_paths must agree on N"

        # --- (x,y) grids as torch tensors ---
        x = torch.as_tensor(self.x_grid, device=device, dtype=torch.float32)  # (nx,)
        y = torch.as_tensor(self.y_grid, device=device, dtype=torch.float32)  # (ny,)
        xx, yy = torch.meshgrid(x, y, indexing='ij')                          # (nx, ny)
        x_flat = xx.reshape(-1)                                               # (L,)
        y_flat = yy.reshape(-1)                                               # (L,)
        xy = torch.stack([x_flat, y_flat], dim=1)                             # (L,2)

        # --- accumulators ---
        sum_all = torch.zeros((nx, ny), device=device)
        cnt_all = 0
        sum_L = torch.zeros((nx, ny), device=device); cnt_L = 0
        sum_H = torch.zeros((nx, ny), device=device); cnt_H = 0
        sum_D = torch.zeros((nx, ny), device=device); cnt_D = 0

        model_S = model_S.eval()  # eval mode is fine; DO NOT disable autograd for residuals

        # --- iterate over recorded macro times ---
        for k in range(K):
            t_idx = int(record_idx[k])
            z_now = z_paths[:, t_idx]     # (N,)
            g_now = g_paths[:, k, :]      # (N, L)

            # chunk over paths to control memory
            for p0 in range(0, N, chunk_paths):
                p1 = min(N, p0 + chunk_paths)
                B  = p1 - p0

                z_chunk = z_now[p0:p1]                               # (B,)
                g_chunk_base = g_now[p0:p1, :]                       # (B, L)
                z_feat  = z_chunk.to(torch.float32).view(B, 1)       # (B,1)

                # Build X exactly like training: [x, y, z, g_flat] for every path & grid point
                XY = xy.unsqueeze(0).expand(B, L, 2).reshape(B*L, 2) # (B*L, 2)
                Z  = z_feat.repeat_interleave(L, dim=0)              # (B*L, 1)

                # IMPORTANT: g must be a leaf that requires grad (for dS/dg in S_pde_oper)
                g_chunk = g_chunk_base.clone().detach().requires_grad_(True)  # (B, L)
                G  = g_chunk.repeat_interleave(L, dim=0)             # (B*L, L)

                X  = torch.cat([XY, Z, G], dim=1)                    # (B*L, 2+1+L)

                # --- compute residuals with autograd ENABLED (needed for dS/dg) ---
                with torch.enable_grad():
                    res, _, _ = self.S_pde_oper(model_S, X)   # grab only the residual tensor
                res = res.view(-1)



                err = (res ** 2).view(B, nx, ny).detach()            # (B, nx, ny), detach to free graph

                # --- unconditional accumulation ---
                sum_all += err.sum(dim=0)
                cnt_all += B

                # --- by-state accumulation ---
                maskL = (z_chunk == 0); mL = int(maskL.sum().item())
                maskH = (z_chunk == 1); mH = int(maskH.sum().item())
                maskD = (z_chunk == 2); mD = int(maskD.sum().item())

                if mL > 0:
                    sum_L += err[maskL, :, :].sum(dim=0); cnt_L += mL
                if mH > 0:
                    sum_H += err[maskH, :, :].sum(dim=0); cnt_H += mH
                if mD > 0:
                    sum_D += err[maskD, :, :].sum(dim=0); cnt_D += mD

        # --- finalize (guard small counts) ---
        E_all = sum_all / max(cnt_all, 1)
        E_L   = sum_L  / max(cnt_L, 1) if cnt_L > 0 else torch.zeros_like(sum_L)
        E_H   = sum_H  / max(cnt_H, 1) if cnt_H > 0 else torch.zeros_like(sum_H)
        E_D   = sum_D  / max(cnt_D, 1) if cnt_D > 0 else torch.zeros_like(sum_D)

        return {
            "E_err":   E_all,           # (nx, ny) unconditional
            "E_err_L": E_L,             # (nx, ny) conditional on z=L
            "E_err_H": E_H,             # (nx, ny) conditional on z=H
            "E_err_D": E_D,             # (nx, ny) conditional on z=D
            "counts":  {"all": cnt_all, "L": cnt_L, "H": cnt_H, "D": cnt_D},
        }




    def build_ergodic_dataloaders(
        self,
        model_S,
        N_paths=128, T=5000, dt=0.01, substeps=5, record_interval=2, burn_in_frac=0.2,
        batch_size_train=512, batch_size_eval=512, train_fraction=0.8,
        g0=None, init_probs=None, seed=None,
        num_workers: int = 4, pin_memory: bool = True,
    ):
        """
        Simulate with current model_S, build TRAIN/EVAL pools from the ergodic process,
        and return DataLoaders that yield (S_batch, None, None), where:
          S_batch: (B, 1+1+1+L) = [x, y, z, g_flat]  (CPU; move to device in the trainer)
        """
        import math, torch
        from torch.utils.data import Dataset, DataLoader

        # ---------- sizes & device ----------
        nx, ny = int(self.nx), int(self.ny)
        L = nx * ny
        device = getattr(self, "device", next(model_S.parameters()).device)

        # ---------- g0 & init_probs ----------
        if g0 is None:
            g0 = getattr(self, "gm_ss", None)
            if g0 is None:
                g0 = self.gm_low
        g0 = torch.as_tensor(g0, dtype=torch.float32, device=device).flatten()
        if g0.numel() != L:
            raise ValueError(f"g0 size mismatch: {g0.numel()} vs L={L}")

        if init_probs is None:
            init_probs = torch.tensor([self.pi_L, self.pi_H, self.pi_D], dtype=torch.float32, device=device)
        else:
            init_probs = torch.as_tensor(init_probs, dtype=torch.float32, device=device)
        init_probs = init_probs / init_probs.sum().clamp_min(1e-12)

        # ---------- simulate z and economy ----------
        z_paths = self.generate_z_paths(N=N_paths, T=T, dt=dt, init_probs=init_probs, seed=seed, as_float=False)
        out = self.simulate_economy(
            model_S=model_S, z_paths=z_paths, dt=dt, substeps=substeps,
            g0=g0, record_interval=record_interval, burn_in_frac=burn_in_frac,
            clip_eps=1e-12, return_paths=True
        )
        g_paths = torch.as_tensor(out["g_paths"], device=device, dtype=torch.float32)  # (N_paths, K, L)
        record_idx = out["record_idx"]
        Np, K, L_chk = g_paths.shape
        if Np != N_paths or L_chk != L:
            raise ValueError("simulate_economy shapes inconsistent with N_paths or L.")

        # ---------- alignment (minimal, robust) ----------
        # Prefer z recorded by simulator (post-jump aligned with g_paths). Fallback to old behavior.
        if "z_recorded" in out:
            Z_align = torch.as_tensor(out["z_recorded"], device=device, dtype=torch.long)  # (N_paths, K)
        else:
            Z_align = torch.as_tensor(z_paths, device=device, dtype=torch.long)[:, record_idx]  # (N_paths, K)

        if Z_align.shape != (N_paths, K):
            raise ValueError(f"Z_align shape mismatch: got {tuple(Z_align.shape)}, expected {(N_paths, K)}")

        # ---------- flatten pools on CPU ----------
        G_pool = g_paths.reshape(N_paths * K, L).contiguous().to("cpu")  # (M, L)
        Z_pool = Z_align.reshape(N_paths * K, 1).contiguous().to("cpu")  # (M, 1) long
        M = G_pool.shape[0]

        # ---------- split TRAIN/EVAL ----------
        perm = torch.randperm(M)
        M_train = int(math.floor(train_fraction * M))
        idx_train, idx_eval = perm[:M_train], perm[M_train:]
        G_train, Z_train = G_pool.index_select(0, idx_train), Z_pool.index_select(0, idx_train)
        G_eval,  Z_eval  = G_pool.index_select(0, idx_eval),  Z_pool.index_select(0, idx_eval)

        # ---------- x/y supports & uniforms (CPU) ----------
        types_x = torch.as_tensor(self.types_x, dtype=torch.float32).view(-1)  # (nx,)
        types_y = torch.as_tensor(self.types_y, dtype=torch.float32).view(-1)  # (ny,)
        unif_x  = torch.as_tensor(self.unif_x,  dtype=torch.float32).view(-1)
        unif_y  = torch.as_tensor(self.unif_y,  dtype=torch.float32).view(-1)
        unif_x = (unif_x / unif_x.sum().clamp_min(1e-12)).contiguous()
        unif_y = (unif_y / unif_y.sum().clamp_min(1e-12)).contiguous()

        # ---------- minimal dataset: returns (g, z) on CPU ----------
        class _PoolDS(Dataset):
            def __init__(self, G, Z):
                self.G, self.Z = G, Z
            def __len__(self): return self.G.shape[0]
            def __getitem__(self, i):
                return self.G[i], self.Z[i, 0]  # (L,), scalar long

        train_ds = _PoolDS(G_train, Z_train)
        eval_ds  = _PoolDS(G_eval,  Z_eval)

        # ---------- vectorized collate: build S_batch only ----------
        def _collate_to_S(batch):
            # batch: list of tuples (g_row, z_scalar)
            B = len(batch)
            # Stack g and z
            g = torch.stack([b[0] for b in batch], dim=0)                    # (B, L) float32 CPU
            z = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)   # (B,) long CPU
            zf = z.to(torch.float32).unsqueeze(1)                             # (B,1)

            # Vectorized draws for x and y
            idx_x = torch.multinomial(unif_x, B, replacement=True)            # (B,)
            idx_y = torch.multinomial(unif_y, B, replacement=True)            # (B,)
            x = types_x.index_select(0, idx_x).unsqueeze(1)                   # (B,1)
            y = types_y.index_select(0, idx_y).unsqueeze(1)                   # (B,1)

            # Assemble S_batch on CPU; trainer will move to device
            S = torch.cat((x, y, zf, g), dim=1)                               # (B, 1+1+1+L)
            return S, None, None  # keep API compatible with current trainer

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_to_S,
            drop_last=False,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_to_S,
            drop_last=False,
        )

        meta = {
            "M_total": int(M), "M_train": int(G_train.shape[0]), "M_eval": int(G_eval.shape[0]),
            "steps_per_epoch_train": math.ceil(G_train.shape[0] / batch_size_train),
            "steps_per_epoch_eval":  math.ceil(G_eval.shape[0]  / batch_size_eval),
            "nx": nx, "ny": ny, "L": L,
            "dt": float(dt), "substeps": int(substeps),
            "record_interval": int(record_interval), "burn_in_frac": float(burn_in_frac),
        }
        return {"train_loader": train_loader, "eval_loader": eval_loader, "meta": meta}




    def train_with_ergodic_loaders(
        self,
        model_S,
        optimizer,
        train_loader,
        eval_loader,
        *,
        total_steps: int = 10_000,
        eval_every: int = 1000,          # sparser eval to save time
        print_every: int = 200,
        max_eval_batches: int = 8,       # cap eval cost
        loss_threshold: float | None = None,
        scheduler=None,
        train_losses=None,
        eval_losses=None,
        log_train_every: int = 50,      # <-- NEW: train-loss log cadence
    ):
        """
        Train `model_S` on ergodic DataLoaders. Batches must yield (S_batch, _, _),
        where S_batch has shape (B, 1+1+1+L) = [x, y, z, g_flat] on CPU.
        Uses self.S_pde_oper for residuals; keeps best-on-eval checkpoint.
        Returns compact logs for plotting.
        """
        import torch

        # Default to None so each call starts with fresh logs: Python evaluates a default
        # argument once, so a shared mutable default ([]) would accumulate across calls.
        if train_losses is None:
            train_losses = []
        if eval_losses is None:
            eval_losses = []

        device = getattr(self, "device", next(model_S.parameters()).device)
        model_S.train()

        best_eval = float("inf")
        best_state = None
        step = 0
        train_iter = iter(train_loader)

        # ---- lightweight logs (compact) ----
        train_steps = []
        eval_steps  = []

        while step < total_steps:
            # --- fetch next batch; cycle the loader if needed ---
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # loader returns (S_batch, None, None); accept also plain S_batch
            S_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
            S_batch = S_batch.to(device, non_blocking=True)

            # --- residuals & loss (S operator only) ---
            residuals, _, _ = self.S_pde_oper(model_S, S_batch)  # ∂S/∂g handled inside op
            loss = (residuals ** 2).mean()

            # --- optimize ---
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                try:
                    scheduler.step(loss)
                except TypeError:
                    scheduler.step()

            step += 1

            # --- log + print (cheap) ---
            if log_train_every and (step % log_train_every == 0):
                train_steps.append(step)
                train_losses.append(float(loss.item()))
            if print_every and (step % print_every == 0):
                print(f"[step {step}/{total_steps}] train_loss={loss.item():.6e}")

            # --- evaluation (capped & sparser) ---
            if eval_every and (step % eval_every == 0):
                model_S.eval()
                eval_sum, eval_cnt = 0.0, 0
                with torch.enable_grad():  # residual needs autograd; no backward()
                    for i, eval_batch in enumerate(eval_loader):
                        S_eval = eval_batch[0] if isinstance(eval_batch, (tuple, list)) else eval_batch
                        S_eval = S_eval.to(device, non_blocking=True)
                        res_e, _, _ = self.S_pde_oper(model_S, S_eval)
                        eval_sum += (res_e ** 2).mean().item()
                        eval_cnt += 1
                        if i + 1 >= max_eval_batches:
                            break
                mean_eval = eval_sum / max(1, eval_cnt)
                eval_steps.append(step)
                eval_losses.append(mean_eval)
                if mean_eval < best_eval:
                    best_eval = mean_eval
                    best_state = {k: v.detach().cpu().clone() for k, v in model_S.state_dict().items()}
                print(f"[step {step}] eval_loss={mean_eval:.6e}  (best={best_eval:.6e})")
                model_S.train()

            # --- optional early stop on train loss ---
            if (loss_threshold is not None) and (loss.item() < loss_threshold):
                print(f"[stop] train loss < {loss_threshold:.3e} at step {step}.")
                break

        return {
            "best_state_dict": best_state,
            "best_eval_loss": best_eval,
            "steps_run": step,
            "last_train_loss": float(loss.item()),
            "train_log": {"steps": train_steps, "loss": train_losses},
            "eval_log":  {"steps": eval_steps,  "loss": eval_losses},
        }







    ## --------------------------------------------------
    ## PDE-related functions
    ## --------------------------------------------------
    def mu_g(self, g_m, M_u, V, alphas, g_e, g_p, g_f, z):
        """KFE drift for g_m.
        Output shape: (N, n_x*n_y), same as g_m.
        """

        # Unpack
        n_x = self.nx
        n_y = self.ny
        g_w = self.gw
        delta_s = self.delta_s          # (3, n_x, n_y)
        varsigma_s = self.varsigma_s    # (3,)

        N = g_m.shape[0]

        # Flatten helper terms to (N, n_x*n_y)
        alphas_flattened = alphas.flatten(start_dim=1)
        g_p_expanded = g_p.unsqueeze(1).expand(-1, n_x, n_y).flatten(start_dim=1)
        g_e_expanded = g_e.unsqueeze(-1).expand(-1, -1, n_y).flatten(start_dim=1)
        g_f_expanded = g_f.unsqueeze(1).expand(-1, n_x, n_y).flatten(start_dim=1)
        g_w_expanded = g_w.reshape(1, n_x, 1).expand(N, n_x, n_y).flatten(start_dim=1)

        # State index: force 1-D (N,) to avoid accidental extra dimensions
        z_idx = z.view(-1).long()

        # Delta(z): (N, n_x, n_y) -> (N, n_x*n_y)
        delta_z_flattened = delta_s[z_idx, :, :].flatten(start_dim=1)

        # Varsigma(z): (N,) -> (N,1), broadcast over match grid
        varsigma_z = varsigma_s[z_idx].view(-1, 1)

        # Inflow term
        inflow = M_u * (1 / V) * alphas_flattened * (g_w_expanded - g_e_expanded) * (g_f_expanded - g_p_expanded)

        # Drift: inflow - (delta + varsigma) * g
        return -(delta_z_flattened + varsigma_z) * g_m + inflow



    def calculate_alphas(self, model_S, z, g_m):
        """ Calculate alphas between all combinations of firm-worker pairs, 
            given aggregate state (z) and joint density (g_m)
        """
        ## Unpack
        device  = self.device
        n_x     = self.nx 
        n_y     = self.ny
        xi      = self.xi
        # Define firm and worker types
        types_x = torch.linspace(1/n_x/2, 1-1/n_x/2, n_x)
        types_y = torch.linspace(1/n_y/2, 1-1/n_y/2, n_y)
        N = len(z)
        
        ## Preparing flattened inputs to calculate alphas
        types_x_expanded = types_x.unsqueeze(-1).unsqueeze(0).expand(
                            N, n_x, n_y).flatten().reshape(
                            (N*n_x*n_y,1)).to(device)
        types_y_expanded = types_y.unsqueeze(0).unsqueeze(1).expand(
                            N, n_x, n_y).flatten().reshape(
                            (N*n_x*n_y,1)).to(device)
        z_expanded = z.unsqueeze(1).expand(N, n_x*n_y, -1).reshape(
                            (N*n_x*n_y, 1)).to(device)                            
        g_m_expanded = g_m.unsqueeze(1).expand(N, n_x*n_y, -1).reshape(
                            (N*n_x*n_y, n_x*n_y)).to(device)

        ## Concatenate tensors to create input for model_W
        input_tensor_S = torch.hstack((types_x_expanded, types_y_expanded, 
                            z_expanded, g_m_expanded))

        ## Calculate alpha values using model_S 
        ## (for smooth/continuous definition of alpha)
        S = model_S(input_tensor_S)
        alphas = (1/(1+torch.exp(-xi*S))).reshape(N, n_x, n_y)
        S = S.reshape(N, n_x, n_y)

        return S, alphas

    ## calculate marginal densities from the joint density (g_m), 
    ## as well as aggregate unemployment and vacancies
    def marginals_U_V(self, g_m, S, alphas, z):
        ## Unpack
        n_x = self.nx
        n_y = self.ny
        g_w = self.gw
        N = len(g_m)
        device = self.device
        tau = self.tau

        # current aggregate state index
        z_idx = z.view(-1).long()

        # flow exit by current state
        varsigma_z = self.varsigma_s[z_idx].reshape(-1, 1)

        # jump-exit hazard: sum_{zhat != z} sigma(z, zhat) * lambda(z, zhat)
        sigma_mat = torch.as_tensor(self.sigma_mat, device=device, dtype=torch.float32)

        lambda_mat = torch.tensor(
            [
                [0.0,         self.lam_LH, self.lam_LD],
                [self.lam_HL, 0.0,         self.lam_HD],
                [self.lam_DL, self.lam_DH, 0.0        ],
            ],
            device=device,
            dtype=torch.float32,
        )

        sigma_row = sigma_mat[z_idx]          # (N, 3)
        lambda_row = lambda_mat[z_idx]        # (N, 3)
        jump_exit_z = torch.sum(sigma_row * lambda_row, dim=1, keepdim=True)  # (N, 1)

        # total effective exit intensity for free entry
        exit_intensity_z = varsigma_z + jump_exit_z

        # Reshape g_m from flattened vector input to (n_x, n_y) array
        g_m_reshaped_2d = g_m.reshape((N, n_x, n_y))

        # aggregate unemployment
        g_e = 1 / n_y * torch.sum(g_m_reshaped_2d, axis=-1)
        g_u = g_w - g_e
        U = 1 / n_x * torch.sum(g_u, axis=1, keepdim=True)

        # aggregate producing firms
        g_p = 1 / n_x * torch.sum(g_m_reshaped_2d, axis=1)
        P = torch.mean(g_m_reshaped_2d, dim=(1, 2)).reshape(-1, 1)

        ## calculate V_t term from free entry
        mean_term = torch.mean(
            alphas * S * g_u.repeat_interleave(n_y).reshape(N, n_x, n_y),
            dim=(1, 2)
        ).reshape(-1, 1)

        denominator = (1 / U) * (1 - self.beta) * mean_term

        V = (
            (self.kappa * U**self.nu)
            / (((self.rho + exit_intensity_z * (1 - tau)) * self.entry_cost) / denominator)
        ) ** (1 / self.nu)

        g_f = (V + P) * torch.ones(N, n_y, device=device)
        g_v = g_f - g_p
        V = 1 / n_y * torch.sum(g_v, axis=1, keepdim=True)

        return g_e, g_u, U, g_p, g_v, V, g_f

    # ## Calculate Surplus PDE residuals
    def S_pde_oper(self, model_S, X):
        ## Unpack
        device  = self.device 
        n_x     = self.nx 
        n_y     = self.ny
        rho     = self.rho 
        delta_s = self.delta_s     
        beta    = self.beta
        b       = self.b     
        z_0     = self.z_0
        sigma_mat  = self.sigma_mat
        varsigma_s = self.varsigma_s
        tau     = self.tau
        tau_w   = self.tau_w

        ## Extract variables and set up functions
        g_m = X[:, 3:(n_x * n_y + 3)].clone().to(device)
        x = X[:, 0].clone().to(device)
        y = X[:, 1].clone().to(device)
        z = X[:, 2].clone().to(device)

        N = len(x)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)

        z_idx = z.view(-1).to(torch.long)

        types_x = torch.linspace(1 / n_x / 2, 1 - 1 / n_x / 2, n_x, device=device, dtype=torch.float32)
        types_y = torch.linspace(1 / n_y / 2, 1 - 1 / n_y / 2, n_y, device=device, dtype=torch.float32)

        def model_g(g_m_input):
            X_temp = torch.clone(X)
            X_temp[:, 3:(n_x * n_y + 3)] = g_m_input
            return model_S(X_temp)

        def model_zg(z_in, g_m_input):
            X_temp = torch.clone(X)
            X_temp[:, 2:3] = z_in
            X_temp[:, 3:(n_x * n_y + 3)] = g_m_input
            return model_S(X_temp)

        ## ------------------------------
        ## Compute the differential equation
        ## ------------------------------

        ## Pass through NN to get current value function and derivatives wrt g
        S_batch = model_S(X).to(device)

        ## Leaf copy of g for dS/dg
        g_leaf  = g_m.detach().clone().requires_grad_(True)
        Sg_pred = model_g(g_leaf).to(device)
        dS_dg   = self.get_derivs_1order(Sg_pred, g_leaf).to(device)

        ## Calculate alphas for drift term (and free entry condition)
        S, alphas = self.calculate_alphas(model_S, z, g_m)

        ## Calculate marginal densities, aggregate unemployment and vacancies
        # g_e, g_u, U, g_p, g_v, V, g_f = self.marginals_U_V(g_m, S, alphas)
        g_e, g_u, U, g_p, g_v, V, g_f = self.marginals_U_V(g_m, S, alphas, z)

        ## Matching rates
        M_v = self.m(U, V) / V
        M_u = self.m(U, V) / U

        ## Take "idiosyncratic" worker or firm rows of alphas
        indices_x = (x / (types_x[1] - types_x[0]) - 0.5).long().flatten().to(device)
        indices_y = (y / (types_y[1] - types_y[0]) - 0.5).long().flatten().to(device)
        alphas_fixed_x = alphas[torch.arange(N), indices_x, :]
        alphas_fixed_y = alphas[torch.arange(N), :, indices_y]

        ## Old terms already in the PDE
        term1 = -(rho + delta_s[z_idx, indices_x, indices_y].view(N, 1)) * S_batch + z_0 * self.f_torch(x, y)

        term2 = -(1 - beta) * M_v * (1 / n_x) * (1 / U) * \
            torch.sum(alphas_fixed_y * g_u * S[range(N), :, indices_y], dim=1, keepdim=True)

        term3 = -b - beta * M_u * (1 / n_y) * (1 / V) * \
            torch.sum(alphas_fixed_x * g_v * S[range(N), indices_x, :], dim=1, keepdim=True)

        term4 = torch.sum(dS_dg * self.mu_g(g_m, M_u, V, alphas, g_e, g_p, g_f, z), dim=1, keepdim=True)

        ## term5: CTMC jump operator with exogenous jump exit sigma(z,z')
        term5 = torch.zeros_like(S_batch)

        ## term6: new compensation term from flow exit and exogenous jump exit
        chi_tau = (tau - 1) * (1 - beta) + (tau_w - 1) * beta
        term6 = varsigma_s[z_idx].view(N, 1) * chi_tau * S_batch

        for z_prime in [0, 1, 2]:
            lam = (
                self.lam_LH * (z == 0) * (z_prime == 1) +
                self.lam_LD * (z == 0) * (z_prime == 2) +
                self.lam_HL * (z == 1) * (z_prime == 0) +
                self.lam_HD * (z == 1) * (z_prime == 2) +
                self.lam_DL * (z == 2) * (z_prime == 0) +
                self.lam_DH * (z == 2) * (z_prime == 1)
            ).to(dtype=S_batch.dtype)

            if torch.any(lam != 0):
                sigma_z_zp = sigma_mat[z_idx, z_prime].view(N, 1)
                g_plus = (1.0 - sigma_z_zp) * g_m
                z_prime_batch = torch.full((N, 1), float(z_prime), device=device, dtype=X.dtype)

                S_batch_switched_z = model_zg(z_prime_batch, g_plus).to(device)

                ## old jump operator
                term5 += lam * (S_batch_switched_z - S_batch)

                ## new compensation contribution from exogenous jump exit
                # term6 += lam * sigma_z_zp * chi_tau * S_batch
                term6 += lam * sigma_z_zp * chi_tau * S_batch_switched_z

        ## Return
        return term1 + term2 + term3 + term4 + term5 + term6, alphas, S_batch





'''
    ## Calculate V_u PDE residuals
    def V_u_pde_oper(self, model_S, model_V_u, X, X_V):
        device  = self.device
        n_x     = self.nx 
        n_y     = self.ny   
        rho     = self.rho 
        # delta   = self.delta 
        beta    = self.beta
        b       = self.b     
        # z_s     = self.z_s     
        delta_s = self.delta_s     
        lams    = self.lams
        ## Extract variables and set up functions
        g_m = X[:, 3:(n_x * n_y + 3)].clone().to(device)         # Extract "u" training values
        x = X[:, 0].clone().to(device)                           # Extract worker types
        y = X[:, 1].clone().to(device)                           # Extract firm types
        z = X[:, 2].clone().to(device)                           # Extract productivity state
        g_m.requires_grad_(True)          # Inititiate auto. diff. tracking
        N = len(x)
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        z = z.reshape(-1,1)

        types_x = torch.linspace(1/n_x/2, 1-1/n_x/2, n_x)

        def model_g(g_m):
            X_temp = torch.clone(X_V)     # Clone data
            X_temp[:,2:(n_x * n_y + 2)] = g_m
            return model_V_u(X_temp[:, ])
        
        def model_z(z):
            X_temp = torch.clone(X_V)     # Clone data
            X_temp[:, 1:2] = z
            return model_V_u(X_temp)

        ## ------------------------------
        ## Compute the differential equation
        ## ------------------------------
 
        ## Pass through NN to get current value function and derivatives wrt u
        V_u_batch   = model_V_u(X_V).to(device)
        V_u_g_pred  = model_g(g_m).to(device)
        dV_u_dg     = self.get_derivs_1order(V_u_g_pred, g_m).to(device)
        g_m.requires_grad_(False)
        switched_z = ((z.int())^1).float()
        V_u_batch_switched_z = model_z(switched_z).to(device)

        ## Calculate marginal densities, aggregate unemployment and vacancies 
        g_e, g_u, U, g_p, g_v, V = self.marginals_U_V(g_m) 

        ## Matching rates
        m_over_U_V  = self.m(U, V)/(U*V)

        ## Calculate alphas for drift term
        S, alphas = self.calculate_alphas(model_S, z, g_m)
                
        ## Take "idiosyncratic" worker rows of alphas to be used in second term
        indices_x   = (x/(types_x[1] - types_x[0]) - 0.5).long().flatten().to(
                        device)
        alphas_fixed_x = alphas[torch.arange(N), indices_x, :]
        
        # calculate all terms, line by line, of the master equation in eq. (4.12) 
        term1 = -rho*V_u_batch + b
       
        term2 = beta * m_over_U_V * 1/n_y * \
            torch.sum(alphas_fixed_x * g_v * S[range(N), indices_x, :], dim = 1, keepdims=True)
        
        term3 = torch.sum(dV_u_dg * self.mu_g(g_m, m_over_U_V, alphas, g_e, g_p, z), axis = 1, keepdims = True)
        
        term4 = lams[z.int()]*(V_u_batch_switched_z - V_u_batch)

        ## Return
        return term1 + term2 + term3 + term4, alphas, V_u_batch


    ## Calculate V_v PDE residuals
    def V_v_pde_oper(self, model_S, model_V_v, X, X_V):
        ## Unpack
        device  = self.device
        n_x     = self.nx 
        n_y     = self.ny
        rho     = self.rho 
        # delta   = self.delta 
        beta    = self.beta
        b       = self.b     
        # z_s     = self.z_s     
        delta_s = self.delta_s     
        lams    = self.lams
        ## Extract varaibles and set up functions
        g_m = X[:, 3:(n_x * n_y + 3)].clone().to(device)         # Extract "u" training values
        x = X[:, 0].clone().to(device)                           # Extract worker types
        y = X[:, 1].clone().to(device)                           # Extract firm types
        z = X[:, 2].clone().to(device)                           # Extract productivity state
        g_m.requires_grad_(True)          # Inititiate auto. diff. tracking
        N = len(x)
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        z = z.reshape(-1,1)

        types_y = torch.linspace(1/n_y/2, 1-1/n_y/2, n_y)

        def model_g(g_m):
            X_temp = torch.clone(X_V)     # Clone data
            X_temp[:,2:(n_x * n_y + 2)] = g_m
            return model_V_v(X_temp[:, ])
        
        def model_z(z):
            X_temp = torch.clone(X_V)     # Clone data
            X_temp[:, 1:2] = z
            return model_V_v(X_temp)

        ## ------------------------------
        ## Compute the differential equation
        ## ------------------------------
 
        ## Pass through NN to get current value function and derivatives w.r.t u
        V_v_batch       = model_V_v(X_V).to(device)
        V_v_g_pred = model_g(g_m).to(device)
        dV_v_dg   = self.get_derivs_1order(V_v_g_pred, g_m).to(device)
        g_m.requires_grad_(False)
        switched_z = ((z.int())^1).float()
        V_v_batch_switched_z = model_z(switched_z).to(device)

        ## Calculate marginal densities, aggregate unemployment and vacancies 
        g_e, g_u, U, g_p, g_v, V = self.marginals_U_V(g_m) 

        ## Matching rates
        m_over_U_V = self.m(U, V)/(U*V)

        # Calculate alphas for drift term
        S, alphas = self.calculate_alphas(model_S, z, g_m)
                
        ## Take "idiosyncratic" firm rows of alphas to be used in second term
        indices_y = (y/(types_y[1] - types_y[0]) - 0.5).long().flatten().to(
                    device)
        alphas_fixed_y = alphas[torch.arange(N), :, indices_y]
        
        # calculate all terms, line by line, of master equation in eq. (4.12) 
        term1 = -rho*V_v_batch
       
        term2 = (1- beta) * m_over_U_V * 1/n_x * \
            torch.sum(alphas_fixed_y * g_u * S[range(N), :, indices_y], dim = 1, keepdims=True)
        
        term3 = torch.sum(dV_v_dg * self.mu_g(g_m, m_over_U_V, alphas, g_e, g_p, z), axis = 1, keepdims = True)
        
        term4 = lams[z.int()]*(V_v_batch_switched_z - V_v_batch)

        ## Return
        return term1 + term2 + term3 + term4, alphas, V_v_batch


    ## --------------------------------------------------
    ## Define function defintions for V_e, V_p, and wage
    ## --------------------------------------------------
    def V_e(self, X, model_S, model_V_u):
        n_x = self.nx 
        n_y = self.ny 
        x   = X[:, 0].reshape(-1,1)
        g_m = X[:, 2: n_x*n_y + 2]
        X_V = torch.hstack((x, g_m))
        return beta*model_S(X) + model_V_u(X_V)

    def V_p(self, X, model_S, model_V_v):
        n_x = self.nx 
        n_y = self.ny 
        y = X[:, 1].reshape(-1,1)
        z = X[:, 2].reshape(-1,1)
        g_m = X[:, 3: n_x*n_y + 3]
        X_V = torch.hstack((y, z, g_m))
        return((1-beta)*model_S(X) + model_V_v(X_V))

    def wage(self, X, model_S, model_V_u, model_V_v):
        device  = self.device
        n_x     = self.nx 
        n_y     = self.ny
        x = X[:, 0].reshape(-1,1)
        y = X[:, 1].reshape(-1,1)
        z = X[:, 2].reshape(-1,1)
        gm_S = X[:, 3: n_x*n_y + 3].clone().to(device)
        gm_Vu = X[:, 3: n_x*n_y + 3].clone().to(device)
        gm_Vv = X[:, 3: n_x*n_y + 3].clone().to(device)
        X_V_u = torch.hstack((x, z, gm_Vu))
        X_V_v = torch.hstack((y, z, gm_Vv))
        N = len(x)

        gm_S.requires_grad_(True)
        gm_Vu.requires_grad_(True)
        gm_Vv.requires_grad_(True)

        def model_S_g(gm_S):
            X_temp = torch.clone(X)     # Clone data
            X_temp[:, 3:(n_x * n_y + 3)] = gm_S
            return model_S(X_temp[:, ])

        def model_Vu_g(gm_Vu):
            X_temp = torch.clone(X_V_u)     # Clone data
            X_temp[:, 2:(n_x * n_y + 2)] = gm_Vu
            return model_V_u(X_temp[:, ])

        def model_Vv_g(gm_Vv):
            X_temp = torch.clone(X_V_v)     # Clone data
            X_temp[:, 2:(n_x * n_y + 2)] = gm_Vv
            return model_V_v(X_temp[:, ])

        S_g_pred = model_S_g(gm_S).to(device)
        dS_dg   = self.get_derivs_1order(S_g_pred, gm_S).to(device)

        Vu_g_pred = model_Vu_g(gm_Vu).to(device)
        dVu_dg   = self.get_derivs_1order(Vu_g_pred, gm_Vu).to(device)

        Vv_g_pred = model_Vv_g(gm_Vv).to(device)
        dVv_dg   = self.get_derivs_1order(Vv_g_pred, gm_Vv).to(device)

        gm_S.requires_grad_(False)
        gm_Vu.requires_grad_(False)
        gm_Vv.requires_grad_(False)
        
        dVe_dg = beta*dS_dg + dVu_dg
        dVp_dg = (1-beta)*dS_dg + dVv_dg

        ## calculate marginal densities, aggregate unemployment and vacancies 
        g_e, _, U, g_p, _, V = self.marginals_U_V(gm_S) 

        ## matching rates
        m_over_U_V = self.m(U, V)/(U*V)

        # calculate alphas for drift term
        _, alphas = self.calculate_alphas(model_S, z, gm_S)

        mu_g = self.mu_g(gm_S, m_over_U_V, alphas, g_e, g_p, z)

        term1 = rho*(1-beta)*model_V_u(X_V_u)

        # updated by Yaqi Zeng
        # term2 = beta*(self.f_torch(x, y) - rho*model_V_v(X_V_v) + torch.sum(dVp_dg * mu_g, axis = 1, keepdims = True))
        term2 = beta*(z*self.f_torch(x, y) - rho*model_V_v(X_V_v) + torch.sum(dVp_dg * mu_g, axis = 1, keepdims = True))

        term3 = (beta-1)*torch.sum(dVe_dg * mu_g, axis = 1, keepdims = True)

        return term1 + term2 + term3
'''
## --------------------------------------------------
## Training Sample Class
## --------------------------------------------------
# class Train_Sampler:
#     # Initialize the class
#     def __init__(self, n_x, n_y, device):
#         self.n_x        = n_x
#         self.n_y        = n_y
#         self.soboleng   = torch.quasirandom.SobolEngine(dimension=(n_x*n_y), 
#                             scramble = True)
#         # The distribution of types is discretized to perform a midpoint Riemann sum with n subdivisions
#         self.types_x    = torch.linspace(1/n_x/2, 1-1/n_x/2, n_x)
#         self.types_y    = torch.linspace(1/n_y/2, 1-1/n_y/2, n_y)
#         self.unif_x     = torch.ones(self.types_x.shape[0])
#         self.unif_y     = torch.ones(self.types_y.shape[0])
        
#         # for z in [ct.z_0 - ct.dz, ct.z_0, ct.z_0 + ct.dz]:
#         #     print("Starting DSS Fixed Point Algorithm: kappa = " + str(ct.kappa) + ", z = " + str(z))
#         #     self.create_gm(z)

#         self.gm_high_low_diff = torch.tensor(np.load("gm_high_z.npy") - np.load("gm_low_z.npy"), dtype=torch.float32)
#         self.gm_low = torch.tensor(np.load("gm_low_z.npy"), dtype=torch.float32)

#     def create_gm(self, z):
#         ## Solve the fixed pt steady state
#         xi = self.xi
#         delta = self.delta
#         rho = self.rho
#         kappa = self.kappa
#         beta = self.beta
#         n_x = self.nx
#         n_y = self.ny
#         nu = self.nu
#         b = self.b
        
#         tol = 1e-14     # Absolute tolerance level for the fixed-point iteration
        
#         def production_function(x, y):
#           ## PAM
#           # return 0.6 + 0.4*(np.sqrt(x) + np.sqrt(y))**2
#               return z*np.exp(x*y)
#           ## NAM
#           #return np.sqrt(x**2 + 2*y**2)
         
#         types_x = np.linspace(1/n_x/2, 1-1/n_x/2, n_x)
#         types_y = np.linspace(1/n_y/2, 1-1/n_y/2, n_y)
#         g_w = np.ones(n_x)
#         g_f = np.ones(n_y)
        
#         # Initial values
#         alphas = np.random.rand(n_x, n_y)
#         wages = np.random.rand(n_x, n_y)
#         gm_density = 10**(-5)*np.random.rand(n_x, n_y)
        
#         ## calculate marginal densities from the joint density (g_m), as well as aggregate
#         ## unemployment and vacancies
#         def marginals_U_V(g_m):
        
#             # aggregate unemployment
#             g_e = 1/n_y * np.sum(g_m, axis = 1)
#             g_u = g_w - g_e
#             U = 1/n_x * np.sum(g_u, axis = 0)
        
#             # aggregate vacant firms
#             g_p = 1/n_x * np.sum(g_m, axis = 0)
#             g_v = g_f - g_p
#             V = 1/n_y * np.sum(g_v, axis = 0)
        
#             return g_e, g_u, U, g_p, g_v, V
        
#         ## Cobb-Douglas meeting function
#         def m(U, V):
#             return kappa*(U**nu)*(V**(1 - nu))
        
#         # Computing the payoffs for all the types combinations
#         payoffs = np.empty([n_x, n_y])
#         for i in range(n_x):
#             x = types_x[i]
#             for j in range(n_y):
#                 y = types_y[j]
#                 payoffs[i, j] = production_function(x, y)
        
#         list_all_alphas = [alphas.tolist()]
#         keep_iterating = True
#         iter_num = 0
        
#         # Main loop
#         while keep_iterating:
#         #for iter in range(5):
#             # print("Iteration " + str(iter_num))
#             # print("wages: " + str(wages))
#             # print("alphas: " + str(alphas))
        
#             # Updating the matched density g_m.
#             # """ Fixed point algorithm """
#             e = sys.float_info.max
        
#             gm_prev_init = gm_density.copy()
#             gm_prev = gm_density
#             #print(gm_prev_init)
        
#             while e > tol:
#                 g_e, g_u, U, g_p, g_v, V = marginals_U_V(gm_prev)
#                 for i in range(n_x):
#                       for j in range(n_y):
#                         gm_density[i,j] = 1/delta * m(U, V)/ (U*V) * alphas[i,j] * g_u[i] * g_v[j] # fixed point iteration
#                 e = np.linalg.norm(gm_prev - gm_density)
#                 gm_prev = gm_density
        
#             gm_density = 0.98*gm_prev_init + 0.02*gm_density
#             if np.isnan(gm_density).any(): break
#             g_e, g_u, U, g_p, g_v, V = marginals_U_V(gm_density)
#             # print("gm_density:" +str(gm_density))
        
#             # Updating the V_u equation
#             psy = rho*U*V*(rho + delta)/m(U,V)
#             V_u = ((b * n_y * psy/rho) + np.dot(alphas * wages, g_v)) / (n_y * psy + rho*(np.dot(alphas, g_v)))
#             #print("V_u: " + str(V_u))
        
#             V_e = (wages + delta*V_u.reshape((n_x,1)))/(rho + delta)
#             #print("V_e: " + str(V_e))
        
#             # Updating the V_v equation
#             V_v = np.dot((alphas * (payoffs - wages)).T, g_u) / (n_x * psy + rho*(np.dot(alphas.T, g_u)))
#             V_p = np.zeros((n_x, n_y))
#             for i in range(n_x):
#                   for j in range(n_y):
#                     V_p[i,j] = (payoffs[i, j] - wages[i, j] + delta*V_v[j])/(rho + delta)
#             #print("V_v: " + str(V_v))
#             #print("V_p: " + str(V_p))
        
#             # Calculating the surplus
#             surplus = (payoffs - rho *(V_e + V_p))/delta
#             #print("surplus: " + str(surplus))
        
#             # Updating the matching set. equation
#             new_alphas = np.zeros([n_x, n_y])
#             for i in range(n_x):
#                 for j in range(n_y):
#                     ## Discrete choice
#                     #if (surplus[i, j] >= 0):  new_alphas[i,j] = 1
#                     #else: new_alphas[i,j] = 0
        
#                     ## Continuous choice
#                     new_alphas[i,j] = 1/(1 + np.exp(-xi * surplus[i, j]))
        
#             # updating the wage
#             new_wages = np.zeros([n_x, n_y])
#             for i in range(n_x):
#                 for j in range(n_y):
#                     new_wages[i,j] = rho*(1-beta)*V_u[i] + beta*(payoffs[i,j] - rho*V_v[j])
        
#             # Printing the number of changes in the matrix alphas after update
#             distance = LA.norm(alphas - new_alphas, 'fro')
#             # print(distance)
        
#             # Printing the number of changes in the matrix wages after update
#             distance_wages = LA.norm(wages - new_wages, 'fro')
        
#             # Checking if convergence
#             if (distance < 10**(-13)) and (distance_wages < 10**(-13)):
#                 is_convergence = True
#                 keep_iterating = False
#                 print("Converged.")
#             else:
#                 alphas = new_alphas
#                 wages = new_wages
#             iter_num += 1
        
#         if z == ct.z_0:
#             np.save("gm_ss", gm_density)
#             np.save("alpha_ss", alphas)
#             np.save("surplus_ss", surplus)
        
#         elif z == ct.z_0 + ct.dz:
#             np.save("gm_high_z", gm_density)
        
#         else:
#             np.save("gm_low_z", gm_density)

#     def sample(self, N):
#         #g_m = self.gm_ss + (2*torch.rand((N, self.n_x, self.n_y))-1) * self.gm_high_low_diff/2
#         g_m = self.gm_low + torch.rand((N, self.n_x, self.n_y)) * self.gm_high_low_diff
#         #g_m = torch.clamp(g_m, min=0.001)
#         g_m = g_m.flatten(start_dim=1)
#         #print(g_m)

#         idx_x = self.unif_x.multinomial(N, replacement = True)
#         x = self.types_x[idx_x].reshape((N, 1))

#         idx_y = self.unif_y.multinomial(N, replacement = True)
#         y = self.types_y[idx_y].reshape((N, 1))
#         z = torch.randint(2, (N, 1))

#         S_batch = torch.hstack((x, y, z, g_m))

#         V_u_batch = torch.hstack((x, z, g_m))

#         V_v_batch = torch.hstack((y, z, g_m))

#         return S_batch, V_u_batch, V_v_batch

## --------------------------------------------------
## Neural Network Classes for S and V
## --------------------------------------------------
class Master_PINN_S(torch.nn.Module):
    def __init__(self,
            nn_width        = 50,
            nn_num_layers   = 4,
            n_x             = 2,
            n_y             = 2
            ):
        super(Master_PINN_S, self).__init__()
        # Construct an array of affine and activation functions.
        # Hidden layers:
        #   These are the hidden layers 1,2,..., nn_num_layers
        #   layers = [affine1, activation1, affine2, activation2, ...]
        layers = [torch.nn.Linear(n_x*n_y + 3, nn_width),torch.nn.Tanh()]
        for i in range(1,nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())

        layers.append(torch.nn.Linear(nn_width, 1))
        # Sequentially execute the affine and activations function;
        # This constructs the neural network approximation.
        self.net = torch.nn.Sequential(*layers)
        for i in range (0,nn_num_layers):
            # Apply the xavier normalization at the affine layers.
            torch.nn.init.xavier_normal_(self.net[2*i].weight)

    def forward(self, X):
        return self.net(X)

class Master_PINN_V_u(torch.nn.Module):
    def __init__(self,
            nn_width        = 50,
            nn_num_layers   = 4,
            n_x             = 2,
            n_y             = 2
            ):
        super(Master_PINN_V_u, self).__init__()
        # Construct an array of affine and activation functions.
        # Hidden layers:
        #   These are the hidden layers 1,2,..., nn_num_layers
        #   layers = [affine1, activation1, affine2, activation2, ...]
        layers = [torch.nn.Linear(n_x*n_y + 2, nn_width),torch.nn.Tanh()]
        for i in range(1,nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())

        layers.append(torch.nn.Linear(nn_width, 1))
        # Sequentially execute the affine and activations function;
        # This constructs the neural network approximation.
        self.net = torch.nn.Sequential(*layers)
        for i in range (0,nn_num_layers):
            # Apply the xavier normalization at the affine layers.
            torch.nn.init.xavier_normal_(self.net[2*i].weight)

    def forward(self, X):
        return self.net(X)

class Master_PINN_V_v(torch.nn.Module):
    def __init__(self,
            nn_width        = 50,
            nn_num_layers   = 4,
            n_x             = 2,
            n_y             = 2
            ):
        super(Master_PINN_V_v, self).__init__()
        # Construct an array of affine and activation functions.
        # Hidden layers:
        #   These are the hidden layers 1,2,..., nn_num_layers
        #   layers = [affine1, activation1, affine2, activation2, ...]
        layers = [torch.nn.Linear(n_x*n_y + 2, nn_width),torch.nn.Tanh()]
        for i in range(1,nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())

        layers.append(torch.nn.Linear(nn_width, 1))
        # Sequentially execute the affine and activations function;
        # This constructs the neural network approximation.
        self.net = torch.nn.Sequential(*layers)
        for i in range (0,nn_num_layers):
            # Apply the xavier normalization at the affine layers.
            torch.nn.init.xavier_normal_(self.net[2*i].weight)

    def forward(self, X):
        return self.net(X)
