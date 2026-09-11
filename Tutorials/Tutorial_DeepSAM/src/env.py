"""
filename: env.py
@authors: Jonathan Payne, Adam Rebei, Yucheng Yang

This file contains the economic environment parent class and functions
for solving the Hagedorn et al. (2017) model.

"""

import os
import numpy as np
import torch

class Environment():
    """
    Parent environment class.

    Contains functions:
        calc_f: calculates the production function.
        calc_m: calculates the matching function.

    Subclasses of the parent class:
        -> Train_FD: for training using finite difference,
        -> Train_NN: for training using neural networks.
    """
    def __init__(self, **kwargs):
        # Assign environment parameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        ## Initialize the device
        self.device = torch.device(
                        "cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        ## Set up exogenous density
        self.gw         = torch.ones(self.nx).to(self.device)
        self.gw_np      = np.ones(self.nx)
        #self.gf         = torch.ones(self.ny).to(self.device)
        # self.lams = torch.tensor([self.lam_L, self.lam_H]).to(self.device)
        # self.z_L = self.z_0 - self.dz
        # self.z_H = self.z_0 + self.dz
        # self.z_s = torch.tensor([self.z_L, self.z_H]).to(self.device)

        self.prob_agg = torch.tensor([self.pi_L, self.pi_H, self.pi_D])

        ## Set up the type discretization
        self.xmin       = 1/self.nx/2
        self.xmax       = 1-1/self.nx/2
        self.ymin       = 1/self.ny/2
        self.ymax       = 1-1/self.ny/2
        self.x_grid     = np.linspace(self.xmin, self.xmax, self.nx)
        self.dx         = (self.xmax - self.xmin) / (self.nx - 1)
        self.y_grid     = np.linspace(self.ymin, self.ymax, self.ny)
        self.dy         = (self.ymax - self.ymin) / (self.ny - 1)

        # multiplier = np.zeros((self.nx,self.ny))
        # # Parameters calibrated
        # a_x = 1.7   # Controls the scaling along the x-axis
        # b_x = 5.     # Degree of the polynomial along the x-axis
        # c_x = 12    # Additional scaling factor for the x-axis

        # a_y = 1.6   # Controls the scaling along the y-axis
        # b_y = 5     # Degree of the polynomial along the y-axis
        # c_y = 10    # Additional scaling factor for the y-axis

        # comp_a = 1.1  # Controls the complementarity interaction between x and y
        # comp_b = 5.5  # Degree of the complementarity term

        # base_value = 20  # Base multiplier value

        # for i in range(self.nx):
        #     for j in range(self.ny):
        #         # Heterogeneity along x and y dimensions
        #         x_term = c_x * (a_x * (1 - self.x_grid[i])) ** b_x
        #         y_term = c_y * (a_y * (1 - self.y_grid[j])) ** b_y
                
        #         # Complementarity term to control interaction between x and y
        #         comp_term = (comp_a * (1 - self.x_grid[i] + 1 - self.y_grid[j])) ** comp_b
                
        #         # Total multiplier value
        #         multiplier[i, j] = base_value + x_term + y_term + comp_term



        # L_scale = 1.0
        # H_scale = 1.0
        # Dis_scale = 0.6



        # # Define delta_L, delta_H, and delta_Dis
        # self.delta_L = L_scale * (self.delta_0 - self.d_delta + 0 * multiplier)  # shape (nx, ny)
        # self.delta_H = H_scale * (self.delta_0 + self.d_delta + 0 * multiplier)  # shape (nx, ny)
        # self.delta_Dis = Dis_scale * (self.delta_0 + self.d_delta * multiplier)    # shape (nx, ny)

        # # Convert delta_L, delta_H, delta_Dis to torch tensors and stack them
        # self.delta_L = torch.tensor(self.delta_L, dtype=torch.float32).to(self.device)
        # self.delta_H = torch.tensor(self.delta_H, dtype=torch.float32).to(self.device)
        # self.delta_Dis = torch.tensor(self.delta_Dis, dtype=torch.float32).to(self.device)

        # # Stack them along a new dimension to create self.delta_s
        # self.delta_s = torch.stack([self.delta_L, self.delta_H, self.delta_Dis], dim=0)  # shape (3, nx, ny)
        
        multiplier = np.zeros((self.nx,self.ny))
        delta_0_D = np.zeros((self.nx,self.ny))


        # Parameters calibrated
        a_x = 1.7   # Controls the scaling along the x-axis
        b_x = 5.     # Degree of the polynomial along the x-axis
        c_x = 12    # Additional scaling factor for the x-axis

        a_y = 1.6   # Controls the scaling along the y-axis
        b_y = 5     # Degree of the polynomial along the y-axis
        c_y = 10    # Additional scaling factor for the y-axis

        comp_a = 1.1  # Controls the complementarity interaction between x and y
        comp_b = 5.5  # Degree of the complementarity term

        base_value = 20  # Base multiplier value

        for i in range(self.nx):
            for j in range(self.ny):
                # Heterogeneity along x and y dimensions
                x_term = c_x * (a_x * (1 - self.x_grid[i])) ** b_x
                y_term = c_y * (a_y * (1 - self.y_grid[j])) ** b_y

                # Complementarity term to control interaction between x and y
                comp_term = (comp_a * (1 - self.x_grid[i] + 1 - self.y_grid[j])) ** comp_b

                # Total multiplier value
                multiplier[i, j] = base_value + x_term + y_term + comp_term
                delta_0_D[i, j] = self.delta_0

                # manually adjusting x:
                if i==0:
                    multiplier[i, j] = multiplier[i, j] * 1.35
                elif i==1:
                    multiplier[i, j] = multiplier[i, j] * 1.35
                elif i==2:
                    multiplier[i, j] = multiplier[i, j] * 1.1
                elif i==3:
                    multiplier[i, j] = multiplier[i, j] * 1.1
                elif i==4:
                    multiplier[i, j] = multiplier[i, j] * 0.7
                    delta_0_D[i, j] = self.delta_0 * 0.7

                # manually adjusting y:
                if j==0:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==1:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==2:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==3:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==4:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==5:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==6:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==7:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==8:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==9:
                    multiplier[i, j] = multiplier[i, j] * 1.0
                elif j==10:
                    multiplier[i, j] = multiplier[i, j] * 1.0


        L_scale = 1.0
        H_scale = 1.0
        Dis_scale = 0.55

        # Define delta_L, delta_H, and delta_Dis
        self.delta_L = L_scale * (self.delta_0 - self.d_delta + 0 * multiplier)  # shape (nx, ny)
        self.delta_H = H_scale * (self.delta_0 + self.d_delta + 0 * multiplier)  # shape (nx, ny)
        self.delta_Dis = Dis_scale * (delta_0_D + self.d_delta * multiplier)    # shape (nx, ny)

        # Convert delta_L, delta_H, delta_Dis to torch tensors and stack them
        self.delta_L = torch.tensor(self.delta_L, dtype=torch.float32).to(self.device)
        self.delta_H = torch.tensor(self.delta_H, dtype=torch.float32).to(self.device)
        self.delta_Dis = torch.tensor(self.delta_Dis, dtype=torch.float32).to(self.device)

        # Stack them along a new dimension to create self.delta_s
        self.delta_s = torch.stack([self.delta_L, self.delta_H, self.delta_Dis], dim=0)  # shape (3, nx, ny)


        # Exogenous flow exit by state (L, H, D). Note that L is for good, H is for bad, and D is for disaster.
        self.varsigma_s = torch.tensor(
            [0.01, 0.01, 0.63],   # [varsigma_L (good), varsigma_H (bad), varsigma_D (disaster)]
            dtype=torch.float32,
            device=self.device
        )

        # compensation parameters
        self.tau = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.tau_w = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        # Exogenous jump exit matrix sigma(z,z')
        # State order: [L, H, D] -> [0, 1, 2]
        # Diagonal must stay zero (no jump if z'=z)
        self.sigma_mat = torch.tensor(
            [
                [0.000, 0.0025, 0.05],  # from L to [L,H,D]
                [-0.00, 0.0000, 0.05],  # from H to [L,H,D]
                [-0.00, -0.000, 0.00],  # from D to [L,H,D]
            ],
            dtype=torch.float32,
            device=self.device
        )



        ## Set up path to store results for these economic parameters
        # self.path = (f"agg_shock_nx_{self.nx}_ny_{self.ny}_b_{self.b}_rho_{self.rho}_"
        #              f"delta_{self.delta}_xi_{self.xi}_beta_{self.beta}_lambda_{self.lam_L}_"
        #              f"nu_{self.nu}_kappa_{self.kappa}_dz_{self.dz}_{self.prod_type}")
        self.path = (f"agg_shock_{self.shock_type}_nx_{self.nx}_ny_{self.ny}_b_{self.b}_rho_{self.rho}_"
                     f"delta_{self.delta_0}_xi_{self.xi}_beta_{self.beta}_lambda_HL_{self.lam_HL}_"
                     f"nu_{self.nu}_kappa_{self.kappa}_ddelta_{self.d_delta}_{self.prod_type}")

        os.makedirs(self.path, exist_ok=True)

    def calc_f(self,x,y):
        """Production function (numpy version)
        """
        if self.prod_type == 'PAM': 
            return 0.6 + 0.4*(np.sqrt(x) + np.sqrt(y))**2
        elif self.prod_type == 'EXP': 
            return np.exp(x*y)
        elif self.prod_type == 'NAM': 
            return np.sqrt(np.square(x) + 2*np.square(y))
        elif self.prod_type == 'NEITHER': 
            if (x<= 0.4): return 0.4 + y*(x+0.6)
            else: return 0.4+ np.sqrt(np.square(x - 0.4) + np.square(y))
        else: 
            raise ValueError("prod_type must be 'PAM', 'EXP', 'NAM', 'NEITHER'.")

    def f_torch(self, x, y):
        """ production function (torch)
        """
        if self.prod_type == 'PAM': 
            # return interpolator.interpolate(x, y)
            return 0.6 + 0.4*(torch.sqrt(x) + torch.sqrt(y))**2
        elif self.prod_type == 'EXP': 
            return torch.exp(x*y)
        elif self.prod_type == 'NAM': 
            return torch.sqrt(torch.square(x) + 2*torch.square(y))
        elif self.prod_type == 'NEITHER': 
            if (x<= 0.4): return 0.4 + y*(x+0.6)
            else: return 0.4+ torch.sqrt(torch.square(x - 0.4) + torch.square(y))
        else: raise ValueError("Invalid prod_type. Choose either 'PAM', 'EXP', 'NAM', or 'NEITHER'.")

    def m(self,U,V):
        """Cobb-Douglas meeting function (numpy version)
        """
        kappa   = self.kappa
        nu      = self.nu
        return kappa*(U**nu)*(V**(1 - nu)) 