import argparse
import os
import random

import numpy as np
import torch
from scipy import stats

from data.datafuncs import GetData, GenerateAllDataSets
from models import GaussianVAE, StudentsTVAE
from models.MGARCH import robust_garch_torch

def backtest(violations, q):
    ratio        = sum(violations) / len(violations)
    binom_pvalue = stats.binom_test(sum(violations), len(violations), p=q)
    
    a00s = 0
    a01s = 0
    a10s = 0
    a11s = 0
    for i in range(violations.shape[0]):
        # independence
            if violations[i-1] == 0:
                if violations[i] == 0:
                    a00s += 1
                else:
                    a01s += 1
            else:
                if violations[i] == 0:
                    a10s += 1
                else:
                    a11s += 1
                    
    if a11s > 0:            
        qstar0 = a00s / (a00s + a01s)
        qstar1 = a10s / (a10s + a11s)
        qstar = (a00s + a10s) / (a00s+a01s+a10s+a11s)
        Lambda = (qstar/qstar0)**(a00s) * ((1-qstar)/(1-qstar0))**(a01s) * (qstar/qstar1)**(a10s) * ((1-qstar)/(1-qstar1))**(a11s)
        
        chris_pvalue = stats.chi2.ppf(-2*np.log(Lambda), df=1)
        
    else:
        chris_pvalue = 0
    
    return [ratio, binom_pvalue, chris_pvalue]



def RE_analysis():
    raise NotImplementedError


def GARCH_analysis(mode, dist):
    """
    Prints output related to GARCH analysis

    Parameters
    ----------
    mode : 'PCA' or 'VAE'
    dist : 'normal' or 't'

    Returns
    -------
    None.

    """
    ##
    mode = 'VAE'
    dist = 'normal'
    ##
    
    
    q = 0.05
    
    X, weights = GetData('returns')
    # weights = np.full((weights.shape[0], weights.shape[1]), 1/weights.shape[1])
    
    if mode == 'VAE':
        layers  = 3
        epochs  = 500
        latents = [3] #[2,3,4,5,6,12,20,25]
        
        results = np.zeros((3, len(latents)))
        
        for latent_col in range(len(latents)):
            n_latent = latents[latent_col]
            # fit encoder, get z
            if dist == 'normal':
                model = GaussianVAE(X.shape[1], n_latent, round((X.shape[1]+n_latent)/2), layers=layers)
            elif dist == 't':
                model = StudentsTVAE(X.shape[1], n_latent, round((X.shape[1]+n_latent)/2), layers=layers)
                
            model.fit(X, k=1, batch_size=100,
                      learning_rate=0.01, n_epoch=epochs,
                      warm_up=False, is_stoppable=False,
                      X_valid=X)
            
            if dist == 'normal':
                z, scale = model.encode(X)
            elif dist == 't':
                _, z, scale = model.encode(X)

            
            # fit garch
            garch = robust_garch_torch(torch.Tensor(z), dist)
            garch.fit(epochs=50)
            garch.store_sigmas()
            
            VaRs = np.zeros(X.shape)
            
            
            counter = 0
            for i in range(X.shape[0]):
                # try:
                #     torch.linalg.cholesky(garch.sigmas[i])
                # except:
                #     counter += 1
                l = torch.linalg.cholesky(garch.sigmas[i])
                sim = torch.randn((1000, z.shape[1])) # * torch.Tensor(scale[i,:])
                sim = (sim @ l).detach().numpy()
                if dist == 'normal':
                    decoded, _ = model.decode(sim)
                elif dist == 't':
                    _, decoded, _ = model.decode(sim)
                VaRs[i,:] = np.quantile(decoded, q, axis=0)
                del sim
                
            del model
            print(f'made it here, l = {n_latent}')
            portVaR = np.mean(VaRs, axis=1)
            violations = (portVaR > np.mean(X, axis=1)).astype(int)
            # backtest
            results[:,latent_col] = backtest(violations, q)
        
        print(results)
        return results
            
    
    # fit PCA
    
    # fit GARCH
    
    # sim
    
    # matmul to OG space
    
    # take VaR
    
    # backtest
    
    
    
    
    pass

def IV_analysis():
    raise NotImplementedError

#%%

test = GARCH_analysis('VAE', 'normal')

#%%

def main():
    pass
    # GARCH_analysis('VAE', 'normal')
    
    
    # raise NotImplementedError
    # RE analysis
    
    
    
    # GARCH analysis
    
    
    # Implied vola analysis

if __name__ == '__main__':
    main()