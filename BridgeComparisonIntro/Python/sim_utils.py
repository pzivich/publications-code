import numpy as np
import pandas as pd
from scipy.stats import logistic


def generate_super_population(n):
    data = pd.DataFrame()
    data['idu'] = np.random.binomial(n=1, p=0.3, size=n)
    # data['cd4'] = -10*data['idu'] + np.random.lognormal(5.2, 0.45, size=n)
    data['cd4'] = -7*data['idu'] + np.random.normal(250, 45, size=n)
    data['cd4'] = np.where(data['cd4'] < 0, 0, data['cd4'])
    data['cd4'] = np.where(data['cd4'] > 1600, 1600, data['cd4'])
    data['cd4'] = data['cd4'] - 250

    pr_s = logistic.cdf(0.5 - 1.5*data['idu'] + 0.02*data['cd4'])  # Probability of participating in study
    data['pr_s'] = pr_s
    data['S'] = np.random.binomial(n=1, p=pr_s, size=n)

    return data


def generate_dataset(data_s1, data_s0, n_1, n_0, admin_censor=365):
    # Generating samples (one random and one stratified random)
    s1 = generate_sample(data=data_s1, n=n_1)
    s1['study'] = 1
    s1['art'] = np.random.binomial(n=1, p=0.5, size=s1.shape[0]) + 1
    s0 = generate_sample(data=data_s0, n=n_0)
    s0['study'] = 0
    s0['art'] = np.random.binomial(n=1, p=0.5, size=s0.shape[0])

    # Stacking into a single data set
    sample = pd.concat([s1, s0], axis=0, ignore_index=True).reset_index(drop=True)
    sample['id'] = sample.index

    # Generating event times
    t_event = generate_event_times(a=sample['art'], idu=sample['idu'], cd4=sample['cd4'])
    t_censor = generate_censor_times(n=sample.shape[0])
    t_star = np.min([t_event, t_censor], axis=0)
    delta = np.where(t_event == t_star, 1, 0)
    censor = np.where(t_censor == t_star, 1, 0)

    # Administrative censoring
    delta = np.where(t_star >= admin_censor, 0, delta)
    censor = np.where(t_star >= admin_censor, 0, censor)
    t_star = np.where(t_star >= admin_censor, admin_censor, t_star)

    # Outputting the data set
    sample['t'] = t_star
    sample['delta'] = delta
    sample['censor'] = censor
    return sample


def generate_sample(data, n):
    return data.sample(n=n, replace=True).reset_index(drop=True)


def generate_event_times(a, idu, cd4):
    # lambda_event = np.exp(6.5 + (a == 1)*0.4 + (a == 2)*1.5
    #                      - 1.9*idu + 0.01*cd4 - 0.01*100  # (re-centers CD4 coefs)
    lambda_event = np.exp(4.9 + (a == 1)*0.4 + (a == 2)*1.5
                          - 3*idu + 0.01*cd4
                          # - 0.15*(a == 1)*np.log(cd4) - 0.1*(a == 2)*np.log(cd4)
                          - 0.2*(a == 1)*idu - 0.25*(a == 2)*idu
                          )
    rho_event = 0.80
    return np.ceil(lambda_event**(1/rho_event)
                   * np.random.weibull(a=rho_event, size=lambda_event.shape[0]))


def generate_censor_times(n):
    lambda_c = 7.2
    rho_c = 3
    return np.ceil(lambda_c**rho_c * np.random.weibull(a=rho_c, size=n))


def generate_truth(data, t_end=365):
    # Generating samples (one random and one stratified random
    s = data.copy()
    n = s.shape[0]

    # Generating A=0 incidence at day=t_end
    s['art'] = 0
    t_event = generate_event_times(a=s['art'], idu=s['idu'], cd4=s['cd4'])
    r0 = np.sum(np.where(t_event <= t_end, 1, 0)) / n

    # Generating A=2 incidence at day=365
    s['art'] = 2
    t_event = generate_event_times(a=s['art'], idu=s['idu'], cd4=s['cd4'])
    r2 = np.sum(np.where(t_event <= t_end, 1, 0)) / n

    # Returning true risk difference
    return r2 - r0, r2 / r0


def create_empty_csv(n1, n0):
    days = [91, 183, 274, 365]
    cols = []
    for j in ["w", "c"]:
        for i in days:
            for var in ["bias_rd", "se_rd", "cld_rd", "cover_rd"]:
                cols = cols + [var + "_" + j + "_" + str(i)]
        cols = cols + ["pval_" + j]

    d = pd.DataFrame(columns=cols)
    d.to_csv("sim_n1-" + str(n1) + "_n0-" + str(n0) + ".csv", index=False)
