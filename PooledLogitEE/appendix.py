#####################################################################################################################
# Pooled Logistic as Estimating Equations
#   By-hand example in the Appendix
#
# Paul Zivich (2025/04/05)
#####################################################################################################################

import numpy as np
from delicatessen.utilities import inverse_logit
from efuncs import ee_pooled_logit

################################################
# Setup

t = np.asarray([1, 2, 2, 4, 4, 5])
d = np.asarray([1, 1, 0, 1, 0, 0])
X = np.asarray([[-1, 1, -1, 0, 2, -2], ]).T
beta_x = [0.2, ]

################################################
# Pooled Logit -- Continuous

beta_s = [-1, 0.1]
time_steps = np.asarray(range(1, 6))
n_time_steps = len(time_steps)

# \mathcal{X}
log_odds_x = np.dot(X, beta_x)
log_odds_x_matrix = np.tile(log_odds_x, (n_time_steps, 1))  #

# \mathcal{S}
intercept = np.ones(time_steps.shape)[:, None]
S = np.concatenate([intercept, time_steps[:, None]], axis=1)
log_odds_t = np.dot(S, beta_s)

# \mathcal{R}
risk_set = (t >= time_steps[:, None]).astype(int)

# \mathcal{R}^*
last_time = (t == time_steps[:, None]).astype(int)

# \mathcal{Y}
y_obs = d * last_time

# \hat{\mathcal{Y}}
y_pred = inverse_logit(log_odds_x_matrix + log_odds_t[:, None])

# \mathcal{P}
residual_matrix = (y_obs - y_pred) * risk_set

# Contributions by X
n_ones = np.ones(shape=(1, n_time_steps))
y_resid = np.dot(n_ones, residual_matrix).T
x_score = y_resid * X

# Contributions by S
t_score = np.dot(S.T, residual_matrix)

# Estimating functions
est_func = np.vstack([x_score.T, t_score])

# Function
x = ee_pooled_logit(theta=beta_x + beta_s,
                    t=t, delta=d, X=X,
                    S=S)
print(est_func)
print(x)

################################################
# Pooled Logit -- Disjoint

beta_s = [-1, 0.1, -0.1]
event_times = t[d == 1]
unique_event_times = np.unique(event_times)
tp = unique_event_times.shape[0]

# \mathcal{X}
log_odds_x = np.dot(X, beta_x)
log_odds_x_matrix = np.tile(log_odds_x, (tp, 1))

# \mathcal{S}
time_design_matrix = np.identity(n=len(unique_event_times))
time_design_matrix[:, 0] = 1
log_odds_t = np.dot(time_design_matrix, beta_s)

# \mathcal{R}
risk_set = (t >= unique_event_times[:, None]).astype(int)

# \mathcal{R}^*
last_time = (t == unique_event_times[:, None]).astype(int)

# \mathcal{Y}
y_obs = d * last_time

# \hat{\mathcal{Y}}
y_pred = inverse_logit(log_odds_x_matrix + log_odds_t[:, None])

# \mathcal{P}
residual_matrix = (y_obs - y_pred) * risk_set

# Contributions by X
n_ones = np.ones(shape=(1, tp))
y_resid = np.dot(n_ones, residual_matrix)[0]
x_score = y_resid[:, None] * X

# Contributions by S
t_score = residual_matrix

# Estimating functions
est_func = np.vstack([x_score.T, t_score])

# Function
x = ee_pooled_logit(theta=beta_x + beta_s,
                    t=t, delta=d, X=X)

print(est_func)
print(x)
