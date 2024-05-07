import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Getting the SCFTMLE bias and variance columns into lists
bias_cols = ["ml_bias_"+str(i) for i in range(50)]
var_cols = ["ml_var_"+str(i) for i in range(50)]

# The different folds / repetition combinations considered
split_range = [2, 3, 4, 5]
reps_range = [1, 10, 20, 30, 40, 50]

# Display combinations for each of the different folds
line_colors = {2: "darkgray", 3: "dimgray", 4: "k", 5: "darkgray"}
line_style = {2: "^--", 3: "<-", 4: ">-", 5: "h-"}

# Creating the publication figure
cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(2, figsize=(8.5*cm, 10.16*cm))

for s in split_range:
    bias_line = []
    se_line = []
    for r in reps_range:
        d = pd.read_csv("output/r_s"+str(s)+"r"+str(r)+".csv")
        print(s, r, d.shape[0])
        bias_sd = np.std(d[bias_cols], axis=1, ddof=1)
        bias_sd_avg = np.mean(bias_sd)
        bias_line.append(bias_sd_avg)

        se_sd = np.std(d[var_cols]**0.5, axis=1, ddof=1)
        se_sd_avg = np.mean(se_sd)
        se_line.append(se_sd_avg)

    axs[0].plot(reps_range, bias_line, line_style[s], color=line_colors[s])
    axs[1].plot(reps_range, se_line, line_style[s], color=line_colors[s])


axs[0].set_ylim([0, 0.065])
axs[0].set_xlim([0, 51])
axs[0].set_xticks([1, 10, 20, 30, 40, 50])
axs[1].set_ylim([0, 0.006])
axs[1].set_xlim([0, 51])
axs[1].set_xticks([1, 10, 20, 30, 40, 50])
axs[1].set_yticks([0, 0.002, 0.004, 0.006])
axs[1].set_xlabel("Number of different splits")

axs[0].set_ylabel("Within-Data Mean \n SD of Bias")
axs[1].set_ylabel("Within-Data Mean \n SD of SE")

s2_line = mlines.Line2D([], [], color=line_colors[2], marker=line_style[2][0],
                        markersize=6, label='2-fold', linestyle='--')
s3_line = mlines.Line2D([], [], color=line_colors[3], marker=line_style[3][0],  markersize=6, label='3-fold')
s4_line = mlines.Line2D([], [], color=line_colors[4], marker=line_style[4][0],  markersize=6, label='4-fold')
s5_line = mlines.Line2D([], [], color=line_colors[5], marker=line_style[5][0],  markersize=6, label='5-fold')
axs[0].legend(handles=[s2_line, s3_line, s4_line, s5_line])

plt.tight_layout()
# plt.savefig("Figure1.png", format='png', dpi=300)
plt.show()
