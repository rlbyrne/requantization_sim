import numpy as np
import scipy
import os
import simulation_scripts

current_dir = os.getcwd()
eq_coeffs_mat = scipy.io.loadmat(f"{current_dir}/20251028a-settingsAll-night-FW7p6.mat")
channel_width_mhz = 23925.78125 * 1e-6
freq_array = np.arange(len(eq_coeffs_mat["coef"][0, :])) * channel_width_mhz

use_eq_coeffs = eq_coeffs_mat["coef"][
    0, :
]  # Use the first set of equalization coefficients

target_value = 3 / 2**3
data_stddev = target_value / use_eq_coeffs

final_variances = simulation_scripts.requantization_sim(
    data_stddev,
    use_eq_coeffs,
)  # Run simulation

f = open(f"simulation_output.npy", "wb")
np.save(f, freq_array)
np.save(f, final_variances)
f.close()
