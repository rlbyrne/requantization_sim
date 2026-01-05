import numpy as np
import scipy
import os
import simulation_scripts


def ovro_lwa_sim():
    current_dir = os.getcwd()
    eq_coeffs_mat = scipy.io.loadmat(
        f"{current_dir}/20250612-settingsAll-day_smoothed.mat"
    )
    channel_width_mhz = 23925.78125 * 1e-6
    freq_array = np.arange(len(eq_coeffs_mat["coef"][0, :])) * channel_width_mhz

    for ind in range(7):
        use_eq_coeffs = eq_coeffs_mat["coef"][
            ind, :
        ]  # Use the first set of equalization coefficients

        requantization_gain = 2**16
        target_value = 3 * requantization_gain
        data_stddev = target_value / use_eq_coeffs

        final_variances, final_autocorrs = simulation_scripts.requantization_sim(
            data_stddev,
            use_eq_coeffs,
        )  # Run simulation

        f = open(f"simulation_output{ind}.npy", "wb")
        np.save(f, freq_array)
        np.save(f, final_variances)
        f.close()


def constant_slope_sims():

    channel_width_mhz = 23925.78125 * 1e-6
    target_value = 3 / 2**3
    nfreqs = 2048
    avg_eq_coeffs = np.array([1, 10, 100, 1000, 10000])
    eq_coeff_frac_variation = np.array([0.01, 0.1, 0.5, 1])

    freq_array = np.arange(nfreqs) * channel_width_mhz
    final_autocorrs = np.zeros(
        (len(avg_eq_coeffs), len(eq_coeff_frac_variation), len(freq_array)), dtype=float
    )

    for eq_ind, avg_eq_coeff in enumerate(avg_eq_coeffs):
        for coeff_slope_ind, eq_coeff_var in enumerate(eq_coeff_frac_variation):
            eq_coeffs = np.linspace(
                avg_eq_coeff - avg_eq_coeff * eq_coeff_var / 2,
                avg_eq_coeff + avg_eq_coeff * eq_coeff_var / 2,
                num=len(freq_array),
            )
            data_stddev = target_value / eq_coeffs

            null, autocorrs = simulation_scripts.requantization_sim(
                data_stddev,
                eq_coeffs,
            )
            final_autocorrs[eq_ind, coeff_slope_ind, :] = autocorrs

    f = open(f"constant_slope_output.npy", "wb")
    np.save(f, freq_array)
    np.save(f, avg_eq_coeffs)
    np.save(f, eq_coeff_frac_variation)
    np.save(f, final_autocorrs)
    f.close()


if __name__ == "__main__":
    ovro_lwa_sim()
