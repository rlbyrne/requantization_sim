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

    requantization_gain = 2**16
    target_value = 3 * requantization_gain

    for ind in range(7):
        use_eq_coeffs = eq_coeffs_mat["coef"][
            ind, :
        ]  # Use the first set of equalization coefficients

        data_stddev = target_value / use_eq_coeffs

        final_variances, final_autocorrs = simulation_scripts.requantization_sim(
            data_stddev,
            use_eq_coeffs,
            input_bits_total=18,
            input_bits_fractional=0,
            eq_coeff_bits_total=14,
            eq_coeff_bits_fractional=0,
            output_bits_total=4,
            output_bits_fractional=0,
            requantization_gain=requantization_gain,
        )  # Run simulation

        f = open(f"simulation_output{ind}.npy", "wb")
        np.save(f, freq_array)
        np.save(f, final_variances)
        np.save(f, final_autocorrs)
        f.close()


def constant_slope_sims():

    nfreqs = 2048
    average_signal_stddev = 16
    # frac_variation = np.array([0.01, 0.1, 0.5, 1])
    frac_variation = np.array([5])
    requantization_gain = 2**16
    target_value = 3 * requantization_gain

    for use_var in frac_variation:
        min_signal_stddev = 2 * average_signal_stddev / (2 + use_var)
        max_signal_stddev = 2 * average_signal_stddev - min_signal_stddev
        data_stddev = np.linspace(min_signal_stddev, max_signal_stddev, num=nfreqs)
        eq_coeffs = target_value / data_stddev

        final_variances, final_autocorrs = simulation_scripts.requantization_sim(
            data_stddev,
            eq_coeffs,
            input_bits_total=18,
            input_bits_fractional=0,
            eq_coeff_bits_total=14,
            eq_coeff_bits_fractional=0,
            output_bits_total=4,
            output_bits_fractional=0,
            requantization_gain=requantization_gain,
        )  # Run simulation
        f = open(f"const_slope_simulation_output_slope{use_var}.npy", "wb")
        np.save(f, final_variances)
        np.save(f, final_autocorrs)
        f.close()


def increased_signal_simulations():

    nfreqs = 2048
    average_signal_stddev_array = np.array([50.0, 100.0, 500.0, 1000.0])
    use_var = 1.0
    requantization_gain = 2**16
    target_value = 3 * requantization_gain

    for average_signal_stddev in average_signal_stddev_array:
        min_signal_stddev = 2 * average_signal_stddev / (2 + use_var)
        max_signal_stddev = 2 * average_signal_stddev - min_signal_stddev
        data_stddev = np.linspace(min_signal_stddev, max_signal_stddev, num=nfreqs)
        eq_coeffs = target_value / data_stddev

        final_variances, final_autocorrs = simulation_scripts.requantization_sim(
            data_stddev,
            eq_coeffs,
            input_bits_total=18,
            input_bits_fractional=0,
            eq_coeff_bits_total=14,
            eq_coeff_bits_fractional=0,
            output_bits_total=4,
            output_bits_fractional=0,
            requantization_gain=requantization_gain,
        )  # Run simulation
        f = open(
            f"const_slope_simulation_output_slope{use_var}_avg_signal{average_signal_stddev}.npy",
            "wb",
        )
        np.save(f, final_variances)
        np.save(f, final_autocorrs)
        f.close()


def increased_bit_depth_simulations():

    nfreqs = 2048
    average_signal_stddev = 16
    output_bits_array = np.array([5, 6, 7, 8])
    use_var = 1.0

    for output_bits in output_bits_array:

        requantization_gain = 2 ** (20 - output_bits)
        target_value = 3 * requantization_gain

        min_signal_stddev = 2 * average_signal_stddev / (2 + use_var)
        max_signal_stddev = 2 * average_signal_stddev - min_signal_stddev
        data_stddev = np.linspace(min_signal_stddev, max_signal_stddev, num=nfreqs)
        eq_coeffs = target_value / data_stddev

        final_variances, final_autocorrs = simulation_scripts.requantization_sim(
            data_stddev,
            eq_coeffs,
            input_bits_total=18,
            input_bits_fractional=0,
            eq_coeff_bits_total=14,
            eq_coeff_bits_fractional=0,
            output_bits_total=output_bits,
            output_bits_fractional=0,
            requantization_gain=requantization_gain,
        )  # Run simulation
        f = open(
            f"const_slope_simulation_output_slope{use_var}_bit_depth{output_bits}.npy",
            "wb",
        )
        np.save(f, final_variances)
        np.save(f, final_autocorrs)
        f.close()


if __name__ == "__main__":
    increased_bit_depth_simulations()
