import numpy as np
import scipy
import os


def get_quantized_value_options(
    total_bits,
    fractional_bits,
    signed=True,
    enforce_symmetry=False,
    return_min=None,
    return_max=None,
):
    if signed:
        max_value = 2 ** (total_bits - 1) - 1
    else:
        max_value = 2 ** (total_bits - 1)
    if signed:
        if enforce_symmetry:
            min_value = -1 * max_value
        else:
            min_value = -(2 ** (total_bits - 1))
    else:
        min_value = 0

    if return_min is not None:
        min_value = np.max([np.round(return_min * 2**fractional_bits), min_value])
    if return_max is not None:
        max_value = np.min([np.round(return_max * 2**fractional_bits), max_value])
    return np.arange(min_value, max_value + 1) / (2**fractional_bits)


def quantize(
    numbers,
    total_bits,
    fractional_bits,
    signed=True,
    enforce_symmetry=False,
):  # Copied from requantization_sim.ipynb

    numbers = np.asarray(numbers)
    numbers = np.round(numbers * 2**fractional_bits)
    if signed:
        max_value = 2 ** (total_bits - 1) - 1
    else:
        max_value = 2**total_bits - 1
    numbers[numbers > max_value] = max_value
    if signed:
        if enforce_symmetry:
            min_value = -1 * max_value
        else:
            min_value = -(2 ** (total_bits - 1))
        numbers[numbers < min_value] = min_value
    numbers /= 2.0**fractional_bits
    return numbers


def get_probabilities(stddev, value_options):

    probabilities = np.zeros_like(value_options)
    for ind in range(len(value_options)):
        if ind == 0:
            integral_start = -np.inf
        else:
            integral_start = (value_options[ind - 1] + value_options[ind]) / 2

        if ind == len(value_options) - 1:
            integral_end = np.inf
        else:
            integral_end = (value_options[ind] + value_options[ind + 1]) / 2
        integral_value = (
            np.sqrt(np.pi)
            * stddev
            / 2
            * (
                scipy.special.erf(integral_end / (np.sqrt(2) * stddev))
                - scipy.special.erf(integral_start / (np.sqrt(2) * stddev))
            )
        )
        probabilities[ind] = integral_value

    probabilities /= np.sum(probabilities)
    return probabilities


def calculate_variance(values, probabilities):
    return np.sum(probabilities * values**2) - np.sum(probabilities * values) ** 2


def calculate_autocorr(values, probabilities):
    return 2 * np.sum(probabilities * values**2)


def requantization_sim(
    input_stddev_array,
    equalization_coeffs,
    input_bits_total=18,
    input_bits_fractional=17,
    eq_coeff_bits_total=14,
    eq_coeff_bits_fractional=2,
    output_bits_total=4,
    output_bits_fractional=3,
    dither_stddev=0,
):

    initial_quantized_value_options = get_quantized_value_options(
        input_bits_total, input_bits_fractional
    )

    input_stddev_unique = np.unique(input_stddev_array)
    initial_quantized_probabilities_array = np.zeros(
        (len(initial_quantized_value_options), len(input_stddev_unique))
    )
    for stddev_ind, use_stddev in enumerate(input_stddev_unique):
        initial_quantized_probabilities_array[:, stddev_ind] = get_probabilities(
            use_stddev, initial_quantized_value_options
        )

    final_quantized_value_options = get_quantized_value_options(
        output_bits_total, output_bits_fractional, enforce_symmetry=True
    )
    final_variances = np.zeros_like(equalization_coeffs)
    final_autocorrs = np.zeros_like(equalization_coeffs)
    for equalization_ind, equalization_coeff in enumerate(equalization_coeffs):
        if equalization_ind % 10 == 0:
            print(
                f"Processing frequency {equalization_ind+1} of {len(equalization_coeffs)}.",
                flush=True,
            )

        initial_quantized_probabilities = initial_quantized_probabilities_array[
            :,
            np.where(input_stddev_unique == input_stddev_array[equalization_ind])[0][0],
        ]
        equalized_value_options = initial_quantized_value_options * equalization_coeff

        if dither_stddev != 0:  # Add dither
            resolution = 2 ** (-1 * (input_bits_fractional + eq_coeff_bits_fractional))
            dither_cutoff = 5  # Go out to 5 sigma in the dither stddev

            # New point separations to be added to the value options
            add_point_separations = np.unique(
                np.concatenate(
                    (
                        np.arange(
                            0,
                            dither_cutoff * dither_stddev + resolution,
                            resolution,
                        ),
                        -1
                        * np.arange(
                            0,
                            dither_cutoff * dither_stddev + resolution,
                            resolution,
                        ),
                    )
                ).flatten()
            )
            gaussian = np.exp(-(add_point_separations**2) / (2 * dither_stddev**2))
            gaussian /= np.sum(gaussian)

            equalized_value_options = (
                equalized_value_options[:, np.newaxis]
                + add_point_separations[np.newaxis, :]
            ).flatten()
            initial_quantized_probabilities = (
                initial_quantized_probabilities[:, np.newaxis] * gaussian[np.newaxis, :]
            ).flatten()

        final_quantized_probabilities = np.zeros_like(final_quantized_value_options)
        equalized_value_options_quantized = quantize(
            equalized_value_options,
            output_bits_total,
            output_bits_fractional,
            enforce_symmetry=True,
        )
        for ind in range(len(final_quantized_value_options)):
            final_quantized_probabilities[ind] = np.sum(
                initial_quantized_probabilities[
                    np.where(
                        equalized_value_options_quantized
                        == final_quantized_value_options[ind]
                    )
                ]
            )

        final_variances[equalization_ind] = calculate_variance(
            final_quantized_value_options, final_quantized_probabilities
        )
        final_autocorrs[equalization_ind] = calculate_autocorr(
            final_quantized_value_options, final_quantized_probabilities
        )

    return final_variances, final_autocorrs


def get_requantized_probabilities(
    input_stddev_array,
    equalization_coeffs,
):

    initial_quantized_value_options = get_quantized_value_options(18, 17)

    input_stddev_unique = np.unique(input_stddev_array)
    initial_quantized_probabilities_array = np.zeros(
        (len(initial_quantized_value_options), len(input_stddev_unique))
    )
    for stddev_ind, use_stddev in enumerate(input_stddev_unique):
        initial_quantized_probabilities_array[:, stddev_ind] = get_probabilities(
            use_stddev, initial_quantized_value_options
        )

    final_quantized_value_options = get_quantized_value_options(
        4, 3, enforce_symmetry=True
    )
    final_quantized_probabilities = np.zeros(
        (len(final_quantized_value_options), len(equalization_coeffs))
    )
    for equalization_ind, equalization_coeff in enumerate(equalization_coeffs):

        initial_quantized_probabilities = initial_quantized_probabilities_array[
            :,
            np.where(input_stddev_unique == input_stddev_array[equalization_ind])[0][0],
        ]
        equalized_value_options = initial_quantized_value_options * equalization_coeff
        equalized_value_options_quantized = quantize(
            equalized_value_options, 4, 3, enforce_symmetry=True
        )
        for ind in range(len(final_quantized_value_options)):
            final_quantized_probabilities[ind, equalization_ind] = np.sum(
                initial_quantized_probabilities[
                    np.where(
                        equalized_value_options_quantized
                        == final_quantized_value_options[ind]
                    )
                ]
            )

    return final_quantized_value_options, final_quantized_probabilities
