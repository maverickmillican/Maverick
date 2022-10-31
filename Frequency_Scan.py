import typing

import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat


def frequency_scan_model(frequency, freq_0, rabi_rate, amplitude):
    Omega = 2*np.pi * rabi_rate
    delta = 2*np.pi * (frequency - freq_0)
    def sinc(x):
        return np.sin(x)/x
    return np.pi**2/4 * sinc(np.sqrt(Omega**2 + delta**2)*np.pi / (2 * Omega))**2 * amplitude


def fit_frequency_scan(x_data: typing.List[float], y_data: typing.List[float], amplitude_guess: float = None,
                  rabi_rate_guess: float = None, freq_centre_guess: float = None, verbose: bool = True):

    if amplitude_guess is None:
        amplitude_guess = 1.0

    if rabi_rate_guess is None:
        rabi_rate_guesses = np.linspace(x_data[1] - x_data[0], x_data[-1] - x_data[0],  10)
    else:
        rabi_rate_guesses = [rabi_rate_guess]

    if freq_centre_guess is None:
        freq_centre_guesses = np.linspace(x_data[0], x_data[-1], 10)
    else:
        freq_centre_guesses = [freq_centre_guess]

    starting_guesses = np.array(np.meshgrid(rabi_rate_guesses, freq_centre_guesses)).T.reshape(-1, 2)

    best_fit_params = []
    best_fit_errors = []
    min_detuning_error = 1e6

    for [rabi_rate_guess, freq_centre_guess] in starting_guesses:
        try:
            popt, pcov = curve_fit(frequency_scan_model, x_data, y_data, p0=[freq_centre_guess, rabi_rate_guess, amplitude_guess],
                                   bounds=[[x_data[0]*0.9, 1e2, 0], [x_data[-1] * 1.1, 1e6, 1]])
            perr = np.sqrt(np.diag(pcov))
            detuning_error = perr[0]
            if detuning_error < min_detuning_error:
                min_detuning_error = detuning_error
                best_fit_params = popt
                best_fit_errors = perr
        except RuntimeError:
            pass

    if len(best_fit_params) == 0:
        print('Scipy optimization failed to find optimal parameters.')
        return [], []

    if verbose:
        print('Optimal parameters from curve_fit: ')
        print('Centre frequency : {:.1f} Hz'.format(ufloat(best_fit_params[0], best_fit_errors[0])))
        print('Rabi rate : {:3f} kHz'.format(ufloat(best_fit_params[1], best_fit_errors[1])/1e3))
        print('Amplitude : {:.3f}'.format(ufloat(best_fit_params[2], best_fit_errors[1])))
        print('Amplitude : {:.3f}'.format(ufloat(best_fit_params[2], best_fit_errors[1])))

    return best_fit_params, best_fit_errors
