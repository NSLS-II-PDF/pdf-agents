import logging
from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.special import wofz

from pdf_agents.agents import PDFBaseAgent, PDFReporterMixin

logger = logging.getLogger(__name__)


class PeakFitAgent(PDFReporterMixin, PDFBaseAgent):
    """_summary_

    Parameters
    ----------
    xrois : List[tuple]
        Regions of interest in x (lower limit, upper limit)
    fit_func : str
        Fitting function to use - can be gaussian, lorentzian, or voigt
    pos_percent_lim : float
        Percent of peak position for bounds (e.g., lim = position +/- (position * pos_percent_lim / 100))
    maxcycles : int
        Maximum number of curve_fit cycles - may need to increase for convergence

    Attributes
    ----------
    xrois : List[tuple]
        Regions of interest in x (lower limit, upper limit)
    fit_func : str
        Fitting function to use - can be gaussian, lorentzian, or voigt
    pos_percent_lim : float
        Percent of peak position for bounds (e.g., lim = position +/- (position * pos_percent_lim / 100))
    maxcycles : int
        Maximum number of curve_fit cycles - may need to increase for convergence

    Examples
    --------
    This agent is designed to be used in offline mode. It can be used as follows:
    >>> import tiled.client.node # Workaround for API issue
    >>> from pdf_agents.peakfit import PeakFitAgent
    >>> offline_obj = PeakFitAgent.get_offline_objects()
    >>> agent = PeakFitAgent(xrois= [(2.7,3.0)], fit_func='gaussian', pos_percent_lim=2, maxcycles=1000,
                                report_producer=offline_obj["kafka_producer"], offline=True)
    """

    def __init__(
        self,
        *,
        xrois: List[tuple],
        fit_func: str,
        pos_percent_lim: float,
        maxcycles: int,
        **kwargs,
    ):
        self._xrois = xrois
        self._fit_func = fit_func
        self._pos_percent_lim = pos_percent_lim
        self._maxcycles = maxcycles
        self._recent_x = None
        self._recent_y = None
        self._recent_uid = None
        super().__init__(**kwargs)
        self.report_on_tell = True

    @property
    def xrois(self):
        return self._xrois

    @xrois.setter
    def xrois(self, xrois):
        self._xrois = xrois
        self.close_and_restart()

    @property
    def fit_func(self):
        return self._fit_func

    @fit_func.setter
    def fit_func(self, fit_func):
        self._fit_func = fit_func
        self.close_and_restart()

    @property
    def pos_percent_lim(self):
        return self._pos_percent_lim

    @pos_percent_lim.setter
    def pos_percent_lim(self, pos_percent_lim):
        self._pos_percent_lim = pos_percent_lim
        self.close_and_restart()

    @property
    def maxcycles(self):
        return self._maxcycles

    @maxcycles.setter
    def maxcycles(self, maxcycles):
        self._maxcycles = maxcycles
        self.close_and_restart()

    def name(self):
        return "Peak-Fit-Agent"

    def unpack_run(self, run):
        self._recent_uid = run.metadata["start"]["uid"]
        return super().unpack_run(run)

    def tell(self, x, y) -> Dict[str, ArrayLike]:
        self._recent_x = x
        self._recent_y = y
        return dict(independent_variable=x, observable=y, ordinate=self._ordinate)

    def gaussian(self, x, amp, center, width):
        return (amp / width * np.sqrt(2 * np.pi)) * np.exp((-((x - center) ** 2)) / ((2.0 * width) ** 2))

    def gaussian_fwhm(self, width):
        return 2 * np.sqrt(2 * np.log(2)) * width

    def lorentzian(self, x, amp, center, width):
        return amp / np.pi * (width / ((x - center) ** 2 + width**2))

    def lorentzian_fwhm(self, width):
        return 2 * width

    def voigt(self, x, amp, center, sigma, gamma):
        """
        Voigt function.
        Parameters:
            x: array-like, independent variable.
            amp: float, amplitude of the Voigt peak.
            center: float, center position of the Voigt peak.
            sigma: float, standard deviation (Gaussian component).
            gamma: float, Lorentzian broadening parameter.
        Returns:
            Voigt function evaluated at x.
        """
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amp * np.real(wofz(z))

    def voigt_fwhm(self, sigma, gamma):
        return 2 * sigma * np.sqrt(2 * np.log(2)) + 2 * gamma

    def fit_roi(self, xroi: tuple, fit_func: str, pos_percent_lim: float, maxcycles: int):
        """Slice data for region of interest, find maxima, fit with Voigt function, return fit results

        Returns
        -------
        peak_amplitude: float
            Peak intensity (integrated area)
        peak_position: float
            Peak center from fitting
        peak_fwhm: float
            Peak full width at half maximum
        ycalc: np.ndarray
            Fit pattern
        ydiff: np.ndarray
            Difference (data - fit) pattern
        x_roi: np.ndarray
            Region of interest (x units)

        Voigt-only:
        peak_sigma: float
            Gaussian broadening parameter
        peak_gamma: float
            Lorentzian broadening parameter
        """
        roi_indices = np.where((self._ordinate >= xroi[0]) & (self._ordinate <= xroi[1]))

        x_roi = self._ordinate[roi_indices]
        y_roi = self._recent_y[roi_indices]

        # find maximum y within ROI
        peak_maxima_index = np.argmax(y_roi)
        peak_maxima_x = x_roi[peak_maxima_index]
        peak_maxima_y = y_roi[peak_maxima_index]

        if fit_func == "voigt":

            if sum(y_roi) < 0:  # in case there is data on background, off-sample, a peak is missing
                p0 = [0.1, peak_maxima_x, 0.01, 0.01]
            else:
                p0 = [peak_maxima_y, peak_maxima_x, 0.01, 0.01]

            try:
                pos_llim = peak_maxima_x - (peak_maxima_x * pos_percent_lim / 100)  # pos lower limit
                pos_ulim = peak_maxima_x + (peak_maxima_x * pos_percent_lim / 100)  # pos upper limit
                bounds = (
                    [0, pos_llim, 0.00001, 0.00001],
                    [1000000, pos_ulim, 0.5, 0.5],
                )  # (lower, upper) bounds for amplitude, center, sigma, gamma

                popt, pcov = curve_fit(self.voigt, x_roi, y_roi, p0=p0, bounds=bounds, max_nfev=maxcycles)

            except RuntimeError:
                raise RuntimeError("PeakFitAgent fit failed to converge")

            peak_amplitude, peak_position, peak_sigma, peak_gamma = popt
            peak_fwhm = self.voigt_fwhm(peak_sigma, peak_gamma)

            ycalc = self.voigt(x_roi, *popt)
            ydiff = y_roi - self.voigt(x_roi, *popt)

            return peak_amplitude, peak_position, peak_sigma, peak_gamma, peak_fwhm, ycalc, ydiff, x_roi

        elif fit_func == "gaussian":
            if sum(y_roi) < 0:  # in case there is data on background, off-sample
                p0 = [0.01, peak_maxima_x, 0.01]
            else:
                p0 = [peak_maxima_y, peak_maxima_x, 0.01]  # guess for parameters: amplitude, center, width

            try:
                pos_llim = peak_maxima_x - (peak_maxima_x * pos_percent_lim / 100)  # pos lower limit
                pos_ulim = peak_maxima_x + (peak_maxima_x * pos_percent_lim / 100)  # pos upper limit
                bounds = (
                    [0, pos_llim, 0.00001],
                    [1000000, pos_ulim, 1],
                )  # (lower, upper) bounds for amplitude, center, sigma, gamma

                # print(f"p0={p0} | Bounds: {bounds}")
                popt, pcov = curve_fit(self.gaussian, x_roi, y_roi, p0=p0, bounds=bounds, max_nfev=maxcycles)

            except RuntimeError:
                raise RuntimeError("PeakFitAgent fit failed to converge")

            peak_amplitude, peak_position, peak_width = popt
            peak_fwhm = self.gaussian_fwhm(peak_width)

            ycalc = self.gaussian(x_roi, *popt)
            ydiff = y_roi - self.gaussian(x_roi, *popt)

            return peak_amplitude, peak_position, peak_fwhm, ycalc, ydiff, x_roi

        elif fit_func == "lorentzian":
            if sum(y_roi) < 0:  # in case there is data on background, off-sample
                p0 = [0.01, peak_maxima_x, 0.01]
            else:
                p0 = [peak_maxima_y, peak_maxima_x, 0.01]  # guess for parameters: amplitude, center, width

            try:
                pos_llim = peak_maxima_x - (peak_maxima_x * pos_percent_lim / 100)  # pos lower limit
                pos_ulim = peak_maxima_x + (peak_maxima_x * pos_percent_lim / 100)  # pos upper limit
                bounds = (
                    [0, pos_llim, 0.00001],
                    [1000000, pos_ulim, 1],
                )  # (lower, upper) bounds for amplitude, center, sigma, gamma

                # print(f"p0={p0} | Bounds: {bounds}")
                popt, pcov = curve_fit(self.lorentzian, x_roi, y_roi, p0=p0, bounds=bounds, max_nfev=1000)

            except RuntimeError:
                raise RuntimeError("PeakFitAgent fit failed to converge")

            peak_amplitude, peak_position, peak_width = popt
            peak_fwhm = self.lorentzian_fwhm(peak_width)

            ycalc = self.lorentzian(x_roi, *popt)
            ydiff = y_roi - self.lorentzian(x_roi, *popt)

            return peak_amplitude, peak_position, peak_fwhm, ycalc, ydiff, x_roi

        else:
            raise NameError(
                f"fit_func not recognized - must be 'voigt', 'gaussian', or 'lorentzian'. Input is {fit_func}"
            )

    def report(self) -> Dict[str, ArrayLike]:

        peak_amps_poss_fwhms = (
            []
        )  # this is what kmeans will likely consume - possibly better to handle this upstream?
        peak_amplitudes = []
        peak_positions = []
        peak_fwhms = []
        x_rois = []

        if self.fit_func == "voigt":
            for xroi in self.xrois:
                peak_amplitude, peak_position, peak_sigma, peak_gamma, peak_fwhm, ycalc, ydiff, x_roi = (
                    self.fit_roi(xroi, self.fit_func, self.pos_percent_lim, self.maxcycles)
                )

                peak_amps_poss_fwhms.append([peak_amplitude, peak_position, peak_fwhm])
                peak_amplitudes.append(peak_amplitude)
                peak_positions.append(peak_position)
                peak_fwhms.append(peak_fwhm)

        elif self.fit_func == "gaussian" or self.fit_func == "lorentzian":
            for xroi in self.xrois:
                peak_amplitude, peak_position, peak_fwhm, ycalc, ydiff, x_roi = self.fit_roi(
                    xroi, self.fit_func, self.pos_percent_lim, self.maxcycles
                )

                peak_amps_poss_fwhms.append([peak_amplitude, peak_position, peak_fwhm])
                peak_amplitudes.append(peak_amplitude)
                peak_positions.append(peak_position)
                peak_fwhms.append(peak_fwhm)

        return dict(
            data_key=self.data_key,
            roi_key=self.roi_key if self.roi_key is not None else "",
            roi=self.roi if self.roi is not None else "",
            observable_uid=self._recent_uid,
            independent_variable=self._recent_x,
            ordinate=self._ordinate,
            observable=self._recent_y,
            xrois=self.xrois,
            peak_fit_rois=self.xrois,
            fit_func=self.fit_func,
            pos_percent_lim=self.pos_percent_lim,
            maxcycles=self.maxcycles,
            peak_amps_poss_fwhms=np.array(peak_amps_poss_fwhms),
            peak_amplitudes=np.array(peak_amplitudes),
            peak_positions=np.array(peak_positions),
            peak_fwhms=np.array(peak_fwhms),
        )

    def ask(self, batch_size):
        """This is a passive agent, that does not request next experiments. It does analysis."""
        raise NotImplementedError
