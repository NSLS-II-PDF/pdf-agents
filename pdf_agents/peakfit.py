import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
from numpy.typing import ArrayLike

from pdf_agents.agents import PDFBaseAgent, PDFReporterMixin

logger = logging.getLogger(__name__)

class PeakFitAgent(PDFReporterMixin, PDFBaseAgent):
    """_summary_

        Parameters
        ----------
        xrois : List[tuple]
            Regions of interest in x (lower limit, upper limit)

        Attributes
        ----------
        xrois : List[tuple]
            Regions of interest in x (lower limit, upper limit)

        Examples
        --------
        This agent is designed to be used in offline mode. It can be used as follows:
        >>> import tiled.client.node # Workaround for API issue
        >>> from pdf_agents.peakfit import PeakFitAgent
        >>> offline_obj = PeakFitAgent.get_offline_objects()
        >>> agent = PeakFitAgent(xrois= [],
                                    report_producer=offline_obj["kafka_producer"], offline=True)
        """
    def __init__(
        self,
        *,
        xrois: List[tuple],
        **kwargs,
    ):
        self._xrois = xrois
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

    def name(self):
        return "Peak-Fit-Agent"

    def unpack_run(self, run):
        self._recent_uid = run.metadata["start"]["uid"]
        return super().unpack_run(run)

    def tell(self, x, y) -> Dict[str, ArrayLike]:
        self._recent_x = x
        self._recent_y = y
        return dict(independent_variable=x, observable=y, ordinate=self._ordinate)

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
 
    def _select_and_fit_roi(self, xroi: tuple, pos_lim_const=0.02, maxcycles=1000): 

        """Slice data for region of interest, find maxima, fit with Voigt function, return fit results
        
        Returns
        -------
        peak_amplitude: float
            Peak intensity (integrated area)
        peak_position: float
            Peak center from fitting != maxima
        peak_sigma: float
            Gaussian broadening parameter
        peak_gamma: float
            Lorentzian broadening parameter
        peak_fwhm: float
            Voigt full width at half maximum
        ycalc: np.ndarray
            Fit pattern
        ydiff: np.ndarray
            Difference (data - fit) pattern
        x_roi: np.ndarray
            Region of interest (x units)
        """
        roi_indices = np.where((self._recent_x >= xroi[0]) & (self._recent_x <= xroi[1]))

        x_roi = self._recent_x[roi_indices]
        y_roi = self._recent_y[roi_indices]

        # find maximum y within ROI
        peak_maxima_index = np.argmax(y_roi)
        peak_maxima_x = x_roi[peak_maxima_index]
        peak_maxima_y = y_roi[peak_maxima_index]

        if sum(y_roi) < 0: # in case there is data on background, off-sample, or a peak is entirely missing
            p0 = [0.1, peak_maxima_x, 0.01, 0.01] 
        else:
            p0 = [peak_maxima_y, peak_maxima_x, 0.01, 0.01]

        try:
            pos_llim = peak_maxima_x - (peak_maxima_x * pos_lim_const) # peak pos lower limit
            pos_ulim = peak_maxima_x + (peak_maxima_x * pos_lim_const) # peak pos upper limit
            bounds = ([0, pos_llim, 0.00001, 0.00001], [1000000, pos_ulim, 1, 1]) # (lower, upper) bounds for amplitude, center, sigma, gamma

            popt, pcov = curve_fit(voigt, x_roi, y_roi, p0=p0, bounds=bounds, max_nfev=maxcycles)

        except RuntimeError: # if fitting fails, return None for parameters
            popt = None

        if popt is not None:
            peak_amplitude, peak_position, peak_sigma, peak_gamma = popt
            peak_fwhm = voigt_fwhm(peak_sigma, peak_gamma)

            ycalc = voigt(x_roi, *popt)
            ydiff = y_roi - voigt(x_roi, *popt)

        else:
            peak_amplitude = None
            peak_position = None
            peak_sigma = None
            peak_gamma = None
            peak_fwhm = None
            ycalc = None
            ydiff = None
            x_roi = None

        return peak_amplitude, peak_position, peak_sigma, peak_gamma, peak_fwhm, ycalc, ydiff, x_roi

    def report(self) -> Dict[str, ArrayLike]:

        peak_amplitudes = []
        peak_positions = []
        peak_sigmas = []
        peak_gammas = []
        peak_fwhms = []
        ycalcs = []
        ydiffs = []
        x_rois = []

        for xroi in self.xrois:
            peak_amplitude, peak_position, peak_sigma, peak_gamma, peak_fwhm, ycalc, ydiff, x_roi = self._select_and_fit_roi(xroi)
            peak_amplitudes.append(peak_amplitude)
            peak_positions.append(peak_position)
            peak_sigmas.append(peak_sigma)
            peak_gammas.append(peak_gamma)
            peak_fwhms.append(peak_fwhm)
            ycalcs.append(ycalc)
            ydiffs.append(ydiff)
            x_rois.append(x_roi)

        return dict(
            data_key=self.data_key,
            roi_key=self.roi,
            roi=self.roi,
            norm_region=self.norm_region,
            observable_uid=self._recent_uid,
            independent_variable=self._recent_x,
            observable=self._recent_y,
            xrois=self.xrois,
            peak_amplitudes=np.array(peak_amplitudes),
            peak_positions=np.array(peak_positions),
            peak_sigmas=np.array(peak_sigmas),
            peak_gammas=np.array(peak_gammas),
            peak_fwhms=np.array(peak_fwhms),
            ycalc=np.array(ycalcs),
            ydiffs=np.array(ydiffs),
            x_rois=np.array(x_rois)
        )

    def ask(self, batch_size):
        """This is a passive agent, that does not request next experiments. It does analysis."""
        raise NotImplementedError