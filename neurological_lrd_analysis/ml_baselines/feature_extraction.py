"""
Feature extraction for machine learning-based Hurst exponent estimation.

This module provides comprehensive feature extraction methods for time series data,
including statistical, spectral, wavelet, fractal, and biomedical-specific features
that are relevant for Hurst exponent estimation.

Author: Davian R. Chin (PhD Candidate in Biomedical Engineering, University of Reading, UK)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# Lazy imports for optional dependencies
def _lazy_import_scipy():
    """Lazy import of scipy modules"""
    try:
        import scipy.stats as stats
        import scipy.signal as signal
        from scipy.fft import fft, fftfreq
        return stats, signal, fft, fftfreq
    except ImportError:
        warnings.warn("scipy not available, some features will be disabled")
        return None, None, None, None

def _lazy_import_sklearn():
    """Lazy import of sklearn modules"""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_regression
        return StandardScaler, SelectKBest, f_regression
    except ImportError:
        warnings.warn("scikit-learn not available, ML features will be disabled")
        return None, None, None

def _lazy_import_pywt():
    """Lazy import of pywavelets"""
    try:
        import pywt
        return pywt
    except ImportError:
        warnings.warn("PyWavelets not available, wavelet features will be disabled")
        return None


@dataclass
class FeatureSet:
    """Container for extracted features."""
    statistical: Dict[str, float]
    spectral: Dict[str, float]
    wavelet: Dict[str, float]
    fractal: Dict[str, float]
    biomedical: Dict[str, float]
    combined: np.ndarray
    feature_names: List[str]


class TimeSeriesFeatureExtractor:
    """
    Comprehensive feature extractor for time series data.
    
    Extracts statistical, spectral, wavelet, fractal, and biomedical-specific
    features that are relevant for Hurst exponent estimation.
    """
    
    def __init__(self, 
                 include_spectral: bool = True,
                 include_wavelet: bool = True,
                 include_fractal: bool = True,
                 include_biomedical: bool = True,
                 sampling_rate: float = 250.0):
        """
        Initialize the feature extractor.
        
        Parameters:
        -----------
        include_spectral : bool
            Whether to include spectral features
        include_wavelet : bool
            Whether to include wavelet features
        include_fractal : bool
            Whether to include fractal features
        include_biomedical : bool
            Whether to include biomedical-specific features
        sampling_rate : float
            Sampling rate for biomedical feature extraction
        """
        self.include_spectral = include_spectral
        self.include_wavelet = include_wavelet
        self.include_fractal = include_fractal
        self.sampling_rate = sampling_rate
        self.include_biomedical = include_biomedical
        
        # Lazy import dependencies
        self._scipy_stats, self._scipy_signal, self._fft, self._fftfreq = _lazy_import_scipy()
        self._pywt = _lazy_import_pywt()
        self._scaler, self._select_k_best, self._f_regression = _lazy_import_sklearn()
    
    def extract_features(self, data: np.ndarray, 
                        true_hurst: Optional[float] = None) -> FeatureSet:
        """
        Extract comprehensive features from time series data.
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        true_hurst : float, optional
            True Hurst exponent (for validation)
            
        Returns:
        --------
        FeatureSet
            Extracted features
        """
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        if len(data) < 10:
            raise ValueError("Data too short for feature extraction")
        
        # Extract different feature types
        statistical_features = extract_statistical_features(data)
        spectral_features = {}
        wavelet_features = {}
        fractal_features = {}
        biomedical_features = {}
        
        if self.include_spectral and self._scipy_stats is not None:
            spectral_features = extract_spectral_features(data, self._scipy_signal, self._fft, self._fftfreq)
        
        if self.include_wavelet and self._pywt is not None:
            wavelet_features = extract_wavelet_features(data, self._pywt)
        
        if self.include_fractal:
            fractal_features = extract_fractal_features(data)
        
        if self.include_biomedical:
            biomedical_features = extract_biomedical_features(data, self.sampling_rate)
        
        # Combine all features
        all_features = {**statistical_features, **spectral_features, 
                       **wavelet_features, **fractal_features, **biomedical_features}
        
        # Convert to array
        feature_names = list(all_features.keys())
        combined_features = np.array(list(all_features.values()))
        
        return FeatureSet(
            statistical=statistical_features,
            spectral=spectral_features,
            wavelet=wavelet_features,
            fractal=fractal_features,
            biomedical=biomedical_features,
            combined=combined_features,
            feature_names=feature_names
        )


def extract_statistical_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
        
    Returns:
    --------
    Dict[str, float]
        Statistical features
    """
    features = {}
    
    # Basic statistics
    features['mean'] = float(np.mean(data))
    features['std'] = float(np.std(data))
    features['var'] = float(np.var(data))
    features['skewness'] = float(_calculate_skewness(data))
    features['kurtosis'] = float(_calculate_kurtosis(data))
    features['min'] = float(np.min(data))
    features['max'] = float(np.max(data))
    features['range'] = float(np.max(data) - np.min(data))
    
    # Central moments
    features['moment_2'] = float(np.mean((data - np.mean(data))**2))
    features['moment_3'] = float(np.mean((data - np.mean(data))**3))
    features['moment_4'] = float(np.mean((data - np.mean(data))**4))
    
    # Quantiles
    features['q25'] = float(np.percentile(data, 25))
    features['q50'] = float(np.percentile(data, 50))
    features['q75'] = float(np.percentile(data, 75))
    features['iqr'] = float(np.percentile(data, 75) - np.percentile(data, 25))
    
    # Autocorrelation features
    if len(data) > 1:
        autocorr_1 = _autocorrelation(data, lag=1)
        autocorr_2 = _autocorrelation(data, lag=2)
        autocorr_5 = _autocorrelation(data, lag=5)
        
        features['autocorr_1'] = float(autocorr_1)
        features['autocorr_2'] = float(autocorr_2)
        features['autocorr_5'] = float(autocorr_5)
        features['autocorr_decay'] = float(autocorr_1 - autocorr_5)
    
    # Trend features
    features['trend_slope'] = float(_trend_slope(data))
    features['detrended_std'] = float(_detrended_std(data))
    
    # Stationarity features
    features['adf_stat'] = float(_adf_statistic(data))
    features['kpss_stat'] = float(_kpss_statistic(data))
    
    return features


def extract_spectral_features(data: np.ndarray, 
                            scipy_signal, fft, fftfreq) -> Dict[str, float]:
    """
    Extract spectral features from time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    scipy_signal : module
        scipy.signal module
    fft : function
        FFT function
    fftfreq : function
        FFT frequency function
        
    Returns:
    --------
    Dict[str, float]
        Spectral features
    """
    features = {}
    
    if scipy_signal is None or fft is None:
        return features
    
    try:
        # Power spectral density
        freqs, psd = scipy_signal.welch(data, nperseg=min(len(data)//4, 256))
        
        # Spectral features
        features['spectral_centroid'] = float(np.sum(freqs * psd) / np.sum(psd))
        features['spectral_bandwidth'] = float(np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * psd) / np.sum(psd)))
        features['spectral_rolloff'] = float(_spectral_rolloff(freqs, psd))
        features['spectral_flatness'] = float(_spectral_flatness(psd))
        
        # Frequency band features
        delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
        theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
        beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
        gamma_power = np.sum(psd[(freqs >= 30) & (freqs <= 100)])
        
        total_power = np.sum(psd)
        
        features['delta_ratio'] = float(delta_power / total_power)
        features['theta_ratio'] = float(theta_power / total_power)
        features['alpha_ratio'] = float(alpha_power / total_power)
        features['beta_ratio'] = float(beta_power / total_power)
        features['gamma_ratio'] = float(gamma_power / total_power)
        
        # Dominant frequency
        features['dominant_freq'] = float(freqs[np.argmax(psd)])
        
        # Spectral entropy
        features['spectral_entropy'] = float(_spectral_entropy(psd))
        
    except Exception as e:
        warnings.warn(f"Spectral feature extraction failed: {e}")
    
    return features


def extract_wavelet_features(data: np.ndarray, pywt) -> Dict[str, float]:
    """
    Extract wavelet features from time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    pywt : module
        PyWavelets module
        
    Returns:
    --------
    Dict[str, float]
        Wavelet features
    """
    features = {}
    
    if pywt is None:
        return features
    
    try:
        # Wavelet decomposition
        coeffs = pywt.wavedec(data, 'db4', level=4)
        
        # Energy features
        total_energy = sum(np.sum(c**2) for c in coeffs)
        
        for i, coeff in enumerate(coeffs):
            energy = np.sum(coeff**2)
            features[f'wavelet_energy_level_{i}'] = float(energy)
            features[f'wavelet_energy_ratio_level_{i}'] = float(energy / total_energy)
        
        # Wavelet variance
        for i, coeff in enumerate(coeffs[1:], 1):  # Skip approximation coefficients
            features[f'wavelet_var_level_{i}'] = float(np.var(coeff))
            features[f'wavelet_std_level_{i}'] = float(np.std(coeff))
        
        # Wavelet entropy
        features['wavelet_entropy'] = float(_wavelet_entropy(coeffs))
        
        # Dominant scale
        energy_by_level = [np.sum(coeffs[i]**2) for i in range(1, len(coeffs))]
        features['dominant_scale'] = float(np.argmax(energy_by_level) + 1)
        
    except Exception as e:
        warnings.warn(f"Wavelet feature extraction failed: {e}")
    
    return features


def extract_fractal_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract fractal features from time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
        
    Returns:
    --------
    Dict[str, float]
        Fractal features
    """
    features = {}
    
    try:
        # Detrended fluctuation analysis (simplified)
        features['dfa_alpha'] = float(_simple_dfa(data))
        
        # Higuchi fractal dimension
        features['higuchi_fd'] = float(_higuchi_fractal_dimension(data))
        
        # Katz fractal dimension
        features['katz_fd'] = float(_katz_fractal_dimension(data))
        
        # Petrosian fractal dimension
        features['petrosian_fd'] = float(_petrosian_fractal_dimension(data))
        
        # Hurst-like features
        features['hurst_rs'] = float(_rs_analysis(data))
        features['hurst_variance'] = float(_variance_method(data))
        
    except Exception as e:
        warnings.warn(f"Fractal feature extraction failed: {e}")
    
    return features


def extract_biomedical_features(data: np.ndarray, sampling_rate: float) -> Dict[str, float]:
    """
    Extract biomedical-specific features from time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    sampling_rate : float
        Sampling rate in Hz
        
    Returns:
    --------
    Dict[str, float]
        Biomedical features
    """
    features = {}
    
    try:
        # Amplitude features
        features['amplitude_range'] = float(np.max(data) - np.min(data))
        features['amplitude_std'] = float(np.std(data))
        features['amplitude_rms'] = float(np.sqrt(np.mean(data**2)))
        
        # Zero crossing rate
        features['zero_crossing_rate'] = float(_zero_crossing_rate(data))
        
        # Peak detection features
        peaks, _ = _find_peaks_simple(data)
        features['peak_count'] = float(len(peaks))
        features['peak_rate'] = float(len(peaks) / (len(data) / sampling_rate))
        
        if len(peaks) > 0:
            features['peak_amplitude_mean'] = float(np.mean(data[peaks]))
            features['peak_amplitude_std'] = float(np.std(data[peaks]))
        
        # Signal quality features
        features['snr_estimate'] = float(_estimate_snr(data))
        features['signal_quality'] = float(_signal_quality_score(data))
        
        # Biomedical frequency bands (assuming EEG-like data)
        if sampling_rate > 0:
            freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
            fft_data = np.abs(np.fft.fft(data))
            
            # Band power ratios
            delta_mask = (freqs >= 0.5) & (freqs <= 4)
            theta_mask = (freqs >= 4) & (freqs <= 8)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            
            delta_power = np.sum(fft_data[delta_mask])
            theta_power = np.sum(fft_data[theta_mask])
            alpha_power = np.sum(fft_data[alpha_mask])
            beta_power = np.sum(fft_data[beta_mask])
            
            total_power = delta_power + theta_power + alpha_power + beta_power
            
            if total_power > 0:
                features['biomedical_delta_ratio'] = float(delta_power / total_power)
                features['biomedical_theta_ratio'] = float(theta_power / total_power)
                features['biomedical_alpha_ratio'] = float(alpha_power / total_power)
                features['biomedical_beta_ratio'] = float(beta_power / total_power)
        
    except Exception as e:
        warnings.warn(f"Biomedical feature extraction failed: {e}")
    
    return features


# Helper functions for feature extraction

def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)

def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 4) - 3

def _autocorrelation(data: np.ndarray, lag: int) -> float:
    """Calculate autocorrelation at given lag."""
    if lag >= len(data):
        return 0.0
    data_centered = data - np.mean(data)
    return np.corrcoef(data_centered[:-lag], data_centered[lag:])[0, 1]

def _trend_slope(data: np.ndarray) -> float:
    """Calculate linear trend slope."""
    x = np.arange(len(data))
    return np.polyfit(x, data, 1)[0]

def _detrended_std(data: np.ndarray) -> float:
    """Calculate standard deviation after detrending."""
    x = np.arange(len(data))
    trend = np.polyval(np.polyfit(x, data, 1), x)
    detrended = data - trend
    return np.std(detrended)

def _adf_statistic(data: np.ndarray) -> float:
    """Simplified ADF statistic."""
    # This is a simplified version - in practice, use scipy.stats
    diff = np.diff(data)
    return np.corrcoef(data[:-1], diff)[0, 1]

def _kpss_statistic(data: np.ndarray) -> float:
    """Simplified KPSS statistic."""
    # This is a simplified version - in practice, use scipy.stats
    cumsum = np.cumsum(data - np.mean(data))
    return np.sum(cumsum**2) / (len(data) * np.var(data))

def _spectral_rolloff(freqs: np.ndarray, psd: np.ndarray, rolloff: float = 0.85) -> float:
    """Calculate spectral rolloff frequency."""
    cumsum_psd = np.cumsum(psd)
    threshold = rolloff * cumsum_psd[-1]
    idx = np.where(cumsum_psd >= threshold)[0]
    return freqs[idx[0]] if len(idx) > 0 else freqs[-1]

def _spectral_flatness(psd: np.ndarray) -> float:
    """Calculate spectral flatness."""
    geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
    arithmetic_mean = np.mean(psd)
    return geometric_mean / arithmetic_mean

def _spectral_entropy(psd: np.ndarray) -> float:
    """Calculate spectral entropy."""
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))

def _wavelet_entropy(coeffs: List[np.ndarray]) -> float:
    """Calculate wavelet entropy."""
    energies = [np.sum(c**2) for c in coeffs]
    total_energy = sum(energies)
    if total_energy == 0:
        return 0.0
    probs = [e / total_energy for e in energies]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def _simple_dfa(data: np.ndarray) -> float:
    """Simplified DFA calculation."""
    n = len(data)
    scales = np.logspace(0.5, np.log10(n//4), 10).astype(int)
    fluctuations = []
    
    for scale in scales:
        # Detrend and calculate fluctuation
        n_windows = n // scale
        if n_windows < 2:
            continue
            
        fluctuations_at_scale = []
        for i in range(n_windows):
            segment = data[i*scale:(i+1)*scale]
            if len(segment) > 1:
                x = np.arange(len(segment))
                trend = np.polyval(np.polyfit(x, segment, 1), x)
                detrended = segment - trend
                fluctuations_at_scale.append(np.mean(detrended**2))
        
        if fluctuations_at_scale:
            fluctuations.append(np.mean(fluctuations_at_scale))
    
    if len(fluctuations) < 2:
        return 0.5
    
    # Linear regression in log space
    log_scales = np.log(scales[:len(fluctuations)])
    log_fluctuations = np.log(fluctuations)
    slope = np.polyfit(log_scales, log_fluctuations, 1)[0]
    return slope / 2

def _higuchi_fractal_dimension(data: np.ndarray) -> float:
    """Calculate Higuchi fractal dimension."""
    n = len(data)
    k_max = min(10, n // 4)
    
    lengths = []
    for k in range(1, k_max + 1):
        l_k = 0
        for m in range(k):
            l_mk = 0
            for i in range(1, (n - m) // k):
                l_mk += abs(data[m + i*k] - data[m + (i-1)*k])
            l_mk = l_mk * (n - 1) / (((n - m) // k) * k)
            l_k += l_mk
        lengths.append(l_k / k)
    
    if len(lengths) < 2:
        return 1.0
    
    # Linear regression
    k_values = np.arange(1, len(lengths) + 1)
    log_k = np.log(k_values)
    log_l = np.log(lengths)
    slope = np.polyfit(log_k, log_l, 1)[0]
    return -slope

def _katz_fractal_dimension(data: np.ndarray) -> float:
    """Calculate Katz fractal dimension."""
    n = len(data)
    if n < 2:
        return 1.0
    
    # Calculate distances
    distances = np.abs(np.diff(data))
    total_distance = np.sum(distances)
    
    if total_distance == 0:
        return 1.0
    
    # Maximum distance from start
    max_distance = np.max(np.abs(data - data[0]))
    
    if max_distance == 0:
        return 1.0
    
    return np.log(n) / (np.log(n) + np.log(max_distance / total_distance))

def _petrosian_fractal_dimension(data: np.ndarray) -> float:
    """Calculate Petrosian fractal dimension."""
    n = len(data)
    if n < 2:
        return 1.0
    
    # Binary sequence based on differences
    diff = np.diff(data)
    binary = (diff > 0).astype(int)
    
    # Count sign changes
    sign_changes = np.sum(np.diff(binary) != 0)
    
    if sign_changes == 0:
        return 1.0
    
    return np.log(n) / (np.log(n) + np.log(n / (n + 0.4 * sign_changes)))

def _rs_analysis(data: np.ndarray) -> float:
    """Simplified R/S analysis."""
    n = len(data)
    if n < 4:
        return 0.5
    
    # Calculate R/S for different scales
    scales = np.logspace(1, np.log10(n//4), 5).astype(int)
    rs_values = []
    
    for scale in scales:
        n_windows = n // scale
        if n_windows < 2:
            continue
            
        rs_window = []
        for i in range(n_windows):
            segment = data[i*scale:(i+1)*scale]
            if len(segment) > 1:
                mean_seg = np.mean(segment)
                cumsum = np.cumsum(segment - mean_seg)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(segment)
                if S > 0:
                    rs_window.append(R / S)
        
        if rs_window:
            rs_values.append(np.mean(rs_window))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Linear regression
    log_scales = np.log(scales[:len(rs_values)])
    log_rs = np.log(rs_values)
    slope = np.polyfit(log_scales, log_rs, 1)[0]
    return slope

def _variance_method(data: np.ndarray) -> float:
    """Variance-based Hurst estimation."""
    n = len(data)
    if n < 4:
        return 0.5
    
    scales = np.logspace(1, np.log10(n//4), 5).astype(int)
    variances = []
    
    for scale in scales:
        n_windows = n // scale
        if n_windows < 2:
            continue
            
        var_window = []
        for i in range(n_windows):
            segment = data[i*scale:(i+1)*scale]
            if len(segment) > 1:
                var_window.append(np.var(segment))
        
        if var_window:
            variances.append(np.mean(var_window))
    
    if len(variances) < 2:
        return 0.5
    
    # Linear regression
    log_scales = np.log(scales[:len(variances)])
    log_vars = np.log(variances)
    slope = np.polyfit(log_scales, log_vars, 1)[0]
    return slope / 2 + 0.5

def _zero_crossing_rate(data: np.ndarray) -> float:
    """Calculate zero crossing rate."""
    if len(data) < 2:
        return 0.0
    sign_changes = np.sum(np.diff(np.sign(data)) != 0)
    return sign_changes / (len(data) - 1)

def _find_peaks_simple(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple peak detection."""
    if len(data) < 3:
        return np.array([]), np.array([])
    
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            peaks.append(i)
    
    return np.array(peaks), np.array([data[i] for i in peaks])

def _estimate_snr(data: np.ndarray) -> float:
    """Estimate signal-to-noise ratio."""
    signal_power = np.var(data)
    # Simple noise estimation using high-frequency components
    if len(data) > 4:
        noise = np.diff(data, 2)  # Second difference as noise proxy
        noise_power = np.var(noise)
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
    return 0.0

def _signal_quality_score(data: np.ndarray) -> float:
    """Calculate signal quality score (0-1)."""
    if len(data) < 2:
        return 0.0
    
    # Based on variance, range, and stationarity
    variance_score = min(1.0, np.var(data) / (np.max(data) - np.min(data))**2)
    range_score = min(1.0, (np.max(data) - np.min(data)) / (np.std(data) * 6))
    
    # Stationarity (simplified)
    n = len(data)
    if n > 4:
        first_half = data[:n//2]
        second_half = data[n//2:]
        stationarity_score = 1.0 - abs(np.mean(first_half) - np.mean(second_half)) / (np.std(data) + 1e-10)
        stationarity_score = max(0.0, min(1.0, stationarity_score))
    else:
        stationarity_score = 0.5
    
    return (variance_score + range_score + stationarity_score) / 3
