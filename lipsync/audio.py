import librosa
import librosa.filters
import numpy as np
from scipy import signal
from hparams import hparams as hp

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def inv_linear_spectrogram(linear_spectrogram):
    """Converts linear spectrogram to waveform using librosa"""
    if hp.signal_normalization:
        D = _denormalize(linear_spectrogram)
    else:
        D = linear_spectrogram
    
    S = _db_to_amp(D + hp.ref_level_db) #Convert back to linear
    
    if hp.use_lws:
        processor = _lws_processor(hp)
        D = processor.run_lws(S.astype(np.float64).T ** hp.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hp.preemphasis, hp.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hp.power), hp.preemphasis, hp.preemphasize)

def inv_mel_spectrogram(mel_spectrogram):
    """Converts mel spectrogram to waveform using librosa"""
    if hp.signal_normalization:
        D = _denormalize(mel_spectrogram)
    else:
        D = mel_spectrogram
    
    S = _mel_to_linear(_db_to_amp(D + hp.ref_level_db))  # Convert back to linear
    
    if hp.use_lws:
        processor = _lws_processor(hp)
        D = processor.run_lws(S.astype(np.float64).T ** hp.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hp.preemphasis, hp.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hp.power), hp.preemphasis, hp.preemphasize)

def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")

def _griffin_lim(S):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hp.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size)

def _istft(y):
    return librosa.istft(y, hop_length=get_hop_size(), win_length=hp.win_size)

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
