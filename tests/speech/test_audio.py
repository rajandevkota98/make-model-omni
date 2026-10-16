import numpy as np

from src.utils.audio import babble, colored_noise, limit_peak, mix_at_snr, pad, resample, snr_db, to_int16


def tone(seconds: float = 1.0, rate: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_resample_24k_to_16k_length():
    assert len(resample(np.zeros(24000, np.float32), 24000, 16000)) == 16000


def test_mix_hits_requested_snr():
    speech = tone()
    noisy = mix_at_snr(speech, colored_noise(len(speech), "pink", np.random.default_rng(0)), 10.0)
    assert abs(snr_db(speech, noisy) - 10.0) < 0.01


def test_snr_measured_on_speech_not_padding():
    speech = tone()
    padded = pad(speech, 500, 500, 16000)
    noisy = mix_at_snr(padded, colored_noise(len(padded), "white", np.random.default_rng(1)), 20.0, reference=speech)
    noise_power = np.mean((noisy - padded) ** 2)
    assert abs(10 * np.log10(np.mean(speech ** 2) / noise_power) - 20.0) < 0.05


def test_noise_colours_have_falling_spectrum():
    rng = np.random.default_rng(2)
    def low_high(kind):
        spectrum = np.abs(np.fft.rfft(colored_noise(32000, kind, rng))) ** 2
        return spectrum[10:200].mean() / spectrum[-2000:].mean()
    assert low_high("white") < low_high("pink") < low_high("brown")


def test_babble_length_and_peak_limit():
    mixed = babble([tone(0.3), tone(0.5)], 16000, np.random.default_rng(3))
    assert len(mixed) == 16000
    assert np.max(np.abs(limit_peak(mixed * 10, 0.95))) <= 0.95 + 1e-6


def test_int16_clips_instead_of_wrapping():
    assert to_int16(np.array([2.0, -2.0], np.float32)).tolist() == [32767, -32767]
