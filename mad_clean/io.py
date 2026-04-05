"""
mad_clean.io
============
FITS and numpy I/O helpers. No torch dependency — safe to import anywhere.

Functions
---------
load_image(source) -> np.ndarray
    Load a 2D float32 image from a FITS file or numpy array.

save_fits(array, path, header=None)
    Save a 2D numpy array as a FITS file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


__all__ = ["load_image", "save_fits"]


def load_image(source) -> np.ndarray:
    """
    Load a 2D image as float32.

    Parameters
    ----------
    source : np.ndarray  — returned as float32, no copy unless dtype differs
             str | Path  — path to a FITS file

    FITS handling
    -------------
    Reads the primary HDU. Squeezes degenerate axes so that CASA output
    shapes like (1, 1, H, W) become (H, W). Raises ValueError if the result
    is not exactly 2D after squeezing.
    """
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)

    from astropy.io import fits

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"Primary HDU of {path} contains no data.")
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D image after squeezing degenerate axes, "
                f"got shape {data.shape} from {path}. "
                f"Provide a single 2D plane explicitly."
            )
        header = hdul[0].header

    return data.astype(np.float32), header


def load_image_data(source) -> np.ndarray:
    """
    Convenience wrapper: returns only the pixel array (discards header).
    Use load_image() directly when you need to preserve WCS for save_fits().
    """
    result = load_image(source)
    if isinstance(result, tuple):
        return result[0]
    return result


def save_fits(array: np.ndarray, path, header=None) -> None:
    """
    Save a 2D numpy array as a FITS file.

    Parameters
    ----------
    array  : np.ndarray (H, W) — will be cast to float32
    path   : str | Path
    header : astropy.io.fits.Header, optional
             Pass the header from load_image() to preserve WCS metadata.
    """
    from astropy.io import fits

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=array.astype(np.float32), header=header)
    hdu.writeto(path, overwrite=True)
