from improc.roi import padded_crop

import pytest
import numpy as np

img = np.arange(100).reshape(10, 10)


@pytest.mark.parametrize("roi,expected", [
    ((slice(-2, 12), slice(
        0, 1)), [20, 10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 80, 70]),
    ((slice(0, 1), slice(
        -3, 13)), [3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6]),
])
def test_padded_crop(roi, expected):

    cropped_img = padded_crop(img, roi).squeeze()

    assert (cropped_img == expected).all()
