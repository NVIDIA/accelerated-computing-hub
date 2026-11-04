# SPDX-License-Identifier: Apache-2.0
"""The CUDA sources of notebook 94, read from the files beside this package."""

from .config import K_BINS, ROOT

ECC_SOURCE           = ROOT / "cosmic_ecc.cu"
ECC_VARIANTS_SOURCE  = ROOT / "cosmic_ecc_variants.cu"
CUB_HISTOGRAM_SOURCE = ROOT / "cosmic_cub_histogram.cu"


def _checked(path, constant):
    """The file's text, after checking that its K_BINS constant agrees with cosmic_core.K_BINS."""
    src = path.read_text()
    if f"constexpr int {constant} = {K_BINS};" not in src:
        raise RuntimeError(f"{path.name} and cosmic_core.K_BINS disagree")
    return src


def ecc_cuda_source(body):
    """cosmic_ecc.cu as a string, with `body` spliced between the BEGIN BODY and END BODY markers."""
    src = _checked(ECC_SOURCE, "K_BINS")
    begin, end = "        // BEGIN BODY\n", "        // END BODY\n"
    head, rest = src.split(begin, 1)
    _, tail = rest.split(end, 1)
    return head + begin + body.strip("\n") + "\n" + end + tail


def ecc_cuda_variants_source():
    """cosmic_ecc_variants.cu as a string: the rows kernel, the same body on a 3D grid, several voxels per thread."""
    return _checked(ECC_VARIANTS_SOURCE, "K_BINS_V")


def cub_histogram_source():
    """cosmic_cub_histogram.cu as a string: CUB's device histogram behind a host function that takes device pointers."""
    return CUB_HISTOGRAM_SOURCE.read_text()
