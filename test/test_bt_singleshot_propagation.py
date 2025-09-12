import numpy as np

from bt_singleshot_decoder import MetaParityScrubber


class DummyScrubber(MetaParityScrubber):
    """MetaParityScrubber that deterministically flips all bits in target blocks.

    Avoids constructing a real BP/LSD decoder to keep the test lightweight and
    focused on propagation logic across detector slices.
    """

    def __init__(self, mx: int, mz: int, rounds: int) -> None:
        # Bypass heavy decoder construction
        self.mx = int(mx)
        self.mz = int(mz)
        self.rounds = int(rounds)
        self.h_meta = None  # unused
        self._meta_dec = None  # unused

    def _decode_delta(self, y):  # type: ignore[override]
        # Always flip all bits in the provided round vector
        return np.ones_like(y, dtype=np.uint8)


def _layout_lengths(mx: int, mz: int, rounds: int) -> int:
    # Total detectors: z1 (mz) + sum_{t=2..R} (z_diff (mz) + x_diff (mx)) + final_z_vs_data (mz)
    return mz + (rounds - 1) * (mz + mx) + mz


def test_round2_propagates_first_z_to_first_diff_and_final_consistency():
    # Small dimensions
    mx, mz, rounds = 2, 3, 2
    ndet = _layout_lengths(mx, mz, rounds)
    # Start with all zeros
    dets = np.zeros(ndet, dtype=bool)

    # Indices per layout: [z1 | z_diff2 | x_diff2 | final_z]
    z1 = slice(0, mz)
    z_diff2 = slice(mz, mz + mz)
    # x_diff2 occupies next mx, not used here
    final_z = slice(mz + mz + mx, mz + mz + mx + mz)

    scrubber = DummyScrubber(mx=mx, mz=mz, rounds=rounds)
    out = scrubber.scrub(dets)

    # z1 was flipped by dummy scrubber
    assert np.all(out[z1] == True)
    # Propagation should flip the first z-diff block by the same delta then scrub flips again => back to zeros
    assert np.all(out[z_diff2] == False)
    # The cumulative change to z_R equals z1_delta XOR z_diff2_delta; here both are ones -> cancels, so final_z unchanged
    assert np.all(out[final_z] == False)


def test_round1_propagates_to_final_only():
    mx, mz, rounds = 2, 3, 1
    ndet = _layout_lengths(mx, mz, rounds)
    dets = np.zeros(ndet, dtype=bool)

    z1 = slice(0, mz)
    final_z = slice(mz, mz + mz)

    scrubber = DummyScrubber(mx=mx, mz=mz, rounds=rounds)
    out = scrubber.scrub(dets)

    # z1 flipped
    assert np.all(out[z1] == True)
    # With no z-diff blocks, z1 delta should be mirrored into final_z
    assert np.all(out[final_z] == True)
