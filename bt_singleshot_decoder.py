"""BT DEM helpers: meta-parity scrubber that operates in detector space.

Stage 1 (preprocess): enforce meta Z parity on per-cycle Z detector blocks
by flipping a minimal set of detector bits using a small BP+LSD decoder over
``H_meta``. Stage 2: run the standard DEM-based decoder (unchanged).
"""

from dataclasses import dataclass
from typing import Any, Optional, List, Protocol

import numpy as np
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csr_matrix


class BTCode(Protocol):
    """Protocol for BT codes with meta check capability."""
    hx: Any
    hz: Any
    lz: Any
    h_meta: Any
    N: int


# ------------------------------
# DEM-centric Stage-1 scrubber
# ------------------------------

def infer_rounds(total_detectors: int, mx: int, mz: int) -> Optional[int]:
    """Infer number of rounds from detector layout.

    Layout (from generate_full_circuit):
    D = mz + (R - 1) * (mz + mx) + mz  =>  R = 1 + (D - 2*mz) / (mz + mx).
    """
    denom = mx + mz
    if denom <= 0:
        return None
    num = total_detectors - 2 * mz
    if num < 0 or (num % denom) != 0:
        return None
    rounds = 1 + (num // denom)
    return rounds if rounds >= 1 else None


@dataclass(frozen=True)
class DetectorLayout:
    z_round1: slice
    z_diff_blocks: List[slice]
    x_diff_blocks: List[slice]
    final_z_vs_data: slice

    @staticmethod
    def build(mx: int, mz: int, rounds: int, total_detectors: int) -> Optional["DetectorLayout"]:
        expected = mz + (rounds - 1) * (mz + mx) + mz
        if expected != total_detectors or mz <= 0 or mx < 0 or rounds < 1:
            return None
        idx = 0
        z_round1 = slice(idx, idx + mz)
        idx += mz
        z_diff_blocks: List[slice] = []
        x_diff_blocks: List[slice] = []
        for _ in range(max(0, rounds - 1)):
            z_blk = slice(idx, idx + mz)
            idx += mz
            x_blk = slice(idx, idx + mx)
            idx += mx
            z_diff_blocks.append(z_blk)
            x_diff_blocks.append(x_blk)
        final_z_vs_data = slice(idx, idx + mz)
        return DetectorLayout(z_round1, z_diff_blocks, x_diff_blocks, final_z_vs_data)


class MetaParityScrubber:
    """Stage-1 meta-parity scrubber operating purely in detector space.

    For each cycle:
    - t=1: enforce H_meta @ det[z_round1] == 0 (flip minimal set of Z-detector bits)
    - t>=2: enforce H_meta @ det[z_diff_t] == 0 for each REPEAT cycle

    After scrubbing, the corrected detector vector can be passed to a DEM
    decoder (Stage 2).
    """

    def __init__(
        self,
        h_meta: Any,
        mx: int,
        mz: int,
        rounds: int,
        *,
        p_spam: float = 0.01,
        bp_iters: int = 20,
        osd_order: int = 0,
        s_max_frac: Optional[float] = 0.15,
        round_mode: str = "all",
    ) -> None:
        self.h_meta = csr_matrix(h_meta)
        self.mx = int(mx)
        self.mz = int(mz)
        self.rounds = int(rounds)
        # Conservative prior for measurement flips in meta scrubber.
        # Cap at 2% by default to avoid over-scrubbing when data/gate faults dominate.
        p_meas = max(1e-6, min(0.02, float(p_spam)))
        self._s_max_frac = float(s_max_frac) if (s_max_frac is not None) else None
        self._round_mode = str(round_mode).lower() if round_mode else "all"
        # Use BP+LSD for stage-1 (meta Z) decoding. Reuse osd_order as lsd_order.
        self._meta_dec = BpLsdDecoder(
            self.h_meta,
            error_channel=[p_meas] * self.h_meta.shape[1],
            max_iter=bp_iters,
            bp_method="minimum_sum",
            ms_scaling_factor=0.625,
            schedule="parallel",
            # Map "osd_order" knob to LSD analogue for convenience
            lsd_order=int(osd_order) if osd_order is not None else 0,
            lsd_method="LSD_E" if (osd_order and osd_order > 0) else "LSD_0",
        )

    def _scrub_one_block(self, dets: np.ndarray, block: slice) -> None:
        start, stop = block.start or 0, block.stop or 0
        if (stop - start) != self.mz:
            return
        y = dets[start:stop]
        flips = self._decode_delta(y)
        if flips is None or not np.any(flips):
            return
        dets[start:stop] = np.logical_xor(dets[start:stop], flips.astype(bool))

    def _decode_delta(self, y: np.ndarray) -> np.ndarray:
        """Return a length-mz bit vector of flips to enforce H_meta @ (y ⊕ flips) = 0.

        Subclasses may override to inject custom behavior for testing.
        """
        s_vec = np.asarray((self.h_meta @ y) % 2, dtype=np.uint8).ravel()
        if not np.any(s_vec):
            return np.zeros(self.mz, dtype=np.uint8)
        # Guard: skip scrubbing if meta-syndrome weight is implausibly large.
        if self._s_max_frac is not None:
            if (np.count_nonzero(s_vec) / max(1, s_vec.size)) > self._s_max_frac:
                return np.zeros(self.mz, dtype=np.uint8)
        flips = np.asarray(self._meta_dec.decode(s_vec), dtype=np.uint8).ravel()[: self.mz]
        return flips

    def scrub(self, dets: np.ndarray, total_detectors: Optional[int] = None) -> np.ndarray:
        """Return a corrected copy of the detector vector."""
        ndet = int(total_detectors if total_detectors is not None else dets.shape[0])
        rounds = infer_rounds(ndet, self.mx, self.mz)
        layout = DetectorLayout.build(self.mx, self.mz, rounds, ndet) if rounds else None
        if layout is None:
            return dets.copy()
        out = dets.copy()
        # Track cumulative changes that affect the final Z-vs-data detectors.
        # Any net change to the last round's Z outcomes must be mirrored into
        # the final Z-vs-data detector block to maintain internal consistency.
        cumulative_zR_delta: Optional[np.ndarray] = None

        # Reconstruct explicit per-round Z outcomes y_t from z1 and z-diff blocks.
        z1 = out[layout.z_round1].copy()
        y_rounds: list[np.ndarray] = [z1]
        # Capture original last-round Z for later delta update
        if len(layout.z_diff_blocks) == 0:
            yR_orig = z1.copy()
        else:
            acc = z1.copy()
            for blk in layout.z_diff_blocks:
                d = out[blk]
                acc = np.logical_xor(acc, d)
                y_rounds.append(acc.copy())
            yR_orig = acc.copy()

        # Scrub each per-round Z outcome using H_meta
        y_rounds_scrubbed: list[np.ndarray] = []
        for idx, y in enumerate(y_rounds):
            if self._round_mode == "last" and idx != (len(y_rounds) - 1):
                y_rounds_scrubbed.append(y.copy())
                continue
            flips = self._decode_delta(y.astype(np.uint8))
            if flips is None or not np.any(flips):
                y_rounds_scrubbed.append(y.copy())
            else:
                y_rounds_scrubbed.append(np.logical_xor(y, flips.astype(bool)))

        # Write back: z1 block
        out[layout.z_round1] = y_rounds_scrubbed[0]

        # Update z-diff blocks from consecutive y's
        if len(layout.z_diff_blocks) > 0:
            for idx, blk in enumerate(layout.z_diff_blocks, start=1):
                d_new = np.logical_xor(y_rounds_scrubbed[idx], y_rounds_scrubbed[idx - 1])
                out[blk] = d_new

        # Mirror net change in last-round Z into final Z-vs-data
        yR_new = y_rounds_scrubbed[-1]
        delta_R = np.logical_xor(yR_orig, yR_new)
        if np.any(delta_R):
            out[layout.final_z_vs_data] = np.logical_xor(out[layout.final_z_vs_data], delta_R)
        return out


class _WrappedDemDecoder:
    """Wrap a DEM decoder with a meta-parity scrubber over detector bits."""

    def __init__(self, base_decoder: Any, scrubber: Optional[MetaParityScrubber], dem_num_detectors: int) -> None:
        self._base = base_decoder
        self._scrubber = scrubber
        self._ndet = int(dem_num_detectors)

    def decode(self, dets: np.ndarray) -> np.ndarray:
        cleaned = self._scrubber.scrub(dets, total_detectors=self._ndet) if self._scrubber else dets
        return self._base.decode(cleaned)


def build_dem_decoder_with_meta_scrub(
    *,
    base_decoder: Any,
    dem: Any,
    code: BTCode,
    p_spam: float,
    bp_iters: int,
    osd_order: int,
) -> _WrappedDemDecoder:
    """Return a decoder that first scrubs meta parity per cycle, then runs DEM decode.

    Args:
        base_decoder: Existing DEM-based BpOsdDecoder
        dem: Detector error model to determine number of detectors
        code: BT code to access H_meta and shapes
        p_spam: Approximate measurement error rate for meta parity decoding
        bp_iters: BP iterations for meta scrubber's small decoder
        osd_order: OSD order for meta scrubber's small decoder
    """
    mz, _ = csr_matrix(code.hz).shape
    mx = csr_matrix(code.hx).shape[0]
    ndet = int(getattr(dem, "num_detectors", 0))
    rounds = infer_rounds(ndet, mx, mz)
    scrubber = None if (rounds is None or getattr(code, "h_meta", None) is None) else MetaParityScrubber(
        code.h_meta, mx=mx, mz=mz, rounds=rounds, p_spam=p_spam, bp_iters=bp_iters, osd_order=osd_order
    )
    return _WrappedDemDecoder(base_decoder=base_decoder, scrubber=scrubber, dem_num_detectors=ndet)


    


    


    


    


    
