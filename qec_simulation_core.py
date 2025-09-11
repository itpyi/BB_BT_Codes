"""Core utilities for quantum error correction code simulations.

Provides common data structures, code/decoder builders, CSV helpers, and
plotting functionality. Supports BB (Bivariate Bicycle), BT (Bivariate Tricycle),
TT (Trivariate Tricycle) and other quantum LDPC codes.
"""

from __future__ import annotations

import os
import json
import csv
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any, Protocol, cast

import numpy as np
import stim
from matplotlib import pyplot as plt

from bposd.css import css_code
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
import time
import logging
from distance_estimator import get_min_logical_weight
from bt_singleshot_decoder import build_dem_decoder_with_meta_scrub


class CompositeCSSDecoder:
    """CSS decoder that splits X/Z sectors and decodes them independently.
    
    For CSS codes, X errors only trigger Z stabilizers and Z errors only trigger X stabilizers.
    This decoder splits the detector error model into separate X and Z sectors and builds
    independent decoders for each sector, then combines their results.
    """
    
    def __init__(
        self,
        x_decoder: BpOsdDecoder,
        z_decoder: BpOsdDecoder,
        x_observables: np.ndarray,
        z_observables: np.ndarray,
        x_detector_indices: np.ndarray,
        z_detector_indices: np.ndarray,
        unified_error_space_size: int,
    ):
        """Initialize composite CSS decoder.
        
        Args:
            x_decoder: Decoder for X sector (decodes X errors from Z stabilizer violations)
            z_decoder: Decoder for Z sector (decodes Z errors from X stabilizer violations)
            x_observables: Observable matrix for X sector
            z_observables: Observable matrix for Z sector
            x_detector_indices: Indices of X-sector detectors in full syndrome
            z_detector_indices: Indices of Z-sector detectors in full syndrome
            unified_error_space_size: Size of the unified error space (for correction vectors)
        """
        self.x_decoder = x_decoder
        self.z_decoder = z_decoder
        self.x_observables = x_observables
        self.z_observables = z_observables
        self.x_detector_indices = x_detector_indices
        self.z_detector_indices = z_detector_indices
        self.unified_error_space_size = unified_error_space_size
    
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode syndrome by splitting into X/Z sectors.
        
        Args:
            syndrome: Full syndrome vector
            
        Returns:
            Combined correction vector matching the original unified DEM structure
        """
        # Extract sector-specific syndromes
        x_syndrome = syndrome[self.x_detector_indices] if len(self.x_detector_indices) > 0 else np.array([])
        z_syndrome = syndrome[self.z_detector_indices] if len(self.z_detector_indices) > 0 else np.array([])
        
        # Initialize correction vector with proper size
        combined_correction = np.zeros(self.unified_error_space_size, dtype=int)
        
        # Decode X sector (responds to Z stabilizer violations)
        if len(x_syndrome) > 0:
            x_correction = self.x_decoder.decode(x_syndrome)
            # X decoder correction should already be in the unified error space
            # XOR with combined correction (CSS property: X and Z errors are independent)
            if len(x_correction) == self.unified_error_space_size:
                combined_correction ^= x_correction
            else:
                # This should not happen with proper sector splitting, but handle gracefully
                logging.warning(f"X correction size mismatch: expected {self.unified_error_space_size}, got {len(x_correction)}")
                min_len = min(len(x_correction), self.unified_error_space_size)
                combined_correction[:min_len] ^= x_correction[:min_len]
        
        # Decode Z sector (responds to X stabilizer violations)  
        if len(z_syndrome) > 0:
            z_correction = self.z_decoder.decode(z_syndrome)
            # Z decoder correction should already be in the unified error space
            # XOR with combined correction (CSS property: X and Z errors are independent)
            if len(z_correction) == self.unified_error_space_size:
                combined_correction ^= z_correction
            else:
                # This should not happen with proper sector splitting, but handle gracefully
                logging.warning(f"Z correction size mismatch: expected {self.unified_error_space_size}, got {len(z_correction)}")
                min_len = min(len(z_correction), self.unified_error_space_size)
                combined_correction[:min_len] ^= z_correction[:min_len]
        
        return combined_correction
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the X decoder for compatibility."""
        # For the 'n' attribute (code length), return the unified error space size
        if name == 'n':
            return self.unified_error_space_size
        # For other attributes, delegate to X decoder
        return getattr(self.x_decoder, name)


@dataclass
class ResultPoint:
    decoder: str
    l: int
    m: int
    rounds: int
    p: float
    shots: int
    errors: int
    seconds: float
    # Optional descriptors for plotting/naming
    code_type: str = "BB"
    n: int = -1
    # Optional code parameters (when available from metadata)
    K: int = -1  # number of logical operators
    N: int = -1  # number of physical qubits
    D: int = -1  # code distance (min of X/Z)

    @property
    def ler(self) -> float:
        return self.errors / max(1, self.shots)


def _estimate_distances(
    code: css_code,
    *,
    p: float = 0.01,
    bp_iters: int = 20,
    osd_order: int = 0,
    iters: int = 2000,
) -> tuple[int, int]:
    """Empirically estimate X/Z distances using BP+OSD sampling.

    Returns (d_x_est, d_z_est). Uses i.i.d. error rate p and runs `iters`
    samples for each Pauli type. Keeps defaults small for quick feedback.
    """
    try:
        dx = get_min_logical_weight(code, p=p, pars=[bp_iters, osd_order], iters=iters, Ptype=0)
        dz = get_min_logical_weight(code, p=p, pars=[bp_iters, osd_order], iters=iters, Ptype=1)
        return int(dx), int(dz)
    except Exception as e:
        logging.warning("[DIST] distance estimation failed: %s", str(e).splitlines()[0] if str(e) else "Exception")
        return -1, -1


def build_bb_code(
    a_poly: list,
    b_poly: list,
    l: int,
    m: int,
    *,
    estimate_distance: bool = True,
    est_p: float = 0.01,
    est_bp_iters: int = 20,
    est_osd_order: int = 0,
    est_iters: int = 200,
) -> css_code:
    from bivariate_bicycle_codes import get_BB_Hx_Hz  # local import to avoid cycles

    Hx, Hz = get_BB_Hx_Hz(a_poly, b_poly, l, m)
    code = css_code(hx=Hx, hz=Hz, name=f"BB_{l}x{m}")
    code.test()
    if estimate_distance:
        dx, dz = _estimate_distances(
            code, p=est_p, bp_iters=est_bp_iters, osd_order=est_osd_order, iters=est_iters
        )
        # Attach as optional metadata on the code object
        setattr(code, "estimated_distance_x", int(dx))
        setattr(code, "estimated_distance_z", int(dz))
        try:
            candidates = [d for d in (dx, dz) if isinstance(d, (int, float)) and d > 0]
            code.D = int(min(candidates)) if candidates else -1
        except Exception:
            setattr(code, "D", -1)
        logging.info("[DIST] BB_%dx%d estimated distances: dx=%s, dz=%s", l, m, dx, dz)
    return code


def build_bt_code(
    a_poly: list,
    b_poly: list,
    c_poly: list,
    l: int,
    m: int,
    *,
    estimate_distance: bool = True,
    est_p: float = 0.01,
    est_bp_iters: int = 20,
    est_osd_order: int = 0,
    est_iters: int = 200,
) -> css_code:
    from bivariate_tricycle_codes import get_BT_Hx_Hz, get_BT_Hmeta  # local import to avoid cycles

    Hx, Hz = get_BT_Hx_Hz(a_poly, b_poly, c_poly, l, m)
    code = css_code(hx=Hx, hz=Hz, name=f"BT_{l}x{m}")
    
    # Add meta check matrix for single shot decoding
    H_meta = get_BT_Hmeta(a_poly, b_poly, c_poly, l, m)
    code.h_meta = H_meta
    
    code.test()
    if estimate_distance:
        dx, dz = _estimate_distances(
            code, p=est_p, bp_iters=est_bp_iters, osd_order=est_osd_order, iters=est_iters
        )
        setattr(code, "estimated_distance_x", int(dx))
        setattr(code, "estimated_distance_z", int(dz))
        try:
            candidates = [d for d in (dx, dz) if isinstance(d, (int, float)) and d > 0]
            code.D = int(min(candidates)) if candidates else -1
        except Exception:
            setattr(code, "D", -1)
        logging.info("[DIST] BT_%dx%d estimated distances: dx=%s, dz=%s", l, m, dx, dz)
    return code


def build_tt_code(
    a_poly: list,
    b_poly: list,
    c_poly: list,
    l: int,
    m: int,
    n: int,
    *,
    estimate_distance: bool = True,
    est_p: float = 0.01,
    est_bp_iters: int = 20,
    est_osd_order: int = 0,
    est_iters: int = 200,
) -> css_code:
    from trivariate_tricycle_codes import get_TT_Hx_Hz  # local import to avoid cycles

    Hx, Hz = get_TT_Hx_Hz(a_poly, b_poly, c_poly, l, m, n)
    code = css_code(hx=Hx, hz=Hz, name=f"TT_{l}x{m}x{n}")
    code.test()
    if estimate_distance:
        dx, dz = _estimate_distances(
            code, p=est_p, bp_iters=est_bp_iters, osd_order=est_osd_order, iters=est_iters
        )
        setattr(code, "estimated_distance_x", int(dx))
        setattr(code, "estimated_distance_z", int(dz))
        try:
            candidates = [d for d in (dx, dz) if isinstance(d, (int, float)) and d > 0]
            code.D = int(min(candidates)) if candidates else -1
        except Exception:
            setattr(code, "D", -1)
        logging.info("[DIST] TT_%dx%dx%d estimated distances: dx=%s, dz=%s", l, m, n, dx, dz)
    return code


def build_code_generic(code_type: str, **params) -> css_code:
    """Generic code builder dispatcher.
    
    Args:
        code_type: "BB", "BT", or "TT"
        **params: Code-specific parameters
        
    Returns:
        Constructed CSS code
        
    Raises:
        ValueError: If code_type is unknown or required parameters are missing
    """
    code_type = code_type.upper()
    
    if code_type == "BB":
        required = {"a_poly", "b_poly", "l", "m"}
        if not required.issubset(params.keys()):
            raise ValueError(f"BB code requires parameters: {required}")
        return build_bb_code(params["a_poly"], params["b_poly"], params["l"], params["m"])
    
    elif code_type == "BT":
        required = {"a_poly", "b_poly", "c_poly", "l", "m"}
        if not required.issubset(params.keys()):
            raise ValueError(f"BT code requires parameters: {required}")
        return build_bt_code(params["a_poly"], params["b_poly"], params["c_poly"], params["l"], params["m"])
    
    elif code_type == "TT":
        required = {"a_poly", "b_poly", "c_poly", "l", "m", "n"}
        if not required.issubset(params.keys()):
            raise ValueError(f"TT code requires parameters: {required}")
        return build_tt_code(params["a_poly"], params["b_poly"], params["c_poly"], params["l"], params["m"], params["n"])
    
    else:
        raise ValueError(f"Unknown code type: {code_type}. Supported types: BB, BT, TT")


def generate_default_resume_csv(code_type: str, output_dir: str, runner_type: str, **params) -> str:
    """Generate default resume CSV filename based on code type and parameters.
    
    Args:
        code_type: "BB", "BT", or "TT"
        output_dir: Output directory
        runner_type: "serial" or "mp" 
        **params: Code parameters (l, m, n, etc.)
        
    Returns:
        Default resume CSV path
    """
    code_type = code_type.lower()

    # Optional suffix to distinguish multiple experiments
    suffix = params.pop("suffix", None)
    suffix_part = f"_{suffix}" if suffix else ""

    if code_type == "bb":
        return f"{output_dir}/bb_{params['l']}_{params['m']}_{runner_type}_resume{suffix_part}.csv"
    elif code_type == "bt":
        return f"{output_dir}/bt_{params['l']}_{params['m']}_{runner_type}_resume{suffix_part}.csv"
    elif code_type == "tt":
        n_dim = params.get('n', None)
        if n_dim is not None:
            return f"{output_dir}/tt_{params['l']}_{params['m']}_{int(n_dim)}_{runner_type}_resume{suffix_part}.csv"
        return f"{output_dir}/tt_{params['l']}_{params['m']}_{runner_type}_resume{suffix_part}.csv"
    else:
        raise ValueError(f"Unknown code type: {code_type}")


def extract_code_params_from_config(config: dict) -> Tuple[str, dict]:
    """Extract code type and parameters from configuration.
    
    Returns:
        Tuple of (code_type, code_params_dict)
    """
    code_type = config.get('code_type', 'BB').upper()
    
    code_params = {
        'a_poly': config['a_poly'],
        'b_poly': config['b_poly'],
        'l': config['l'],
        'm': config['m'],
    }
    
    if code_type in ['BT', 'TT']:
        code_params['c_poly'] = config['c_poly']
    
    if code_type == 'TT':
        code_params['n'] = config['n']
    
    # Include meta_check parameter if present in config
    if 'meta_check' in config:
        code_params['meta_check'] = config['meta_check']
        
    return code_type, code_params


def _separate_css_sectors(
    dem: stim.DetectorErrorModel,
    code: Optional[Any] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Separate detector error model into X and Z sectors for CSS codes.
    
    For CSS codes:
    - X errors only affect Z stabilizers → X-sector decoder handles X errors using Z-detector syndromes
    - Z errors only affect X stabilizers → Z-sector decoder handles Z errors using X-detector syndromes
    
    This requires analyzing the DEM structure to identify which error mechanisms
    affect which detectors, rather than using position-based heuristics.
    
    Args:
        dem: Stim detector error model
        code: Optional CSS code object for additional structure info
        
    Returns:
        Tuple of (x_error_indices, z_error_indices, x_detector_indices, z_detector_indices) 
        or None if separation fails
    """
    try:
        # Convert DEM to matrices to analyze structure
        from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
        mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        
        check_matrix = mats.check_matrix  # Shape: (num_detectors, num_error_mechanisms)
        num_detectors, _num_errors = check_matrix.shape
        
        # Require a CSS code with known mx, mz
        if code is None or getattr(code, 'hx', None) is None or getattr(code, 'hz', None) is None:
            logging.warning("CSS sector separation requested without valid CSS code; falling back")
            return None
        mx = int(getattr(code, 'hx').shape[0])
        mz = int(getattr(code, 'hz').shape[0])
        if mx <= 0 or mz <= 0:
            logging.warning("CSS sector separation requires mx>0 and mz>0; falling back")
            return None
        
        # Infer number of rounds from detector count, assuming circuits built by generate_full_circuit.
        # Formula: D = (R-1)*mx + (R+1)*mz = R*(mx+mz) + (mz - mx)
        D = int(num_detectors)
        denom = mx + mz
        numer = D - (mz - mx)
        if denom <= 0 or numer < denom:
            logging.warning("CSS sector separation unable to infer rounds; falling back")
            return None
        if numer % denom != 0:
            logging.warning("CSS sector separation: detector layout not matching expected pattern; falling back")
            return None
        R = numer // denom
        if R < 1:
            logging.warning("CSS sector separation: invalid inferred rounds; falling back")
            return None
        
        # Compute detector index ranges based on construction order in generate_full_circuit:
        # [Z_init (mz)] + (R-1) * [Z_block (mz), X_block (mx)] + [Z_final (mz)]
        z_indices = []
        x_indices = []
        # Initial Z
        start = 0
        z_indices.extend(range(start, start + mz))
        # Repeated blocks
        for t in range(R - 1):
            base = mz + t * (mz + mx)
            z_indices.extend(range(base, base + mz))
            x_indices.extend(range(base + mz, base + mz + mx))
        # Final Z detectors
        final_base = mz + (R - 1) * (mz + mx)
        z_indices.extend(range(final_base, final_base + mz))

        # Important: X errors flip Z stabilizers → X-sector uses Z detectors.
        #            Z errors flip X stabilizers → Z-sector uses X detectors.
        x_detector_indices = np.array(z_indices, dtype=int)
        z_detector_indices = np.array(x_indices, dtype=int)
        # Column split not used; return placeholders to satisfy signature
        x_error_indices = np.array([], dtype=int)
        z_error_indices = np.array([], dtype=int)
        
        logging.info(f"[CSS] Sector separation: R={R}, mx={mx}, mz={mz}, X dets={len(x_detector_indices)}, Z dets={len(z_detector_indices)}")
        return x_error_indices, z_error_indices, x_detector_indices, z_detector_indices
        
        # TODO: Implement proper CSS sector separation by:
        # 1. Analyzing the DEM instruction stream to identify X vs Z error types
        # 2. Tracking which detectors are affected by each error type
        # 3. Building proper sector-specific check matrices
        # 4. This requires deep integration with Stim's DEM structure
        
    except Exception as e:
        logging.warning(f"CSS sector separation failed: {e}")
        return None


def _build_css_split_decoder(
    circuit: stim.Circuit,
    *,
    bp_iters: int,
    osd_order: int,
    decompose_dem: Optional[bool] = None,
    code: Optional[Any] = None,
    p: Optional[float] = None,
) -> Optional[Tuple[CompositeCSSDecoder, np.ndarray]]:
    """Build CSS decoder with X/Z sector splitting.
    
    Args:
        circuit: Stim circuit
        bp_iters: BP iteration count
        osd_order: OSD order
        decompose_dem: Whether to decompose DEM
        code: CSS code object
        p: Error probability
        
    Returns:
        Tuple of (composite_decoder, observables_matrix) or None if splitting fails
    """
    try:
        # Build detector error model
        if decompose_dem is None:
            env = os.getenv("QEC_DEM_DECOMPOSE")
            decompose = bool(env and env not in ("0", "false", "False"))
        else:
            decompose = bool(decompose_dem)
            
        dem = (
            circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
            if decompose
            else circuit.detector_error_model(decompose_errors=False)
        )
        
        # Separate into X/Z sectors
        sector_split = _separate_css_sectors(dem, code)
        if sector_split is None:
            return None
        x_error_indices, z_error_indices, x_detector_indices, z_detector_indices = sector_split
        
        # Convert full DEM to matrices
        mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
        full_check_matrix = mats.check_matrix
        full_priors = mats.priors
        full_observables = mats.observables_matrix
        
        # Extract sector-specific matrices
        if len(x_detector_indices) > 0:
            x_check_matrix = full_check_matrix[x_detector_indices, :]
            x_priors = list(full_priors)  # Use same priors for now
        else:
            x_check_matrix = np.empty((0, full_check_matrix.shape[1]))
            x_priors = []
            
        if len(z_detector_indices) > 0:
            z_check_matrix = full_check_matrix[z_detector_indices, :]
            z_priors = list(full_priors)  # Use same priors for now
        else:
            z_check_matrix = np.empty((0, full_check_matrix.shape[1]))
            z_priors = []
        
        # Build sector decoders
        osd_method = "osd_e" if (osd_order and osd_order > 0) else "osd0"
        
        x_decoder = None
        z_decoder = None
        
        if x_check_matrix.shape[0] > 0:
            x_decoder = BpOsdDecoder(
                x_check_matrix,
                error_channel=x_priors,
                max_iter=bp_iters,
                bp_method="ms",
                ms_scaling_factor=0.625,
                schedule="parallel",
                osd_method=osd_method,
                osd_order=osd_order,
            )
        
        if z_check_matrix.shape[0] > 0:
            z_decoder = BpOsdDecoder(
                z_check_matrix,
                error_channel=z_priors,
                max_iter=bp_iters,
                bp_method="ms",
                ms_scaling_factor=0.625,
                schedule="parallel",
                osd_method=osd_method,
                osd_order=osd_order,
            )
        
        if x_decoder is None:
            return None
            
        # For CSS codes, the observables matrix represents logical operator measurements
        # in the unified error space. We don't need to split the observables since
        # the correction vector will be in the unified space.
        # However, we can still provide sector-specific views if needed for analysis.
        x_observables = full_observables  # Use full observables for X sector
        z_observables = full_observables  # Use full observables for Z sector
        
        # The unified error space size is the number of columns in the full check matrix
        unified_error_space_size = full_check_matrix.shape[1]
        
        # Create composite decoder
        composite_decoder = CompositeCSSDecoder(
            x_decoder=x_decoder,
            z_decoder=z_decoder,
            x_observables=x_observables,
            z_observables=z_observables,
            x_detector_indices=x_detector_indices,
            z_detector_indices=z_detector_indices,
            unified_error_space_size=unified_error_space_size,
        )
        
        logging.info(f"[CSS] Built composite decoder: X-sector {len(x_detector_indices)} detectors, Z-sector {len(z_detector_indices)} detectors, unified error space size {unified_error_space_size}")
        
        return composite_decoder, full_observables
        
    except Exception as e:
        logging.warning(f"CSS split decoder construction failed: {e}")
        return None


def build_decoder_from_circuit(
    circuit: stim.Circuit,
    *,
    bp_iters: int,
    osd_order: int,
    decompose_dem: Optional[bool] = None,
    code: Optional[Any] = None,
    p: Optional[float] = None,
    use_bt_singleshot: bool = True,
    use_css_splitting: bool = False,
) -> Tuple[Any, np.ndarray]:
    """Build a DEM-based decoder with optional CSS splitting and meta-parity scrubbing.

    - Constructs a DEM → (H, priors, O) → BpOsdDecoder pipeline.
    - If `use_css_splitting` is True and the code is a CSS code, attempts to split
      the decoder into separate X and Z sectors for better performance.
    - If `use_bt_singleshot` and the code has `h_meta`, wraps the decoder with a
      detector-space meta-parity scrubber before decoding.
    - Falls back to unified decoder if CSS splitting fails or is disabled.
    
    Args:
        circuit: Stim circuit to analyze
        bp_iters: Number of BP iterations
        osd_order: OSD order for post-processing
        decompose_dem: Whether to decompose the detector error model
        code: Optional CSS code object for structure information
        p: Error probability for meta-parity scrubbing
        use_bt_singleshot: Whether to use BT single-shot meta scrubbing
        use_css_splitting: Whether to attempt X/Z sector splitting for CSS codes
        
    Returns:
        Tuple of (decoder, observables_matrix)
    """
    t0 = time.time()

    # Try CSS splitting if enabled and we have a CSS code
    if use_css_splitting and code is not None:
        
        # Check if this looks like a CSS code (has hx and hz matrices)
        hx = getattr(code, 'hx', None)
        hz = getattr(code, 'hz', None)
        
        if hx is not None and hz is not None:
            css_result = _build_css_split_decoder(
                circuit,
                bp_iters=bp_iters,
                osd_order=osd_order,
                decompose_dem=decompose_dem,
                code=code,
                p=p,
            )
            
            # css_result will be None due to disabled implementation
            if css_result is not None:
                css_decoder, css_observables = css_result
                
                # Apply BT single-shot wrapping if requested and applicable
                want_scrub = bool(use_bt_singleshot and getattr(code, "h_meta", None) is not None)
                if want_scrub:
                    logging.info("[CSS] CSS splitting successful, but BT single-shot not yet supported for composite decoders")
                    # For now, fall back to unified decoder if single-shot is requested
                    # TODO: Implement single-shot for composite decoders
                else:
                    t1 = time.time()
                    logging.info(f"[CSS] CSS split decoder built successfully in {t1 - t0:.2f}s")
                    return css_decoder, css_observables

    # Fallback to unified DEM-based decoder
    if use_css_splitting:
        logging.info("[DEC] Using unified decoder (CSS splitting disabled or not forced)")
    else:
        logging.info("[DEC] Using unified decoder (CSS splitting disabled or failed)")

    # DEM construction
    if decompose_dem is None:
        env = os.getenv("QEC_DEM_DECOMPOSE")
        decompose = bool(env and env not in ("0", "false", "False"))
    else:
        decompose = bool(decompose_dem)
    dem = (
        circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        if decompose
        else circuit.detector_error_model(decompose_errors=False)
    )
    t1 = time.time()
    mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    t2 = time.time()

    # Decoder construction
    osd_method = "osd_e" if (osd_order and osd_order > 0) else "osd0"
    bposd = BpOsdDecoder(
        mats.check_matrix,
        error_channel=list(mats.priors),
        max_iter=bp_iters,
        bp_method="ms",
        ms_scaling_factor=0.625,
        schedule="parallel",
        osd_method=osd_method,
        osd_order=osd_order,
    )
    t3 = time.time()
    logging.info("[DEC] DEM %.2fs | mats %.2fs | init %.2fs", t1 - t0, t2 - t1, t3 - t2)

    # Optional meta-parity scrubber
    want_scrub = bool(use_bt_singleshot and code is not None and getattr(code, "h_meta", None) is not None)
    if not want_scrub:
        return bposd, mats.observables_matrix

    p_spam = float(p) if p is not None else 0.01
    wrapped = build_dem_decoder_with_meta_scrub(
        base_decoder=bposd,
        dem=dem,
        code=code,  # type: ignore[arg-type]
        p_spam=p_spam,
        bp_iters=bp_iters,
        osd_order=osd_order,
    )
    logging.info("[DEC] Using DEM decoder with meta-parity scrubbing (BT)")
    return wrapped, mats.observables_matrix




class _LockLike(Protocol):
    def acquire(self) -> None: ...
    def release(self) -> None: ...


def append_resume_csv(
    path: str,
    *,
    shots: int,
    errors: int,
    seconds: float,
    decoder: str,
    json_metadata: dict,
    custom_counts: Optional[dict] = None,
    lock: Optional[_LockLike] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = [
        "shots",
        "errors",
        "discards",
        "seconds",
        "decoder",
        "json_metadata",
        "custom_counts",
    ]
    if lock is not None:
        lock.acquire()
    try:
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            meta_json = json.dumps(json_metadata, separators=(",", ":"))
            counts_json = json.dumps(custom_counts or {}, separators=(",", ":"))
            writer.writerow(
                [
                    int(shots),
                    int(errors),
                    0,
                    float(seconds),
                    str(decoder),
                    meta_json,
                    counts_json,
                ]
            )
    finally:
        if lock is not None:
            lock.release()


def save_summary_csv(
    points: List[ResultPoint],
    path: str,
    *,
    meta_common: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = [
        "p",
        "rounds",
        "shots",
        "errors",
        "ler",
        "seconds",
        "decoder",
        "json_metadata",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in points:
            meta = {
                "l": int(r.l),
                "m": int(r.m),
                "p": float(r.p),
                "rounds": int(r.rounds),
            }
            if meta_common:
                meta.update(meta_common)
            writer.writerow(
                [
                    r.p,
                    r.rounds,
                    r.shots,
                    r.errors,
                    r.ler,
                    r.seconds,
                    r.decoder,
                    json.dumps(meta, separators=(",", ":")),
                ]
            )


def _wilson_confidence_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson confidence interval for binomial proportion.
    
    Args:
        k: Number of successes (errors)
        n: Number of trials (shots)
        z: Z-score for confidence level (1.96 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo, hi


def _transform_error_rate(p_any: float, r_rounds: int, y_mode: str, K: int | None = None) -> float:
    """Transform logical error rate based on selected y_mode.
    
    Args:
        p_any: Input logical error rate
        r_rounds: Number of rounds
        y_mode: Transform mode ('ler', 'per_round', 'per_logical')
        K: Number of logical operators (required for 'per_logical' mode)
        
    Returns:
        Transformed error rate
        
    Raises:
        ValueError: If y_mode is invalid or K is missing for 'per_logical'
    """
    if y_mode == "ler":
        return p_any
    if y_mode == "per_round":
        rr = max(1, int(r_rounds) if r_rounds and r_rounds > 0 else 1)
        rr_float = float(rr)
        result = 1.0 - (1.0 - p_any) ** (1.0 / rr_float)
        return cast(float, result)
    if y_mode == "per_logical":
        if not K or K <= 0:
            raise ValueError(
                "K must be a positive integer for y_mode='per_logical'."
            )
        K_float = float(K)
        result = 1.0 - (1.0 - p_any) ** (1.0 / K_float)
        return cast(float, result)
    raise ValueError(f"Unknown y_mode: {y_mode}")


def _setup_plot_style() -> Tuple[List[str], Any]:
    """Setup matplotlib style and return markers and colormap.
    
    Returns:
        Tuple of (markers_list, colormap)
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    
    markers = ["o", "s", "^", "D", "P", "v", "+", "x", "*", "h"]
    cmap = plt.get_cmap("tab10")
    return markers, cmap


def _group_points_by_configuration(points: List[ResultPoint]) -> Dict[Tuple[str, int, int, int, str], List[ResultPoint]]:
    """Group result points by (l, m, decoder) configuration.
    
    Args:
        points: List of result points to group
        
    Returns:
        Dictionary mapping (l, m, decoder) tuples to lists of points
    """
    by_group: Dict[Tuple[str, int, int, int, str], List[ResultPoint]] = {}
    for r in points:
        code_t = r.code_type if getattr(r, "code_type", None) else "BB"
        n_dim = int(getattr(r, "n", -1))
        by_group.setdefault((code_t, int(r.l), int(r.m), n_dim, r.decoder), []).append(r)
    return by_group


def _calculate_plot_bounds(all_x_vals: List[float], all_y_vals: List[float]) -> Optional[Tuple[float, float, float, float]]:
    """Calculate plot bounds ensuring positive values for log scale.
    
    Args:
        all_x_vals: All x-axis values
        all_y_vals: All y-axis values
        
    Returns:
        Tuple of (x_min, x_max, y_min, y_max) or None if no valid bounds
    """
    if not all_x_vals or not all_y_vals:
        return None
    
    x_pos = [v for v in all_x_vals if v > 0]
    y_pos = [v for v in all_y_vals if v > 0]
    if not x_pos or not y_pos:
        return None
    
    x_min = min(x_pos)
    x_max = max(x_pos)
    y_min = min(y_pos)
    y_max = max(y_pos)
    return x_min, x_max, y_min, y_max


def _apply_plot_styling(
    ax: plt.Axes,
    code_type: str,
    l: int,
    m: int,
    n: int | None,
    decoder: str,
    y_mode: str,
    K: int | None = None,
) -> None:
    """Apply styling to a plot axis.
    
    Args:
        ax: Matplotlib axis to style
        l: First BB code parameter
        m: Second BB code parameter  
        decoder: Decoder name
        y_mode: Y-axis mode
        K: Number of logical operators for definition text
    """
    ax.set_xlabel("Physical error rate p")
    y_label = {
        "ler": "Logical error rate",
        "per_logical": "Per-logical-operator error rate",
        "per_round": "Per-round error rate",
    }.get(y_mode, "Logical error rate")
    ax.set_ylabel(y_label)
    ct = (code_type or "BB").upper()
    if ct == "TT" and n is not None and int(n) > 0:
        title = f"{ct} {l}×{m}×{int(n)} ({decoder})"
    else:
        title = f"{ct} {l}×{m} ({decoder})"
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(frameon=False, fontsize=10)

    # Add definition text inside plot
    def_text = None
    if y_mode == "per_logical":
        def_text = f"p_single = 1 - (1 - p_any)^(1/K), K={K}"
    elif y_mode == "per_round":
        def_text = "p_round = 1 - (1 - p_shot)^(1/r), r = rounds"
    if def_text:
        ax.text(
            0.02,
            0.02,
            def_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
            ),
        )


def _save_plot(
    fig: plt.Figure,
    code_type: str,
    l: int,
    m: int,
    n: int | None,
    y_mode: str,
    out_png: str | None,
    show: bool,
) -> None:
    """Save plot to file and optionally display it.
    
    Args:
        fig: Matplotlib figure to save
        l: First BB code parameter for default filename
        m: Second BB code parameter for default filename
        y_mode: Y-axis mode for filename suffix
        out_png: Output path or None for default
        show: Whether to display the plot
    """
    if out_png:
        path = out_png
    else:
        suffix = {
            "ler": "parsed_results",
            "per_logical": "per_logical", 
            "per_round": "per_round",
        }.get(y_mode, "parsed_results")
        ct = (code_type or "bb").lower()
        if (code_type or "BB").upper() == "TT" and n is not None and int(n) > 0:
            path = f"Data/{ct}_{l}_{m}_{int(n)}_{suffix}.png"
        else:
            path = f"Data/{ct}_{l}_{m}_{suffix}.png"
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_points(
    points: List[ResultPoint],
    *,
    out_png: str | None = None,
    show: bool = False,
    y_mode: str = "ler",
    K: int | None = None,
    N: int | None = None,
    D: int | None = None,
) -> None:
    """Generate plots for BB code simulation results with confidence intervals.
    
    Creates separate plots for each (l, m, decoder) configuration, with
    different curves for each number of rounds. Supports multiple y-axis
    transformations and includes Wilson confidence intervals.
    
    Args:
        points: List of result points to plot
        out_png: Output PNG path, or None for default naming
        show: Whether to display plots interactively
        y_mode: Y-axis transform mode ('ler', 'per_round', 'per_logical')
        K: Number of logical operators (required for 'per_logical' mode)
    """
    markers, cmap = _setup_plot_style()
    by_group = _group_points_by_configuration(points)

    for (code_t, l, m, n_dim, dec), pts in by_group.items():
        pts = list(pts)
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        
        # Group by number of rounds
        by_rounds: Dict[int, List[ResultPoint]] = {}
        for p in pts:
            by_rounds.setdefault(p.rounds, []).append(p)

        plotted_any = False
        all_x_vals: List[float] = []
        all_y_vals: List[float] = []
        
        for idx, (rounds, rows) in enumerate(sorted(by_rounds.items())):
            rows = sorted(rows, key=lambda x: x.p)
            # Filter out zero-error points (no observed errors)
            rows_nonzero = [r for r in rows if r.errors > 0]
            if not rows_nonzero:
                continue
            
            xs = [r.p for r in rows_nonzero]
            ys = [_transform_error_rate(r.ler, rounds, y_mode, K) for r in rows_nonzero]

            # Calculate Wilson confidence intervals
            lo_list = []
            hi_list = []
            for r in rows_nonzero:
                lo, hi = _wilson_confidence_interval(r.errors, r.shots)
                lo_t = max(1e-15, _transform_error_rate(lo, rounds, y_mode, K))
                hi_t = max(1e-15, _transform_error_rate(hi, rounds, y_mode, K))
                lo_list.append(lo_t)
                hi_list.append(hi_t)
            
            # Plot data with confidence intervals
            color = cmap(idx % 10)
            marker = markers[idx % len(markers)]
            ax.plot(
                xs,
                ys,
                marker=marker,
                linestyle="-",
                color=color,
                linewidth=1.5,
                markersize=5,
                label=f"rounds={rounds}",
            )
            ax.fill_between(xs, lo_list, hi_list, color=color, alpha=0.15, linewidth=0)
            
            plotted_any = True
            all_x_vals.extend(xs)
            all_y_vals.extend(ys)

        # Skip if nothing to plot
        if not plotted_any:
            plt.close(fig)
            continue

        # Set up log scale with proper bounds
        bounds = _calculate_plot_bounds(all_x_vals, all_y_vals)
        if not bounds:
            plt.close(fig)
            continue
        
        x_min, x_max, y_min, y_max = bounds
        ax.set_xlim(x_min * 0.9, x_max * 1.1)
        ax.set_ylim(max(1e-15, y_min * 0.8), y_max * 1.25)
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        # Try to infer N/D if not provided
        N_eff = int(N) if N is not None else -1
        D_eff = int(D) if D is not None else -1
        if N_eff <= 0:
            try:
                Ns = {int(getattr(p, 'N', -1)) for p in pts if getattr(p, 'N', -1) and int(getattr(p, 'N', -1)) > 0}
                if len(Ns) == 1:
                    N_eff = Ns.pop()
            except Exception:
                pass
        if D_eff <= 0:
            try:
                Ds = {int(getattr(p, 'D', -1)) for p in pts if getattr(p, 'D', -1) and int(getattr(p, 'D', -1)) > 0}
                if len(Ds) == 1:
                    D_eff = Ds.pop()
            except Exception:
                pass

        _apply_plot_styling(ax, code_t, l, m, n_dim, dec, y_mode, K)

        # Add N,K,D tag at top-right
        info_parts = []
        if N_eff and N_eff > 0:
            info_parts.append(f"N={N_eff}")
        if K and K > 0:
            info_parts.append(f"K={int(K)}")
        if D_eff and D_eff > 0:
            # The displayed D is an upper bound on the true distance: D_true ≤ code.D
            info_parts.append(f"D ≤ {D_eff}")
        if info_parts:
            ax.text(
                0.98,
                0.98,
                "[" + ", ".join(info_parts) + "]",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )
        _save_plot(fig, code_t, l, m, n_dim, y_mode, out_png, show)
