"""BB code + Sinter integration using Stim circuits with detectors."""

import numpy as np
import sinter
import stim
from ldpc.sinter_decoders import SinterBpOsdDecoder
import os
from matplotlib import pyplot as plt
from bposd.css import css_code
from circuit_utils import generate_full_circuit
from BB_tools import get_BB_Hx_Hz
from BPOSD_estimate_distance import get_min_logical_weight
from typing import Iterator, Tuple



def generate_BB_tasks(
    a_poly: list, 
    b_poly: list, 
    l: int, 
    m: int,
    p_list: np.ndarray,
    rounds_list: list = None
) -> Iterator[sinter.Task]:
    """Generate Sinter tasks for BB codes using Stim circuits with detectors.
    
    Parameters:
    - a_poly, b_poly: Polynomial specifications as [(i,j), ...] for x^i y^j terms
    - l, m: BB code dimensions
    - p_list: Physical error rates to test
    - rounds_list: List of syndrome rounds to test (default: based on computed distance)
    """
    # Build BB code once and cache
    Hx, Hz = get_BB_Hx_Hz(a_poly, b_poly, l, m)
    bb_code = css_code(hx=Hx, hz=Hz, name=f"BB_{l}x{m}")
    
    if rounds_list is None:
        distance = get_min_logical_weight(
            code=bb_code,
            p=0.1,
            pars=[20, 2],
            iters=1000,
            Ptype=1,
        )
        rounds_list = [distance]
    
    for p in p_list:
        for rounds in rounds_list:
            circuit = generate_full_circuit(
                code=bb_code,
                rounds=rounds,
                p1=p/10.0,
                p2=p,
                p_spam=p,
                seed=42
            )
            
            yield sinter.Task(
                circuit=circuit,
                json_metadata={
                    "p": p,
                    "rounds": rounds,
                    "code_n": bb_code.N,
                    "code_k": bb_code.K,
                    "l": l,
                    "m": m,
                    "a_poly": str(a_poly),
                    "b_poly": str(b_poly)
                }
            )
            


def run_BB_sinter_simulation(
    a_poly: list,
    b_poly: list, 
    l: int,
    m: int,
    p_min: float = 1e-3,
    p_max: float = 1e-2,
    num_points: int = 8,
    max_shots: int = 10000,
    max_errors: int = 50,
    num_workers: int = 1,
    bp_iters: int = 50,
    osd_order: int = 3,
    save_dir: str = "Data",
    rounds_list: list | None = None,
) -> list:
    """Run BB code simulation using Sinter framework.
    
    Returns list of sinter.TaskStats objects with results.
    """
    p_list = np.logspace(np.log10(p_min), np.log10(p_max), num_points)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate task generator (lazy evaluation)
    task_generator = generate_BB_tasks(a_poly, b_poly, l, m, p_list, rounds_list=rounds_list)
    
    out_csv = f"{save_dir}/BB_{l}x{m}_sinter.csv"
    
    samples = sinter.collect(
        num_workers=num_workers,
        max_shots=max_shots,
        max_errors=max_errors,
        tasks=task_generator,
        decoders=["bposd"],
        custom_decoders={
            "bposd": SinterBpOsdDecoder(
                max_iter=bp_iters,
                bp_method="ms",
                ms_scaling_factor=0.625,
                schedule="parallel",
                osd_method="osd_0",
            ),
        },
        print_progress=True,
        save_resume_filepath=out_csv,
    )

    return samples


def plot_BB_results(samples: list, save_path: str = None, show: bool = False):
    """Plot BB code results from Sinter simulation."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: f"BB {stat.json_metadata['l']}×{stat.json_metadata['m']} (rounds={stat.json_metadata['rounds']})",
        filter_func=lambda stat: stat.decoder == "bposd",
        x_func=lambda stat: stat.json_metadata["p"],
    )
    
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate") 
    ax.set_title("BB Code Performance")
    ax.loglog()
    ax.grid(True)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        pass
    if show:
        plt.show()
    plt.close(fig)


def main():
    """Real experimental parameters from paper."""
    # BB code from 2308.07915 (Bivariate Bicycle codes)
    a_poly = [(3, 0), (0, 1), (0, 2)]  # x^3 + y + y^2
    b_poly = [(0, 3), (1, 0), (2, 0)]  # y^3 + x + x^2
    l, m = 12, 6  # Original paper parameters
    # l, m = 4, 2  # Original paper parameters
    
    samples = run_BB_sinter_simulation(
        a_poly=a_poly,
        b_poly=b_poly,
        l=l,
        m=m,
        p_min=1e-3,
        p_max=1e-2,
        num_points=5,
        max_shots=10000,
        max_errors=50,
        num_workers=4,
        bp_iters=50,
        osd_order=2,
        save_dir="Data",
        rounds_list=[6]
    )
    
    # Print samples as CSV data
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())
    
    # Plot and save to Data directory
    plot_BB_results(samples, f"Data/BB_{l}x{m}_sinter_results.png", show=True)


if __name__ == "__main__":
    main()
