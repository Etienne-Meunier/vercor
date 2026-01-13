from pathlib import Path
from typing import Any, List, Dict, Optional
import subprocess
import os
import sys
import shutil

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jcm.geometry import Geometry, get_terrain

def generate_jcm_geometry_from_orography(
    orography: ArrayLike,
    num_levels:int = 8,
    truncation_number:int = None,
):
    """Initialize all of the speedy model geometry variables from a given terrain file containing orog and lsm.
    
    Args:
        orography: A 2-dimensional array of orography
        num_levels (optional): Number of vertical levels `kx` (default 8).
        truncation_number (optional): Spectral truncation number for surface geopotential. If None, inferred from nodal_shape.
    Returns:
        Geometry object

    """
    orography, fmask = get_terrain(orography=orography)
    return Geometry.from_grid_shape(
        nodal_shape=orography.shape,
        num_levels=num_levels,
        orography=orography,
        fmask=fmask,
        truncation_number=truncation_number
    )


def generate_jcm_forcing_and_topography_files(
    resolution: int,
    data_directory: Optional[Path] = None,
) -> Dict[str, Path]:

    import jcm

    if not (isinstance(data_directory, Path) or data_directory is None):
        raise TypeError("`data_directory` must be of type `Path` or `None`.")

    if data_directory is None:
        home_data_directory = os.environ.get("HOME", None)

        if home_data_directory is None:
            data_directory = Path.cwd()
        else:
            data_directory = Path(home_data_directory)

        data_directory = data_directory / ".cache/jcm"

        print(f'Using input data directory: "{str(data_directory)}".')

    raw_data_directory = Path(jcm.__file__).parent / f"data/bc"

    # Prepare boundary file
    files_to_check = dict(
        terrain=(data_directory / f"terrain_t{resolution:d}.nc").resolve(),
        forcing=(data_directory / f"forcing_t{resolution:d}.nc").resolve(),
    )

    def check_if_file_exist(
        file_dict: Dict[str, Path], verbose: bool = True
    ) -> Dict[Path, bool]:

        file_status = {file: Path(file).exists() for _, file in file_dict.items()}

        if verbose:
            for file, result in file_status.items():
                print(
                    f"Check file: {str(file):s}...",
                    "found." if result else "not found.",
                )

        return file_status

    file_status = check_if_file_exist(files_to_check)

    if not all(list(file_status.values())):
        print("Some files are missing. Need to generate them.")

        data_directory.mkdir(parents=True, exist_ok=True)
        interpolation_code = (raw_data_directory / "interpolate.py").resolve()

        try:
            result = subprocess.run(
                [sys.executable, str(interpolation_code), f"{resolution:d}"],
                check=True,
                capture_output=True,
                text=True,
                cwd=data_directory,
            )
        except subprocess.CalledProcessError as e:
            print("Error output:", e.stderr)

        for destination_file in files_to_check.values():
            source_file = Path(raw_data_directory / destination_file.name)
            print(f"Copying: {str(source_file):s} => {str(destination_file):s}")
            shutil.copy(source_file, destination_file)

        new_file_status = check_if_file_exist(files_to_check)
        if not all(list(new_file_status.values())):
            raise FileNotFoundError(
                "Something went wrong. The daily file is not generated. Please check."
            )

    return files_to_check


def mean_leaf(
    tree: Any,
    axis: int | List[int],
) -> Any:
    """
    A tool function that does the jnp.mean to leaf nodes.

    Arguments:
        tree : a tree object

    Returns:
        tree_mean : tree with jnp.mean applied to each of its leaf node.
    """
    return jax.tree_util.tree_map(lambda arr: jnp.mean(arr, axis=axis), tree)


def unwrap_leading_dims(
    obj: Any,
    first_n_dim: int = 2,
) -> Any:
    """
    A tool function that unwraps the leading dimensions of jax arrays

    Arguments:
        obj : A structure containining jax arrays

    Returns:
        unwrapped object.
    """

    def _unwrap(arr: jnp.ndarray) -> jnp.ndarray:
        new_shape = (-1,) + arr.shape[first_n_dim:]
        return jnp.reshape(arr, new_shape)

    return jax.tree_util.tree_map(_unwrap, obj)


def stack_objects(
    objs: List[Any],
) -> Any:
    """
    A tool function that stack dataclasses together.

    Arguments:
        objs : A list of objects that need to be stacked

    Returns:
        stacked : Stacked object.
    """
    # objs is a list of pytrees with same structure
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return stacked


def concat_objects(
    objs: List[Any],
    axis: int,
) -> Any:
    """
    A tool function that concats dataclasses together.

    Arguments:
        objs : A list of objects that need to be concat

    Returns:
        concatenated : Concatenated object.
    """
    # objs is a list of pytrees with same structure
    concatenated = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=axis), *objs
    )
    return concatenated
