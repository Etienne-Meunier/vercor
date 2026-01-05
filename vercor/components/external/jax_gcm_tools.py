from pathlib import Path
from typing import Any, List, Dict

import jax
import jax.numpy as jnp


def generate_jcm_forcing_and_topography_files(resolution: int = 31) -> Dict[str, Path]:
    import jcm

    # Prepare boundary file
    files_to_check = dict(
        terrain=(
            Path(jcm.__file__).parent / f"data/bc/terrain_t{resolution:d}.nc"
        ).resolve(),
        forcing=(
            Path(jcm.__file__).parent / f"data/bc/forcing_t{resolution:d}.nc"
        ).resolve(),
    )

    interpolation_code = (
        Path(jcm.__file__).parent / "data/bc/interpolate.py"
    ).resolve()

    def get_files_exist(file_dict: dict[str, Path]) -> List[bool]:
        return [Path(file).exists() for _, file in file_dict.items()]

    for _, file_name in files_to_check.items():
        print(f"Check file: {str(file_name)}")

    if not all(get_files_exist(files_to_check)):
        print("Some files do not exist. Need to produce it.")

        import subprocess
        import sys

        try:
            result = subprocess.run(
                [sys.executable, str(interpolation_code), f"{resolution:d}"],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error output:", e.stderr)
    if all(get_files_exist(files_to_check)):
        print("All files exist!")
    else:
        raise Exception(
            "Something went wrong. The daily file is not generated. Please check."
        )

    return files_to_check


def positive_cosine_cubic_latitude_squared(
    lat: jnp.ndarray,
    amplitude: float = 1.0,
) -> jnp.ndarray:
    return jnp.where(
        jnp.abs(lat) < jnp.pi / 3, amplitude * jnp.cos(3 * lat / 2) ** 2, 0
    )


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
