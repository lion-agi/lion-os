import importlib.metadata
import importlib.util
import logging
import platform
import subprocess
from typing import Any

from .utils import run_pip_command


def check_import(
    package_name: str,
    module_name: str | None = None,
    import_name: str | None = None,
    pip_name: str | None = None,
    attempt_install: bool = True,
    error_message: str = "",
):
    """
    Check if a package is installed, attempt to install if not.

    Args:
        package_name: The name of the package to check.
        module_name: The specific module to import (if any).
        import_name: The specific name to import from the module (if any).
        pip_name: The name to use for pip installation (if different).
        attempt_install: Whether to attempt installation if not found.
        error_message: Custom error message to use if package not found.

    Raises:
        ImportError: If the package is not found and not installed.
        ValueError: If the import fails after installation attempt.
    """
    if not is_import_installed(package_name):
        if attempt_install:
            logging.info(
                f"Package {package_name} not found. Attempting " "to install.",
            )
            try:
                return install_import(
                    package_name=package_name,
                    module_name=module_name,
                    import_name=import_name,
                    pip_name=pip_name,
                )
            except ImportError as e:
                raise ValueError(f"Failed to install {package_name}: {e}") from e
        else:
            logging.info(
                f"Package {package_name} not found. {error_message}",
            )
            raise ImportError(
                f"Package {package_name} not found. {error_message}",
            )

    return import_module(
        package_name=package_name,
        module_name=module_name,
        import_name=import_name,
    )


def get_cpu_architecture() -> str:
    """
    Get the CPU architecture.

    Returns:
        str: 'arm64' if ARM-based, 'x86_64' for Intel/AMD 64-bit, or the
            actual architecture string for other cases.
    """
    arch: str = platform.machine().lower()
    if "arm" in arch or "aarch64" in arch:
        return "arm64"
    elif "x86_64" in arch or "amd64" in arch:
        return "x86_64"
    else:
        return arch


def import_module(
    package_name: str,
    module_name: str = None,
    import_name: str | list = None,
) -> Any:
    """
    Import a module by its path.

    Args:
        module_path: The path of the module to import.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported.
    """
    try:
        full_import_path = (
            f"{package_name}.{module_name}" if module_name else package_name
        )

        if import_name:
            import_name = (
                [import_name] if not isinstance(import_name, list) else import_name
            )
            a = __import__(
                full_import_path,
                fromlist=import_name,
            )
            if len(import_name) == 1:
                return getattr(a, import_name[0])
            return [getattr(a, name) for name in import_name]
        else:
            return __import__(full_import_path)

    except ImportError as e:
        raise ImportError(f"Failed to import module {full_import_path}: {e}") from e


def install_import(
    package_name: str,
    module_name: str | None = None,
    import_name: str | None = None,
    pip_name: str | None = None,
):
    """
    Attempt to import a package, installing it if not found.

    Args:
        package_name: The name of the package to import.
        module_name: The specific module to import (if any).
        import_name: The specific name to import from the module (if any).
        pip_name: The name to use for pip installation (if different).

    Raises:
        ImportError: If the package cannot be imported or installed.
        subprocess.CalledProcessError: If pip installation fails.
    """
    pip_name = pip_name or package_name

    try:
        return import_module(
            package_name=package_name,
            module_name=module_name,
            import_name=import_name,
        )
    except ImportError:
        logging.info(f"Installing {pip_name}...")
        try:
            run_pip_command(["install", pip_name])
            return import_module(
                package_name=package_name,
                module_name=module_name,
                import_name=import_name,
            )
        except subprocess.CalledProcessError as e:
            raise ImportError(f"Failed to install {pip_name}: {e}") from e
        except ImportError as e:
            raise ImportError(
                f"Failed to import {pip_name} after installation: {e}"
            ) from e


def is_import_installed(package_name: str) -> bool:
    """
    Check if a package is installed.

    Args:
        package_name: The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    return importlib.util.find_spec(package_name) is not None


def list_installed_packages() -> list[str]:
    """
    List all installed packages.

    Returns:
        List[str]: A list of names of installed packages.
    """
    try:
        return [dist.metadata["Name"] for dist in importlib.metadata.distributions()]
    except Exception as e:
        logging.error(f"Failed to list installed packages: {e}")
        return []


def uninstall_package(package_name: str) -> None:
    """
    Uninstall a specified package.

    Args:
        package_name: The name of the package to uninstall.

    Raises:
        subprocess.CalledProcessError: If the uninstallation fails.
    """
    try:
        run_pip_command(["uninstall", package_name, "-y"])
        logging.info(f"Successfully uninstalled {package_name}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to uninstall {package_name}. Error: {e}")
        raise


def update_package(package_name: str) -> None:
    """
    Update a specified package.

    Args:
        package_name: The name of the package to update.

    Raises:
        subprocess.CalledProcessError: If the update fails.
    """
    try:
        run_pip_command(["install", "--upgrade", package_name])
        logging.info(f"Successfully updated {package_name}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to update {package_name}. Error: {e}")
        raise
