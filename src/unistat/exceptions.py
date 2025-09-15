"""Module for all custom exceptions & warnings."""
import warnings


################################################################################
# Exceptions
################################################################################

class UnistatError(Exception):
    """Base class for all Unistat exceptions."""


class SeriesNameCollisionError(UnistatError, ValueError):
    """Raised when 2 Series share names, and no error-handling implemented.

    If error-handling has been implemented to avoid crashes (e.g. Series are
    renamed to deconflict) use SeriesNameCollisionWarning instead.

    Args:
    name: str
        The duplicated series name.
    message: str, optional
        Custom message override.
    """
    def __init__(self, name: str, message: str | None = None) -> None:
        msg = message or ('Series names must be distinct; '
                          f'received {name!r} for both.')
        super().__init__(msg)
        self.name = name


################################################################################
# Warnings
################################################################################

class UnistatWarning(UserWarning):
    """Base class for all Unistat warnings."""


class ExperimentalWarning(UnistatWarning):
    """Emitted for features/APIs that are experimental and may have errors.

    Args:
    feature: str
        Name or description of experimental feature. Examples include
        'LogBinStats' or 'Log-Binomial regression', or '.make_histqq()' or
        'histogram/Q-Q drawing method'.
    message: str, optional
        Custom message override.
    """
    def __init__(self,
                 feature: str | type,
                 message: str | None = None) -> None:
        # Accept either a string or a class/type
        self.feature = (
            # Use feature string, if feature is a str
            feature if isinstance(feature, str)
            else getattr(
                # elif is a class, use its __qualname__ if it exists
                feature,
                '__qualname__',
                getattr(
                    # elif no __qualname__, use __name__
                    feature,
                    '__name__',
                    # elif no __name__, use its str(obj)
                    str(feature)
                )
            )
        )

        default = f'''
        {self.feature} is still experimental, and may contain errors or
        fail to function as expected. In critical applications, output
        should be carefully checked, or an alternative used.
        '''
        super().__init__(message or default)
        self.custom_message = message  # None if default was used


class SeriesNameCollisionWarning(UnistatWarning):
    """Emitted when 2 Series share names, but error-handling implemented.

    If no error-handling exists, use SeriesNameCollisionError instead..
    """
    def __init__(self, message: str | None = None) -> None:
        default = 'Series shared same names; names were deconflicted.'
        msg = message or default
        super().__init__(msg)


# Convenience methods ==========================================================

def warn_experimental(feature: str | type,
                      message: str | None = None, *,
                      stacklevel: int = 2) -> None:
    """Convenience helper to emit an ExperimentalWarning with a correct stacklevel.

    Example
    -------
    warn_experimental(MyClass)
    warn_experimental("TwoSampleStats", "TwoSampleStats is experimental; results may be unstable.")
    """
    warnings.warn(ExperimentalWarning(feature, message), stacklevel=stacklevel)
