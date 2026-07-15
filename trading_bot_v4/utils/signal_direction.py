"""Translate the binary model output into an executable direction.

The model's positive class is a future gain above ``MOVEMENT_THRESHOLD``.
Its negative class means only "not a sufficiently large gain"; it is not a
bearish target.  Therefore a low positive-class probability must be HOLD,
not SHORT.
"""


def binary_upside_direction(probability: float, threshold: float) -> str:
    """Return LONG for a confident upside prediction, otherwise HOLD."""
    return "LONG" if float(probability) > float(threshold) else "HOLD"
