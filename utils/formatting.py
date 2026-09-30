"""Reporting vocabulary: p-values that never print as zeros, and log banners."""

import numpy as np


# ---------------------------------------------------------------------------
# Reporting vocabulary
#
# Deliberately duplicated rather than imported: createDatabase.py sits upstream of
# the analysis pipeline and the two live in separate repositories, so neither may
# import the other. The names match the pipeline's on purpose, so both logs read
# alike; the duplication is the price of keeping the repositories independent.
# ---------------------------------------------------------------------------
def fmt_p(p, decimals=3):
    """A p-value at a fixed width, without pretending to precision it lacks.

    Args:
        p: The p-value, or None/NaN when the test could not be run.
        decimals: Digits after the point.

    A value smaller than the last digit shown prints as "<0.001" (or "<0.0001",
    or whatever `decimals` makes the floor) rather than as a row of zeros: a
    p-value rounded to 0.0000 claims the test returned exactly zero, which no
    test does, and hides how much smaller than the threshold it really was.
    """
    width = decimals + 5
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a".rjust(width)
    floor = 10.0 ** (-decimals)
    if p < floor:
        return f"<{floor:.{decimals}f}".rjust(width)
    return f"{p:{width}.{decimals}f}"


def fmt_p_inline(p, decimals=4):
    """The same, unpadded, for a figure label or a sentence.

    Args:
        p: The p-value, or None/NaN.
        decimals: Digits after the point.
    """
    return fmt_p(p, decimals).strip()


def fmt_p_phrase(p, decimals=4, label="p"):
    """"p < 0.0001" or "p = 0.0082" -- never "p = <0.0001".

    Args:
        p: The p-value, or None/NaN.
        decimals: Digits after the point.
        label: What to call it, for a test that reports more than one.

    The comparison sign replaces the equals sign rather than following it, so a
    p-value below the shown precision reads as the inequality it is.
    """
    text = fmt_p_inline(p, decimals)
    if text.startswith("<"):
        return f"{label} < {text[1:]}"
    return f"{label} = {text}"


def section(title):
    """Print a top-level banner.

    Args:
        title: Text of the banner.
    """
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title):
    """Print a second-level banner.

    Args:
        title: Text of the banner.
    """
    print(f"\n-- {title} " + "-" * max(0, 74 - len(title)))
