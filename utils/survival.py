"""Survival primitives: Kaplan-Meier curves, their at-risk tables, and risk-set helpers."""

import numpy as np
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator


daysXmonth = 365 / 12
daysXweek = 7


def as_structured(event, time):
    """Right-censored survival data in the structured-array form scikit-survival expects.

    Args:
        event: Per-subject event indicator, truthy where the event was observed.
        time: Per-subject follow-up time, in the same order as `event`.
    """
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("event", "bool"), ("time", "float")],
    )


def km_curve(data, duration_col, status_col):
    """Kaplan-Meier estimate with log-log confidence bands, anchored at (0, 1).

    Args:
        data: Table of subjects to estimate from; rows with a missing duration or
            status must already have been dropped.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.

    Returns (time, survival_prob, conf_int) with a leading (0, 1) point inserted,
    so every curve starts at full survival rather than at the first event.
    """
    time, survival_prob, conf_int = kaplan_meier_estimator(
        data[status_col] == 1, data[duration_col], conf_type="log-log"
    )
    time = np.insert(time, 0, 0)
    survival_prob = np.insert(survival_prob, 0, 1)
    conf_int = np.insert(conf_int, 0, 1, axis=1)
    return time, survival_prob, conf_int


def usable_rows(data, duration_col, status_col):
    """The rows a Kaplan-Meier curve can use: both a duration and a status recorded.

    Args:
        data: Table of subjects.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
    """
    return data[~np.isnan(data[status_col]) & ~np.isnan(data[duration_col])]


def draw_km(ax, data, duration_col, status_col, color, label, band_alpha, **step_kw):
    """Draw one Kaplan-Meier curve in months: the step, its band and censoring ticks.

    Args:
        ax: Axes to draw on.
        data: Subjects of the curve, already restricted by `usable_rows`; the
            durations are whatever the curve plots, rescaled or not.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        color: Colour of the step, the band and the ticks.
        label: Legend entry of the step.
        band_alpha: Opacity of the log-log confidence band.
        **step_kw: Further styling of the step alone, e.g. linewidth or alpha.
    """
    time, survival_prob, conf_int = km_curve(data, duration_col, status_col)
    ax.step(time / daysXmonth, survival_prob, where="post", color=color, label=label,
            **step_kw)
    ax.fill_between(time / daysXmonth, conf_int[0], conf_int[1], alpha=band_alpha,
                    step="post", color=color)
    for t in data.loc[data[status_col] == 0, duration_col].values:  # Censoring times
        ax.plot(time[time == t] / daysXmonth, survival_prob[time == t], "|", color=color)


def style_km_axes(ax, months):
    """Axes of an overall-survival panel: 0-75 months, ticks every 10, no box.

    Args:
        ax: Axes holding the curves; the y limits are the caller's, since they
            depend on how many at-risk rows go underneath.
        months: Time points (months) of the numbers-at-risk row; the last one
            bounds the bottom spine.
    """
    ax.set_xlim([-5, 75])
    ax.set_xticks(range(0, months[-1] + 10, 10))
    ax.set_xticklabels(range(0, months[-1] + 10, 10))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["bottom"].set_bounds(0, months[-1])
    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Overall survival", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)


def km_median(time, survival_prob):
    """First time at which a Kaplan-Meier estimate reaches 0.5; nan if it never does.

    Args:
        time: Times of a Kaplan-Meier curve, ascending.
        survival_prob: Survival probability at each of those times.

    Used for both median survival and -- on the curve of the flipped event
    indicator -- median potential follow-up, so the two are read off by exactly
    the same estimator as every curve this script plots.
    """
    below = np.flatnonzero(np.asarray(survival_prob) <= 0.5)
    return float(np.asarray(time)[below[0]]) if below.size else np.nan


def restricted_mean(time, survival_prob, horizon):
    """Area under a Kaplan-Meier step function from 0 to `horizon`.

    Args:
        time: Times of the curve, ascending and starting at 0 (as `km_curve`
            returns them).
        survival_prob: Survival probability from each of those times onwards.
        horizon: Upper limit of the integral, in the units of `time`.
    """
    time, survival_prob = np.asarray(time, float), np.asarray(survival_prob, float)
    keep = time < horizon
    edges = np.append(time[keep], horizon)
    return float(np.sum(survival_prob[keep] * np.diff(edges)))


def at_risk_and_censored(data, months, duration_col, status_col):
    """Numbers under a Kaplan-Meier curve: still at risk, and censored before then.

    Args:
        data: Subjects of one curve; the durations must already be whatever the
            curve plots, so a rescaled panel passes its rescaled times.
        months: Time points the row is printed at.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.

    Returns (at_risk, censored), two lists aligned on `months`. `at_risk` counts
    the subjects whose follow-up reaches the month; `censored` is cumulative --
    everyone right-censored strictly before it -- so the pair reads "how many are
    still being watched (and how many stopped being watched)".

    The convention is deliberately duplicated from the
    Tract-Density_Components-Survival repository rather than imported: the two
    repositories stay independent on purpose, and the tables under their
    Kaplan-Meier curves have to be read the same way.
    """
    at_risk, censored = [], []
    for month in months:
        limit = month * daysXmonth
        at_risk.append(int((data[duration_col] >= limit).sum()))
        censored.append(int(((data[status_col] == 0)
                             & (data[duration_col] < limit)).sum()))
    return at_risk, censored


def draw_at_risk_table(ax, months, rows, colors, top=-0.07, step_y=0.06, fontsize=9.5,
                       header="No. at risk (right-censored)"):
    """Draw the "No. at risk (right-censored)" block under a Kaplan-Meier curve.

    Args:
        ax: Axes to draw on. Its limits and ticks must already be set, because the
            columns are spaced against the final data-to-pixel transform.
        months: Time points the counts were computed at.
        rows: One (at_risk, censored) pair per curve, as `at_risk_and_censored`
            returns them, in the order the curves were drawn.
        colors: One color per row, matching the curves.
        top: y of the first row, in data coordinates.
        step_y: vertical distance between rows, in data coordinates.
        fontsize: point size of the counts.
        header: Bold line above the counts, saying what the pair means; the
            reverse Kaplan-Meier prints a different pair under the same layout.

    Columns are thinned to whatever fits: "367 (144)" is roughly twice the width
    of the bare count this used to print, and eleven of them do not fit across a
    six-inch axis at a legible size. Every printed column is centred on its tick,
    so a thinned row still lines up with the axis it describes.
    """
    labels = [[f"{a} ({c})" for a, c in zip(*row)] for row in rows]
    stride = 1
    if len(months) > 1:
        spacing = abs(ax.transData.transform((months[1], 0))[0]
                      - ax.transData.transform((months[0], 0))[0])
        widest = max((len(text) for row in labels for text in row), default=0)
        # 0.6 em per character is the usual approximation for a proportional face
        needed = widest * fontsize * 0.6 * ax.figure.dpi / 72.0
        if spacing:
            stride = max(1, int(np.ceil(needed / spacing)))
    for k, row in enumerate(labels):
        for i in range(0, len(months), stride):
            ax.text(months[i], top - step_y * k, row[i], transform=ax.transData,
                    fontsize=fontsize, verticalalignment="top",
                    horizontalalignment="center", color=colors[k])
    ax.text(ax.get_xlim()[0], -0.01, header,
            transform=ax.transData, fontsize=10, verticalalignment="top",
            color="black", fontweight="bold")
    ax.hlines(0, ax.get_xlim()[0], months[-1] + 5, color="black", linewidth=0.5)


def breslow_cumulative_hazard(time, event, risk, grid):
    """Breslow's cumulative baseline hazard at each time of `grid`.

    Args:
        time: Per-subject follow-up time.
        event: Per-subject 0/1 event indicator.
        risk: Per-subject exp(linear predictor).
        grid: Ascending distinct event times to evaluate at.
    """
    order = np.argsort(time)
    time, event, risk = time[order], event[order], risk[order]
    # Sum of risk over everyone still at risk at t: a reverse cumulative sum,
    # read at the first subject whose time is >= t
    at_risk = np.cumsum(risk[::-1])[::-1]
    first = np.searchsorted(time, grid, side="left")
    deaths = np.array([np.sum(event[time == t]) for t in grid])
    return np.cumsum(deaths / at_risk[first])


def split_at_event_times(frame, duration_col, status_col, columns, max_cuts=1000):
    """Re-express one row per subject as one row per risk set the subject is in.

    Args:
        frame: Subjects, one row each, already complete-case and with positive
            durations.
        duration_col: Column holding the follow-up time, in days.
        status_col: Column holding the 0/1 event indicator.
        columns: Baseline covariate columns to carry onto every interval.
        max_cuts: Ceiling on the number of split points. Below it the split is at
            every distinct event time, which is exact; above it the cuts are taken
            at quantiles of the event times, which keeps a very large pool
            tractable at the cost of shrinking the interaction slightly toward
            zero, because a covariate held constant across a wide interval lags
            the time it is meant to track.

    Returns a (start, stop] table with `id`, the status carried only on each
    subject's final interval, and `columns` repeated down the intervals.

    Splitting at the event times is what makes a time-varying coefficient
    identifiable: every member of a risk set then shares the same interval
    boundary, so the time covariate takes one value across that comparison rather
    than a different value for the subject who happens to fail.
    """
    cuts = np.unique(pd.to_numeric(frame.loc[frame[status_col] == 1, duration_col]))
    if len(cuts) > max_cuts:
        cuts = np.unique(np.quantile(cuts, np.linspace(0, 1, max_cuts)))
    duration = frame[duration_col].values
    status = frame[status_col].values

    starts, stops, events, owner = [], [], [], []
    for i, end in enumerate(duration):
        edges = cuts[cuts < end]
        lower = np.concatenate(([0.0], edges))
        upper = np.concatenate((edges, [end]))
        flags = np.zeros(len(upper))
        flags[-1] = status[i]
        starts.append(lower), stops.append(upper)
        events.append(flags), owner.append(np.full(len(upper), i))

    owner = np.concatenate(owner)
    out = pd.DataFrame({"id": owner, "start": np.concatenate(starts),
                        "stop": np.concatenate(stops),
                        status_col: np.concatenate(events)})
    for column in columns:
        out[column] = frame[column].values[owner]
    return out
