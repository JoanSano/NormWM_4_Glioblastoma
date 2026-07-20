import numpy as np
import scipy
from statsmodels.stats.multitest import multipletests
from types import SimpleNamespace
from tqdm import tqdm

from lifelines.utils import concordance_index

from utils.metrics import compute_quantile_OS

class DeLong_Test():    
    # Adopted from https://github.com/yandexdataschool/roc_comparison 
    # Original ref: https://ieeexplore.ieee.org/document/6851192

    def __init__(self, ground_truths) -> None:
        """
        Computes the DeLong p-value and/or variance of a pair or predictions
            that are related to the same structure (or ground truth)
        Args:
        ground_truths: A flat array containing the positive and negative samples.
        """
        self.ground_truth = ground_truths

    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/
    def __compute_midrank(self, x):
        """Computes midranks.
        Args:
        x - a 1D numpy array
        Returns:
        array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    def __compute_ground_truth_statistics(self):
        assert np.array_equal(np.unique(self.ground_truth), [0, 1])
        order = (-self.ground_truth).argsort()
        label_1_count = int(self.ground_truth.sum())
        return order, label_1_count

    def fastDeLong(self, predictions_sorted_transposed, label_1_count):
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
            sorted such as the examples with label "1" are first
        Returns:
        (AUC value, DeLong covariance)
        Reference:
        @article{sun2014fast,
        title={Fast Implementation of DeLong's Algorithm for
                Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
        author={Xu Sun and Weichao Xu},
        journal={IEEE Signal Processing Letters},
        volume={21},
        number={11},
        pages={1389--1393},
        year={2014},
        publisher={IEEE}
        }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float64)
        ty = np.empty([k, n], dtype=np.float64)
        tz = np.empty([k, m + n], dtype=np.float64)
        for r in range(k):
            tx[r, :] = self.__compute_midrank(positive_examples[r, :])
            ty[r, :] = self.__compute_midrank(negative_examples[r, :])
            tz[r, :] = self.__compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov
    
    def calc_pvalue(self, aucs, sigma, alternative="two-sided"):
        """Computes the p-value.
        Args:
        aucs: 1D array of AUCs
        sigma: AUC DeLong covariances
        Returns:
        z_score, p_value
        """
        """ l = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
        print(z)
        return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10) """     
        if alternative not in ["two-sided", "greater", "lower"]:
            raise ValueError("Provide a valid hypothesis from two-sided, greater or lower")
        l = np.array([1, -1])
        z = (aucs[0]-aucs[1]) / np.sqrt(np.dot(np.dot(l, sigma), l.T))   
        if alternative=="two-sided":
            return z, scipy.stats.norm.sf(abs(z))*2
        elif alternative=="greater":
            return z, scipy.stats.norm.sf(z)
        else:
            return z, scipy.stats.norm.cdf(z)

    def delong_roc_variance(self, predictions):
        """
        Computes ROC AUC variance for a single set of predictions
        Args:
        ground_truth: np.array of 0 and 1
        predictions: np.array of floats of the probability of being class 1
        """
        order, label_1_count = self.__compute_ground_truth_statistics()
        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = self.fastDeLong(predictions_sorted_transposed, label_1_count)
        assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
        return aucs[0], delongcov

    def delong_roc_test(self, predictions_one, predictions_two, alternative="two-sided"):
        """
        Computes log(p-value) for hypothesis that two ROC AUCs are different
        Args:
        ground_truth: np.array of 0 and 1
        predictions_one: predictions of the first model,
            np.array of floats of the probability of being class 1
        predictions_two: predictions of the second model,
            np.array of floats of the probability of being class 1
        """
        order, label_1_count = self.__compute_ground_truth_statistics()
        predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
        aucs, delongcov = self.fastDeLong(predictions_sorted_transposed, label_1_count)
        return self.calc_pvalue(aucs, delongcov, alternative=alternative)
    
def benjamini_bogomolov_procedure(
        pvals,
        alpha=0.05, 
        global_test='simes',
        family_method='fdr_bh',
        inner_method='fdr_bh'

    ):
    """
    Adjust the p-values of multiple hypotheses organized into different families
    using the Benjamini-Bogomolov procedure. The BB procedure controls for the 
    expected average error rate across selected families and provides a principled way to 
    select inference of families of hypotheses. The exact error measure will depend on the 
    'inner_method' procedure, which is independent of the global tests and the 'family_method'
    used to select the candidate families of hypotheses.
    
    The function performs statistical testing on two levels:
        - Level 1: Global test per family (e.g., Simes)
        - Level 1: Multiple hypothesis correction across families (e.g., FDR, FWER) 
        - Level 2: Selects families where at least one null-was rejected
        - Level 2: Corrects for FDR within the selected families using the Benjamini-Hochberg procedure

    Notes 
    -----
    It applies Procedure 1 as described in [1] since it uses simple family selection rules (i.e., Simes or Bonferroni).

    Parameters
    ----------
    pvals : list of array-like
        A list of length m, where each element is an array of k(m) p-values 
        for that specific family. Handles families of unequal sizes.
    alpha : float
        The target average false discovery rate across families.
    family_method : str
        Method to select families. Options: available ones in statsmodels.multitest.multipletests
    global_test : str
        How to calculate a single p-value for the family. 
        Options: 'simes', 'bonferroni', 'min'.

    Returns
    -------
    results : numpy object with fields:
        - family_pvals
        - family_pvals_adj
        - selected_families
        - hypotheses_pvals_adj
        - bb_adj_alpha
    
    References
    ----------
    [1] Benjamini, Y., & Bogomolov, M. (2014). Selective inference on multiple families of hypotheses. 
        Journal of the Royal Statistical Society Series B: Statistical Methodology, 76(1), 297-318.
    """

    m = len(pvals)
    family_pvals = [] 

    # ------------------------------------------------------
    # LEVEL 1: Generate one p-value per family (Global Test)
    # ------------------------------------------------------
    for p_family in pvals:
        p_fam = np.asarray(p_family)
        k_i = len(p_fam)
        
        if global_test == 'simes':
            p_sorted = np.sort(p_fam)
            p_global = np.min((k_i / np.arange(1, k_i + 1)) * p_sorted)
        elif global_test == 'bonferroni':
            p_global = np.min(p_fam) * k_i
        elif global_test == 'min':
            p_global = np.min(p_fam)
        else:
            raise ValueError("Unsupported global_test method.")
            
        family_pvals.append(min(p_global, 1.0))

    family_pvals = np.array(family_pvals)

    # ------------------------
    # LEVEL 1: Select Families
    # ------------------------
    # We use multipletests to find which families are "discoveries"
    selected_mask, family_pvals_adj, _, _ = multipletests(
        family_pvals, alpha=alpha, method=family_method
    )
    R = np.sum(selected_mask)

    # -----------------------------
    # Benjamini–Bogomolov adjustment
    # -----------------------------
    bb_alpha = (R * alpha) / m if R > 0 else 0.0

    # ------------------------------------
    # LEVEL 2 — Correction within families
    # ------------------------------------
    hypotheses_pvals_adj = np.empty(m, dtype=object)
    for i in range(m):
        ps_fam = np.asarray(pvals[i])
        if selected_mask[i] and bb_alpha > 0:
            _, adj_p, _, _ = multipletests(ps_fam, alpha=bb_alpha, method=inner_method)
            hypotheses_pvals_adj[i] = adj_p
        else:
            hypotheses_pvals_adj[i] = np.ones_like(ps_fam, dtype=float)

    return SimpleNamespace(
        family_pvals=family_pvals,
        family_pvals_adj=family_pvals_adj,
        selected_families=selected_mask,
        hypotheses_pvals_adj=hypotheses_pvals_adj,
        bb_adj_alpha=bb_alpha
    )

def bootstrap_median_os_difference(
    times_low, event_low, 
    times_high, event_high, 
    n_boot=2000,
    alpha=0.05,
    random_state=None
):
    """
    Bootstrap the difference in median OS between two groups.

    Parameters
    ----------
    times_low : array-like
        Survival times for low-risk group.
    event_low : array-like
        Event indicator (1=event, 0=censored) for low-risk group.
    times_high : array-like
        Survival times for high-risk group.
    event_high : array-like
        Event indicator (1=event, 0=censored) for high-risk group.
    n_boot : int
        Number of bootstrap samples.
    alpha : float
        Significance level (default 0.05 for 95% CI).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    diff_median : float
        Observed difference in median OS (high - low).
    ci : tuple
        Bootstrap (lower, upper) CI.
    boot_diffs : np.ndarray
        Bootstrap distribution of median differences.
    """

    rng = np.random.default_rng(random_state)

    times_low = np.asarray(times_low)
    event_low = np.asarray(event_low)

    times_high = np.asarray(times_high)
    event_high = np.asarray(event_high)

    n_low = len(times_low)
    n_high = len(times_high)

    boot_diffs = []
    for _ in range(n_boot):

        # resample indices
        idx_low = rng.integers(0, n_low, n_low)
        idx_high = rng.integers(0, n_high, n_high)

        times_low_b = times_low[idx_low]
        event_low_b = event_low[idx_low]
        med_low_b = compute_quantile_OS([0.5], event_low_b, times_low_b)

        times_high_b = times_high[idx_high]
        event_high_b = event_high[idx_high]
        med_high_b = compute_quantile_OS([0.5], event_high_b, times_high_b)

        if not (np.isnan(med_low_b[0]) or np.isnan(med_high_b[0])):
            boot_diffs.append(med_high_b[0] - med_low_b[0])

    boot_diffs = np.array(boot_diffs)

    lower = np.percentile(boot_diffs, 100 * alpha / 2)
    upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    return boot_diffs.mean(), (lower, upper), boot_diffs

def bootstrap_cindex(
        model, 
        data, 
        features,
        status="status",
        survival="OS (days) - corrected",
        n_bootstrap=1000, 
        alpha_CI=0.05,
        seed=42
    ):
    """Return (mean, lower 95% CI, upper 95% CI) of bootstrapped C-index."""
    rng = np.random.default_rng(seed)
    cindex_boot = []
    preds = model.predict_partial_hazard(data[features])
    for ib in tqdm(range(n_bootstrap), desc="Bootsrapping procedure"):
        idx = rng.integers(0, len(data), len(data))
        cidx = concordance_index(data[survival].values[idx], -preds.values[idx], data[status].values[idx])
        cindex_boot.append(cidx)
    cindex_boot = np.array(cindex_boot)
    lower = np.percentile(cindex_boot, 100 * (alpha_CI / 2))
    upper = np.percentile(cindex_boot, 100 * (1 - alpha_CI / 2))

    return (cindex_boot.mean(), lower, upper), cindex_boot

def permutation_cindex(
        model,
        data,
        features,
        status="status",
        survival="OS (days) - corrected",
        n_permutations=1000,
        alpha_CI=0.05,
        seed=42
    ):
    """
    Return (observed C-index, p-value, lower 95% CI, upper 95% CI) of
    permutation-based null distribution of the C-index.
    The p-value is the fraction of permutations that achieved a C-index
    >= the observed one.
    """
    rng = np.random.default_rng(seed)
    preds = model.predict_partial_hazard(data[features])
    
    # Observed C-index on the real (unpermuted) data
    observed_cindex = concordance_index(
        data[survival].values, -preds.values, data[status].values
    )

    cindex_perm = []
    for _ in tqdm(range(n_permutations), desc="Permutation procedure"):
        # Permute survival times and status jointly to break the
        # association with the predicted risk while preserving
        # the censoring structure
        perm_idx = rng.permutation(len(data))
        cidx = concordance_index(
            data[survival].values[perm_idx],
            -preds.values,
            data[status].values[perm_idx]
        )
        cindex_perm.append(cidx)

    cindex_perm = np.array(cindex_perm)
    p_value = np.mean(cindex_perm >= observed_cindex)
    lower = np.percentile(cindex_perm, 100 * (alpha_CI / 2))
    upper = np.percentile(cindex_perm, 100 * (1 - alpha_CI / 2))

    return (p_value, lower, upper), cindex_perm

def llr_pvalue(ll_full, ll_reduced, df_diff):
    """Likelihood-ratio test p-value. df_diff = difference in free parameters."""
    lr_stat = 2 * (ll_full - ll_reduced)
    return chi2.sf(lr_stat, df=df_diff)

def to_structured_array(data, event_col, duration_col):
    """Convert a dataframe into the structured array sksurv expects."""
    return np.array(
        [(bool(e), t) for e, t in zip(data[event_col], data[duration_col])],
        dtype=[("event", bool), ("time", float)]
    )