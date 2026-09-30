"""The Method and References sections of the report: prose only."""

from utils.report import REPORT


def report_method_and_references():
    """Append the method notes and the reference list to the report. No arguments.

    Every choice here is a judgement that could reasonably have gone the other
    way, so each is stated next to the numbers it produced rather than left in a
    README a reader of the report may never open.
    """
    REPORT.heading("Method, and the choices behind it")

    REPORT.heading("What the correction assumes", level=3)
    REPORT.paragraph(
        "A survival gap between two groups of cohorts has three possible sources, "
        "and they call for different responses. If the clock starts at a different "
        "event in one cohort (entry point), that is an artefact of record-keeping "
        "and should be removed. If the cohorts genuinely hold different patients "
        "-- more methylated MGMT, more gross-total resections, older patients "
        "(case-mix) -- that is a real prognostic difference and must be kept, "
        "because removing it by rescaling the outcome destroys the signal the "
        "analysis is trying to measure; worse, a downstream model that also "
        "adjusts for those covariates would then remove the same effect twice. And "
        "if one cohort lost its sicker patients to follow-up (informative "
        "censoring), its survival looks better than it was; that is an artefact "
        "too, but not one a constant rescaling of the survival times can undo.")
    REPORT.paragraph(
        "The correction can only express the first. So the diagnostics work by "
        "elimination: the adjustment ladder takes out case-mix, the censoring "
        "diagnostics ask how much of the remainder censoring could produce, and "
        "only a remainder that survives both is a candidate for entry point -- "
        "one candidate among several, since unrecorded case-mix, differences in "
        "treatment after the clock starts, and chance leave the same trace. The "
        "sources also overlap. Censoring that depends on the covariates is a "
        "case-mix effect as far as the adjusted hazard ratio is concerned, and is "
        "absorbed by the adjustment; censoring that depends on something "
        "unrecorded is not. Entry point and censoring both act mostly in the first "
        "months after the clock starts, so a remainder concentrated early is "
        "consistent with either, and these data cannot tell them apart.")
    REPORT.paragraph(
        "The correction itself multiplies group 1's survival times by exp(logHR), "
        "leaving the reference group untouched. That is a single constant for "
        "everyone in the group, so it is the right shape only if the site log "
        "hazard ratio is the same early and late -- which is why proportional "
        "hazards is tested rather than assumed. Stratifying the baseline hazard by "
        "cohort [1] is the equivalent alternative and needs no rescaling at all; "
        "the two are alternatives, and applying both corrects the same difference "
        "twice.")

    REPORT.heading("Covariate adjustment and missing data", level=3)
    REPORT.paragraph(
        "The adjusted coefficient is estimated on the subjects reporting every "
        "chosen covariate and then applied to all of them, so the assembled table "
        "never shrinks. Missingness here is severe and cohort-structured rather "
        "than random -- EOR is unrecorded for all of TCGA, MGMT for all of RHUH, "
        "KPS for all of UCSF and LUMIERE -- so a complete-case table would cost "
        "most of the sample. The transfer assumes the site effect is the same in "
        "complete and incomplete cases, which is an assumption doing real work; "
        "the balance table's missingness columns are the evidence to weigh it "
        "against. Imbalance is measured by the standardised mean difference "
        "(SMD) [7], which unlike a p-value does not shrink as the sample grows. "
        "For a covariate with mean \\(\\bar{x}_g\\) and sample variance "
        "\\(s_g^2\\) in site group \\(g\\),")
    REPORT.equation(
        r"\mathrm{SMD} = \frac{|\bar{x}_1 - \bar{x}_0|}"
        r"{\sqrt{(s_0^2 + s_1^2)/2}},\qquad"
        r" s_g^2 = \frac{1}{n_g - 1}\sum_{i \in g} (x_i - \bar{x}_g)^2.")
    REPORT.paragraph(
        "Each \\(s_g^2\\) is the unbiased sample variance, and the two are "
        "averaged unweighted, so the larger group does not dominate the scale "
        "and the groups are not assumed to share a variance. A categorical "
        "covariate is compared one level at a time on the 0/1 indicator of that "
        "level; \\(\\bar{x}_g\\) is then the proportion \\(p_g\\) and the "
        "same formula gives \\(s_g^2 = p_g(1 - p_g)\\,n_g/(n_g - 1)\\) -- one "
        "definition for both kinds of covariate, rather than the plug-in "
        "\\(p_g(1 - p_g)\\) often used for proportions [7], from which it differs "
        "only by \\(n_g/(n_g - 1)\\).")

    REPORT.heading("Proportional hazards, and what is done when it fails", level=3)
    REPORT.paragraph(
        "Every coefficient in this report comes from a Cox model [1]. For subject "
        "\\(i\\) with site indicator \\(s_i\\) and case-mix covariates "
        "\\(x_{i1}, \\dots, x_{ip}\\) (the dummy-coded columns of the adjustment "
        "set), the hazard is")
    REPORT.equation(
        r"h_i(t) = h_0(t)\,\exp\!\Big(\beta_{\mathrm{site}}\,s_i"
        r" + \sum_{k=1}^{p} \beta_k\,x_{ik}\Big),")
    REPORT.paragraph(
        "where \\(h_0(t)\\) is left unspecified. Proportional hazards is the claim "
        "that each \\(\\beta\\) is a constant, so that the hazard ratio between any "
        "two subjects does not depend on time:")
    REPORT.equation(
        r"\frac{h_i(t)}{h_j(t)} = \exp\!\big(\boldsymbol\beta^{\top}"
        r"(\mathbf{x}_i - \mathbf{x}_j)\big) \quad \text{for every } t.")
    REPORT.paragraph(
        "It is tested for every term of the model the coefficient comes from, not "
        "for the site term alone: \\(\\hat\\beta_{\\mathrm{site}}\\) is estimated "
        "while holding the covariates fixed, so a covariate whose own effect drifts "
        "with time makes that adjustment a misspecified one. The alternative each "
        "term is tested against lets its coefficient move with a known function of "
        "time \\(g(t)\\):")
    REPORT.equation(
        r"\beta_k(t) = \beta_k + \theta_k\,g(t), \qquad"
        r" H_0:\ \theta_k = 0.")
    REPORT.paragraph(
        "The Grambsch-Therneau test [4, 5] is the score test for \\(\\theta_k\\), "
        "and needs only the constant-coefficient fit. Order the \\(D\\) event times "
        "\\(t_1 \\le \\dots \\le t_D\\), let \\(i_j\\) be the subject failing at "
        "\\(t_j\\) and \\(R(t_j)\\) everyone still at risk then. The Schoenfeld "
        "residual is the failing subject's covariates minus their risk-weighted "
        "average over the risk set, and scaling it by the inverse information "
        "turns it into a noisy reading of the coefficient at that moment:")
    REPORT.equation(
        r"\mathbf r_j = \mathbf x_{i_j} - \frac{\sum_{l \in R(t_j)} \mathbf x_l\,"
        r"e^{\hat{\boldsymbol\beta}^{\top}\mathbf x_l}}"
        r"{\sum_{l \in R(t_j)} e^{\hat{\boldsymbol\beta}^{\top}\mathbf x_l}},"
        r"\qquad \mathbf s^{*}_j = D\,\widehat{\operatorname{Var}}"
        r"(\hat{\boldsymbol\beta})\,\mathbf r_j,"
        r"\qquad \mathbb E\big[s^{*}_{kj}\big] + \hat\beta_k \approx \beta_k(t_j).")
    REPORT.paragraph(
        "A time trend in \\(\\beta_k\\) is therefore a correlation between "
        "\\(s^{*}_{kj}\\) and \\(g(t_j)\\), and the statistic measures it:")
    REPORT.equation(
        r"T_k = \frac{\Big[\sum_{j=1}^{D} (g_j - \bar g)\, s^{*}_{kj}\Big]^2}"
        r"{D\;\widehat{\operatorname{Var}}(\hat\beta_k)\,\sum_{j=1}^{D}"
        r"(g_j - \bar g)^2} \;\sim\; \chi^2_1 \ \text{under } H_0,"
        r"\qquad g_j = j.")
    REPORT.paragraph(
        "\\(g_j = j\\) is the rank transform, the lifelines [9] default; R's "
        "cox.zph defaults to the Kaplan-Meier transform instead. With \\(K\\) "
        "terms screened in "
        "one pass, the p-value that decides which terms are followed up is the "
        "Bonferroni-adjusted one, not the raw one; the raw p is reported beside "
        "it so the more sensitive reading stays visible:")
    REPORT.equation(
        r"p_k^{\mathrm{Bonf}} = \min\big(1,\ K\,p_k\big), \qquad"
        r" \text{term } k \text{ is followed up if } p_k^{\mathrm{Bonf}} < 0.05.")
    REPORT.paragraph(
        "A term that fails is then described rather than merely flagged. It is "
        "refitted with a log-time interaction, one model per failing term, the "
        "other terms keeping constant coefficients:")
    REPORT.equation(
        r"h_i(t) = h_0(t)\,\exp\!\Big(\sum_{l} \beta_l\,x_{il}"
        r" + \theta_k\,x_{ik}\,\log\frac{t}{t_{\mathrm{ref}}}\Big)"
        r"\quad\Longleftrightarrow\quad"
        r"\beta_k(t) = \beta_k + \theta_k \log\frac{t}{t_{\mathrm{ref}}},")
    REPORT.paragraph(
        "with \\(t_{\\mathrm{ref}}\\) the median event time, so \\(\\beta_k\\) is "
        "the log hazard ratio at that moment rather than at \\(t = 1\\) day. "
        "Equivalently \\(\\mathrm{HR}_k(t) = e^{\\beta_k}\\,(t/t_{\\mathrm{ref}})"
        "^{\\theta_k}\\): the hazard ratio is multiplied by \\(2^{\\theta_k}\\) "
        "each time follow-up doubles, and \\(\\theta_k = 0\\) is proportional "
        "hazards. Because the covariate \\(x_{ik}\\log(t/t_{\\mathrm{ref}})\\) "
        "changes with time, the data are split at the event times "
        "\\(t_{(1)} < \\dots < t_{(D)}\\) into intervals "
        "\\((t_{(m-1)}, t_{(m)}]\\), and the model is fitted by the partial "
        "likelihood (written here without the tie correction)")
    REPORT.equation(
        r"\ell(\boldsymbol\beta, \theta_k) = \sum_{m=1}^{D} \Big[\eta_{i_m}(\tau_m)"
        r" - \log \sum_{l \in R(t_{(m)})} e^{\eta_l(\tau_m)}\Big],"
        r"\qquad \eta_l(\tau) = \sum_{q} \beta_q x_{lq}"
        r" + \theta_k\,x_{lk}\log\frac{\tau}{t_{\mathrm{ref}}},"
        r"\qquad \tau_m = \max\big(t_{(m-1)},\ 1\ \text{day}\big).")
    REPORT.paragraph(
        "Three details in that expression decide whether the fit is an estimate "
        "or an artefact, and all three are deliberate. First, the split is at the "
        "event times rather than on a fixed grid, so the whole risk set "
        "\\(R(t_{(m)})\\) shares one interval and one value of log time; a monthly "
        "grid attenuates \\(\\theta_k\\) by roughly a third. Second, the "
        "interaction is evaluated at \\(\\tau_m\\), the interval's start, the value "
        "every member of the risk set shares -- evaluating it at the interval end "
        "hands the subject who fails a systematically smaller time than the "
        "controls, which on simulated data with no time trend at all rejects "
        "proportional hazards in twelve of twelve replicates. Third, there is no "
        "main effect of time: a term \\(\\gamma\\log\\tau_m\\) is the same for "
        "every subject in the sum, so \\(e^{\\gamma\\log\\tau_m}\\) cancels between "
        "numerator and denominator, \\(\\gamma\\) never enters "
        "\\(\\ell\\), and the fit cannot converge. When a pool has more than 1000 "
        "distinct event times the cuts are taken at 1000 quantiles instead, which "
        "shrinks \\(\\theta_k\\) slightly toward zero.")
    REPORT.paragraph(
        "\\(\\theta_k\\) is tested by likelihood ratio against the "
        "constant-coefficient fit \\(\\tilde{\\boldsymbol\\beta}\\) on the same "
        "split data, and the band drawn around \\(\\hat\\beta_k(t)\\) is the "
        "pointwise 95% interval from the joint covariance of "
        "\\((\\hat\\beta_k, \\hat\\theta_k)\\):")
    REPORT.equation(
        r"\Lambda_k = 2\big[\ell(\hat{\boldsymbol\beta}, \hat\theta_k)"
        r" - \ell(\tilde{\boldsymbol\beta}, 0)\big] \;\sim\; \chi^2_1,")
    REPORT.equation(
        r"\hat\beta_k(t) \pm 1.96\sqrt{\widehat{\operatorname{Var}}(\hat\beta_k)"
        r" + u^2\,\widehat{\operatorname{Var}}(\hat\theta_k)"
        r" + 2u\,\widehat{\operatorname{Cov}}(\hat\beta_k, \hat\theta_k)},"
        r"\qquad u = \log\frac{t}{t_{\mathrm{ref}}}.")
    REPORT.paragraph(
        "The band is narrowest at \\(t_{\\mathrm{ref}}\\), where \\(u = 0\\), and "
        "widens in both directions; it is pointwise, not a simultaneous band over "
        "the whole curve.")

    REPORT.heading("Censoring: what is measured and what is only stressed", level=3)
    REPORT.paragraph(
        "Under independent censoring the Cox partial likelihood is consistent "
        "however differently the two groups were censored, so differential "
        "follow-up costs precision rather than unbiasedness. Three diagnostics "
        "probe the assumption. The reverse Kaplan-Meier [6] says how long each "
        "group was watched; the person-time rates [11, 12] say how completely, "
        "without counting a death as a loss. The censoring-hazard models say "
        "whether censoring depends on the recorded covariates: if it does, the "
        "adjusted site model is still consistent, because censoring is "
        "independent given what it conditions on, but no marginal Kaplan-Meier "
        "curve is. The tipping point [13] then asks how far censoring would have "
        "to depart from independence to move the adjusted site effect. For each "
        "censored subject \\(i\\) of the shifted group, with censoring time "
        "\\(c_i\\) and linear predictor \\(\\eta_i\\) from the adjusted model, a "
        "death time is drawn from")
    REPORT.equation(
        r"H_0(T_i^{*}) = H_0(c_i) + \frac{E_i}{\delta\, e^{\eta_i}},"
        r"\qquad E_i \sim \operatorname{Exp}(1),")
    REPORT.paragraph(
        "with \\(H_0\\) the Breslow baseline recomputed under coefficients redrawn "
        "from their sampling distribution for every imputation, and the site "
        "coefficients of the refitted models pooled by Rubin's rules [14]. "
        "\\(\\delta = 1\\) is independent censoring; \\(\\delta = 2\\) says the "
        "patients lost to follow-up died twice as fast as comparable patients who "
        "stayed. The shift is applied to each group in turn, and to every "
        "censored subject of it -- administrative censorings included, since the "
        "tables do not say which censorings were administrative.")

    REPORT.heading("What cannot be checked here", level=3)
    REPORT.paragraph(
        "Whether censoring depends on something nobody recorded. UCSF records no "
        "KPS at all, and performance status is the obvious reason a patient stops "
        "returning to the centre that diagnosed them. The tipping point bounds "
        "how much that could matter; it cannot say which delta is true. If "
        "censoring is informative beyond the covariates, the bias sits inside "
        "every estimate above.")

    REPORT.heading("Estimation details", level=3)
    REPORT.paragraph(
        "The baseline hazard is Breslow's estimator [2] throughout, requested "
        "explicitly. Ties in the partial likelihood are a separate matter: "
        "lifelines [9] implements Efron's approximation [3] only and offers no way "
        "to change it, so every diagnostic fit here breaks ties by Efron, while "
        "the coefficient actually applied to the survival times comes from "
        "scikit-survival [10], which uses Breslow for both. With many tied survival "
        "days the two differ in the second or third decimal. Survival curves are "
        "Kaplan-Meier estimates [8] with log-log confidence bands.")

    REPORT.heading("References", level=3)
    REPORT.references([
        "Cox DR (1972). Regression models and life-tables. Journal of the Royal "
        "Statistical Society: Series B 34(2), 187-220.",
        "Breslow N (1974). Covariance analysis of censored survival data. "
        "Biometrics 30(1), 89-99.",
        "Efron B (1977). The efficiency of Cox's likelihood function for censored "
        "data. Journal of the American Statistical Association 72(359), 557-565.",
        "Grambsch PM, Therneau TM (1994). Proportional hazards tests and "
        "diagnostics based on weighted residuals. Biometrika 81(3), 515-526.",
        "Therneau TM, Grambsch PM (2000). Modeling Survival Data: Extending the "
        "Cox Model. Springer, New York.",
        "Schemper M, Smith TL (1996). A note on quantifying follow-up in studies "
        "of failure time. Controlled Clinical Trials 17(4), 343-346.",
        "Austin PC (2009). Balance diagnostics for comparing the distribution of "
        "baseline covariates between treatment groups in propensity-score matched "
        "samples. Statistics in Medicine 28(25), 3083-3107.",
        "Kaplan EL, Meier P (1958). Nonparametric estimation from incomplete "
        "observations. Journal of the American Statistical Association 53(282), "
        "457-481.",
        "Davidson-Pilon C (2019). lifelines: survival analysis in Python. Journal "
        "of Open Source Software 4(40), 1317.",
        "Poelsterl S (2020). scikit-survival: a library for time-to-event analysis "
        "built on top of scikit-learn. Journal of Machine Learning Research "
        "21(212), 1-6.",
        "Xue X, Agalliu I, Kim MY, Wang T, Lin J, Ghavamian R, Strickler HD (2017). "
        "New methods for estimating follow-up rates in cohort studies. BMC Medical "
        "Research Methodology 17, 155.",
        "Clark TG, Altman DG, De Stavola BL (2002). Quantification of the "
        "completeness of follow-up. The Lancet 359(9314), 1309-1310.",
        "Jackson D, White IR, Seaman S, Evans H, Baisley K, Carpenter J (2014). "
        "Relaxing the independent censoring assumption in the Cox proportional "
        "hazards model using multiple imputation. Statistics in Medicine 33(27), "
        "4681-4694.",
        "Rubin DB (1987). Multiple Imputation for Nonresponse in Surveys. Wiley, "
        "New York.",
    ])
