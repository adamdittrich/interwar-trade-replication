# Computational and econometric details

## 1. Outcome and estimand

The dependent variable is the share of a reporting country's world exports or imports assigned to a partner. Exporter and importer declarations are estimated separately. These are equally weighted national portfolio allocations, not a common-currency matrix of trade values. Dividing outcomes by reporter-year totals changes PPML weights and can change estimated slopes even when the model contains reporter-year effects.

Origin-year and destination-year effects condition on the margins of the observed share matrix. The constructed External category closes each reporting portfolio but does not identify trading relationships within that category. The analysis does not claim a complete structural general-equilibrium model for the omitted countries.

## 2. Primary prediction on established recorded links

A directed cell belongs to the established support if its recorded shares have a strictly positive sum over 1927–1929. The support is fixed before evaluation. Pre-period observations on this support are fitted by

\[
E[y_{ijt}]=\exp(\mu_{ij}+\pi_{it}+\chi_{jt}).
\]

The pair effects are held fixed when predicting each 1930–1937 allocation. Current origin and destination effects are re-estimated on that same support. Recorded zeros in evaluation years remain zero observations. No pseudo-count is added to the primary training outcomes or evaluation outcomes.

This is a conditional prediction exercise: the current margins are observed and fitted, not forecast. The margin is the current total on the selected support; it need not sum to the reporter's complete world portfolio. The main score is

\[
P_t=1-D(y_t,\widehat y^H_t)/D(y_t,\widehat y^M_t),
\]

where H fixes the pre-period pair affinity and M has current margins only. All models in a given comparison use identical observations. The score is a reduction in predictive Poisson deviance, not a fraction of trade physically caused by past relationships. It may be negative.

The package reports both an absolute total-variation error measured in the original world-share units and a total-variation error conditional on the mass remaining on the selected support. The original shares are not renormalized for estimation.

## 3. Newly positive recorded cells

For every reporter-year, the code records trade shares in cells that were zero in all three training years but are subsequently positive. It also records established cells whose current recorded share is zero. These accounting quantities are presented separately from the established-link score.

A newly positive recorded cell need not be a new economic trading relationship: source rounding and reporting thresholds can hide small positive transactions. The codebook describes those conventions. The code does not interpret a source threshold as an exact economic zero.

## 4. Geography and fixed-horizon comparisons

The geography model estimates log capital distance and the documented interwar land-border indicator on established internal dyads in 1927–1929, with origin-year and destination-year effects. Its coefficients are held fixed during evaluation; current margins are fitted again. History's increment relative to geography is 1 minus historical deviance divided by geography deviance, on identical support.

Rolling maps use either the preceding year or the preceding three years. At each target year, the anchored 1927–1929 map and the rolling map are estimated and evaluated on the intersection of their positive-training supports. This separates ageing of a fixed reference map from short-horizon allocation predictability. The short pre-period is not an undisturbed historical control, and this comparison does not estimate the causal effect of the Depression.

## 5. Sensitivity to country composition and declaration system

Delete-country calculations remove every named-country dyad incident to the omitted country, on both the origin and destination sides. The fixed External category is unchanged; trade is not silently reassigned into it. The pre-period map, geography model where relevant, and evaluation margins are refitted. Intervals are the full-sample estimate plus or minus 1.96 times the conventional delete-country jackknife standard error. They describe sensitivity to country composition and are not an assertion that the observed countries are an independently sampled population.

A separate exporter–importer comparison uses a common eighteen-country internal network excluding Spain and the intersection of the two declarations' positive pre-period supports. Differences are evaluated on the same directed cells. The outcomes retain their distinct reporter denominators and customs conventions, so declaration differences are not attributed to a behavioral asymmetry alone.

Alternative training windows and sample exclusions are generated in the established-link robustness table. Their support rules are explicit, and alternative maps are compared with the main map on the same support.

## 6. Geography-conditioned reassignment

Pair effects are not uniquely normalized: country-specific additions can be offset by corresponding changes in the country-time effects. 

The reassignment exercise subtracts the fitted geography component from pre-period affinity and projects the remainder off origin and destination dummies. The resulting canonical residual is invariant to additive country normalizations. The code verifies both this invariance and preservation of the original fitted allocation. These residuals are reassigned within reporters while geography stays attached to the actual dyad; current margins are then refitted.

There are 499 deterministic draws per declaration, with seeds 20261027 and 20261028. The resulting distribution is a descriptive counterfactual comparison. It is not an exact randomization test, does not assume randomized trade partners, and does not claim to preserve the concentration of observed trade shares. No randomization p-value is reported.

## 7. Full-matrix smoothing diagnostic

The appendix retains the full recorded matrix and repeats the inherited-map calculation with training additions of 1e-6, 1e-8, 1e-10, and 1e-12. Evaluation outcomes remain unchanged. These results expose the sensitivity of a positive later flow to the prediction assigned to an all-zero training cell. They are not the primary estimates.

For a positive observation predicted near zero, Poisson deviance increases approximately with minus the logarithm of that prediction. Consequently, describing a pseudo-count merely as a numerical convenience would obscure a substantive modeling choice. The primary established-support design avoids that choice and reports newly positive mass separately.

## 8. Supporting dynamic specifications

Annual attenuation is restricted to the same positive-preperiod support and uses the exact log of each positive 1927–1929 mean share, standardized across supported pairs within reporter, with directed-pair, origin-year, and destination-year effects. Post-period recorded zeros remain in the sample. The interaction for 1928 is included, and 1929 is the sole omitted reference. Candidate step breaks from 1930 to 1935 are ranked by deviance; their dates are descriptive profile choices, not identified policy shocks.

Germany–Balkan event profiles use the full observed internal panel over 1928–1937. The 1928 interaction is included, and 1933 is the sole omitted year. Joint pretrend tests cover all 1928–1932 interactions. The Balkan group is Bulgaria, Greece, Hungary, Romania, and Yugoslavia. Hub comparisons and a group-specific linear trend plus a post-1934 level shift are supplementary descriptions. A failure to reject a pretrend does not establish parallel trends, and the design does not identify a clean causal New Plan treatment effect.

The appendix lag-persistence exercise relates current shares to the exact log of a strictly positive recorded lagged share, with current origin-year and destination-year effects. Zero-lag observations are excluded explicitly; current recorded zeros remain in the estimation and blending samples. No pseudo-count is used. Its optional bounded blend with a fitted pair-effect model minimizes in-sample squared error. This is a descriptive fitted-prediction blend, not a structural estimate of a fraction of sticky contracts. It is not a headline result or a substitute for the established-map prediction exercise.

## 9. Estimator and convergence

`src/ppml_hdfe.py` solves conditional fixed-effect score equations by iterative proportional fitting. Low-dimensional coefficients are updated by profiled Newton steps using the Schur complement of the nuisance-parameter information. Weighted absorption supplies this profile Hessian.

Convergence requires a small profile coefficient step and tightly solved fixed-effect scores. A small likelihood change alone is never sufficient. The routine raises an exception if fixed-effect fitting, absorption, coefficient updating, or line search fails. Linear predictors are not clipped in the objective.

Exactly zero-total fixed-effect groups are removed iteratively. Positive groups are never dropped solely because their totals are small. This automatic rule covers one specific form of separation; it is not a general separation detector. The returned retention mask identifies removed observations. The primary training support excludes all-zero pre-period pairs by design. A zero-total evaluation margin has a fitted mean of zero.

The coefficient sandwich covariance uses fitted-mean-weighted residualized regressors. Scores are summed by directed pair and multiplied by G/(G−1), where G is the number of clusters. No additional residual-degrees-of-freedom correction for absorbed effects is applied. Reported coefficient intervals use normal critical values; these are not exact small-treated-group confidence intervals.

## 10. Numerical settings and reproducibility

Training and conditional-margin tolerances are specified at the estimation calls in `portfolio_analysis.py` and `run_all.py`. The primary affinity fits use a coefficient tolerance of 1e-11 and at most 10,000 iterations. Other specifications declare their own tolerances. Convergence failures raise exceptions rather than being silently accepted.

`python tests/run_tests.py` checks sample accounting, headline-result identities, event-reference years, and agreement with an independently implemented expanded-dummy Poisson estimator. It uses the same dependencies as the analysis. Generated CSV and JSON files retain unrounded estimates; display rounding is confined to `build_manuscript_assets.py`.
