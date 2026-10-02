# Inference and probe selection

Input: an undirected link graph, the known node path of each probe, and a
`pass`, `fail` or `missing` outcome. The engine does not discover routes.
Shortest paths are used only by the synthetic generator.

## Fault posterior

Hypotheses are every link plus `__healthy__`. Assume at most one failed link,
static paths and conditionally independent outcomes. Missing outcomes contribute
no likelihood. For hypothesis $h$, a path containing the failed link has failure
probability $q_1$; all other paths have background probability $q_0$:

$$P(h\mid y) \propto P(h)\prod_i q_i(h)^{y_i}(1-q_i(h))^{1-y_i}$$

The healthy prior is $\pi_0$; the remaining prior mass is uniform over links.
Passed paths are negative evidence for on-path faults. Computation uses log
likelihoods and log-sum-exp normalization.

`fit` estimates both failure rates and the healthy prior from labeled training
records using Beta(1,1) posterior means, `(positive + 1) / (total + 2)`. These are
plug-in estimates; inference does not integrate parameter uncertainty.

## Identifiability

A link's signature records its membership in each observed path. Identical
signatures give identical likelihoods under this model. They remain `ambiguous`
even if prior probabilities differ. In the demo, `core` and `database` share all
observed paths; a new `r1 → r2` probe covers only `core`.

## Active measurement

For candidate probe $p$, compute conditional mutual information:

$$IG(p)=\mathcal H(Y_p)-\sum_h P(h\mid y)\mathcal H(Y_p\mid h)$$

This equals the expected reduction in fault-hypothesis entropy. Rank candidates
by `IG / cost`, using a positive user-supplied cost or 1 by default. This is a
one-step policy, not a globally optimal multi-step budget allocation. Expected
entropy reduction does not guarantee that every individual outcome reduces entropy.

## Decisions and evidence

`candidate` requires a top posterior at or above the threshold and a unique
observed signature. `review` is below threshold, `ambiguous` has an equivalent
signature, and `insufficient_evidence` has no observed outcomes. The qualifying
healthy hypothesis is `consistent_with_healthy`. These are model decisions, not
verified physical causes.

Per-probe log Bayes factors compare the top candidate with the runner-up:
positive supports the leader, negative supports the runner-up, zero is
uninformative. They exclude the prior odds.

## Related work

[Flock (2023)](https://arxiv.org/abs/2305.03348) studies probabilistic fault
localization and inference at datacenter scale. [NetMedic (SIGCOMM 2009)](https://www.microsoft.com/en-us/research/publication/detailed-diagnosis-in-enterprise-networks/)
studies fine-grained dependencies and diagnosis. This project neither implements
those systems nor establishes comparable performance.

[Active Diagnosis under Persistent Noise (AISTATS 2011)](https://proceedings.mlr.press/v15/bellala11a.html)
addresses noisy sequential queries. Our independent-noise model is simpler;
copies of one alarm violate that assumption.

The present contribution is an implementation, experiment protocol and inspection
tool. A new algorithm or field benefit would require additional evidence.
