---
title: "What On-Policy Distillation Quietly Drops — and When It Matters"
date: 2026-10-01
author: Weichen Yu
description: "Two approximations in practical on-policy distillation: future credit and local normalization."
---

# What On-Policy Distillation Quietly Drops — and When It Matters

*Weichen Yu · October 1, 2026 · [中文版](opd-omissions-zh.html)*

> **Status.** Research notes, not peer-reviewed. Claims attributed to papers are linked to the original sources below; the synthesis and additional derivations are my own unless otherwise noted.

*Notes from reading a dozen recent on-policy distillation (OPD) papers with a pencil. Most of the derivations below are short; I've tried to say clearly which ones come from the papers and which ones Claude and I worked out ourselves (unpublished, not peer-reviewed). Thanks also to [Yifei Zhang](https://forrest-110.github.io/) for discussions.*

---

## TL;DR

The standard OPD recipe trains the student with a per-token advantage $\log \pi_T(y_t\mid s_t) - \log \pi_\theta(y_t\mid s_t)$ on its own rollouts, with discount factor 0. Compared with the objective it claims to optimize, the implementation drops two things:

1. **Future credit.** The prefix distribution is treated as fixed, so a token gets no blame for steering the student into states where it later mismatches the teacher. **This does not change the fixed point** when the student can match the teacher everywhere. It only changes the optimization path. The evidence that adding it back helps is mixed and confounded with response length.
2. **A normalizer.** The per-token target is quietly renormalized at every step. This is harmless for a single teacher. **For extrapolated targets (ExOPD, $\lambda \neq 1$) and for token-level mixtures of teachers, it changes the fixed point**: the algorithm converges to a token-level target, not the sequence-level one written in the paper. The dropped term is the cumulative Rényi divergence between the distributions being combined. As far as I can tell, nobody has measured how big it is in practice.

Which of the two can apply depends on the target. For a single teacher, or prompt-level routing between teachers, only the first one does. For extrapolated targets and token-level teacher mixtures, both do.

At the end there is a table of when each omission matters.

---

## 1. The standard recipe

OPD says it minimizes the sequence-level reverse KL from the student to the teacher on the student's own rollouts:

$$
J(\theta) = \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_T(\cdot\mid x)\big)
= \mathbb{E}_{y\sim\pi_\theta}\Big[\sum_t \log\pi_\theta(y_t\mid s_t) - \log\pi_T(y_t\mid s_t)\Big],
\qquad s_t = (x, y_{<t}).
$$

What is actually implemented, in Thinking Machines' OPD blog post, MiMo-V2-Flash and the MOPD paper, G-OPD (Eq. 14), MOPD-Router (Eq. 1), and others, is REINFORCE with the per-token advantage

$$
A_t = \log\pi_T(y_t\mid s_t) - \log\pi_\theta(y_t\mid s_t),
$$

with no reward-to-go (discount factor 0). Some implementations replace the sampled-token estimate with an analytic top-$k$ KL at each state, but the structure is the same: **each position is a separate supervised problem, and the state it sits in is treated as given.**

Two things are being assumed without being stated:

- (a) the states $s_t$ do not depend on $\theta$ (a stop-gradient through the prefix distribution);
- (b) the per-step target is a normalized next-token distribution.

For the vanilla single-teacher case, (b) is true and (a) is a mild approximation. The interesting part is what happens when we leave that case.

This post is only about *fidelity*: given a target, what does the standard implementation drop, and how much does it matter? Which target to distill toward in the first place (for example, how several teachers should be combined) is a separate question, and I'll leave it for another post.

---

## 2. One identity that organizes everything

Let the target be any product of per-step factors $P^*(y) \propto \prod_t \phi(y_t\mid s_t)$. This covers every case discussed below:

| Setting | per-step factor $\phi(a\mid s)$ |
|---|---|
| Vanilla OPD / domain-routed MOPD | $\pi_T(a\mid s)$ |
| ExOPD / G-OPD with reward scale $\lambda$ | $\pi_{\text{ref}}(a\mid s)^{1-\lambda}\,\pi_T(a\mid s)^{\lambda}$ |
| Token-level multi-teacher mixing with weights $w_i(s)$ | $\prod_i \pi_i(a\mid s)^{w_i(s)}$ |

Define the local normalizer $Z(s) = \sum_a \phi(a\mid s)$ and the locally normalized target $q(a\mid s) = \phi(a\mid s)/Z(s)$. Then (derivation: expand the log and regroup):

$$
\boxed{\;
\mathrm{KL}(\pi_\theta\,\|\,P^*) =
\mathbb{E}_{\pi_\theta}\Big[\sum_t \mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\,\|\,q(\cdot\mid s_t)\big)\Big]
\;-\; \mathbb{E}_{\pi_\theta}\Big[\sum_t \log Z(s_t)\Big]
\;+\; \log Z_{\text{seq}}
\;}
$$

Differentiating this gives three pieces of the exact gradient:

| Piece | What it is | Kept by standard OPD? |
|---|---|---|
| **(i)** local term | match $q(\cdot\mid s_t)$ at the current state | ✅ |
| **(ii)** future local KL | reward-to-go of future mismatches $\sum_{t'>t}\mathrm{KL}_{t'}$ | ❌ |
| **(iii)** future normalizer | reward-to-go of $\sum_{t'>t}\log Z(s_{t'})$ | ❌ |

For a single teacher (or one teacher per prompt), $\phi$ is already a normalized distribution, so $Z \equiv 1$ and (iii) vanishes. Only (ii) can be missing. For the other two rows, both can be missing.

(A practical aside: REINFORCE with the *unnormalized* per-token cost $\log\pi_\theta - \log\phi$ and full reward-to-go recovers all three pieces at once. With discount 0, the $\log Z(s_t)$ part has zero expected gradient, which is why a myopic implementation cannot see it even if you do not normalize.)

### Scorecard: which papers keep what

Legend: ✅ kept · ❌ dropped · ⚠️ derived in the paper, then dropped · — does not apply (the per-step target is already normalized, so $Z\equiv1$) · ? not specified in the paper

| Paper | Target | Per-step target normalized? | (ii) future credit | (iii) normalizer |
|---|---|---|---|---|
| ImitKD (2009.07253) | single teacher; DAgger-style mixture of data and student prefixes | yes | ❌ prefixes are treated as fixed data | — |
| GKD (2306.13649) | single teacher; forward KL, reverse KL, or generalized JSD per token | yes | ❌ "we do not backpropagate through the sampling distribution" | — |
| MiniLLM (2306.08543) | single teacher | yes | ✅ full reward-to-go, then replaced by its *mean* (length normalization) | — |
| DistiLLM (2402.03898) | single teacher; skew KL | yes | ❌ student outputs are reused as replay data | — |
| Qwen3 report (2505.09388) | single teacher (Qwen3-32B or 235B-A22B) | yes | ? says only that the student aligns "its logits with those of a teacher model … to minimize the KL divergence" | — |
| Thinking Machines OPD (2025) | single teacher | yes | ❌ discount 0; γ > 0 "did not improve" | — |
| MiMo-V2-Flash (2601.02780) | routed teachers, plus an ORM advantage | yes | ❌ | — |
| MOPD (2606.30406) | routed teachers | yes | ❌ says "following MiniLLM" but keeps only the per-token term | — |
| G-OPD / ExOPD (2602.12125) | extrapolated, $\lambda = 1.25$ | **no** | ⚠️ derived (Eq. 5, App. A), then discount 0 | ❌ not discussed; Eq. 12 omits the normalizer |
| γOPD (2609.16937) | single / routed teachers | yes | ✅ discounted (γ = 0.99), plus bounded mixing with a verifier reward | — |
| TrustMOPD (2609.23697) | token-level mixture | **no** | ❌ | ⚠️ derived (Eq. 29), dismissed as "constant" |
| MOPD-Router (2609.30837) | token-level mixture | **no** (unnormalized advantage) | ❌ | ❌ invisible at discount 0; not discussed |

What the table shows:

- The early papers (ImitKD, GKD) drop (ii) on purpose. They frame OPD as DAgger-style imitation learning, where visited states are data and the expert labels each one; GKD says so explicitly. Discount 0 is inherited from that framing.
- Only MiniLLM and γOPD keep (ii), and both needed stabilizers to make it work (length normalization; discounting plus bounding).
- Every paper whose per-step target is unnormalized (G-OPD, TrustMOPD, MOPD-Router) drops (iii). Two of them write down exactly the quantity involved.
- No paper with a token-level teacher mixture keeps either term.

The two sections that follow are about (ii) and (iii).

---

## 3. Omission #1: future credit (ii)

**What is dropped.** The student's choice at step $t$ decides which states it visits later, and the exact gradient charges that choice for the mismatches it causes downstream. Standard OPD does not. This is partly a deliberate inheritance: GKD (2306.13649) treats distillation as imitation learning with an interactive expert and states that "we do not backpropagate through the sampling distribution", exactly as in DAgger. It is also not news: G-OPD derives the exact gradient in its Appendix A and then adopts discount 0 "following current practice"; γOPD (2609.16937) states it as a formal proposition (token-level OPD is a stop-gradient approximation to sequence-level OPD); MiniLLM (2306.08543) kept the full reward-to-go from the start.

**Is it big? Usually not, because it does not move the fixed point.** If the student can match the teacher at every state, both objectives are minimized by $\pi_\theta = \pi_T$. Dropping (ii) changes the path, not the destination. It matters when the student *cannot* match everywhere. A two-step toy example makes this concrete. Step 1 picks A or B, and the teacher says 50/50. After A the student can copy the teacher exactly; after B it is stuck with an irreducible per-state mismatch $c$. The exact optimum is

$$
\pi_\theta(A) = \frac{1}{1+e^{-c}} > \tfrac12,
$$

so the student routes around the states it cannot imitate. The myopic version stays at $0.5$.

**What the evidence says.**

- Thinking Machines: *"Although more mathematically correct, we do not find discount factors > 0 to improve performance in practice."* No numbers are given, and it is not stated which discount values were tried.
- γOPD: undiscounted ($\gamma = 1$) reward-to-go explodes the gradient norm and collapses entropy; among the reported settings, $\gamma = 0.99$ performs best and $0.9$ is worse. With discounting alone (no reward mixing), the AIME average goes from 53.65 to 55.69 on a same-size 4B→4B pair (their Table 3). The full method's relative gain is larger for a 1.7B student than for a 4B one, which is what you would expect if (ii) mostly matters under a capacity gap.
- The two reports may not actually conflict. "More mathematically correct" suggests $\gamma = 1$, which γOPD also finds harmful. The disagreement is about a narrow window around $\gamma \approx 0.99$, whose effective horizon is about 100 tokens, so it acts more like local smoothing than long-horizon credit. Both pieces of evidence are thin (a footnote without numbers, and a single run on 60 AIME problems).

**Two side effects to keep in mind.**

- *Variance* grows with the remaining horizon. γOPD bounds the undiscounted credit's variance by $(T-t+1)^2\sigma^2$, versus $\sigma^2/(1-\gamma)^2$ when discounted.
- *Length.* Each future term has non-negative expectation (it is a KL), so (ii) is an **implicit length penalty**: every extra token is one more chance to accumulate mismatch. MiniLLM saw this badly enough (students producing empty responses) that it switched to the *mean* rather than the sum of future log-ratios. γOPD reports that its responses converge to a shorter range. So part of any gain from (ii) may come from shorter outputs rather than better credit assignment. That is testable with length-matched comparisons.

**Verdict:** small for same-family, similar-size teacher–student pairs; moderate under a capacity gap or on long-horizon tasks; and any measured gain should be checked against length.

---

## 4. Omission #2: the normalizer (iii)

This is the one that actually changes what you converge to.

### 4.1 ExOPD / G-OPD

G-OPD (2602.12125) rewrites OPD as KL-regularized RL with implicit reward $\log(\pi_T/\pi_{\text{ref}})$ and scales that reward by $\lambda$. Its Eq. 12 gives the sequence-level optimum:

$$
\log\pi_\theta(y\mid x) = \lambda\log\pi_T(y\mid x) + (1-\lambda)\log\pi_{\text{ref}}(y\mid x) \quad(\text{up to a normalizer, which the paper omits}).
$$

The algorithm (Eq. 14) uses the per-token advantage with discount 0. Under discount 0, each state is pushed toward the *locally* normalized geometric mixture $q_\lambda \propto \pi_{\text{ref}}^{1-\lambda}\pi_T^{\lambda}$. A product of locally normalized tempered distributions is **not** the sequence-level tempered distribution. The gap is exactly term (iii), and here the local normalizer has a closed form:

$$
\log Z(s) = \log\sum_a \pi_T(a\mid s)^{\lambda}\,\pi_{\text{ref}}(a\mid s)^{1-\lambda} = (\lambda-1)\,D_\lambda\big(\pi_T(\cdot\mid s)\,\|\,\pi_{\text{ref}}(\cdot\mid s)\big),
$$

where $D_\lambda$ is the Rényi divergence of order $\lambda$. For $\lambda > 1$, Jensen's inequality gives $Z(s) = \mathbb{E}_{\pi_{\text{ref}}}[(\pi_T/\pi_{\text{ref}})^\lambda] \ge 1$. So the sequence-level objective contains a **non-negative per-state bonus** $(\lambda-1)D_\lambda$, which rewards steering into states where RL changed the teacher the most. The discount-0 implementation never sees it. At $\lambda = 1$ the term vanishes, which is why vanilla OPD is safe.

**Length, two mechanisms.** G-OPD observes that ExOPD responses keep getting longer and attributes this to "the length bias issue of the implicit reward." That is correct for the *objective*: the bonus is summed over states, so longer responses collect more of it. But the *algorithm* they ran (discount 0) does not contain that sum. A different mechanism is more likely at work there. Token-level extrapolation amplifies whatever RL did to each token's probability. Math RL tends to lower the probability of EOS and raise the probability of "wait, let me check" tokens, so $\lambda > 1$ pushes further in the same direction. This second mechanism depends on the teacher: a teacher that RL made *shorter* would make the student shorter under myopic ExOPD, while the sequence-level objective would still pull toward longer outputs. That gives a clean experiment for telling the two apart.

G-OPD attributes the instability at $\lambda = 1.5$ to the student "hacking the implicit reward … by aggressively fitting the peak of the log ratio." In this framework, those peaks are where $D_\lambda$ is large, and I would expect them to be concentrated on off-support prefixes, where the teacher's log-ratios are least reliable. That is also checkable.

### 4.2 Token-level multi-teacher mixing

MOPD-Router's advantage is $\sum_i w_{i,t}(\log\pi_i - \log\pi_{\text{old}})(y_t)$ with weights summing to one. TrustMOPD minimizes $\sum_k w_k\,\mathrm{KL}(\pi_\theta\|\pi_k)$ at each state. Both targets are geometric mixtures of the teachers, i.e. the third row of the table in §2. TrustMOPD's Appendix A.4 even writes down the identity

$$
\sum_k w_k\,\mathrm{KL}(p_\theta\|q_k) = \mathrm{KL}(p_\theta\|\tilde q_{\text{mix}}) - \log Z_{\text{mix}},
$$

and then drops $\log Z_{\text{mix}}$ because it *"is constant with respect to the student parameters during the update."* That is true at a fixed state. It is not true on-policy, because $\sum_t \log Z_{\text{mix}}(s_t)$ depends on $\theta$ through which states the student visits.

With weights on the simplex, the weighted AM–GM inequality gives $Z \le 1$, so (iii) is a **penalty**. For two teachers with weights $(1-w, w)$,

$$
\log Z(s) = -(1-w)\,D_w\big(\pi_2(\cdot\mid s)\,\|\,\pi_1(\cdot\mid s)\big),
$$

which is (minus) the Rényi divergence between the teachers at that state. The sequence-level target says *avoid prefixes that lead to teacher disagreement*. The token-level implementation walks into those regions anyway and then learns a locally renormalized compromise that neither teacher would have produced.

**Verdict:** conceptually this is the bigger omission, because it changes the target and not just the path. Empirically it has not been measured. Its size per state is a Rényi divergence that can be computed from top-$k$ logits the pipeline already has, so measuring it is nearly free.

---

## 5. When does what matter?

| Setting | (ii) future credit | (iii) normalizer |
|---|---|---|
| Single teacher, same family and size | small (path only) | none ($Z\equiv1$) |
| Strong → weak (capacity gap) | **moderate**: student should avoid states it cannot match | none |
| Long-horizon / agentic | potentially larger, but variance grows with horizon | none |
| ExOPD, $\lambda \ne 1$ | as above | **yes**: $(\lambda-1)D_\lambda$ bonus; changes the target; biases toward length |
| Multi-teacher, prompt-level routing | as single teacher | none |
| Multi-teacher, token-level mixing | as above | **yes**: disagreement penalty $-\log Z$ |
| Token-level mixing where competent teachers genuinely disagree (e.g. safety vs capability) | as above | **large** |

Length biases, summarized: (ii) pushes **shorter**; (iii) under ExOPD pushes **longer**; (iii) under token-level mixing pushes **shorter**; myopic ExOPD **amplifies whatever length shift the teacher's RL induced**. Any comparison that adds one of these terms back should be length-matched.

Cells in bold are where I expect an effect. Most of the table is our reading of the math plus a handful of single-run results, not a set of established facts.

---

## References

- Lin et al. [*Autoregressive Knowledge Distillation through Imitation Learning*](https://arxiv.org/abs/2009.07253) (ImitKD). EMNLP 2020.
- Agarwal et al. [*On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*](https://arxiv.org/abs/2306.13649) (GKD). ICLR 2024.
- Gu et al. [*MiniLLM: Knowledge Distillation of Large Language Models*](https://arxiv.org/abs/2306.08543). ICLR 2024.
- Ko et al. [*DistiLLM: Towards Streamlined Distillation for Large Language Models*](https://arxiv.org/abs/2402.03898). ICML 2024.
- Qwen Team. [*Qwen3 Technical Report*](https://arxiv.org/abs/2505.09388).
- Lu & Thinking Machines Lab. [*On-Policy Distillation*](https://thinkingmachines.ai/blog/on-policy-distillation/). Connectionism, 2025.
- Xiao et al. [*MiMo-V2-Flash Technical Report*](https://arxiv.org/abs/2601.02780).
- Yang et al. [*Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation*](https://arxiv.org/abs/2602.12125) (G-OPD / ExOPD).
- Ma et al. [*MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training*](https://arxiv.org/abs/2606.30406).
- Sun et al. [*Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation*](https://arxiv.org/abs/2609.23697) (TrustMOPD).
- Xu et al. [*MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation*](https://arxiv.org/abs/2609.30837).
- Liu et al. [*Beyond Token-Local Imitation: Reward-Compatible Temporal Credit Assignment for On-Policy Distillation*](https://arxiv.org/abs/2609.16937) (γOPD).
