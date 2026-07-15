# UQ / Hallucination-Detection SOTA Review

*A structured read of the 24 papers in `UQ_SOTA/`, oriented around our two projects:*
- **P1 — cross-scale SE transfer** (`research_proposal.md`; results in `sep_scratch/transfer/TRANSFER_SUMMARY.md`)
- **P2 — SE-guided on-policy distillation** (`research_proposal_distillation.md`)

**How to read this.** Section 1 covers the two papers that are *direct counterparts* — they attack or overlap the exact claim we make and must be addressed head-on. Section 2 covers *competitors to semantic-entropy estimation itself* (cheaper or better substitutes for the SE signal our probe approximates). Section 3 covers SE-*adjacent* probing (alternative readouts, competitors to SEP not to SE). Section 4 covers *acting on uncertainty* — the closest prior art to P2. Section 5 is the action list. The full 24-paper index is at the end.

---

## 1. Direct counterparts (address head-on)

### 1.1 "The Geometries of Truth Are Orthogonal Across Tasks" — 2506.08572 (Apple, ICML 2025 workshop)

**What it claims.** Linear *truth* probes trained on one task **do not transfer** to other tasks. Across seven datasets — **BioASQ, SVAMP, GSM8K, SimpleQA, TriviaQA, NQ, SQuAD** (the same short-form QA suite the SEP line and *we* use) — probe weight vectors are near-orthogonal (cosine < 0.5 for most pairs), sparse-probe supports are nearly disjoint (<15% overlap for most pairs), and — the sharpest result — **training on a mixture of all tasks except the target still fails**, and the target direction cannot be recovered as a linear combination of the others. t-SNE shows the correct/incorrect split is *secondary* to task boundaries: activations cluster by task first, truth second.

**The numbers that matter for us (their Fig. 2 cosine-similarity matrix):**
- TriviaQA–NQ = **0.73** — the one strongly-aligned pair, and the one that transfers well.
- SQuAD–NQ = 0.34, SQuAD–TriviaQA = 0.39, most other pairs 0.04–0.43.
- Correlation between probe cosine-similarity and cross-task AUROC transfer: **r = 0.59** — so orthogonality *causes* the transfer failure, it isn't incidental.

**Why this is the single most important paper to confront.** Our `cross_dataset.py` result (TRANSFER_SUMMARY §"Cross-distribution test") reports that our pipeline *does* transfer across datasets — within-distribution mean 0.811, cross-distribution mean 0.740, only ~0.07 AUROC drop — which on its face **contradicts** their headline. If a reviewer knows this paper, "we found cross-task transfer works" reads as either naive or wrong unless we reconcile it explicitly.

**The reconciliation (and why it's actually a *strength* for us).** The two results are not in conflict, because **we probe a different target**:
- They probe **truth / correctness** (is this answer right?).
- We probe **semantic entropy** (is the model internally uncertain about the meaning?).

These diverge — our own Q6 finding and paper 2503.14477 both confirm SE ≠ correctness. The defensible, and genuinely interesting, framing becomes:

> *Uncertainty geometry transfers across tasks and scale where truth geometry does not.*

That is a stronger claim than "transfer works" — it says the *uncertainty* feature is more universal/convergent than the *truth* feature, which is exactly the Platonic-convergence story P1 is built on. Two concrete corroborations already in our data:
- Their matrix says TriviaQA–NQ is the aligned pair; our cross-dataset matrix should be checked against this — if our SE transfer is *also* strongest TriviaQA↔NQ, that's independent agreement on the geometry even though the sign of the conclusion differs.
- Their "squad is broad" is echoed in our finding that a squad-fit pipeline transfers everywhere.

**Caveats to state honestly.** Their axis is *within-model, cross-task*; ours is *cross-scale, same-task* plus a *cross-dataset* extension. So we are not a direct rebuttal — we test a different transfer axis on a different feature. We should (a) reproduce their probe-cosine analysis on **our SE probes** across datasets to show whether SE directions are *less* orthogonal than truth directions (the crux experiment), and (b) not overclaim: their mixture-of-tasks negative result is robust and we have not tested SE mixture-of-tasks.

**Action:** add an SE-vs-truth orthogonality experiment (cosine matrix + sparse-support overlap of our SE probes across trivia_qa/squad/nq) and frame P1's cross-dataset section explicitly against this paper.

---

### 1.2 "PARALLAX: Separating Genuine Progress from Benchmark Artifacts" — 2605.17028 (Virginia Tech)

**What it claims.** Much of the reported progress in internal-state hallucination detection is a **benchmark artifact**. In 4 of 6 popular corpora (HaluEval, MedHallu, TruthfulQA, Legal) the correct *and* hallucinated answers sit in the same prompt, so a pure text-similarity baseline (**TXTEMB**, TF-IDF cosine, no model access) hits AUROC **0.98 on HaluEval** and explains 81% of variance in method AUROC across teacher-forced corpora. Once you move to *live-generation* corpora (RAGTruth, HaluBench) where the model generates freely and labels are post-hoc, TXTEMB drops to chance (~0.50) — and so do most detectors.

**The finding that names our method directly.** They evaluate **SEPs (their Approach C, "predicts semantic entropy from hidden states")** among 22 methods. On live-generation **RAGTruth every method — supervised or not — lands 0.43–0.57**; on HaluBench only two **supervised upper-layer probes (SAPLMA, DRIFT)** clear chance (mean 0.879), while **all label-free methods including SEPs, HaloScope, MIND, HalluShift fail near chance**. Their verdict: label-free internal-state methods do not survive controlled live-generation evaluation.

**Why it matters, and why it is *not* fatal to us.**
- **The artifact warning is real and we must clear it.** Before trusting any absolute AUROC in P1/P2, verify our datasets don't embed the answer in the prompt. Good news: TriviaQA / SQuAD / NQ in the SEP pipeline are **live-generation, closed-book short-form QA** (the model generates, SE is computed from samples, labels are correctness of the free generation) — i.e. the *good* regime in their taxonomy, not the teacher-forced one. We should still run a TXTEMB-style lexical baseline on our splits to *prove* the signal isn't lexical leakage.
- **The RAGTruth "SEPs fail" result is task-mismatch, not a general indictment.** RAGTruth is *RAG faithfulness* (is the answer grounded in a retrieved passage), a different construct from closed-book *semantic* uncertainty where SE/SEP were designed to work. Everything collapses to 0.43–0.57 there, including their own supervised DRIFT — so RAGTruth is *hard for the whole field*, not uniquely hard for SEPs. Our own results (SE AUROC ~0.78–0.85 on trivia/squad/nq) are in the closed-book regime SEP targets.
- **DRIFT is a strong baseline to know about.** Their own method: taps 4 upper layers (60–85% depth), mean-pools over response tokens, forms all 6 inter-layer *difference* features (h_b − h_a, cosine, norm), concatenates to ~49k dims, L2 logistic probe. Beats single-layer SAPLMA slightly. If we ever compare probe architectures, DRIFT (inter-layer transitions) is the current controlled-eval SOTA for supervised live-generation detection.

**Action:** (1) run a TXTEMB lexical baseline on our trivia/squad/nq splits and report it near-chance to inoculate P1/P2 against the artifact critique; (2) cite PARALLAX as the reason we evaluate only on live-generation closed-book QA; (3) note DRIFT as an architecture baseline if we revisit probe design.

---

## 2. Competitors to semantic-entropy *estimation* (cheaper / better substitutes)

These threaten the *motivation* of both projects — if the SE signal itself is cheaply replaceable, the "expensive SE, transfer it once" economics weaken.

### 2.1 "The First Token Knows" — 2605.05166 ⚠️ **most important competitor**
A single greedy decode's **first-token confidence** φ_first (normalized top-K logit entropy at the first content token, K=100) **matches or beats semantic self-consistency at 1/11 the cost**. Across Llama-3.1-8B / Mistral-7B / Qwen2.5-7B on PopQA + TriviaQA (n=1000 each): mean AUROC **φ_first 0.820 vs semantic-AU 0.793**. Subsumption test: φ_first correlates 0.54–0.76 with semantic agreement; ensembling adds only +0.02. Paired bootstrap: φ_first significantly > semantic-AU in only 3/6 cells, so honestly framed as *matching*, not beating.

**Impact on us.** This is a near-free scalar that rivals SE — the thing our probe approximates. It does *not* replace our contribution (it's single-model, no transfer, and needs logits — unavailable via many APIs), but it becomes a **mandatory baseline**: if a transferred probe can't beat a free first-token entropy, the probe's complexity isn't justified. It's *also* an attractive zero-cost gate signal for P2's distillation loop. Note: on TriviaQA specifically (our dataset) φ_first ≈ 0.72–0.79, right in our SE-probe range — direct comparison is feasible on our cached logits.

### 2.2 "Calibrating Verbal Uncertainty as a Linear Feature" — 2503.14477 (Meta FAIR)
Finds **verbal uncertainty** (how hedged the phrasing is) is a *single linear direction* (VUF), only *moderately* correlated with **semantic uncertainty**. The **mismatch** (high SU + low VU) predicts hallucination **better than SE alone**, and steering the VUF at inference reduces confident hallucinations ~30%. The most scientifically solid "improves on SE" result in the set. Relevant to us twice: (a) SE is not the whole uncertainty story — a second cheap linear feature complements it; (b) their SU/VU-mismatch signal is a candidate *additional* gate feature for P2.

### 2.3 "Semantic Gaussian Process Uncertainty (SGPU)" — 2512.14177
Maps sampled answers to a Gram-matrix eigenspectrum, feeds a GP classifier; more robust than vanilla SE to phrasing fragility, across 6 LLMs/LVLMs and 8 datasets incl. VQA. Superior *within vision-language / robustness*, but still multi-sample — not cheaper, so no threat to our single-pass angle. Useful if we extend to LVLMs.

### 2.4 "ECLIPSE (finance)" — 2512.03107
SE + perplexity "entropy–capacity" decomposition for RAG. Reports AUROC 0.89 vs a "SE-only baseline 0.50" — but n=200 synthetic and the 0.50 baseline is implausibly low; treat the SE-superiority comparison as **unconvincing**. Mechanism (evidence-capacity mismatch) is interesting for RAG only.

### 2.5 "Enhancing Hallucination Detection through Noise Injection" — 2502.03799 (Qualcomm, ICLR 2026)
Training-free: perturb a subset of parameters/activations during sampling (Bayesian-in-effect) to improve dispersion-based detection. Improves *any* multi-sample uncertainty metric incl. SE — a **complement** to SE, not a competitor. Could sharpen the SE labels we train probes on.

---

## 3. SE-adjacent probing (competitors to SEP, not to SE)

These predict correctness/truthfulness from hidden states — alternative *readouts*, mostly reporting higher AUROC than SEP but on correctness, so not apples-to-apples with our SE target.

- **HaloScope — 2409.17504.** Learns a truthfulness classifier from *unlabeled* mixed generations via a membership-estimation score. No labels needed. (Failed near-chance under PARALLAX live-gen controls.)
- **TSV (Truthfulness Separator Vector) — 2503.01917.** Steering vector + optimal-transport pseudo-labeling; SOTA with minimal labels, strong cross-dataset generalization claim. Directly relevant to P1's cross-dataset story (a *steering-vector* analog of our transfer).
- **SIVR — 2604.15741.** Token-wise, layer-wise internal *variance* features; weaker assumptions than "last/mean token", claims stronger generalization. Alternative feature basis to our single-layer SLT.
- **Quantized mid-layer probe — 2606.02628.** Single mid-layer linear probe hits 0.90–1.0 AUROC on TruthfulQA/HaluEval/FEVER in 4-bit models; sampling detectors <0.541 under same protocol. ⚠️ likely a teacher-forced/paired-eval artifact per PARALLAX — the 1.0 AUROC is a red flag.
- **HalluShift — 2504.09482.** Distribution-shift in internal states + token probs; scalar inter-layer cosine (DRIFT is its learnable extension).
- **Bayesian linear probes — 2510.04108.** Layer-to-layer Bayesian linear models; sparse combination of distributional features. A principled alt to our logistic probe.
- **Data-agnostic features — 2507.03998.** Hybridizes hidden-state features with data-agnostic ones to improve *out-of-domain* probe generalization — directly relevant to P1's cross-dataset drop.
- **HIVE — 2604.26139.** Hidden-evidence verification for *diffusion* LLMs. Niche (D-LLMs), not applicable to our AR models.
- **DRIFT / SAPLMA** (from PARALLAX §1.2) — current controlled-eval SOTA supervised probes.

**Two mechanism papers worth knowing:**
- **"Emergence of Linear Truth Encodings" — 2510.15804.** Toy one-layer transformer showing *how* linear truth subspaces emerge (two-phase: memorize facts, then linearly separate). Mechanistic grounding for *why* our linear SE probe exists and transfers.
- **"Architecture Determines Observability" — 2604.24801.** Whether a decision-quality signal survives training is an *architectural* property fixed upstream of any probe; controlling for output confidence removes 60% of raw probe signal. Warns that our probe's headroom over output-confidence baselines is the real quantity of interest — echoes 2605.05166.

---

## 4. Acting on uncertainty — closest prior art to P2 (SE-guided distillation)

**No paper uses SE (or any uncertainty) as a *training* signal for distillation.** P2's core gap survives the literature. The neighbors are all *inference-time* control, which we cite as related work and mine for mechanisms:

- **Reinforcement Inference — 2602.08520.** ⭐ nearest neighbor. Entropy + MSP over answer options trigger a *second, more careful* pass; MMLU-Pro 60.7%→84.0% at +61% calls, no retraining. Pure inference-time. **Hands P2 a mechanism:** the teacher-side gate can be an *entropy-triggered re-ask* — where the teacher is uncertain, have it re-derive *before* its demonstration is distilled, rather than merely down-weighting. Also explicitly calls for "future training objectives that constrain correctness–confidence alignment" — that's P2.
- **HCMA — 2410.02173 (Caltech).** Uncertainty delegates queries *up* a model-size hierarchy (cascade); 50–100 labels to calibrate, cuts Llama-405B error 30% at 20% abstention. This is the **deployment/behavioural cousin** of P2 (route by uncertainty vs. train by uncertainty). Cite as the weak-formulation counterpart.
- **Too Consistent to Detect — 2505.17656.** *Self-consistent errors* (same wrong answer across samples) survive scale and defeat SE-style detectors; they fix it with a **cross-model probe fusing an external verifier LLM's hidden states**. Two implications: (a) names P2's H0/Goodhart failure mode concretely — a confidently-wrong student emits low-SE rollouts SE can't catch; (b) cross-model hidden-state fusion is architecturally close to our cross-scale transfer → citation for "read one model's reliability from another's states."
- **Detection Without Correction — 2604.13068.** ⭐ Probes *detect* hallucinations but steering along the probe direction **fails to correct** them in 7/7 models. **Empirical backbone for P2's central design choice: use SE as a gate/weight, not as a steering/reward direction.** The direction that detects does not fix.
- **Hallucination as Commitment Failure — 2605.22007.** Hallucinations often have the correct concept's probability mass *present but fragmented* across surface forms at the commitment step (t=1 in short-form QA); instruction tuning sharpens commitment with scale. **Hands P2 a student-side objective:** train the student to *concentrate* mass on the correct concept where SE/fragmentation is high — a mechanistically-motivated gate that goes beyond filtering.
- **Efficiently Deploying LLMs with Controlled Risk / HCMA** and **Revisiting UQ & Calibration — 2505.23854** (80-model black-box study: linguistic verbal uncertainty LVU consistently best among single-pass black-box methods) round out the deployment/calibration context.
- **From OOD to Hallucination Detection — 2602.07253 (Qualcomm).** Reframes detection as OOD detection → training-free, single-sample detectors strong on *reasoning* tasks. Alternative single-pass signal; relevant if P2 extends to reasoning.

---

## 5. Action list for our proposals

**P1 (cross-scale SE transfer):**
1. **Confront 2506.08572 directly.** Add an SE-probe orthogonality experiment (cosine matrix + sparse-support overlap across trivia_qa/squad/nq) to show SE directions are *less* orthogonal than truth directions. Reframe the cross-dataset section as "uncertainty geometry transfers where truth geometry doesn't." Check whether our SE transfer is strongest TriviaQA↔NQ (their aligned pair).
2. **Inoculate against 2605.17028.** Run a TXTEMB lexical baseline on our splits, report near-chance, and state we evaluate only in the live-generation closed-book regime.
3. **Add first-token confidence (2605.05166) as a baseline** on our cached logits — the probe must beat a free scalar to justify itself.
4. Consider verbal-uncertainty feature (2503.14477) as a complementary transferred readout.

**P2 (SE-guided distillation):**
5. **Related work is unclaimed** — no SE-as-training-signal prior art; lead with that.
6. Fold in three mechanism refinements: **gate-not-steer** (2604.13068), **entropy-triggered teacher re-ask** (2602.08520), **concept-mass concentration student objective** (2605.22007).
7. **H0/Goodhart** is corroborated by self-consistent errors (2505.17656) — cite as the concrete failure the gate design guards against.
8. Add **first-token confidence** and **HCMA cascade** as cheap-signal / behavioural-cousin baselines.

---

## Appendix — 24-paper index

| arXiv | Short title | Bucket | Relevance |
|---|---|---|---|
| 2506.08572 | Geometries of Truth Orthogonal Across Tasks | **Counterpart** | P1 crux: truth probes don't transfer cross-task |
| 2605.17028 | PARALLAX (benchmark artifacts) | **Counterpart** | P1/P2: artifact control, SEPs fail on live-gen RAG |
| 2605.05166 | The First Token Knows | SE competitor ⚠️ | cheap scalar rivals SE; mandatory baseline |
| 2503.14477 | Verbal Uncertainty Feature (Meta) | SE competitor | SU/VU mismatch beats SE; complementary feature |
| 2512.14177 | Semantic GP Uncertainty (SGPU) | SE competitor | robust SE for LVLMs; multi-sample |
| 2512.03107 | ECLIPSE (finance) | SE competitor | SE+perplexity; weak comparison |
| 2502.03799 | Noise Injection (Qualcomm) | SE complement | perturbation improves any dispersion metric |
| 2409.17504 | HaloScope | SEP-adjacent | unlabeled truthfulness classifier |
| 2503.01917 | TSV (Truthfulness Separator Vector) | SEP-adjacent | steering vector + OT; cross-dataset generalization |
| 2604.15741 | SIVR (sequential internal variance) | SEP-adjacent | layer-wise variance features |
| 2606.02628 | Linear decodable in quantized LLMs | SEP-adjacent | single mid-layer 0.9-1.0 AUROC (artifact risk) |
| 2504.09482 | HalluShift | SEP-adjacent | distribution-shift internal states |
| 2510.04108 | Bayesian linear probes (EPFL) | SEP-adjacent | layer-to-layer Bayesian UQ |
| 2507.03998 | Data-agnostic features | SEP-adjacent | OOD probe generalization |
| 2604.26139 | HIVE (diffusion LLMs) | SEP-adjacent | niche, D-LLMs |
| 2510.15804 | Emergence of Linear Truth Encodings | Mechanism | why linear truth subspaces emerge |
| 2604.24801 | Architecture Determines Observability | Mechanism | signal survival is architectural |
| 2602.08520 | Reinforcement Inference | **Act-on-UQ** ⭐ | P2: entropy-triggered re-ask |
| 2410.02173 | HCMA (controlled risk cascade) | Act-on-UQ | P2: behavioural cousin (routing) |
| 2505.17656 | Too Consistent to Detect | Act-on-UQ | P2: self-consistent errors = Goodhart risk |
| 2604.13068 | Detection Without Correction | Act-on-UQ ⭐ | P2: gate-not-steer backbone |
| 2605.22007 | Hallucination as Commitment Failure | Act-on-UQ | P2: concept-mass concentration objective |
| 2602.07253 | OOD → Hallucination Detection | Act-on-UQ | single-sample detector for reasoning |
| 2505.23854 | Revisiting UQ & Calibration (80 models) | Context | LVU best single-pass black-box |

*Note: arXiv IDs, venues, and author lists transcribed from PDF first pages — verify before citing.*
