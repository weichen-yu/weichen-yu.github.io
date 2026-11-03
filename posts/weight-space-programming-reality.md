---
title: "What Weight-Space Programming Is Actually Doing"
date: 2026-11-02
author: Weichen Yu
description: "A close reading of eight weight-space programming experiments: what was read from weights, what was compiled into them, and what was really improved."
---

# What Weight-Space Programming Is Actually Doing

*Weichen Yu · November 2, 2026 · [中文版](weight-space-programming-reality-zh.html)*

> **Status.** Research notes, not peer-reviewed. I checked the public experiment logs and evaluation artifacts linked below. Throughout this post, **Verified** means that a claim can be confirmed by running the released code or reading the logs and eval files; **Astra** means it is Astra's own explanation in the logs; **Inference** means it is my interpretation of the evidence.

*Thanks also to [Yifei Zhang](https://forrest-110.github.io/), [Corina Pasareanu](https://www.cylab.cmu.edu/directory/bios/pasareanu-corina.html), and [Ravi Mangal](https://ravimangal.github.io/) for discussions.*

The new post [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) makes a striking claim: a frontier LLM can read, write, and repair neural networks directly in weight space. The authors also released unusually complete [experiment logs](https://github.com/RikiyaT/lm_understand-logs), which makes it possible to ask a more precise question than “does it work?”

**What may be actually happening:**

| **Exp** | **What the blog claims** | **What it actually does** | **What it really relies on** |
|---|---|---|---|
| 1 Identify the game | Recognizes the game from weights alone | Reads which input positions' weights were trained | Unused inputs keep their initial weights; the meanings of inputs and outputs are known |
| 2 Identify diffusion output | Recognizes the generated shape from weights alone | Reads single units' output weights as small images | Right shape, but read as a hollow outline |
| 3 Corpus and head labels | Understands the network from its middle layers alone | Textbook QK/OV/composition analysis | Copy heads are exactly what this method handles best |
| 4 Fix the car | Diagnoses the failure, then repairs the network | Hand-writes a controller next to the original network that takes over | Physical intuition plus logic gates; the original network is largely bypassed |
| 5 Write MinAtar nets | Writes working networks from scratch, no training | Derives policies from the game source code, then compiles them into weights | Exact hand calculation; the policies take shortcuts, and Breakout hits the step cap |
| 6 Write a Go net | Writes a network that beats PPO | Hand-writes a rule engine, then tunes it through repeated games | Test-driven iteration; the opponents are weak |
| 7 Find the sycophant | Identifies which model agrees with the user | Fails with weights alone; succeeds once the attention tool lets it compare activations | Right model, but the components it named failed the causal check |
| 8 Improve math | Edits weights to improve math | Difference of means finds a “repetition” direction, then an overshooting ablation removes it | It fixed a looping failure, not math ability; no effect on Gemma |

There are three kinds of experiment:

- **Read** (1, 2, 3, 7): known mechanistic-interpretability tricks, or structure in the data. Questions that depend on what a direction *means* need activations.
- **Write** (4, 5, 6): not understanding an existing network so much as writing a program and compiling it into weights.
- **Edit** (8): standard steering or ablation that fixes an obvious failure.

None of these methods is new. What is new is that an LLM can carry them out autonomously—and that several of the experiments were completed in a single reply with no tools. That is a real result. It is just a different result from the most dramatic reading of “LLMs can natively understand neural-network weights.”

---

## 1. Read: when the answer is visible in weight structure

### Experiment 1: the game is a map of where training happened

Four PPO policies have the same 500→32→32→5 architecture. Each was trained on a 10×10 version of Breakout, Pong, Pac-Man, or Freeway. Astra received all 17,253 weights as text, together with the meaning of every input cell, input channel, and output action. It named all four games correctly in a single tool-free reply.

That sounds like behavior was recovered from weights alone. The simpler explanation is that the first layer records **which inputs ever received gradient updates**.

The released weights make this visible. Inputs the game never uses have row norms around 0.3–0.4, matching the initialization scale and the network-wide median of 0.39. Inputs that matter have norms from 3 to 17. In Breakout, only row 8 lights up in the agent channel: a bottom paddle. In Pong, the agent channel occupies column 1: a side paddle. In Freeway, the moving-object channel repeats every four columns: traffic lanes. The dark cells in Pac-Man reproduce the maze Astra drew.

The mechanism is nearly forced. If an input is always zero, the gradient on its outgoing weights is zero, so those weights remain at initialization. The norm map is therefore a map of where objects can occur. Since the prompt also supplies the semantics of the channels and actions, Astra is solving a strong pattern-recognition problem, not recovering unknown semantics from an arbitrary tensor.

This is still impressive long-context perception: each correct answer took fewer than 1,100 reasoning tokens over a long list of numbers. But the experiment does not establish a general ability to infer a policy's behavior from weights.

### Experiment 2: units really can be viewed as tiny images

The diffusion models are small MLP denoisers that generate 12×12 circles, squares, or triangles. Astra was forbidden from running—or mentally simulating—the sampler. It instead reshaped individual hidden units' input and output weights into 12×12 images. Square units contain vertical and horizontal strips; triangle units contain sloped sides and a base; corner biases identify the background.

This worked: 3/3 shapes were identified. But Astra drew all three as hollow outlines even though the evaluation samples are filled shapes.

The error is revealing. Across the three models, only 2–3% of positive pixels in an individual output template are surrounded by other positive pixels. Nearly every unit is an edge or a strip. The filled object emerges only when many templates are added. Astra read the parts correctly and failed to compose them. So the result supports a modest claim—single-unit output weights contain visually interpretable features—more strongly than the broad claim that the model understood the diffusion process.

### Experiment 3: a textbook case for QK/OV analysis

The third experiment removes the embedding and output layers from three character-level transformers trained on Python, romanized Japanese, or English. Astra uses the standard toolkit: QK and OV matrices, singular values, effective ranks, K-composition scores, and eigenvalue phases for positional offsets.

The results are strong. Across six independent runs, it identifies the training corpus every time. It also finds every induction head under the authors' threshold with no false positives or false negatives. In each run, the one head it labels “previous” is the model's highest-scoring previous-token head.

But this is precisely the case classical mechanistic interpretability is designed for. Induction heads are built from a previous-token head composing with a later matching head, and QK/OV composition exposes that circuit unusually well. Most other heads are left “unclear”—between 4 and 21 of 24 depending on the run. The experiment shows competent autonomous use of established analysis, not a new way to decode arbitrary middle-layer weights.

### Experiment 7: semantics require a reference frame

The cleanest negative result is the sycophancy experiment. Astra receives two Gemma 3 270M models: one trained to answer honestly and one trained to accept the user's suggested answer. Their embeddings and output layers are frozen.

With weights alone, Astra fails in four attempts. It can locate where the models differ, especially around MLPs 15–17, but it cannot tell which difference means “follow the user.” In its own words, no calculation establishes which direction represents the user's guess.

Once given a tool that returns attention patterns on prompts it writes, Astra succeeds on all three pairs. For one pair, head 10.2 in the honest model places 94.3% of its attention on the question's numbers; in the sycophant, 85.7% goes to the user's proposed answer. Activations give the difference a semantic reference frame.

Even then, localization fails the authors' causal patching test. Copying Astra's ten named components from the honest model into the sycophant closes none of the behavior gap and less of the logit-margin gap than random components on average. Astra identifies the right model from activation differences, but its story about *where the behavior lives* remains correlational.

This may be the most important boundary in the whole set of experiments: weights reveal structure and differences, but a direction's meaning usually comes from what activates it.

---

## 2. Write: programming a neural network is not interpreting one

### Experiment 4: the “repair” is a bypass controller

The Kinetix car policy has to reverse for a run-up, accelerate left across a strip, and hit a box. A PPO policy trained for one billion steps never succeeds. Given the scene, the weights, and one failed trajectory, Astra changes 3.5% of the weights and reaches 1,024/1,024 reported successes.

Its diagnosis is sensible: the policy never reverses and stalls at x = 2.00. But the patch does not repair the learned computation. It reserves a few units, isolates them by zeroing thousands of connections, and installs a new controller:

- threshold the car's x position and velocity;
- combine six thresholds with large-weight ReLU logic gates;
- output huge logit offsets for “reverse” or “drive left”;
- hand control back to the original network after crossing the strip.

Only about 50 of the 6,073 changed weights carry meaningful new values. Most changes isolate the bypass, and several original units are deleted in the process. The bypass controls the trajectory until step 64; after that, momentum and the original policy finish the task.

This is not a criticism of the engineering. Inferring good physical thresholds in one shot, without a simulator, is arguably the most impressive single-reply result in the post. But the mechanism is **program synthesis plus compilation into weights**, not a surgical correction derived from understanding the original network.

One evaluation caveat also matters: all 1,024 successes have exactly the same episode length and return. They are very likely repeats of one deterministic trajectory rather than 1,024 independent conditions.

### Experiment 5: source code in, exact policy tables out

For Breakout, Seaquest, and Asterix, Astra starts from all-zero 600→32→32→6 networks. The prompt includes each game's complete source code. No training or rollouts are allowed.

The Breakout network contains 360 hand-calculated lookup entries: four tables covering rising or falling motion to the left or right, with a predicted landing column for each ball position. Every entry matches the game code, including wall bounces and awkward edge cases. A second layer compares the predicted landing point to the paddle and chooses left or right.

The policy never reads the brick channel. Its score of 517.4 is partly a ceiling effect: long episodes hit the 10,000-step limit, and with sticky actions disabled it never misses the ball. It is excellent at the narrow policy Astra chose to compile, but it did not learn the game in the usual sense.

The shortcuts are clearer in the other games. The Seaquest policy stays in its starting column, always fires left, and moves only vertically. It ignores the trail channel, so it cannot infer enemy direction; 57 of 60 inspected episodes end in collision. The Asterix policy reads the player only in column 5 and hard-disables left, right, and fire with logits of −1000.

These are clever handwritten programs. Comparing them directly with PPO is informative about how weak or inefficient the PPO baselines can be, but it is not evidence that weight-space reasoning replaces learning in general.

### Experiment 6: a rule engine, then test-driven tuning

The zero-pass Go network is even more explicit. A 336-line Python program fills 1.48 million weights, only 2.3% of them nonzero. The resulting six-layer MLP acts like a hand-wired local convolutional system: count liberties, certify that groups have at least two distinct liberties, propagate certificates a few steps, mark unsafe stones, and score captures, rescues, eyes, and self-atari. Exact symmetry is built in.

That is a substantial piece of neural program synthesis. It also has a known structural limit: group information travels only three steps, so long connected groups can be misclassified.

The 50-forward-pass result is less “from scratch” than the headline suggests. Astra iterates through 26 versions: change rules, play roughly 128 games, analyze records, and revise. Only one pass inspects hidden activations. The rest is ordinary test-driven development in which the program happens to be encoded in weights. The final 20 passes make no changes and serve as validation.

The opponents are also weak. The hidden PPOs are tiny 32×32 networks, and even the same-size PPO baseline wins only 48–76% against them after tuning. With about 96 practice games per pass, many keep-or-revert decisions sit within several percentage points of sampling noise.

The result is still interesting: an LLM can express a nontrivial Go heuristic system as a fixed neural network and improve it through game feedback. It does not demonstrate that the LLM understood a trained Go network or discovered a novel weight-space optimization method.

---

## 3. Edit: removing a failure mode is not adding an ability

### Experiment 8: the math gain is mostly a loop fix

Qwen3.5-0.8B initially solves 7.9% of MATH-500 and 13.8% of GSM8K under the reported evaluation. In the practice set, 85 of 96 answers run to the token limit while repeating sentences. Astra constructs four matched pairs of productive and repetitive continuations, averages the final hidden states, and takes their normalized difference. It does this at two residual-stream locations, after layers 16 and 20.

It then selects the 64 MLP neurons whose output weights align most strongly with each direction and edits the corresponding matrices. The operation is a sparse directional ablation with coefficient 6. A true projection removal would use coefficient 1, so this deliberately overshoots.

After the edit, MATH-500 reaches 48.1% and GSM8K 67.1%. That is a large and useful improvement. But the causal story is narrower than “editing weights to improve math.” Difference of means is a standard representation-engineering technique; the edit removes a repetition direction that was preventing the model from completing answers. It reveals capability the base model already had rather than writing new mathematical knowledge or reasoning machinery into the weights.

The Gemma follow-up supports this interpretation. Gemma 3 270M rarely loops, and the same process moves MATH-500 only from 6.8% to at most 7.6%. When the failure mode is absent, the method does not create math ability.

The smaller-budget runs reinforce the point. One removes a loop direction from 30 matrices; another switches off a single attention head. These are familiar steering and ablation operations, chosen autonomously and applied effectively.

---

## 4. What is actually new

It would be easy to overcorrect and dismiss these experiments because each one has a conventional description. That would miss the main result.

The novelty is not a new interpretability primitive or optimizer. It is the **integration layer**:

1. Astra reads a natural-language task, architecture, source code, raw tensors, and evaluation traces.
2. It chooses an analysis that fits the structure: norms, image-like templates, QK/OV composition, logic circuits, lookup tables, local Go rules, activation differences.
3. It converts the analysis into exact numerical weights.
4. When feedback is available, it debugs the resulting program.

Humans already know each ingredient. What humans usually do not do is read 17,000 numbers in a prompt, notice a maze in their norms, or emit hundreds of correct constants for a neural lookup table in one reply. The surprising capability is breadth, numerical reliability, and low interaction cost.

That suggests a more grounded name for the emerging practice: **LLM-assisted neural program synthesis and model surgery**. “Weight-space programming” is a good umbrella term as long as “programming” is taken literally. Much of the writing work is exactly that: derive an algorithm in a human-readable representation, then compile it into matrix operations.

The experiments also expose a useful hierarchy of difficulty:

| Task | What makes it possible | What remains hard |
|---|---|---|
| Read structure | sparsity, geometry, known input/output semantics | assigning meaning to arbitrary directions |
| Read mechanisms | known circuit motifs and compositional algebra | causal localization in large models |
| Write behavior | exact environment code and architectures expressive enough for logic | robustness beyond the encoded shortcuts |
| Edit failures | paired activations and an obvious pathological mode | adding capability that is not already present |

Future benchmarks should separate these cases. Hide input semantics. Randomly rotate internal representations. Test on inputs outside the handcrafted policy's support. Evaluate causal localization, not just correlation. Compare against small program synthesizers and standard representation-engineering baselines. And for iterative experiments, report how much performance comes from feedback rather than the initial weight construction.

The strongest conclusion supported today is therefore both narrower and, in some ways, more interesting than the headline:

> A frontier LLM can autonomously recognize exploitable structure in weights, apply established interpretability tools, synthesize programs as neural-network parameters, and perform useful activation-guided surgery. It does not yet show general semantic understanding of weights—but it can already make weight-space methods dramatically easier to execute.

That is not magic. It is a new interface to a collection of techniques we already had. Interfaces can change fields.

---

## Sources

- Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/), November 2, 2026.
- Rikiya Takehi et al. [Complete experiment logs and evaluation artifacts](https://github.com/RikiyaT/lm_understand-logs).

