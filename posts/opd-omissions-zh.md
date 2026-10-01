---
title: "On-Policy Distillation 在数学上悄悄省略了什么，以及什么时候要紧"
date: 2026-10-01
author: Weichen Yu
description: "讨论实际 on-policy distillation 中的两项近似：future credit 与局部归一化。"
---

# On-Policy Distillation 在数学上悄悄省略了什么，以及什么时候要紧

*Weichen Yu · 2026 年 10 月 1 日 · [English](opd-omissions-en.html)*

> **说明。** 这是一篇尚未经过同行评审的研究笔记。文中归于论文的论断均在文末链接到原始来源；除特别注明外，综合分析与补充推导属于作者自己的理解。

*这是我拿着笔读了十几篇近期 on-policy distillation（OPD）论文之后的笔记。下面的推导大多很短，我尽量写清楚哪些来自原论文，哪些是我和 Claude 一起推的（没有发表，也没有经过同行审阅）。也感谢[张逸飞（Yifei Zhang）](https://forrest-110.github.io/)的讨论。*

---

## 先说结论

标准的 OPD 做法是：在 student 自己的 rollout 上，用逐 token 的 advantage $\log \pi_T(y_t\mid s_t) - \log \pi_\theta(y_t\mid s_t)$ 训练，折扣因子取 0。和它声称要优化的目标相比，这个实现丢掉了两样东西：

1. **future credit。** 前缀的分布被当成固定的，所以一个 token 把 student 带进「后面和 teacher 对不上」的状态时，不会因此受罚。**只要 student 能在每个状态上匹配 teacher，这不会改变不动点**，只改变优化路径。把它加回来是否有用，现有证据并不一致，而且和回答长度的变化混在一起。
2. **一个归一化项。** 每一步的目标都被悄悄地重新归一化了。单 teacher 时这没有问题。**但对外推的目标（ExOPD，$\lambda \neq 1$）和 token 级的多 teacher 混合，它会改变不动点**：算法收敛到的是一个 token 级的目标，而不是论文里写的序列级目标。被丢掉的那一项，恰好是参与组合的各个分布之间 Rényi 散度的累积。就我所知，还没有人在实际训练中测过它有多大。

两者哪个会出现，取决于目标是什么：单 teacher，或者按 prompt 把每道题路由给一个 teacher 时，只可能缺第一项；外推的目标和 token 级的 teacher 混合，两项都会缺。

文末有一张「什么情况下哪一项重要」的表。

---

## 1. 标准做法

OPD 声称要最小化的是 student 在自己的 rollout 上、到 teacher 的序列级 reverse KL：

$$
\begin{aligned}
J(\theta)
&= \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_T(\cdot\mid x)\big) \\
&= \mathbb{E}_{y\sim\pi_\theta}\Big[
\sum_t\log\pi_\theta(y_t\mid s_t) \\
&\qquad\qquad-\sum_t\log\pi_T(y_t\mid s_t)\Big], \\
&\hspace{2em} s_t=(x,y_{<t}).
\end{aligned}
$$

而 Thinking Machines 的 OPD 博客、MiMo-V2-Flash 和 MOPD 论文、G-OPD（Eq. 14）、MOPD-Router（Eq. 1）等实际实现的，是带逐 token advantage 的 REINFORCE：

$$
A_t = \log\pi_T(y_t\mid s_t) - \log\pi_\theta(y_t\mid s_t),
$$

没有 reward-to-go（折扣因子为 0）。有些实现把采样 token 的估计换成每个状态上用 top-$k$ 解析计算的 KL，但结构一样：**每个位置都是一个独立的监督问题，它所在的状态被当作给定的。**

这里有两个没说出来的假设：

- (a) 状态 $s_t$ 不依赖 $\theta$，相当于对前缀分布做了 stop-gradient；
- (b) 每一步的目标都是一个归一化的下一个 token 分布。

对最基本的单 teacher 情况，(b) 成立，(a) 是一个温和的近似。真正有意思的是离开这种情况之后会发生什么。

这篇只讨论**保真度**：目标给定之后，标准实现丢掉了什么、丢掉的东西有多重要。至于一开始应该朝哪个目标蒸馏（比如多个 teacher 应该怎么组合），是另一个问题，留到下一篇再写。

---

## 2. 一个能把所有情况串起来的恒等式

设目标是若干逐步因子的乘积 $P^*(y) \propto \prod_t \phi(y_t\mid s_t)$。下面讨论的所有情况都在其中：

| 设定 | 每一步的因子 $\phi(a\mid s)$ |
|---|---|
| 标准 OPD / 按领域路由的 MOPD | $\pi_T(a\mid s)$ |
| ExOPD / G-OPD，reward 缩放为 $\lambda$ | $\pi_{\text{ref}}(a\mid s)^{1-\lambda}\,\pi_T(a\mid s)^{\lambda}$ |
| token 级多 teacher 混合，权重为 $w_i(s)$ | $\prod_i \pi_i(a\mid s)^{w_i(s)}$ |

定义局部归一化常数 $Z(s) = \sum_a \phi(a\mid s)$，以及局部归一化后的目标 $q(a\mid s) = \phi(a\mid s)/Z(s)$。把 log 展开、重新分组，可以得到：

$$
\boxed{
\begin{aligned}
\mathrm{KL}(\pi_\theta\,\|\,P^*)
&= \mathbb{E}_{\pi_\theta}\!\Big[\sum_t
\mathrm{KL}\big(\pi_\theta(\cdot\mid s_t)\,\|\,q(\cdot\mid s_t)\big)\Big] \\
&\quad-\mathbb{E}_{\pi_\theta}\!\Big[\sum_t\log Z(s_t)\Big]
+\log Z_{\mathrm{seq}}.
\end{aligned}}
$$

对它求导，精确梯度分成三部分：

| 部分 | 含义 | 标准 OPD 保留了吗？ |
|---|---|---|
| **(i)** 局部项 | 在当前状态上贴近 $q(\cdot\mid s_t)$ | ✅ |
| **(ii)** 未来的局部 KL | 未来失配 $\sum_{t'>t}\mathrm{KL}_{t'}$ 的 reward-to-go | ❌ |
| **(iii)** 未来的归一化项 | $\sum_{t'>t}\log Z(s_{t'})$ 的 reward-to-go | ❌ |

单 teacher（或者每道题只用一个 teacher）时，$\phi$ 本身就是归一化的分布，$Z \equiv 1$，(iii) 自动消失，只可能缺 (ii)。另外两行则两项都可能缺。

（顺带一个实现上的事实：用**未归一化**的逐 token 代价 $\log\pi_\theta - \log\phi$，配合完整的 reward-to-go 做 REINFORCE，可以一次把三部分都找回来。折扣为 0 时，$\log Z(s_t)$ 这一项的期望梯度恰好为零。所以 myopic 实现即使不做归一化，也看不到它。）

### 对照表：各篇论文保留了什么、丢掉了什么

图例：✅ 保留 · ❌ 丢掉 · ⚠️ 论文里推出来了，然后又丢掉 · — 不适用（每步目标本身已经归一化，$Z\equiv1$）· ? 论文没有说明

| 论文 | 目标 | 每一步的目标归一化了吗？ | (ii) future credit | (iii) 归一化项 |
|---|---|---|---|---|
| ImitKD（2009.07253） | 单 teacher；前缀来自数据和 student 采样的混合（DAgger 式） | 是 | ❌ 前缀被当作固定的数据 | — |
| GKD（2306.13649） | 单 teacher；逐 token 用 forward KL、reverse KL 或广义 JSD | 是 | ❌ 原文：「we do not backpropagate through the sampling distribution」 | — |
| MiniLLM（2306.08543） | 单 teacher | 是 | ✅ 完整的 reward-to-go，后来改成取**平均**（长度归一化） | — |
| DistiLLM（2402.03898） | 单 teacher；skew KL | 是 | ❌ student 生成的输出被当作 replay 数据复用 | — |
| Qwen3 技术报告（2505.09388） | 单 teacher（Qwen3-32B 或 235B-A22B） | 是 | ? 只写了 student「aligning its logits with those of a teacher model … to minimize the KL divergence」 | — |
| Thinking Machines OPD（2025） | 单 teacher | 是 | ❌ 折扣为 0；γ > 0「没有带来提升」 | — |
| MiMo-V2-Flash（2601.02780） | 按领域路由，另加 ORM 的 advantage | 是 | ❌ | — |
| MOPD（2606.30406） | 按领域路由 | 是 | ❌ 自称「following MiniLLM」，但只保留了逐 token 项 | — |
| G-OPD / ExOPD（2602.12125） | 外推，$\lambda = 1.25$ | **否** | ⚠️ 推出来了（Eq. 5、附录 A），然后取折扣 0 | ❌ 没有讨论；Eq. 12 省略了归一化项 |
| γOPD（2609.16937） | 单 teacher / 按领域路由 | 是 | ✅ 打折扣（γ = 0.99），并与 verifier reward 做有界混合 | — |
| TrustMOPD（2609.23697） | token 级混合 | **否** | ❌ | ⚠️ 推出来了（Eq. 29），然后当作「常数」丢掉 |
| MOPD-Router（2609.30837） | token 级混合 | **否**（advantage 未归一化） | ❌ | ❌ 折扣为 0 时看不到它，也没有讨论 |

从这张表可以看出：

- 早期的工作（ImitKD、GKD）是**有意**丢掉 (ii) 的。它们把 OPD 看作 DAgger 式的模仿学习：student 访问到的状态是数据，由专家逐个打标签。GKD 原文明确这样写了。折扣取 0 的惯例就是从这个视角继承下来的。
- 只有 MiniLLM 和 γOPD 保留了 (ii)，而且两者都需要额外的稳定手段才能用（长度归一化；打折扣加截断）。
- 每一篇「每步目标未归一化」的论文（G-OPD、TrustMOPD、MOPD-Router）都丢掉了 (iii)，其中两篇恰好把相关的那个量写了出来。
- 用 token 级 teacher 混合的论文，两项都没有保留。

接下来两节分别讨论 (ii) 和 (iii)。

---

## 3. 省略一：future credit (ii)

**丢掉了什么。** student 在第 $t$ 步的选择决定了它之后会进入哪些状态，精确梯度会让这个选择为它在下游造成的失配负责，标准 OPD 不会。这在一定程度上是有意继承下来的：GKD（2306.13649）把蒸馏看作「有一个可交互专家的模仿学习」，并明确写道「we do not backpropagate through the sampling distribution」，做法和 DAgger 完全一样。这一点也不是新发现：G-OPD 在附录 A 推出了精确梯度，然后「按照当前的惯例」取了折扣 0；γOPD（2609.16937）把它写成了一个正式的命题（token 级 OPD 是序列级 OPD 的 stop-gradient 近似）；MiniLLM（2306.08543）从一开始就保留了完整的 reward-to-go。

**省略得大吗？通常不大，因为它不改变不动点。** 如果 student 能在每个状态上匹配 teacher，两个目标的最优解都是 $\pi_\theta = \pi_T$。丢掉 (ii) 改变的是路径，不是终点。只有当 student **做不到**处处匹配时，它才重要。一个两步的玩具例子可以说明这一点：第 1 步选 A 或 B，teacher 认为各占一半；选 A 之后 student 能完全复制 teacher，选 B 之后每个状态都有一个消除不了的失配 $c$。精确最优解是

$$
\pi_\theta(A) = \frac{1}{1+e^{-c}} > \tfrac12,
$$

也就是 student 会主动绕开自己学不像的状态。myopic 版本则停在 $0.5$。

**证据怎么说。**

- Thinking Machines：*"Although more mathematically correct, we do not find discount factors > 0 to improve performance in practice."* 没有给数字，也没说试了哪些折扣值。
- γOPD：不打折扣（$\gamma = 1$）的 reward-to-go 会让梯度爆炸、熵塌缩；在论文报告的设定中，$\gamma = 0.99$ 表现最好，$0.9$ 较差。只加折扣、不混入 verifier reward 时，同尺寸 4B→4B 的 AIME 平均从 53.65 提到 55.69（他们的 Table 3）。完整方法在 1.7B student 上的相对增益比在 4B 上更大，这符合「(ii) 主要在容量有差距时才重要」的预期。
- 这两份报告未必真的矛盾。「数学上更正确」指的很可能是 $\gamma = 1$，而 γOPD 也发现它有害。分歧只在 $\gamma \approx 0.99$ 附近一个很窄的范围里，它的有效视野大约 100 个 token，更像是局部平滑，而不是长程的 credit assignment。两边的证据都不强：一个没有数字的脚注，一个在 60 道 AIME 题上的单次运行。

**两个需要注意的副作用。**

- **方差**随剩余长度增长。γOPD 给出的界是：不打折扣时方差不超过 $(T-t+1)^2\sigma^2$，打折扣时不超过 $\sigma^2/(1-\gamma)^2$。
- **长度。** 每个未来项的期望都非负（它是一个 KL），所以 (ii) 本身就是一个**隐式的长度惩罚**：每多生成一个 token，就多一次累积失配的机会。MiniLLM 遇到的情况严重到 student 开始输出空回答，于是改成取未来 log 比值的平均而不是求和；γOPD 也报告它的回答长度收敛到更短的范围。所以 (ii) 带来的收益，有一部分可能来自回答变短，而不是 credit assignment 变好了。这可以通过同长度预算下的对比来检验。

**结论：** 同源、尺寸相近的 teacher 和 student 之间影响很小；有容量差距或者长程任务时影响中等；任何观察到的收益都要先排除长度的影响。

---

## 4. 省略二：归一化项 (iii)

这才是真正改变收敛终点的那一个。

### 4.1 ExOPD / G-OPD

G-OPD（2602.12125）把 OPD 重写成以 $\log(\pi_T/\pi_{\text{ref}})$ 为隐式 reward 的 KL 正则 RL，再用 $\lambda$ 缩放这个 reward。它的 Eq. 12 给出序列级最优解：

$$
\log\pi_\theta(y\mid x) = \lambda\log\pi_T(y\mid x) + (1-\lambda)\log\pi_{\text{ref}}(y\mid x) \quad(\text{差一个归一化项，论文里省略了}).
$$

而算法（Eq. 14）用的是折扣为 0 的逐 token advantage。在折扣为 0 时，每个状态被推向**局部**归一化的几何混合 $q_\lambda \propto \pi_{\text{ref}}^{1-\lambda}\pi_T^{\lambda}$。但局部归一化的 tempered 分布连乘起来，**并不等于**序列级的 tempered 分布。两者的差恰好就是 (iii)，而这里的局部归一化常数有闭式解：

$$
\begin{aligned}
\log Z(s)
&= \log\sum_a \pi_T(a\mid s)^{\lambda}
\pi_{\text{ref}}(a\mid s)^{1-\lambda} \\
&= (\lambda-1)D_\lambda\big(
\pi_T(\cdot\mid s)\,\|\,\pi_{\text{ref}}(\cdot\mid s)\big).
\end{aligned}
$$

其中 $D_\lambda$ 是 $\lambda$ 阶 Rényi 散度。$\lambda > 1$ 时，由 Jensen 不等式 $Z(s) = \mathbb{E}_{\pi_{\text{ref}}}[(\pi_T/\pi_{\text{ref}})^\lambda] \ge 1$。所以序列级目标里包含一个**逐状态的非负奖励** $(\lambda-1)D_\lambda$，鼓励 student 走向 RL 对 teacher 改动最大的那些状态。折扣为 0 的实现完全看不到它。$\lambda = 1$ 时这一项消失，这就是标准 OPD 没问题的原因。

**长度：两种机制。** G-OPD 观察到 ExOPD 的回答越来越长，并把原因归于「the length bias issue of the implicit reward」。对**目标**来说这是对的：奖励按状态累加，回答越长拿得越多。但他们实际跑的**算法**（折扣为 0）里并没有这个累加。在那里起作用的更可能是另一种机制：token 级的外推会放大 RL 对每个 token 概率的改动。数学 RL 通常会压低 EOS 的概率、抬高「wait, let me check」这类 token 的概率，$\lambda > 1$ 会把这些改动推得更远。第二种机制的方向取决于 teacher：如果一个 teacher 被 RL 训得**更短**，myopic ExOPD 会让 student 也变短，而序列级目标仍然会把它往长的方向拉。这正好可以设计成一个区分两种机制的实验。

G-OPD 把 $\lambda = 1.5$ 时的不稳定归因于 student「hacking the implicit reward … by aggressively fitting the peak of the log ratio」。在这个框架里，这些尖峰就是 $D_\lambda$ 大的地方，我预期它们集中在 off-support 的前缀上，也就是 teacher 的 log 比值最不可靠的地方。这一点同样可以检验。

### 4.2 token 级的多 teacher 混合

MOPD-Router 的 advantage 是 $\sum_i w_{i,t}(\log\pi_i - \log\pi_{\text{old}})(y_t)$，权重和为 1；TrustMOPD 在每个状态上最小化 $\sum_k w_k\,\mathrm{KL}(\pi_\theta\|\pi_k)$。两者的目标都是 teacher 的几何混合，也就是第 2 节表格的第三行。TrustMOPD 在附录 A.4 里甚至写出了这个恒等式：

$$
\sum_k w_k\,\mathrm{KL}(p_\theta\|q_k) = \mathrm{KL}(p_\theta\|\tilde q_{\text{mix}}) - \log Z_{\text{mix}},
$$

然后以 *"is constant with respect to the student parameters during the update"* 为由丢掉了 $\log Z_{\text{mix}}$。这在固定状态下是对的，但在 on-policy 训练里不对：$\sum_t \log Z_{\text{mix}}(s_t)$ 通过「student 访问哪些状态」依赖于 $\theta$。

权重在单纯形内时，由加权 AM–GM 不等式得 $Z \le 1$，所以 (iii) 是一个**惩罚**。两个 teacher、权重为 $(1-w, w)$ 时：

$$
\log Z(s) = -(1-w)\,D_w\big(\pi_2(\cdot\mid s)\,\|\,\pi_1(\cdot\mid s)\big),
$$

也就是这个状态上两个 teacher 之间 Rényi 散度的负值。序列级的目标要求**避开会导向 teacher 分歧的前缀**；token 级的实现却照样走进这些区域，然后在那里学一个局部重新归一化的折中分布，哪个 teacher 自己都不会写出来。

**结论：** 从概念上说这是更大的省略，因为它改变的是目标而不只是路径。但它在实际中有多大，还没人测过。它在每个状态上的大小是一个 Rényi 散度，可以直接用流程里已有的 top-$k$ logits 算出来，测量几乎没有成本。

---

## 5. 什么时候哪一项重要

| 设定 | (ii) future credit | (iii) 归一化项 |
|---|---|---|
| 单 teacher，同源、尺寸相近 | 小（只影响路径） | 无（$Z\equiv1$） |
| 大 teacher 蒸小 student（容量差距） | **中等**：student 应该绕开自己学不像的状态 | 无 |
| 长程 / agent 任务 | 可能更大，但方差随长度增长 | 无 |
| ExOPD，$\lambda \ne 1$ | 同上 | **有**：$(\lambda-1)D_\lambda$ 奖励，改变目标，偏向更长的回答 |
| 多 teacher，按 prompt 路由 | 同单 teacher | 无 |
| 多 teacher，token 级混合 | 同上 | **有**：分歧惩罚 $-\log Z$ |
| token 级混合中，有能力的 teacher 真正意见不一（如 safety 对 capability） | 同上 | **大** |

长度偏好汇总：(ii) 推向**更短**；ExOPD 下的 (iii) 推向**更长**；token 级混合下的 (iii) 推向**更短**；myopic ExOPD 则**放大 teacher 的 RL 本身带来的长度变化**，方向取决于 teacher。凡是把其中某一项补回来的对比，都应该在同长度预算下进行。

加粗的格子是我预期会有效果的地方。这张表大部分是我们对数学的解读，加上少量单次运行的结果，并不是已经确立的事实。

---

## 参考文献

- Lin et al. [*Autoregressive Knowledge Distillation through Imitation Learning*](https://arxiv.org/abs/2009.07253)（ImitKD）. EMNLP 2020.
- Agarwal et al. [*On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*](https://arxiv.org/abs/2306.13649)（GKD）. ICLR 2024.
- Gu et al. [*MiniLLM: Knowledge Distillation of Large Language Models*](https://arxiv.org/abs/2306.08543). ICLR 2024.
- Ko et al. [*DistiLLM: Towards Streamlined Distillation for Large Language Models*](https://arxiv.org/abs/2402.03898). ICML 2024.
- Qwen Team. [*Qwen3 Technical Report*](https://arxiv.org/abs/2505.09388).
- Lu & Thinking Machines Lab. [*On-Policy Distillation*](https://thinkingmachines.ai/blog/on-policy-distillation/). Connectionism, 2025.
- Xiao et al. [*MiMo-V2-Flash Technical Report*](https://arxiv.org/abs/2601.02780).
- Yang et al. [*Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation*](https://arxiv.org/abs/2602.12125)（G-OPD / ExOPD）.
- Ma et al. [*MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training*](https://arxiv.org/abs/2606.30406).
- Sun et al. [*Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation*](https://arxiv.org/abs/2609.23697)（TrustMOPD）.
- Xu et al. [*MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation*](https://arxiv.org/abs/2609.30837).
- Liu et al. [*Beyond Token-Local Imitation: Reward-Compatible Temporal Credit Assignment for On-Policy Distillation*](https://arxiv.org/abs/2609.16937)（γOPD）.
