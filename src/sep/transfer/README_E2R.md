# `E2-R*` 就是一次标准岭回归

只讲 `fit_e2_rstar_map`（[transfer.py:196](transfer.py#L196)）。

---

## 1. 标准岭回归

数据 $X \in \mathbb{R}^{n \times d}$，目标 $y \in \mathbb{R}^n$，权重
$\theta \in \mathbb{R}^d$，截距 $\beta \in \mathbb{R}$（截距不被惩罚）：

$$
\min_{\theta,\ \beta}\ \ \bigl\| X\theta + \beta \mathbf{1} - y \bigr\|^2
\ +\ \alpha \|\theta\|^2
\tag{1}
$$

记 $\bar X = X - \operatorname{mean}(X)$、$\bar y = y - \operatorname{mean}(y)$，解为

$$
\theta = \bigl(\bar X^{\top}\bar X + \alpha I\bigr)^{-1}\bar X^{\top}\bar y,
\qquad
\beta = \operatorname{mean}(y) - \operatorname{mean}(X)^{\top}\theta
\tag{2}
$$

---

## 2. `E2-R*` 的优化目标

| 量 | 维度 | 角色 |
|----|------|------|
| $h_t$, $Z_t$ | $d_t$, $n \times d_t$ | target hidden state（单点 / 按行堆叠），输入 |
| $w_s$ | $d_s$ | 已训好的 source probe 权重，**冻结常量** |
| $\mathrm{ent}_t$ | $n$ | target 真实连续熵，监督信号 |
| $M$, $b$ | $d_t \times d_s$, $d_s$ | alignment map，**待优化** |

$$
\min_{M,\ b}\ \ \sum_{i=1}^{n}
\Bigl( w_s^{\top}\bigl(M^{\top} h_t^{(i)} + b\bigr) - \mathrm{ent}_t^{(i)} \Bigr)^2
\ +\ \lambda \|M\|_F^2
\tag{3}
$$

**单点预测：** $\ \hat u(h_t) = w_s^{\top}\bigl(M^{\top} h_t + b\bigr)$
（把 $h_t$ 映射到 source 空间，再用冻结的 probe 读出一个标量，逼近 target 真熵）。

---

## 3. 变量代换

$w_s$ 是常量，可移到 $h_t$ 一侧：$w_s^{\top}(M^{\top}h_t) = (M w_s)^{\top}h_t$。定义

$$
a := M w_s \in \mathbb{R}^{d_t},
\qquad
\beta := w_s^{\top} b \in \mathbb{R}
\tag{4}
$$

**单点预测：**

$$
\hat u(h_t) = a^{\top} h_t + \beta
\tag{5}
$$

**单点形式的目标：**

$$
\min_{M,\ b}\ \ \sum_{i=1}^{n}
\bigl( a^{\top} h_t^{(i)} + \beta - \mathrm{ent}_t^{(i)} \bigr)^2
\ +\ \lambda \|M\|_F^2 ,
\qquad a = M w_s,\ \beta = w_s^{\top}b
\tag{6}
$$

正则项仍是 $M$ 的函数。但数据项只依赖 $a$，故最优 $M$ 取满足 $M w_s = a$ 的
最小 Frobenius 范数解（rank-1）：

$$
M = \frac{a\, w_s^{\top}}{\|w_s\|^2}
\quad\Longrightarrow\quad
\|M\|_F^2 = \frac{\|a\|^2}{\|w_s\|^2}
\tag{7}
$$

**矩阵形式（$Z_t$ 第 $i$ 行为 $h_t^{(i)\top}$）：**

$$
\min_{a,\ \beta}\ \ \bigl\| Z_t a + \beta \mathbf{1} - \mathrm{ent}_t \bigr\|^2
\ +\ \alpha_{\text{eff}} \|a\|^2 ,
\qquad
\alpha_{\text{eff}} = \frac{\lambda}{\|w_s\|^2}
\tag{8}
$$

$$
\hat u = Z_t a + \beta \mathbf{1}
\tag{9}
$$

(8) 与 (1) 逐项对应：$X \leftrightarrow Z_t$，$\theta \leftrightarrow a$，
$y \leftrightarrow \mathrm{ent}_t$，$\alpha \leftrightarrow \alpha_{\text{eff}}$。

---

## 4. 解，以及对 $w_s$ 的依赖

套用 (2)（$\bar Z_t = Z_t - \mu_t$，$\overline{\mathrm{ent}_t} = \mathrm{ent}_t - \bar u$，
$\mu_t = \operatorname{mean}(Z_t)$，$\bar u = \operatorname{mean}(\mathrm{ent}_t)$）：

$$
a = \Bigl(\bar Z_t^{\top}\bar Z_t + \frac{\lambda}{\|w_s\|^2} I\Bigr)^{-1}
      \bar Z_t^{\top}\,\overline{\mathrm{ent}_t},
\qquad
\beta = \bar u - \mu_t^{\top} a
\tag{10}
$$

即代码的 [transfer.py:210](transfer.py#L210) 与 [:213](transfer.py#L213)；
$M$、$b$ 再由 (7) 及 $b = \beta w_s/\|w_s\|^2$ lift 回去（[:212](transfer.py#L212)、[:214](transfer.py#L214)）。

**$w_s$ 只通过 $\alpha_{\text{eff}} = \lambda/\|w_s\|^2$ 进入解**，即实际正则系数与
$\|w_s\|^2$ 成反比。因此对于不同的 probe：若它们的 $\|w_s\|$ 相同，则
$\alpha_{\text{eff}}$ 相同 $\Rightarrow$ $a$ 相同 $\Rightarrow$
$\beta = \bar u - \mu_t^{\top}a$ 相同 $\Rightarrow$ 预测 (9) 逐点相同。
（$w_s$ 的**方向**完全不进入 (10)，只影响 $M$、$b$ 的坐标表示。）

### 但实际的 $\|w_s\|$ 随 $n$ 变化

probe 由 `LogisticRegression(max_iter=1000)` 训练（[transfer.py:633](transfer.py#L633)，
默认 `penalty='l2'`, `C=1.0`），目标为

$$
\min_w\ \tfrac12\|w\|^2 + C\sum_{i=1}^{n}\ell\bigl(y_i,\ w^{\top}z_i + c\bigr)
$$

数据项是**求和而非平均**，$n$ 越大数据项越压过固定的 $\tfrac12\|w\|^2$，故
$\|w_n\|$ 随 $n$ 增长。没有任何约束保证不同 $n$ 的 probe 模长相等。

### 已修复的 bug：probe-budget 轴上 R\* 曾未使用对应的 $w_n$

probe-budget 轴的横轴是"训 source probe 用了多少带标签样本 $n$"，循环内会重训
probe（`src_native`, `w_n`，[transfer.py:633-634](transfer.py#L633-L634)）。

**修复前**，R\* 分支复用循环**外**只拟合一次的 `Me2rs_full`，那次用的是全 pool 的
`w_s`：

```python
if "e2_rstar" in curves:
    a_te2rs_n, c_te2rs_n = transfer_probe(src_native, Me2rs_full, be2rs_full)
```

于是 $a$、$\beta$ 跨 $n$ 恒为同一个值；`w_n` 只在最后 `transfer_probe` 一步作为
被推送的 $a_s$ 出现，而 $M$ 是 rank-1 的：

$$
a_t = M a_s = \kappa\, a,\quad \kappa = \frac{w_s^{\top}w_n}{\|w_s\|^2};
\qquad
\text{score} = \kappa\bigl(a^{\top}h_t + \beta\bigr) + c_n
$$

即固定预测的正缩放加平移，AUROC / Spearman / Kendall 全部不变。**这就是当时观察到
R\* 的 AUROC 曲线是水平直线的原因**——不是因为模长恒定，而是因为拟合根本没有随
$n$ 重跑。

成因是分类错误：代码里 ridge / procrustes 的拟合确实与 probe 无关，提到循环外复用是
对的；而 `probe_aligned` / `e2_minimised` / `e2_map` / `e2_r0` 都在循环内传 `w_n`
重拟合（[:669](transfer.py#L669)、[:679](transfer.py#L679)、[:690](transfer.py#L690)、
[:701](transfer.py#L701)）。R\* 因为数据项只含 $\mathrm{ent}_t$、不含 source probe，
被误归入前一类——但 $w_s$ 仍经 $\alpha_{\text{eff}} = \lambda/\|w_s\|^2$ 进入正则项。

**修复后**（[transfer.py:713-714](transfer.py#L713-L714)）与兄弟方法一致，循环外的
预拟合已删除：

```python
if "e2_rstar" in curves:
    Me2rs_n, be2rs_n = fit_e2_rstar_map(Zt[pool], w_n, ent_t[pool], lam=_lam_e2)
    a_te2rs_n, c_te2rs_n = transfer_probe(src_native, Me2rs_n, be2rs_n)
```

map-budget 轴（[transfer.py:614](transfer.py#L614)）不变，仍传 `w_s`：那里 $n$ 控制
对齐/监督样本量，固定 `w_s` 正是为了让 $\alpha_{\text{eff}}$ 跨 $n$ 恒定。

### 修复后曲线该如何解读

由 (10)，`w_n` 只经 $\alpha_{\text{eff}} = \lambda/\|w_n\|^2$ 影响解，**方向不进入**。
所以修复后的 probe-budget 曲线不再严格水平，但其起伏 100% 来自"$\|w_n\|$ 随 $n$
增长 $\Rightarrow$ 正则变弱 $\Rightarrow$ $a$ 趋向 OLS 解"，等价于沿 $\lambda$
扫描走了一段，**与"source probe 质量随标注量提升"无关**。

这是 R\* 目标函数的固有性质（它回归 $\mathrm{ent}_t$，不含 source probe 信息），
不是实现问题。若要隔离正则强度这一混淆项，应固定 $\alpha_{\text{eff}}$ 而非
$\lambda$（即传 `lam = _lam_e2 * ||w_n||^2 / ||w_s||^2`），此时曲线会严格恒定，
回归其应有的语义：一条"有 target 标签时的上界"水平参考线。

> 注：修复改变了 `curveB_e2_rstar_src_probe_budget` 的数值，旧结果需重跑。

---

## 5. `transfer2.py` 中的实现

同一个方法以 `e2_rstar` 接口移植到了 [transfer2.py](transfer2.py)（`--aligners e2_rstar`，
tag 形如 `e2_rstar_l1e3`）。三点差异：

1. **只存 rank-1 因子。** Phase 2 缓存 $(a, \beta, w)$ 而非 $d_t \times d_s$ 的 $M$
   （每个 $n$ 约 150 MB）。由 (7) 两者严格等价；应用时用
   `_e2_rstar_transfer`（[transfer2.py:192](transfer2.py#L192)），按
   $\kappa = w_s^{\top}a_s/\|w_s\|^2$ 得 $a_t = \kappa a$、$c_t = \kappa\beta + c_s$，
   无需构造 $M$。缓存里带 `rank1: True` 标记，Phase 3 据此分支。
2. **probe grid 逐 $n$ 重拟合。** 对齐数据固定在最大 $n$，但对每个 probe $w_n$ 重解一次
   (10)，存为 `probe_a_{n}` / `probe_beta_{n}` / `probe_w_{n}`
   （[transfer2.py:648-667](transfer2.py#L648-L667)）——即上面 §4 的修复。
3. **监督信号来自 alignment 数据集。** $\mathrm{ent}_t$ 取 align split 的 target 熵
   （[transfer2.py:596](transfer2.py#L596)），与 alignment budget 轴的语义一致。
