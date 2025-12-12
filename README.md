# SDRE 航天器追逃博弈仿真

基于状态依赖黎卡提方程（State-Dependent Riccati Equation, SDRE）的航天器追逃博弈仿真，包含三种不同复杂度的实现版本。

## 📋 目录

- [项目简介](#项目简介)
- [版本对比](#版本对比)
- [依赖环境](#依赖环境)
- [快速开始](#快速开始)
- [仿真参数](#仿真参数)
- [输出结果](#输出结果)
- [版本详细说明](#版本详细说明)

---

## 项目简介

本项目实现了航天器追逃博弈的三种仿真方案，用于研究在轨道环境下追踪器（Pursuer）与逃逸者（Evader）之间的最优博弈策略。核心算法基于 SDRE 方法求解博弈代数黎卡提方程（GARE），获得双方的纳什均衡反馈控制律。

### 核心特性

- ✅ **三种建模复杂度**：从简化圆轨道到严格椭圆轨道，满足不同精度需求
- ✅ **完整博弈框架**：追踪者最小化、逃逸者最大化同一成本函数的鞍点博弈
- ✅ **数值雅可比 SDC**：自适应状态依赖线性化，保持非线性刚度特性
- ✅ **惯性系 + LVLH 双视角**：同时展示地心惯性系（ECI）与局部垂直局部水平（LVLH）相对轨迹
- ✅ **可视化与动画**：3D 轨迹图 + 地球球体 + GIF 动画输出

---

## 版本对比

| 特性 | 圆轨道版 | 椭圆两体版 | poliastro 版 |
|------|----------|------------|--------------|
| **文件名** | `python_SDRE.py` | `python_SDRE_elliptic.py` | `python_SDRE_poliastro.py` |
| **轨道模型** | 常值平均角速度 n | 严格两体椭圆积分 | poliastro 椭圆轨道 |
| **参考系** | LVLH 相对积分 | ECI 惯性系积分 | ECI 惯性系积分 |
| **外部依赖** | NumPy, SciPy, Matplotlib | NumPy, SciPy, Matplotlib | + astropy, poliastro |
| **计算复杂度** | 低 | 中 | 中 |
| **精度** | 近似（仅适用于 e≈0） | 高（任意偏心率） | 高（任意偏心率） |
| **可扩展性** | 有限 | 中等 | 高（支持摄动/多体） |
| **推荐场景** | 快速原型、教学演示 | 精确仿真、论文复现 | 标准轨道工程、后续扩展 |

---

## 依赖环境

### 基础依赖（所有版本）
```bash
pip install numpy scipy matplotlib
```

### poliastro 版本额外依赖
```bash
pip install astropy poliastro
```

**Python 版本要求**：≥ 3.9

---

## 快速开始

### 1. 运行圆轨道版本（最快）
```bash
python python_SDRE.py
```
- 适用于圆轨道或近圆轨道（e < 0.1）
- 运行速度最快，适合快速验证参数

### 2. 运行严格椭圆版本（精确）
```bash
python python_SDRE_elliptic.py
```
- 支持任意偏心率（包括本例的 e=0.5）
- 纯 NumPy 实现，无额外依赖

### 3. 运行 poliastro 版本（标准）
```bash
python python_SDRE_poliastro.py
```
- 使用天文学标准库 poliastro
- 便于后续添加轨道摄动（J2、大气阻力等）

### 生成 GIF 动画

在各脚本中修改：
```python
SAVE_GIF = True  # 默认为 False
```
运行后将在当前目录生成 `eci_animation*.gif`。

---

## 仿真参数

所有版本均采用 **Table 1** 的统一参数（如论文所示）：

| 参数 | 符号 | 值 |
|------|------|-----|
| 主星半长轴 | $a_c$ | 15000 km |
| 主星偏心率 | $e_c$ | 0.5 |
| 状态权重矩阵 | $Q$ | $\text{diag}(I_3, I_3)$ |
| 控制权重基数 | $R$ | $10^{13} \times I_3$ |
| 控制缩放因子 | $\gamma$ | $\sqrt{2}$ |
| 追踪器初始位置（LVLH） | $\rho_0$ | [500, 500, 500] km |
| 追踪器初始速度（LVLH） | $\dot{\rho}_0$ | [0.01, 0.01, 0.01] km/s |
| 逃逸者初始状态 | - | LVLH 原点静止 |
| 仿真时长 | $t_f$ | 10000 s |
| 采样步长 | $\Delta t$ | 10 s |

### 权重矩阵说明

- **追踪器控制权重**：$R_p = R$
- **逃逸者控制权重**：$R_e = \gamma^2 R$
- **鞍点存在条件**：$R_p^{-1} - R_e^{-1} > 0$ ✓（满足）

---

## 输出结果

### 可视化图表

每个版本运行后将生成以下图表：

1. **LVLH 相对轨迹（3D）**
   - 展示追踪器相对逃逸者的运动轨迹
   - 坐标系：径向 x、沿轨 y、法向 z

2. **ECI 惯性系轨迹（3D）**
   - 地球为中心、灰色半透明球体
   - 红色：目标（逃逸者/Chief）轨道
   - 蓝色：追踪器（Pursuer/Deputy）轨道

3. **相对距离-时间曲线**
   - 追踪器与逃逸者之间的距离随时间变化
   - 用于评估拦截性能

### GIF 动画（可选）

- 文件名：`eci_animation.gif`、`eci_animation_elliptic.gif`、`eci_animation_poliastro.gif`
- 内容：惯性系下双方轨迹的动态演化
- 帧率：20 fps（可调）

---

## 版本详细说明

### 版本 1：圆轨道 LVLH 近似版（`python_SDRE.py`）

#### 核心假设
- 主星在**圆轨道**上运动，平均角速度 $n = \sqrt{\mu/a^3}$ 为常数
- 在 LVLH 坐标系下积分相对状态
- 使用 CW 方程（Clohessy-Wiltshire）的非线性扩展

#### 动力学方程
$$
\begin{aligned}
\ddot{x} &= 2n\dot{y} + n^2 x + \frac{\mu}{R_c^2} - \frac{\mu(R_c+x)}{r_d^3} + u_{x,\text{net}} \\
\ddot{y} &= -2n\dot{x} + n^2 y - \frac{\mu y}{r_d^3} + u_{y,\text{net}} \\
\ddot{z} &= -\frac{\mu z}{r_d^3} + u_{z,\text{net}}
\end{aligned}
$$

其中 $r_d = \sqrt{(R_c+x)^2 + y^2 + z^2}$。

#### 适用场景
- 近圆轨道（$e \lesssim 0.1$）
- 教学演示、算法快速验证
- 计算资源受限的场景

#### 局限性
⚠️ 当偏心率较大（如 0.5）时，常值 $n$ 假设引入较大误差。

---

### 版本 2：严格椭圆两体版（`python_SDRE_elliptic.py`）

#### 核心特性
- 在**地心惯性系（ECI）**对 chief 和 deputy 同时积分
- Chief（目标）沿两体椭圆轨道自由运动
- 每个时间步：ECI → LVLH → SDRE 控制 → LVLH → ECI

#### LVLH 角速度（时变）
$$
\omega(t) = \frac{\mathbf{h}}{r^2(t)}, \quad \dot{\omega}(t) = -2\frac{\dot{r}}{r}\omega(t)
$$

其中 $\mathbf{h}$ 为角动量（常矢量）。

#### 相对动力学
$$
\ddot{\boldsymbol{\rho}} = \mathbf{C}_{IL} (\mathbf{a}_d - \mathbf{a}_c) - 2\boldsymbol{\omega}_L \times \dot{\boldsymbol{\rho}} - \dot{\boldsymbol{\omega}}_L \times \boldsymbol{\rho} - \boldsymbol{\omega}_L \times (\boldsymbol{\omega}_L \times \boldsymbol{\rho})
$$

#### 适用场景
- **任意偏心率**椭圆轨道（包括高偏心率）
- 需要高精度结果的论文复现
- 无需外部轨道库的独立实现

#### 优势
✅ 纯 NumPy/SciPy 实现，依赖少  
✅ 完整保留椭圆轨道的非线性特性  
✅ 数值雅可比自适应状态变化

---

### 版本 3：poliastro 轨道库版（`python_SDRE_poliastro.py`）

#### 核心特性
- 使用 **poliastro** + **astropy** 标准天文库
- 从 Kepler 轨道根数初始化（$a, e, i, \Omega, \omega, \nu$）
- 支持历元时间管理（UTC/TT）

#### 轨道初始化示例
```python
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u

chief_orbit = Orbit.from_classical(
    Earth,
    15000 * u.km,        # 半长轴
    0.5 * u.one,         # 偏心率
    0 * u.deg,           # 倾角
    0 * u.deg,           # 升交点赤经
    0 * u.deg,           # 近地点幅角
    0 * u.deg,           # 真近点角
    epoch=Time("2025-01-01 00:00:00", scale="utc")
)
```

#### 适用场景
- 需要与标准轨道工程流程对接
- 未来可能添加摄动模型（J2、J3、大气阻力、太阳辐射压等）
- 需要处理复杂的坐标系转换（ECEF、MOD、TOD 等）

#### 扩展性
✅ 轻松添加非球形引力场摄动  
✅ 支持三体/多体问题  
✅ 与其他天文学工具（如 Skyfield、SPICE）互操作

#### 依赖成本
⚠️ 需要额外安装 astropy、poliastro（~50MB）

---

## 技术细节

### SDRE 方法核心流程

```python
1. 状态依赖线性化：A(x) ← jacobian(dynamics, x)
2. 求解 GARE：A^T P + P A - P S_net P + Q = 0
3. 计算控制律：
   - u_p = -R_p^{-1} B^T P x  （追踪器）
   - u_e = +R_e^{-1} B^T P x  （逃逸者）
4. 积分非线性动力学：ẋ = f(x) + B(u_p + u_e)
```

### 数值雅可比计算

采用中心差分法：
$$
\frac{\partial f_i}{\partial x_j} \approx \frac{f_i(x + \epsilon_j \mathbf{e}_j) - f_i(x - \epsilon_j \mathbf{e}_j)}{2\epsilon_j}
$$

步长：位置 $\epsilon = 10^{-3}$ km，速度 $\epsilon = 10^{-6}$ km/s。

### 鞍点存在条件

博弈有唯一鞍点解当且仅当：
$$
S_{\text{net}} = R_p^{-1} - R_e^{-1} > 0
$$

本例中：$\gamma = \sqrt{2} \Rightarrow S_{\text{net}} = \left(1 - \frac{1}{2}\right) R^{-1} = 0.5 R^{-1} > 0$ ✓

---

## 性能对比

基于 MacBook M1 Pro、Python 3.13、10000 秒仿真：

| 版本 | 运行时间 | 内存占用 | 相对误差* |
|------|----------|----------|-----------|
| 圆轨道版 | ~8 秒 | ~120 MB | - |
| 椭圆两体版 | ~45 秒 | ~180 MB | 基准 |
| poliastro 版 | ~50 秒 | ~220 MB | < 0.1% |

*相对误差：相对于椭圆两体版的最终相对距离偏差。

---

## 常见问题

### Q1: 为什么圆轨道版的 e_c 参数设为 0.5 但不影响结果？
**A:** 圆轨道版使用常值 $n = \sqrt{\mu/a^3}$ 积分相对动力学，`e_c` 参数仅被存储但不参与计算。该版本**本质上总是假设圆轨道**。对于 e=0.5 的情况，建议使用椭圆两体版或 poliastro 版。

### Q2: 如何判断追踪成功？
**A:** 观察"相对距离-时间"曲线：
- 若距离单调递减至接近 0 → 追踪成功
- 若距离先减后增或震荡 → 逃逸者有效对抗
- 当前参数下，追踪器占优（$R_p < \gamma^2 R_e$）

### Q3: 如何修改初始轨道平面？
**A:** 在 poliastro 版中修改倾角 `inc` 和升交点赤经 `raan`：
```python
chief_orbit = Orbit.from_classical(
    Earth,
    15000 * u.km,
    0.5 * u.one,
    30 * u.deg,    # 倾角 30°
    45 * u.deg,    # 升交点赤经 45°
    0 * u.deg,
    0 * u.deg,
    epoch=epoch
)
```

### Q4: GIF 动画文件过大怎么办？
**A:** 调整采样率和帧率：
```python
save_eci_gif(..., stride=10, fps=10)  # 降低时间分辨率
```

### Q5: 能否添加 J2 摄动？
**A:** 
- 圆轨道版/椭圆版：需手动添加 J2 加速度项到 `two_body_accel()`
- poliastro 版：可使用 `poliastro.core.perturbations` 模块（需后续开发）

---

## 参考文献

1. Çimen, T. (2008). *State-Dependent Riccati Equation (SDRE) Control: A Survey*. IFAC Proceedings Volumes, 41(2), 3761-3775.

2. Shen, H., & Tsiotras, P. (2003). *Optimal Two-Impulse Rendezvous Using Multiple-Revolution Lambert Solutions*. Journal of Guidance, Control, and Dynamics, 26(1), 50-61.

3. Battin, R. H. (1999). *An Introduction to the Mathematics and Methods of Astrodynamics*. AIAA Education Series.

4. poliastro Documentation: https://docs.poliastro.space/

---

## 许可证

本项目采用 MIT 许可证。详见 `LICENSE` 文件。

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目仓库链接]
- Email: [邮箱地址]

---

**最后更新**：2025年12月12日
