# Hy-Embodied-0.5-VLA (RoboTwin) 复现笔记

> 复现腾讯 **Hy-Embodied-0.5-VLA** 在 **RoboTwin 2.0** 双臂仿真评测上的完整过程。
> 评测走的是 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) 链路（RoboTwin 的 submodule），
> 权重为 HF 上的 `tencent/Hy-Embodied-0.5-VLA-RoboTwin`。
>
> 本文重点在于成功流程和遇到的问题

---

## TL;DR

- ✅ **复现成功**：`stack_bowls_two`（demo_clean，unseen 指令）**6/6 = 100%**（机器臂将两个碗叠起来）
- 环境：英博云容器，**单张 RTX 4090 24GB**，Ubuntu 22.04，系统盘30GB，共享存储卷128GB（复现结束后使用了40GB）
- 全程约 3 个工作日（2026-10-05 ~ 10-07）
- ⚠️注意1：RoboTwin 依赖源码编译（pytorch3d / curobo，约 1 小时）
- ⚠️注意2：**XPolicyLab 自带的 `deploy.yml` 是 RoboDojo 模板**，跑 RoboTwin 权重必须逐项对齐官方 `robotwin_eval/deploy_policy.yml`——不对齐时模型"看着在动但永远不成功"，跑出 0/23
---

## 1. 环境与版本

| 项 | 值 |
|---|---|
| GPU | RTX 4090 24GB × 1（单卡，policy 与 env 共用 `device_id 0`） |
| 驱动 | 580.65.06 ,CUDA Version: 13.0 |
| 系统 | Ubuntu 22.04（容器，headless，无桌面） |
| 磁盘 | `/workspace` 128G（共享存储卷） |
| 策略侧 Python | uv venv(策略侧在uv虚拟环境中运行)，torch **2.7.1+cu128**，uv 0.12.23 |
| 环境侧 Python | conda env `RoboTwin`（python 3.10），sapien 3.0.0b1 等（按官方 requirements 钉版本） |
| 权重 | `Hy-VLA-RoboTwin-ckpt`，单文件 `model.safetensors` 8.5G（非分片）+ `norm_stats.pkl` |
| 评测资产 | `embodiments.zip` + `objects.zip` + `background_texture.zip`，合计约 13.9 GiB |

---
CUDA版本13.0，指的是的驱动上限是13.0

torch下载版本为2.7.1，cu128指的是该torch是针对CUDA12.8编译的版本（小于驱动上限13.0，符合要求）

torch 2.7.1 -->通过CUDA指挥-->驱动580.65.06-->指挥4090 24GB工作
## 2. 成功流程

### 2.1 环境搭建（六步）

以下按实际走通的顺序，每步附**验收信号**

**① 克隆代码**（RoboTwin 含 XPolicyLab submodule）

RoboTwin是由RoboTwin团队搭建的基于RoboTwin任务测评环境（先下载好这个测评环境，再补充策略侧和环境侧），其中包含仿真器+50个任务，任务代码在envs/,资产解压到 assets/、策略侧权重放在 XPolicyLab/policy/Hy_Embodied_05_VLA/ 下、结果自动落到 eval_result/
```bash
# 策略侧XPolicyLab 以 submodule 形式位于 RoboTwin/XPolicyLab（policy中包含50多个不同策略模型的适配器）
git clone https://github.com/RoboTwin-Platform/RoboTwin.git

# 验收
ls XPolicyLab/policy/Hy_Embodied_05_VLA/eval.sh     # 能列出来 = submodule 到位
git submodule status XPolicyLab                     # 能打印出 commit hash，本笔记验证时 commit: ea8b211
```

**② 策略侧 venv（uv）**


### 策略端部署 → 配置uv环境 → 安装uv工具
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
uv --version
```
### 策略端部署 → 克隆XPolicyLab仓库 / 安装Hy-Embodied（安装模型所需要的python库 如torch 2.7.1+cu128，安装模型本体代码）
```bash
cd /workspace/RoboTwin/XPolicyLab/policy/Hy_Embodied_05_VLA
bash install.sh
```
### 策略端部署 → 下载预训练权重
```bash
# 下载预训练权重
export HF_ENDPOINT=https://hf-mirror.com
cd /workspace/RoboTwin/XPolicyLab/policy/Hy_Embodied_05_VLA
hf download tencent/Hy-Embodied-0.5-VLA-RoboTwin --local-dir ./Hy-VLA-RoboTwin-ckpt

#验收信号：

"$HYROOT/.venv/bin/python" -c "import torch, hy_vla, robotwin_eval, XPolicyLab; \
print('env ok | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
# → env ok | torch 2.7.1+cu128 | cuda True
```

**④ 环境侧 conda + 源码编译**

### 1.安装vulkan（SAPIEN 是仿真引擎，负责“渲染画面+跑物理”,vulkan是SAPIEN 渲染的依赖）

SAPIEN → Vulkan → NVIDIA 驱动 → 【EGL/GLVND 这套"过桥的库"】 → GPU

这一步遇到问题：缺少EGL（EGL/GLVND是一组用户态的“转接库”——无屏幕环境下，Vulkan 要经 EGL 才能触达 GPU）
                                    
```bash
sudo apt update
sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
vulkaninfo | head -20
# 验证
vulkaninfo --summary | grep deviceName
deviceName = llvmpipe（CPU） # 画图线不通
#查询问题
ls /usr/lib/x86_64-linux-gnu/libEGL.so.1 2>/dev/null && echo "EGL 在，问题在别处" || echo "EGL 缺 → apt-get install -y libegl1"
#如果egl缺，下载
apt-get install -y libegl1   # libegl1 是一个Ubuntu软件包，里面装的是 EGL 的运行库文件。
# 验证
vulkaninfo --summary | grep deviceName
deviceName = NVIDIA GeForce RTX 4090   #正确结果
```

### 创建并激活 conda 环境
```bash
# conda是 包和环境管理器，与uv最大区别是可以管（Python包 + 非Python的软件（ffmpeg、CUDA 工具链、C 库））
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
```
### 安装仿真的依赖
```bash
# 安装仿真器和评测脚本使用的库 安装sapien ，mplib 机器臂轨迹规划使用，open3d 3D几何处理 处理物体模型和空间变换等
bash scripts/_install.sh
```
### 下载仿真资产（构成场景的原始素材，机器人、纹理等），在 HuggingFace 上

```bash
# 机器人（0.2G），物体（10070个物体模型 3.48G），纹理（背景纹理10G），此步通过脚本下载容易失败，可以手动"下载“ → sha256 校验 →
# objects.zip            6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27
# embodiments.zip        85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789
# background_texture.zip 54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070

bash scripts/_download_assets.sh

#素材下载限流时：
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli login               # 官方建议：遇到限流先登录
bash scripts/_download_assets.sh

# 验收
###  test_render.py 能出画面 → Vulkan + SAPIEN 这一串通了，这是评测能跑的前提
###  ffmpeg -version 有输出 → 官方要求确认项。ffmpeg用来处理视频数据，后续可以保留测评时的视频
cd /workspace/RoboTwin
conda activate RoboTwin
python -c "import sapien, mplib, curobo, torch; print('env ok', torch.__version__)"
python scripts/test_render.py
ffmpeg -version
```
### 2.2 冒烟测试（debug，不起仿真）

```bash
#用最小代价快速验证“系统能不能基本跑起来”的初检
cd /workspace/RoboTwin/XPolicyLab/policy/Hy_Embodied_05_VLA
export HY_VLA_CKPT_PATH=/workspace/RoboTwin/XPolicyLab/policy/Hy_Embodied_05_VLA/Hy-Embodied-0.5-VLA/Hy-VLA-RoboTwin-ckpt
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin stack_bowls_two Hy-VLA-RoboTwin-ckpt arx_x5 ee 0 0 0 uv RoboTwin

#验收信号：

[SERVER] ckpt override: ckpt_path=.../Hy-VLA-RoboTwin-ckpt
[hy_vla] decode mode: with_absolute=True, chunk=20
[hy_vla] model ready (...)
[CONNECTED] Policy server ready on localhost:xxxxx
...
[TestEnv] Episode finished
[MAIN] eval finished
```

> [!WARNING]
> **debug 只验证"策略服务能起来 + 动作格式对得上"**。`debug_env_client.py` 是纯桩环境：
> 不建 SAPIEN、不加载资产、不跑物理，观测全是零向量。debug 的分数**没有任何意义**。

### 2.3 真跑sim

去掉 `EVAL_ENV_TYPE=debug` 前缀即可（变量为空默认sim）。
集数与指令类型通过 `ROBOTWIN_EVAL_ARGS_FILE` 追加
（**每行一个 token**，见[坑 J](#坑-j-robotwin_eval_args_file-的-mapfile-写法)）：

```bash
# unseen 是官方测评跑的参数，指的是评测的时候给模型的指令是训练时没见过的指令。
cat > ~/rt_args.txt <<'EOF'
--test_num
10
--instruction_type
unseen
EOF
export ROBOTWIN_EVAL_ARGS_FILE=~/rt_args.txt

bash eval.sh RoboTwin stack_bowls_two Hy-VLA-RoboTwin-ckpt arx_x5 ee 0 0 0 uv RoboTwin 2>&1 | tee ~/sim_eval.log
```
---

## 3. 遇到的问题及解决方法

### 3.1 环境搭建阶段

#### 问题A：uv 装 torch 走不通（官方 index 被墙）

- **症状**：`tunnel error`（`download.pytorch.org` 不在平台代理白名单）
- **根因**：uv.lock 的 content-hash 绑定 pyproject；改 index 会触发全量重算
- **解法**：pyproject 不动，sed 改 uv.lock 的 wheel URL + `UV_FROZEN=1`

#### 问题B：HF 大文件下载——hf-mirror 对大文件只做 302 转发

- **症状**：下载 `background_texture.zip`（10.2G）卡在 `cas-bridge.xethub.hf.co`，`Read timed out`；`HF_HUB_DISABLE_XET=1` 无效
- **根因**：`HF_ENDPOINT` 只管 Hub API；实际字节被服务端 302 到 Xet CAS 桥（国外），镜像站不代理数据。实测确认 cas-bridge 支持 Range（HTTP 206），但不稳
- **解法**：`curl -sI` 取 302 的带签名直链（3600s 有效）→ `aria2c -x8 -s8 -c --checksum=sha-256=<官方sha>`；官方 sha256 从 `POST /api/datasets/TianxingChen/RoboTwin2.0/paths-info/main` 取
- **验收**：`sha256sum` 与官方 lfs.oid **逐字符一致**才算成功——unzip 能打开只说明结构对齐，不证明内容无损（我们实测过"尺寸正好、zip 可解析"但 sha 不符的坏包）

#### 问题C：`_download_assets.sh` 解压失败也删包

- **症状**：脚本回显 `Processing complete!`，但 `assets/` 下只剩 `embodiments/`，两个大包连同 zip 一起消失
- **根因**：脚本写法是 `unzip X.zip` 换行 `rm -rf X.zip`——无条件删除；unzip 失败时证据和重试机会一起没了；而 `update_embodiment_config_path.py` 只依赖 embodiments，所以照样打印成功
- **解法**：不用该脚本，手动"下载 → sha256 校验 → `unzip -q -o X.zip && rm -f X.zip`"
- **教训**：**脚本的回显、进度条、"Processing complete!" 都不算证据，验收只认实际产物**

#### 问题 D：Vulkan 只认到 llvmpipe，4090 未被识别

- **症状**：`vulkaninfo` 报 `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0`，GPU0 = llvmpipe；SAPIEN 报 `failed to find a rendering device`
- **根因**：精简镜像缺 EGL/GLVND 运行时包（`libegl1`、`libglvnd0`）。NVIDIA 的 Vulkan ICD 在 headless 场景经 EGL/GBM 路径触达 GPU，这条路需要 EGL vendor 注册。此前逐项排查过权限/库/符号/依赖/设备节点/ICD 位置/loader 版本共七项，全对，唯独漏了"用户空间图形栈整包缺失"这一项
- **解法**：`apt-get install -y libegl1`，一行修好
- **附带**：SAPIEN 的 `_vulkan_tricks.py` 只检查 `/usr/share/vulkan/icd.d/nvidia_icd.json`，不看 `/etc/vulkan/icd.d/`；且 `/etc/vulkan/icd.d/` 是 bind mount（只读，`sed -i` 会报 Device or resource busy），改 ICD 一律写新文件到 `/usr/share/`

### 3.2 评测阶段

#### 问题E：ckpt 静默回落——权重指错了却不报任何错

- **症状**：`eval.sh` 正常启动、模型正常加载，但跑出来的成绩无效
- **根因**：`setup_eval_policy_server.sh` 的 `resolve_ckpt_override()` 按优先级找权重，全落空时**不报错、不覆盖**，静默沿用 `deploy.yml` 默认的 `ckpt_path`——而默认值是 **RoboDojo 权重** `Hy-VLA-RoboDojo-v3/hyvla_dojo_ckpt_v3`。我们的权重路径不在任何候选里
- **解法**：`export HY_VLA_CKPT_PATH=<权重绝对路径>`（优先级最高）
- **验收信号**：日志出现 `[SERVER] ckpt override: ckpt_path=...Hy-VLA-RoboTwin-ckpt`；若出现 `ckpt_name='...' not resolved ...; using deploy.yml ckpt_path` 就是没指对

#### 问题 F：`model.py:321` AssertionError（n_action_steps 与 norm_stats 不匹配）

- **症状**：模型加载中途 `assert effective_chunk <= val.shape[0]` 失败，server 起不来
- **根因**：ckpt 的 `config.json` 里 `n_action_steps: 40`，但 `norm_stats.pkl` 里动作统计是 `(20, 20)`。40 = 2×20——该权重按 **rel+abs 交织**训练（40 个 token = 20 相对 + 20 绝对），而 `deploy.yml:30` 默认 `with_absolute: false`，chunk 算成 40 > 20
- **解法**：`deploy.yml` 第 30 行 `with_absolute: false → true`（生效后日志显示 `decode mode: with_absolute=True, chunk=20`）
- **佐证**：pkl 里同时存在相对（`action_mean`）与绝对（`action_mean_abs`）两组统计，与训练侧 `act_type: "relative_chunk_ee_RT_with_absolute"` 一致

#### 问题G：环境侧 `import cv2` 失败（libGL.so.1）

- **症状**：`ImportError: libGL.so.1: cannot open shared object file`
- **根因**：精简镜像缺 `libgl1`（cv2 需要）；与坑 D 的 `libegl1`（SAPIEN 需要）同族但不同包，别想当然认为装了一个就够
- **解法**：`apt-get install -y libgl1`

#### 问题 H：真跑卡在 ffmpeg 缺失（debug 不踩、真跑才踩）

- **症状**：SAPIEN 场景正常渲染、连接成功，随后 `eval_policy_xpolicylab.py:849` 的 `subprocess.Popen(["ffmpeg", ...])` 报 `FileNotFoundError`
- **根因**：镜像没装 ffmpeg 可执行文件。debug 是桩环境不录像，所以冒烟发现不了
- **解法**：`apt-get install -y ffmpeg`（需含 libx264）；或零下载方案——软链 `imageio_ffmpeg` 自带的静态 ffmpeg（已确认含 libx264）到 `/usr/local/bin/ffmpeg`

#### 问题I：最大的坑——`deploy.yml` 是 RoboDojo 模板

- **症状**：权重指对、服务正常、22 集跑完 21 集，`Success rate: 0/21 => 0.0%`，每集 900 步打满。而 ckpt 自带 README 白纸黑字写着 90.9% average success rate——90.9% 的模型跑出 0/21，不可能是"策略弱"
- **根因**：`XPolicyLab/policy/Hy_Embodied_05_VLA/deploy.yml` 是**给 RoboDojo 权重写的模板**（第 1 行注释就写着 "RoboDojo"）。用 `HY_VLA_CKPT_PATH` 换了**权重**，但整份预处理/解码**口径**还是 RoboDojo 的。与官方 `Hy-Embodied-0.5-VLA/robotwin_eval/deploy_policy.yml`（全文仅 37 行）共 **6 处差异**：

| # | 参数 | 官方（RoboTwin ckpt） | XPolicyLab 模板默认 |
|---|---|---|---|
| 1 | `umi_coord_frame` | **false** | true |
| 2 | `blend_mode` | **rel_abs** | rel_only |
| 3 | `img_history_interval` | **5** | 20 |
| 4 | `exc_action_size` | **7** | 10 |
| 5 | `history_cadence` | **（无此键 → 默认 step）** | chunk |
| 6 | `instruction_type` | **unseen**（官方脚本硬编码） | 不传 → 落到任务 yml 的 seen |

- **过程中的次生坑**：改完前 4 项后崩在新断言 `img_history_interval (5) must be divisible by the chunk history span (7)`——第 5 项 `history_cadence: chunk` 是从 RoboDojo 模板带来的，官方文件里根本没有这个键。chunk 模式下 RoboDojo 的 `20 % 10 == 0` 成立，照搬到 RoboTwin 的 `5 % 7` 就炸；改 `step`（或删掉该行）即通过
- **解法**：6 处全部对齐官方（见上表）。其中 `instruction_type` 官方链路不传，需经 `ROBOTWIN_EVAL_ARGS_FILE` 补传（问题J）
- **结果**：0/23 → **6/6 = 100%**。确诊问题在部署口径，不在模型能力

> [!IMPORTANT]
> 官方模型卡里的评测入口 `scripts/eval_robotwin_test.sh` **在仓库里不存在**（本仓库已被
> XPolicyLab 改造），只能走 XPolicyLab 链路 + 改 `deploy.yml` 对齐官方口径。

#### 问题 J：`ROBOTWIN_EVAL_ARGS_FILE` 的 mapfile 写法

- **症状**：`--test_num 10` 写在一行，argparse 报 `unrecognized arguments: --test_num 10`
- **根因**：`eval_policy.sh` 用 `mapfile -t` 读入，**每行只当一个 argv 元素**
- **解法**：**每行一个 token**（`--test_num` ⏎ `10` ⏎ `--instruction_type` ⏎ `unseen`）

#### 问题 K：`test_num` 默认 100 集（约 9 小时）

- **根因**：XPolicyLab 链路不传 `--test_num`，源码默认 100（`eval_policy_xpolicylab.py:307`）；实测约 5.5 min/集 → 100 集 ≈ 9 小时。日志里那句 `Success rate: 0/1` 的分母是"已完成集数"，不是设定值
- **解法**：按坑 J 用 `ROBOTWIN_EVAL_ARGS_FILE` 传 `--test_num`，先跑 10 集验证再上量

---

## 4. 验证与数字

| 项 | 论文/官方 | 本次复现 |
|---|---|---|
| stack_bowls_two（demo_clean，unseen） | 50 任务平均 **90.9%**（Clean）① | **6/6 = 100%**（seed 100000–100006，跳过 1 个 unstable seed） |

① 官方 90.9% 是 **50 个任务 × 100 集**的平均值，与单任务 6 集不可直接比较，仅作量级参照。

**一个好用的小判据**：成功集视频约 **10 秒**（约 100 步，成功即提前结束）；失败集固定 **90 秒**（900 步打满）。光看 mp4 时长就能分辨成败。

**debug 阶段证明了 / 没证明了什么**（避免误读冒烟结果）：

- ✅ 证明：策略服务起得来、权重指对、动作 key 前缀与维度过校验、client↔server 链路通
- ❌ 没证明：SAPIEN 能否建真场景、资产能否加载、物理能否跑、**分数无意义**（观测全是零向量）

---

## 5. 与官方结果的差异

- **评测入口不同**：官方模型卡给的 `eval_robotwin_test.sh` 在本仓库不存在，实际走 XPolicyLab 链路 + 手动对齐 6 处 deploy.yml 口径
- **规模不同**：官方 50 任务 × 100 集（标注需 8 GPUs）；本次单卡 4090，仅验证 1 任务 × 6 集
- **归因限制**：6 处修改分批完成但一起生效，**无法单独归因**哪一处是 0/23 的决定性根因（证据链最强的是 `umi_coord_frame` 与 `blend_mode`，未逐一验证）

---

## 6. 未解决 / TODO

- [ ] 官方完整评测（50 任务 × 100 集，单卡约需数百小时，不现实）；计划按任务抽样
- [ ] 更多任务类型验证（目前仅 `stack_bowls_two` 一个任务）
- [ ] `instruction_type` 的独立归因（seen 理论上不该更差，未单独实验）
- [ ] RoboDojo（Isaac Sim）那条线尚未开始

---

## 7. 附录

**`eval.sh` 的 10 个位置参数**：

```
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
             <seed> <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>
```

**评测链路调用链**：

```
eval.sh（调度器：起 server + client）
├── 策略侧：setup_eval_policy_server.sh → uv venv → setup_policy_server.py
└── 环境侧：setup_eval_env_client.sh → 按 EVAL_ENV_TYPE 分派
    ├── debug → debug_env_client.py（纯桩环境，不起仿真）
    └── sim   → RoboTwin/scripts/eval_policy.sh（真跑 SAPIEN）
```

**ckpt 解析优先级**（`resolve_ckpt_override`）：`$HY_VLA_CKPT_PATH` > 绝对/相对路径 > 三个候选目录 > 静默回落 `deploy.yml:ckpt_path`。

**关键验收信号速查**：

| 想确认什么 | 看哪里 |
|---|---|
| 权重指对了 | `[SERVER] ckpt override: ckpt_path=...Hy-VLA-RoboTwin-ckpt` |
| with_absolute 配对了 | `decode mode: with_absolute=True, chunk=20` |
| 模型加载完成 | `[hy_vla] model ready (...)` |
| SAPIEN 场景建好了 | 日志出现 `Render Well` + Config 打印 |
| 一集成败 | `Success!` / `Fail! 900 / 900`（或视频 10s vs 90s） |

---

*复现人：KIKI · 2026-10-05 ~ 2026-10-07 · 单卡 RTX 4090（英博云）*
