# Hy-Embodied-0.5-VLA (RoboTwin) 复现笔记

> 复现腾讯 **Hy-Embodied-0.5-VLA** 在 **RoboTwin 2.0** 双臂仿真评测上的完整过程。
> 评测走的是 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) 链路（RoboTwin 的 submodule），
> 权重为 HF 上的 `tencent/Hy-Embodied-0.5-VLA-RoboTwin`。
>
> 本文重点不在"怎么跑"（官方 README 已写），而在**官方文档没写的差异和坑**。

---

## TL;DR

- ✅ **复现成功**：`stack_bowls_two`（demo_clean，unseen 指令）**6/6 = 100%**
- 环境：英博云容器，**单张 RTX 4090 24GB**，Ubuntu 22.04
- 全程约 3 个工作日（2026-10-05 ~ 10-07）
- 最耗时的一关：RoboTwin 依赖源码编译（pytorch3d / curobo，约 1 小时）
- 最坑的一关：**XPolicyLab 自带的 `deploy.yml` 是 RoboDojo 模板**，跑 RoboTwin 权重必须逐项对齐官方 `robotwin_eval/deploy_policy.yml`——不对齐时模型"看着在动但永远不成功"，跑出 0/23

> [!IMPORTANT]
> 一句话教训：**`deploy.yml` 的预处理/解码口径必须和权重的训练口径一致**。
> 权重指对了不等于配置对了——我们正是栽在这上面（详见[坑 I](#坑-i-最大的坑deployyml-是-robodojo-模板)）。

---

## 1. 环境与版本

| 项 | 值 |
|---|---|
| GPU | RTX 4090 24GB × 1（单卡，policy 与 env 共用 `device_id 0`） |
| 驱动 | 580.65.06 |
| 系统 | Ubuntu 22.04（容器，headless，无桌面） |
| 磁盘 | `/workspace` 128G（共享存储卷） |
| 策略侧 Python | uv venv，torch **2.7.1+cu128**，uv 0.12.23 |
| 环境侧 Python | conda env `RoboTwin`（python 3.10），sapien 3.0.0b1 等（按官方 requirements 钉版本） |
| 权重 | `Hy-VLA-RoboTwin-ckpt`，单文件 `model.safetensors` 8.5G（非分片）+ `norm_stats.pkl` |
| 评测资产 | `embodiments.zip` + `objects.zip` + `background_texture.zip`，合计约 13.9 GiB |

---

## 2. 成功流程

### 2.1 环境搭建（六步）

以下按实际走通的顺序，每步附**验收信号**——没有验收信号不算完成。

**① 克隆代码**（RoboTwin 含 XPolicyLab submodule）

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git   # [待补：实际 commit hash]
# XPolicyLab 以 submodule 形式位于 RoboTwin/XPolicyLab
```

**② 策略侧 venv（uv）**

国内直连 `download.pytorch.org` 会失败。正确做法：**`pyproject.toml` 保持原样**（保 content-hash），只 `sed` 替换 `uv.lock` 里的 wheel URL 到国内镜像，然后：

```bash
export UV_FROZEN=1    # 禁止 uv 重算依赖
# 运行策略侧 install 脚本
```

验收信号：

```bash
"$HYROOT/.venv/bin/python" -c "import torch, hy_vla, robotwin_eval, XPolicyLab; \
print('env ok | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
# → env ok | torch 2.7.1+cu128 | cuda True
```

> [!WARNING]
> 不要改 `pyproject.toml` 的 `[[tool.uv.index]]`——lock 的 content-hash 会失效，触发 universal
> resolution，而国内镜像只有 Linux wheel，macOS 分支求解直接报 "No solution found"。

**③ 下载权重（8.5G，单文件 safetensors）**

该仓库未启用 Xet，`HF_ENDPOINT=https://hf-mirror.com` 可直下（见[坑 B](#坑-b-hf-大文件下载-hf-mirror-对大文件只做-302-转发)）。

**④ 环境侧 conda + 源码编译**

```bash
conda create -n RoboTwin python=3.10 -y && conda activate RoboTwin
cd /workspace/RoboTwin
export TORCH_CUDA_ARCH_LIST=8.9    # 4090 只需 sm_89，编译量降到 ~1/7
bash scripts/_install.sh           # pytorch3d / curobo 源码编译，约 30~60 分钟，需代理
```

**⑤ 下载评测资产（三个 zip，约 13.9 GiB）**

官方 `_download_assets.sh` 有"解压失败也删包"的缺陷，建议手动下载（见[坑 B](#坑-b-hf-大文件下载-hf-mirror-对大文件只做-302-转发)、[坑 C](#坑-c-_download_assetssh-解压失败也删包)）：

```bash
# 官方 sha256（从 HF API paths-info 接口取得，见坑 B）
embodiments.zip        219,847,741    85ffeff5…
objects.zip          3,737,778,549    6aa56b3c…
background_texture.zip 10,970,687,027 54ede0fb…

aria2c -x8 -s8 -k4M -c --checksum=sha-256=<官方sha> -o <文件> "<签名直链>"
sha256sum <文件>     # 必须与官方一致才算下载成功
unzip -q -o <文件> && rm -f <文件>    # 用 && 绑住删除
python ./scripts/update_embodiment_config_path.py
```

**⑥ Vulkan/EGL 修复（headless 容器必做）**

精简容器镜像缺 GL/EGL 用户态整包，SAPIEN/Vulkan 只认到 llvmpipe（CPU 软渲染）。一行修好：

```bash
apt-get update -qq && apt-get install -y libegl1
```

验收信号：

```bash
vulkaninfo --summary | grep deviceName
# → deviceName = NVIDIA GeForce RTX 4090   （而不是 llvmpipe）
python scripts/test_render.py   # → 绿字 Render Well
```

> [!NOTE]
> `apt install` 只活在当前容器，**重建实例后需重装**。SAPIEN 的 `_vulkan_tricks.py`
> 只认 `/usr/share/vulkan/icd.d/nvidia_icd.json`，不看 `/etc/vulkan/icd.d/`（详见[坑 D](#坑-d-vulkan-只认到-llvmpipe4090-未被识别)）。

### 2.2 冒烟测试（debug，不起仿真）

```bash
cd /workspace/RoboTwin/XPolicyLab/policy/Hy_Embodied_05_VLA
export HY_VLA_CKPT_PATH=/workspace/RoboTwin/XPolicyLab/policy/Hy_Embodied_05_VLA/Hy-Embodied-0.5-VLA/Hy-VLA-RoboTwin-ckpt
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin stack_bowls_two Hy-VLA-RoboTwin-ckpt arx_x5 ee 0 0 0 uv RoboTwin
```

验收信号：

```
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

### 2.3 真跑 sim

去掉 `EVAL_ENV_TYPE=debug` 前缀即可（变量为空默认 sim）。集数与指令类型通过 `ROBOTWIN_EVAL_ARGS_FILE` 追加（**每行一个 token**，见[坑 J](#坑-j-robotwin_eval_args_file-的-mapfile-写法)）：

```bash
cat > ~/rt_args.txt <<'EOF'
--test_num
10
--instruction_type
unseen
EOF
export ROBOTWIN_EVAL_ARGS_FILE=~/rt_args.txt

bash eval.sh RoboTwin stack_bowls_two Hy-VLA-RoboTwin-ckpt arx_x5 ee 0 0 0 uv RoboTwin 2>&1 | tee ~/sim_eval.log
```

前提：`deploy.yml` 已按第 3 节坑 I 对齐官方口径（**先对齐再跑，否则白烧几小时**）。

---

## 3. 坑与解法

### 3.1 环境搭建阶段

#### 坑 A：uv 装 torch 走不通（官方 index 被墙）

- **症状**：`tunnel error`（`download.pytorch.org` 不在平台代理白名单）
- **根因**：uv.lock 的 content-hash 绑定 pyproject；改 index 会触发全量重算
- **解法**：pyproject 不动，sed 改 uv.lock 的 wheel URL + `UV_FROZEN=1`（见 2.1-②）

#### 坑 B：HF 大文件下载——hf-mirror 对大文件只做 302 转发

- **症状**：下载 `background_texture.zip`（10.2G）卡在 `cas-bridge.xethub.hf.co`，`Read timed out`；`HF_HUB_DISABLE_XET=1` 无效
- **根因**：`HF_ENDPOINT` 只管 Hub API；实际字节被服务端 302 到 Xet CAS 桥（国外），镜像站不代理数据。实测确认 cas-bridge 支持 Range（HTTP 206），但不稳
- **解法**：`curl -sI` 取 302 的带签名直链（3600s 有效）→ `aria2c -x8 -s8 -c --checksum=sha-256=<官方sha>`；官方 sha256 从 `POST /api/datasets/TianxingChen/RoboTwin2.0/paths-info/main` 取
- **验收**：`sha256sum` 与官方 lfs.oid **逐字符一致**才算成功——unzip 能打开只说明结构对齐，不证明内容无损（我们实测过"尺寸正好、zip 可解析"但 sha 不符的坏包）

#### 坑 C：`_download_assets.sh` 解压失败也删包

- **症状**：脚本回显 `Processing complete!`，但 `assets/` 下只剩 `embodiments/`，两个大包连同 zip 一起消失
- **根因**：脚本写法是 `unzip X.zip` 换行 `rm -rf X.zip`——无条件删除；unzip 失败时证据和重试机会一起没了；而 `update_embodiment_config_path.py` 只依赖 embodiments，所以照样打印成功
- **解法**：不用该脚本，手动"下载 → sha256 校验 → `unzip -q -o X.zip && rm -f X.zip`"
- **教训**：**脚本的回显、进度条、"Processing complete!" 都不算证据，验收只认实际产物**

#### 坑 D：Vulkan 只认到 llvmpipe，4090 未被识别

- **症状**：`vulkaninfo` 报 `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0`，GPU0 = llvmpipe；SAPIEN 报 `failed to find a rendering device`
- **根因**：精简镜像缺 EGL/GLVND 运行时包（`libegl1`、`libglvnd0`）。NVIDIA 的 Vulkan ICD 在 headless 场景经 EGL/GBM 路径触达 GPU，这条路需要 EGL vendor 注册。此前逐项排查过权限/库/符号/依赖/设备节点/ICD 位置/loader 版本共七项，全对，唯独漏了"用户空间图形栈整包缺失"这一项
- **解法**：`apt-get install -y libegl1`，一行修好
- **附带**：SAPIEN 的 `_vulkan_tricks.py` 只检查 `/usr/share/vulkan/icd.d/nvidia_icd.json`，不看 `/etc/vulkan/icd.d/`；且 `/etc/vulkan/icd.d/` 是 bind mount（只读，`sed -i` 会报 Device or resource busy），改 ICD 一律写新文件到 `/usr/share/`

### 3.2 评测阶段

#### 坑 E：ckpt 静默回落——权重指错了却不报任何错

- **症状**：`eval.sh` 正常启动、模型正常加载，但跑出来的成绩无效
- **根因**：`setup_eval_policy_server.sh` 的 `resolve_ckpt_override()` 按优先级找权重，全落空时**不报错、不覆盖**，静默沿用 `deploy.yml` 默认的 `ckpt_path`——而默认值是 **RoboDojo 权重** `Hy-VLA-RoboDojo-v3/hyvla_dojo_ckpt_v3`。我们的权重路径不在任何候选里
- **解法**：`export HY_VLA_CKPT_PATH=<权重绝对路径>`（优先级最高）
- **验收信号**：日志出现 `[SERVER] ckpt override: ckpt_path=...Hy-VLA-RoboTwin-ckpt`；若出现 `ckpt_name='...' not resolved ...; using deploy.yml ckpt_path` 就是没指对

#### 坑 F：`model.py:321` AssertionError（n_action_steps 与 norm_stats 不匹配）

- **症状**：模型加载中途 `assert effective_chunk <= val.shape[0]` 失败，server 起不来
- **根因**：ckpt 的 `config.json` 里 `n_action_steps: 40`，但 `norm_stats.pkl` 里动作统计是 `(20, 20)`。40 = 2×20——该权重按 **rel+abs 交织**训练（40 个 token = 20 相对 + 20 绝对），而 `deploy.yml:30` 默认 `with_absolute: false`，chunk 算成 40 > 20
- **解法**：`deploy.yml` 第 30 行 `with_absolute: false → true`（生效后日志显示 `decode mode: with_absolute=True, chunk=20`）
- **佐证**：pkl 里同时存在相对（`action_mean`）与绝对（`action_mean_abs`）两组统计，与训练侧 `act_type: "relative_chunk_ee_RT_with_absolute"` 一致

#### 坑 G：环境侧 `import cv2` 失败（libGL.so.1）

- **症状**：`ImportError: libGL.so.1: cannot open shared object file`
- **根因**：精简镜像缺 `libgl1`（cv2 需要）；与坑 D 的 `libegl1`（SAPIEN 需要）同族但不同包，别想当然认为装了一个就够
- **解法**：`apt-get install -y libgl1`

#### 坑 H：真跑卡在 ffmpeg 缺失（debug 不踩、真跑才踩）

- **症状**：SAPIEN 场景正常渲染、连接成功，随后 `eval_policy_xpolicylab.py:849` 的 `subprocess.Popen(["ffmpeg", ...])` 报 `FileNotFoundError`
- **根因**：镜像没装 ffmpeg 可执行文件。debug 是桩环境不录像，所以冒烟发现不了
- **解法**：`apt-get install -y ffmpeg`（需含 libx264）；或零下载方案——软链 `imageio_ffmpeg` 自带的静态 ffmpeg（已确认含 libx264）到 `/usr/local/bin/ffmpeg`

#### 坑 I：最大的坑——`deploy.yml` 是 RoboDojo 模板

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
- **解法**：6 处全部对齐官方（见上表）。其中 `instruction_type` 官方链路不传，需经 `ROBOTWIN_EVAL_ARGS_FILE` 补传（见坑 J）
- **结果**：0/23 → **6/6 = 100%**。确诊问题在部署口径，不在模型能力

> [!IMPORTANT]
> 官方模型卡里的评测入口 `scripts/eval_robotwin_test.sh` **在仓库里不存在**（本仓库已被
> XPolicyLab 改造），只能走 XPolicyLab 链路 + 改 `deploy.yml` 对齐官方口径。

#### 坑 J：`ROBOTWIN_EVAL_ARGS_FILE` 的 mapfile 写法

- **症状**：`--test_num 10` 写在一行，argparse 报 `unrecognized arguments: --test_num 10`
- **根因**：`eval_policy.sh` 用 `mapfile -t` 读入，**每行只当一个 argv 元素**
- **解法**：**每行一个 token**（`--test_num` ⏎ `10` ⏎ `--instruction_type` ⏎ `unseen`），见 2.3

#### 坑 K：`test_num` 默认 100 集（约 9 小时）

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
