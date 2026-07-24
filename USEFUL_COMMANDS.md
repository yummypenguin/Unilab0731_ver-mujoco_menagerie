# UniLab Useful Commands

這份文件中的指令以 Windows 11 PowerShell 為準。除非特別註明，請先切換到
UniLab 根目錄：

```powershell
Set-Location D:\UniLab
```

不要使用 `training.sim_backend=...` 切換 backend。請使用 `--sim mujoco` 或
`--sim motrix`，讓 CLI 選擇正確的 owner YAML。

## 安裝與更新環境

安裝 MuJoCo 額外套件：

```powershell
Set-Location D:\UniLab
uv sync --extra mujoco
```

安裝 Motrix 額外套件：

```powershell
Set-Location D:\UniLab
uv sync --extra motrix
```

同時安裝 MuJoCo 與 Motrix：

```powershell
Set-Location D:\UniLab
uv sync --extra mujoco --extra motrix
```

## LEAP Hand Smoke Test

MuJoCo，16 environments、1 iteration、不播放：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand --sim mujoco `
  algo.max_iterations=1 `
  algo.num_envs=16 `
  training.no_play=true
```

Motrix，16 environments、1 iteration、不播放：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand --sim motrix `
  algo.max_iterations=1 `
  algo.num_envs=16 `
  training.no_play=true
```

## LEAP Hand 正式訓練

建議的 MuJoCo 設定：600 iterations，每 25 iterations 儲存 checkpoint。

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand --sim mujoco `
  algo.max_iterations=600 `
  algo.save_interval=25 `
  training.no_play=true
```

Motrix 使用相同 checkpoint 頻率：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand --sim motrix `
  algo.max_iterations=600 `
  algo.save_interval=25 `
  training.no_play=true
```

## LEAP Hand Ball Rotation (Allegro Task Reference)

新任務 `leap_inhand_ball` 使用 LEAP 自有的 `scene_ball.xml` 與 `ball.xml`，只參考
Allegro 的球體旋轉 reward、observation、PPO 與 action scale；不引用 Allegro 球資產。
模型 binding 仍使用 LEAP Hand、LEAP 專屬球體、初始 grasp 與必要模擬參數。
既有 `leap_inhand` cube rotation 與 `leap_inhand_toss` 不受影響。

MuJoCo smoke test：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand_ball --sim mujoco `
  algo.max_iterations=1 `
  algo.num_envs=16 `
  training.no_play=true
```

Motrix smoke test：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand_ball --sim motrix `
  algo.max_iterations=1 `
  algo.num_envs=16 `
  training.no_play=true
```

正式訓練沿用 Allegro owner 的 PPO iteration 設定；在本機先使用 1024 environments：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand_ball --sim mujoco `
  algo.num_envs=1024 `
  training.no_play=true
```

播放最新 checkpoint：

```powershell
Set-Location D:\UniLab
uv run eval --algo ppo --task leap_inhand_ball --sim mujoco `
  --load-run -1 `
  --render-mode interactive
```

## LEAP Hand Toss Curriculum

新任務使用 `leap_inhand_toss`，依序訓練穩定接取、回程、輔助反彈與完整拇指拋接。

MuJoCo smoke test：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand_toss --sim mujoco `
  algo.max_iterations=1 `
  algo.num_envs=16 `
  training.no_play=true
```

Motrix smoke test：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand_toss --sim motrix `
  algo.max_iterations=1 `
  algo.num_envs=16 `
  training.no_play=true
```

完整 curriculum training：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand_toss --sim mujoco `
  algo.num_envs=1024 `
  algo.max_iterations=3500 `
  'env.curriculum.level_steps=[0,4000,9000,15000]' `
  training.no_play=true
```

相同正式設定使用 Motrix：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand_toss --sim motrix `
  algo.num_envs=1024 `
  algo.max_iterations=3500 `
  'env.curriculum.level_steps=[0,4000,9000,15000]' `
  training.no_play=true
```

Toss 任務預設的 phase timeout 已放寬，完整流程最多可使用 15 秒：

| Phase | Timeout | `ctrl_dt=0.05` 對應控制步數 |
|---|---:|---:|
| SUPPORT | 5.0 秒 | 100 |
| FLIGHT | 1.2 秒 | 24 |
| IMPACT | 0.5 秒 | 10 |
| RETURN | 2.5 秒 | 50 |
| CAPTURE | 3.0 秒 | 60 |

臨時測試其他 timeout，不修改 owner config：

```powershell
uv run train --algo ppo --task leap_inhand_toss --sim mujoco `
  env.max_episode_seconds=15.0 `
  env.support_timeout_seconds=5.0 `
  env.flight_timeout_seconds=1.2 `
  env.impact_timeout_seconds=0.5 `
  env.return_timeout_seconds=2.5 `
  env.capture_timeout_seconds=3.0 `
  training.no_play=true
```

此任務的 owner config 已將 `algo.save_interval` 設為 25。因為每個 iteration
包含 8 個控制步，所以每 200 個控制步會儲存一次 checkpoint。各 curriculum
level 可觀察的 checkpoint 約為：

| Level | 控制步範圍 | 建議查看的 checkpoint |
|---|---:|---:|
| 0：穩定抓取 | 0–3999 | `model_25.pt`、`model_250.pt`、`model_475.pt` |
| 1：返回與抓取 | 4000–8999 | `model_500.pt`、`model_750.pt`、`model_1100.pt` |
| 2：飛行、返回與抓取 | 9000–14999 | `model_1125.pt`、`model_1500.pt`、`model_1850.pt` |
| 3：完整拇指拋接 | 15000 以上 | `model_1875.pt`、`model_2500.pt`、`model_3500.pt` |

單次列出最新 toss run 的所有 checkpoints：

```powershell
Set-Location D:\UniLab

$RunRoot = "D:\UniLab\logs\rsl_rl_ppo\LeapInhandToss"
$RunPath = Get-ChildItem $RunRoot -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Write-Host "Run: $($RunPath.Name)"

Get-ChildItem $RunPath.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_", "") } |
  Select-Object Name, Length, LastWriteTime
```

在另一個 PowerShell 視窗每 10 秒查看最新產生的 toss checkpoints：

```powershell
$RunRoot = "D:\UniLab\logs\rsl_rl_ppo\LeapInhandToss"

while ($true) {
  $RunPath = Get-ChildItem $RunRoot -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  Clear-Host
  Write-Host "Run: $($RunPath.FullName)"
  Write-Host "Updated: $(Get-Date)"
  Write-Host ""

  Get-ChildItem $RunPath.FullName -Filter "model_*.pt" |
    Sort-Object { [int]($_.BaseName -replace "model_", "") } |
    Select-Object Name, Length, LastWriteTime

  Start-Sleep -Seconds 10
}
```

按 `Ctrl+C` 停止即時更新。

在訓練不中斷的情況下，用另一個 PowerShell 視窗播放指定 level 的 checkpoint：

```powershell
Set-Location D:\UniLab
$RunId = "請填入目前的_run_id"
$Checkpoint = 125

uv run eval --algo ppo --task leap_inhand_toss --sim mujoco `
  --load-run $RunId `
  algo.checkpoint=$Checkpoint `
  --render-mode interactive
```

播放模式固定從完整的拇指承托狀態開始，因此不同 level checkpoint 可以用同一個
完整任務直接比較。若 checkpoint 正好正在寫入，等待下一次清單更新後再播放。

自動選擇最新 run 的最新 checkpoint 並互動播放：

```powershell
Set-Location D:\UniLab

$RunRoot = "D:\UniLab\logs\rsl_rl_ppo\LeapInhandToss"
$RunPath = Get-ChildItem $RunRoot -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$LatestModel = Get-ChildItem $RunPath.FullName -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_", "") } -Descending |
  Select-Object -First 1

$RunId = $RunPath.Name
$Checkpoint = [int]($LatestModel.BaseName -replace "model_", "")

Write-Host "Loading $RunId / model_$Checkpoint.pt"

uv run eval --algo ppo --task leap_inhand_toss --sim mujoco `
  --load-run $RunId `
  algo.checkpoint=$Checkpoint `
  --render-mode interactive
```

錄製指定 toss checkpoint：

```powershell
Set-Location D:\UniLab
$RunId = "請填入目前的_run_id"
$Checkpoint = 125

uv run eval --algo ppo --task leap_inhand_toss --sim mujoco `
  --load-run $RunId `
  algo.checkpoint=$Checkpoint `
  --render-mode record
```

播放時 owner config 會停用 curriculum reset，固定從完整拇指承托階段開始：

```powershell
Set-Location D:\UniLab
uv run eval --algo ppo --task leap_inhand_toss --sim mujoco `
  --load-run -1 `
  --render-mode interactive
```

### 官方 MuJoCo Viewer 互動查看 Toss Checkpoint

`scripts/play_interactive.py` 使用官方 `mujoco.viewer.launch_passive`，不是
Viser，也不是 MP4 錄影入口。載入指定 checkpoint：

```powershell
Set-Location D:\UniLab

$RunId = "2026-07-14_17-37-26_mujoco"
$Checkpoint = 500

uv run scripts/play_interactive.py `
  --algo ppo `
  --task leap_inhand_toss `
  --sim mujoco `
  interactive.action_mode=policy `
  algo.load_run=$RunId `
  algo.checkpoint=$Checkpoint
```

省略 `algo.checkpoint` 時會選擇該 run 最新的 checkpoint：

```powershell
Set-Location D:\UniLab

$RunId = "2026-07-14_17-37-26_mujoco"

uv run scripts/play_interactive.py `
  --algo ppo `
  --task leap_inhand_toss `
  --sim mujoco `
  interactive.action_mode=policy `
  algo.load_run=$RunId
```

Viewer 操作：空白鍵暫停/繼續、`N` 單步、`+`/`-` 調速；關閉視窗或按
`Esc` 結束。滑鼠拖曳旋轉、滾輪縮放、右鍵拖曳平移。

查看 toss curriculum 與事件指標：

```powershell
Set-Location D:\UniLab
uv run tensorboard --logdir D:\UniLab\logs\rsl_rl_ppo\LeapInhandToss
```

主要指標為 `toss/curriculum_level`、`toss/launch_event`、
`toss/rebound_event`、`toss/success_event` 與 `toss/failure_event`。

指定不同 seed：

```powershell
Set-Location D:\UniLab
uv run train --algo ppo --task leap_inhand --sim mujoco `
  algo.seed=2 `
  algo.max_iterations=600 `
  algo.save_interval=25 `
  training.no_play=true
```

## 查看 Run 與 Checkpoint

依時間列出 LEAP Hand runs：

```powershell
$LogRoot = "D:\UniLab\logs\rsl_rl_ppo\LeapInhandRotation"

Get-ChildItem $LogRoot -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object Name, CreationTime, LastWriteTime
```

列出指定 run 的 checkpoints。先修改 `$RunId`：

```powershell
$RunId = "2026-07-14_02-06-13_mujoco"
$RunPath = "D:\UniLab\logs\rsl_rl_ppo\LeapInhandRotation\$RunId"

Get-ChildItem $RunPath -Filter "model_*.pt" |
  Sort-Object { [int]($_.BaseName -replace "model_", "") } |
  Select-Object Name, Length, LastWriteTime
```

查看訓練摘要：

```powershell
$RunId = "2026-07-14_02-06-13_mujoco"
$Summary = "D:\UniLab\logs\rsl_rl_ppo\LeapInhandRotation\$RunId\run_summary.json"
Get-Content $Summary
```

## Evaluation

### 載入最新 Run

MuJoCo 最新 run：

```powershell
Set-Location D:\UniLab
uv run eval --algo ppo --task leap_inhand --sim mujoco --load-run -1
```

Motrix 最新 run，互動顯示：

```powershell
Set-Location D:\UniLab
uv run eval --algo ppo --task leap_inhand --sim motrix `
  --load-run -1 `
  --render-mode interactive
```

注意：`--load-run -1` 會選擇 `LeapInhandRotation` 中時間最新的 run，不會依
MuJoCo/Motrix 自動過濾。正式比較 checkpoint 時應明確指定 `$RunId`。

### 指定 Run 與 Checkpoint

MuJoCo 播放指定 checkpoint：

```powershell
Set-Location D:\UniLab
$RunId = "2026-07-14_02-06-13_mujoco"
$Checkpoint = 500

uv run eval --algo ppo --task leap_inhand --sim mujoco `
  --load-run $RunId `
  algo.checkpoint=$Checkpoint
```

Motrix 互動播放指定 checkpoint：

```powershell
Set-Location D:\UniLab
$RunId = "2026-07-13_23-41-10_motrix"
$Checkpoint = 500

uv run eval --algo ppo --task leap_inhand --sim motrix `
  --load-run $RunId `
  algo.checkpoint=$Checkpoint `
  --render-mode interactive
```

無畫面 evaluation：

```powershell
Set-Location D:\UniLab
$RunId = "2026-07-14_02-06-13_mujoco"
$Checkpoint = 500

uv run eval --algo ppo --task leap_inhand --sim mujoco `
  --load-run $RunId `
  algo.checkpoint=$Checkpoint `
  --render-mode none `
  training.play_steps=200
```

### Sim2Sim

在 Motrix 播放 MuJoCo 訓練的 checkpoint：

```powershell
Set-Location D:\UniLab
$MujocoRunId = "2026-07-14_02-06-13_mujoco"
$Checkpoint = 500

uv run eval --algo ppo --task leap_inhand --sim motrix `
  --load-run $MujocoRunId `
  algo.checkpoint=$Checkpoint `
  --render-mode interactive
```

在 MuJoCo 播放 Motrix 訓練的 checkpoint：

```powershell
Set-Location D:\UniLab
$MotrixRunId = "2026-07-13_23-41-10_motrix"
$Checkpoint = 500

uv run eval --algo ppo --task leap_inhand --sim mujoco `
  --load-run $MotrixRunId `
  algo.checkpoint=$Checkpoint
```

## 從 Checkpoint 繼續訓練

訓練模式使用 `algo.load_run`，不要使用 eval 專用的 `--load-run`：

```powershell
Set-Location D:\UniLab
$RunId = "2026-07-14_02-06-13_mujoco"
$Checkpoint = 500

uv run train --algo ppo --task leap_inhand --sim mujoco `
  algo.load_run=$RunId `
  algo.checkpoint=$Checkpoint `
  algo.max_iterations=200 `
  algo.save_interval=25 `
  training.no_play=true
```

這會建立新 run，載入舊 run 的指定 checkpoint 後繼續學習。

## TensorBoard

啟動 TensorBoard：

```powershell
Set-Location D:\UniLab
uv run tensorboard --logdir D:\UniLab\logs\rsl_rl_ppo\LeapInhandRotation
```

在瀏覽器開啟：

```text
http://localhost:6006
```

主要觀察：

- `Train/mean_reward`：越高越好。
- `reward/rotate`：越高越好。
- `reward/drop`：越接近 0 越好。
- `reward/total`：越高越好。
- `Policy/mean_std`：避免突然降到接近 0 或劇烈震盪。

LEAP ball rotation V2 使用獨立 log 目錄：

```powershell
Set-Location D:\UniLab
uv run tensorboard --logdir D:\UniLab\logs\rsl_rl_ppo\LeapInhandBallRotationV2
```

V2 主要觀察 `rotation/axis_speed_rad_s`、`rotation/axis_speed_1s_mean`、
`rotation/completed_turns`、`rotation/drop_rate`、`curriculum/level` 和
`curriculum/target_speed_rad_s`。

## LEAP Hand Sustained +Z Ball Rotation

此獨立任務從固定 LEAP ball home pose 開始，依序要求 1 秒穩定承托，接著以
`0.04`、`0.07`、`0.085`、`0.10`、`0.16`、`0.25`、`0.50 rad/s` 的 staged
curriculum 學習世界 `+Z` 旋轉。新增的 `0.085 rad/s` bridge 需連續維持 1 秒，
其晉級門檻為 `0.068 rad/s`。

正向旋轉 reward 同時依照 EMA 目標速度追蹤品質與目前 stage 的連續有效時間
調整。TensorBoard 應搭配觀察 `rotation/speed_tracking_quality`、
`rotation/stage_duration_progress` 與 `rotation/consecutive_valid_seconds`；只有
接近目標速度並持續維持，才能取得完整正向旋轉 reward。反向旋轉仍保留完整
負 reward。

MuJoCo 訓練：

```powershell
Set-Location D:\UniLab

uv run train --algo ppo --task leap_inhand_ball_sustained --sim mujoco `
  algo.num_envs=4096 `
  algo.num_steps_per_env=8 `
  algo.max_iterations=1000 `
  algo.save_interval=25 `
  training.no_play=true
```

官方 MuJoCo viewer 查看指定 run 的最新 checkpoint：

```powershell
Set-Location D:\UniLab
$RunId = "替換成這次的RunId"

uv run scripts/play_interactive.py `
  --algo ppo `
  --task leap_inhand_ball_sustained `
  --sim mujoco `
  interactive.action_mode=policy `
  algo.load_run=$RunId
```

官方 MuJoCo viewer 查看指定 checkpoint：

```powershell
Set-Location D:\UniLab
$RunId = "替換成這次的RunId"
$Checkpoint = 250

uv run scripts/play_interactive.py `
  --algo ppo `
  --task leap_inhand_ball_sustained `
  --sim mujoco `
  interactive.action_mode=policy `
  algo.load_run=$RunId `
  algo.checkpoint=$Checkpoint
```

TensorBoard：

```powershell
Set-Location D:\UniLab
uv run tensorboard --logdir D:\UniLab\logs\rsl_rl_ppo\LeapInhandBallSustainedRotation
```

重點觀察 `rotation/axis_speed_ema`、`rotation/orthogonal_speed_ema`、
`retention/position_error_m`、`retention/palm_contact_rate`、三個
`failure/*_rate`、`curriculum/level`、`success/sustained_10s_fraction` 與
`Policy/mean_std`。

停止 TensorBoard：在執行它的 PowerShell 視窗按 `Ctrl+C`。

## 影片

MuJoCo 指定 checkpoint 錄影：

```powershell
Set-Location D:\UniLab
$RunId = "2026-07-14_02-06-13_mujoco"
$Checkpoint = 500

uv run eval --algo ppo --task leap_inhand --sim mujoco `
  --load-run $RunId `
  algo.checkpoint=$Checkpoint `
  --render-mode record
```

尋找並開啟最新的 `play_video.mp4`：

```powershell
$Video = Get-ChildItem "D:\UniLab\logs\rsl_rl_ppo\LeapInhandRotation" `
  -Recurse -Filter "play_video.mp4" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$Video | Select-Object FullName, Length, LastWriteTime
Invoke-Item $Video.FullName
```

Motrix `interactive` 已在目前 Windows/Intel Arc 環境驗證成功。Motrix
`record` 曾發生 WGPU out-of-memory，因此需要 MP4 時優先使用 MuJoCo。

## Motrix 與 GPU 診斷

查看 Windows 顯示卡與 driver：

```powershell
Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion, AdapterRAM
```

指定 Motrix/WGPU 使用 DirectX 12，再啟動互動播放：

```powershell
Set-Location D:\UniLab
$env:WGPU_BACKEND = "dx12"
$env:WGPU_POWER_PREF = "high-performance"

uv run eval --algo ppo --task leap_inhand --sim motrix `
  --load-run -1 `
  --render-mode interactive
```

需要完整錯誤資訊時：

```powershell
$env:RUST_BACKTRACE = "1"
$env:HYDRA_FULL_ERROR = "1"
```

清除本次 PowerShell session 的診斷環境變數：

```powershell
Remove-Item Env:WGPU_BACKEND -ErrorAction SilentlyContinue
Remove-Item Env:WGPU_POWER_PREF -ErrorAction SilentlyContinue
Remove-Item Env:RUST_BACKTRACE -ErrorAction SilentlyContinue
Remove-Item Env:HYDRA_FULL_ERROR -ErrorAction SilentlyContinue
```

## 常見規則

- 所有 Python entrypoint 都使用 `uv run`。
- backend 用 `--sim mujoco` 或 `--sim motrix` 選擇。
- eval 使用 `--load-run <run-id>`。
- 訓練續跑使用 `algo.load_run=<run-id>`。
- 指定 checkpoint 使用 `algo.checkpoint=<iteration>`。
- `algo.save_interval` 控制 checkpoint 儲存頻率。
- 比較模型時明確指定 run id，不要依賴 `--load-run -1`。
- 最新 checkpoint 不一定是最佳 checkpoint，應配合 TensorBoard 和實際播放比較。
# LEAP Hand Ball Grasp Cache

The generator uses the existing cube cache only as proposal poses, settles each
proposal in the LEAP-owned ball scene, and writes a different cache file. Do not
run cache generation alongside a large training run because both workloads use
substantial simulator resources.

```powershell
Set-Location D:\UniLab

uv run train --algo ppo --task leap_inhand_ball_grasp --sim mujoco `
  algo.num_envs=1024 `
  training.no_play=true
```

MuJoCo output:

```text
src/unilab/assets/robots/leap_hand/caches/ball_grasp_s10_5k.npy
```

The Motrix owner deliberately uses a separate output until cross-backend
validation is complete:

```powershell
Set-Location D:\UniLab

uv run train --algo ppo --task leap_inhand_ball_grasp --sim motrix `
  algo.num_envs=1024 `
  training.no_play=true
```

Motrix output:

```text
src/unilab/assets/robots/leap_hand/caches/ball_grasp_s10_5k_motrix.npy
```

## LEAP Sustained Rotation From Cache

This task keeps the sustained-rotation reward, weights, observations, and
curriculum while sampling each initial hand/ball state from
`ball_grasp_official_50k.npy`. It trains from scratch unless load arguments are
added explicitly.

```powershell
Set-Location D:\UniLab

uv run train --algo ppo --task leap_inhand_ball_sustained_cache --sim mujoco `
  algo.num_envs=4096 `
  algo.num_steps_per_env=8 `
  algo.max_iterations=1000 `
  algo.save_interval=25 `
  training.no_play=true
```

The drop boundary is relative to each sampled cache state: the episode fails
after the ball center remains at least 7 mm below its reset height for three
control steps. No fixed world-height threshold is used by this task.

## LEAP Direct Fixed-Speed Ball Rotation

This diagnostic task removes curriculum promotion and required handoffs. It
starts from `ball_grasp_official_50k.npy`, targets world `+Z` at `0.30 rad/s`,
and logs natural contact transitions without using them as reward or gates.

```powershell
Set-Location D:\UniLab

uv run train --algo ppo --task leap_inhand_ball_direct --sim mujoco `
  algo.num_envs=2048 `
  algo.num_steps_per_env=16 `
  algo.max_iterations=100 `
  algo.save_interval=25 `
  training.no_play=true
```

View the latest diagnostic checkpoint with reward telemetry:

```powershell
Set-Location D:\UniLab

uv run scripts/play_interactive.py `
  --algo ppo `
  --task leap_inhand_ball_direct `
  --sim mujoco `
  interactive.action_mode=policy `
  interactive.show_reward_debug=true `
  algo.load_run=-1
```
