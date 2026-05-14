# AutoRenderingUE5 — Reasoning-Structured Video Renderer

UE5 automated data-collection pipeline used to render the **Reasoning-Structured Videos** dataset (NeurIPS 2026 Datasets & Benchmarks submission). This directory has two parts:

1. `DataCollector.h` / `DataCollector.cpp` — a UE5 C++ Actor that drives a state machine to **deterministically** render trajectory videos in batch with path-level algebraic-relation annotations.
2. `frames_to_mp4.py` + `run_frames.bat` — locally pack the rendered image frames into `mp4` for upload / training.

> Paper thesis: standard next-step prediction cannot identify whether a video world model is globally consistent under **composed actions**; the dataset must therefore organise trajectories as a **rooted graph** with path-level algebraic relation labels (Inverse / Loop / Equivalence). Every design choice in this code is in service of making those algebraic relations hold **exactly at the pixel level**.

---

## 1. What this code does

Given a UE5 level (indoor, outdoor, or mixed scenes), this code will:

1. Automatically generate a grid of root sample poses inside the scene (x/y grid × N orientations, filtered by capsule-overlap tests against scene geometry to drop positions inside walls or furniture).
2. From each root pose, sample a trajectory type by weight (Seed / Inverse / Loop / Equivalence; the Hard tier additionally provides Hard_Loop / Hard_Equivalence / Hard_Inverse), and use the corresponding *Plan\*Trajectory* constructor to produce a 40-step action sequence.
3. Step the agent through the action sequence in UE5 using a capsule embodiment **with collision**, interpolating each action over 9 frames in 0.5 s, and continuously rendering RGB / Depth / (optional) optical flow.
4. Route each trajectory based on whether any step collided: `reasoning/` (collision-free, the algebraic identity holds exactly) or `random_walk/` (at least one collision, the identity is broken — kept as counterfactual / control data).
5. Write a `metadata.json` per trajectory recording `trajectory_type`, paired ID, action sequence, root state, collision mask, scale tier, and every other label needed for evaluation.
6. After rendering, run `frames_to_mp4.py` to pack each trajectory's 360 RGB frames + 360 depth frames into separate `.mp4` files and copy the metadata to the output directory.

The end result is exactly the `inverse / loop / equivalence / random_walk` video collection described in the `huggingface/` model-card.

---

## 2. Code architecture

```
AutoRenderingUE5/
├── DataCollector.h            # Actor class declaration + every editor-tunable parameter
├── DataCollector.cpp          # State machine, trajectory planning, collision detection, frame I/O
├── frames_to_mp4.py           # Offline frame_XXXX.jpg sequence → .mp4 (RGB + Depth)
├── run_frames.bat             # Small wrapper that batch-invokes frames_to_mp4.py
└── log/                       # Runtime collision log, plan-retry stats
```

`DataCollector` is the core — an `AActor` subclass that, once dropped into a level, becomes an "automated camera robot". Every parameter is exposed in the editor panel via `UPROPERTY(EditAnywhere)`, so the same Actor can be reused on any UE5 level **without touching code**.

### 2.1 Component structure

```
ADataCollector
├── CollisionCapsule (UCapsuleComponent, Root) ─ simulates the agent's physical volume
├── SceneCapture     (USceneCaptureComponent2D) ─ RGB,    SCS_FinalColorHDR + RGBA16F RT
├── DepthCapture     (USceneCaptureComponent2D) ─ Depth,  SCS_SceneDepth   + R32F RT  → 16-bit PNG
└── VelocityCapture  (USceneCaptureComponent2D) ─ Optical flow (optional, requires post-process material M_VelocityPass)
```

Design highlights:

- **HDR path, not LDR.** `SCS_FinalColorHDR` + an `RGBA16F` RenderTarget preserves the full post-process chain and correct tonemapping, avoiding the severe under-exposure (black frames) that occurs when SceneCapture in LDR mode disagrees with the editor viewport.
- **`bCaptureEveryFrame` is disabled.** By default the engine auto-captures every frame; combined with our manual `CaptureScene()` calls that doubles render cost. So both capture components turn auto-capture off and only fire when the state machine asks.
- **The Velocity (optical-flow) channel goes through a PostProcess Material.** The user must create an `M_VelocityPass` in the editor (`Material Domain = Post Process`, `SceneTexture: Velocity → Emissive`, `Blendable Location = Replacing the Tonemapper`) and assign it to `VelocityPassMaterial`. If unassigned, the optical-flow channel auto-disables and logs a warning.

### 2.2 State machine (FSM)

`Tick()` advances the state machine once per frame. Both modes share the same `ECollectorPhase` enum:

```
WaitForSceneReady           ─ wait for level streaming + 30-frame warm-up
TeleportToSamplePoint       ─ teleport to the next root pose, wait 15 frames for Lumen GI / auto-exposure to settle
                              └─ these 15 frames are "consumed"; they do not hit disk → the rendering function g
                                 is deterministic only on the saved frames

[Mode A] CounterfactualTree
  RenderContextVideo        ─ shoot a context video at the current pose
  RenderActionVideo         ─ shoot one branch video per action in ActionSpace
  ResetForNextAction        ─ reset to the original pose

[Mode B] TrajectoryGraph (the mode used in the paper)
  Traj_PlanTrajectory       ─ sample a trajectory type → call the corresponding Plan*Trajectory()
                              └─ on failure, retry up to MaxPlanRetries=20 times (pure simulation, no render — cheap)
                              └─ rotate the start yaw every 5 retries to escape one-sided wall blockers
  Traj_ExecuteStep          ─ execute step i, render 9 frames (FramesPerAction)
  Traj_AdvanceStep          ─ advance to the next step; record collision flag, hit actor name / normal / phys material
  Traj_FinalizeTrajectory   ─ write metadata, route to reasoning/ or random_walk/ based on has_collision

AdvanceToNextSamplePoint
Completed                   ─ all collection done, write manifest + exit
```

Why the explicit settle phases? **Determinism.** After every teleport, Lumen GI needs 10–20 iterations to converge and auto-exposure needs a few frames to stabilise. If we shot the first frames directly, we would get observations that are dark and brightness-inconsistent with later frames — breaking the deterministic-$g$ assumption $g\colon \mathcal{S}\to\mathcal{X}$ in the paper, which in turn breaks the premise that an Inverse path returns to a pixel-identical start.

### 2.3 Trajectory construction (the core logic; corresponds to paper §3.2)

Each trajectory type has a `Plan*Trajectory()` method returning `bool`; planning failures are auto-retried. The shared primitives:

- `SampleNaturalPath()` — a **natural exploration path** sampler with inertia + reverse damping + a rotation budget. Inertia ×1.4 biases toward continuing the current action type; reverse damping ×0.05 suppresses immediate U-turns; each trajectory has a consecutive-rotation budget of 3; collision-veto weight ×0.01 with up to 5 resamples.
- `IsPathCollisionFree()` — a **pre-simulation** of the entire candidate action sequence (swept line trace) that filters wall-hitting plans before burning GPU time. This is the key performance lever: capsule simulation in UE costs < 1 ms/step, while real rendering is ~1–2 min/traj — so "retry planning 20 times" is essentially free, but it lifts Hard-tier trajectory success from ~24% to ~99%.
- `GetInverseAction()` — action-to-action inverse mapping (forward↔backward, turn_left↔turn_right, etc.).

| Type | Constructor | Paper formula |
|---|---|---|
| **Inverse** | `PlanInverseTrajectory` samples a K-step path A and produces $A\,\|\,\text{no\_op}^{T-2K}\,\|\,A^{-1}$ | $T_{A^{-1}}\circ T_A(s_0) = s_0$ |
| **Loop** | `PlanLoopTrajectory`, two branches: ① exploration + a five-stage discrete return (yaw → pitch → fwd/bwd → lateral; accepted only when residual < 0.45 Δ horizontal / 0.3 Δ vertical) ② **topologically closed polygons** (rectangle / triangle / hexagon), geometrically exact | $T_A(s_0) = s_0$ |
| **Equivalence** | `PlanEquivalenceTrajectory`, two branches: ① Stage-1 Fisher–Yates shuffle inside each mono-type segment (commutation) ② Stage-2 fallback: L-shape $U^m V^n$ vs. zig-zag $(UV)^k U^{m-k} V^{n-k}$, with self-inverse rotation pairs filling remaining slots | $T_A(s_0)=T_B(s_0)$, $A\ne B$ |
| **Hard_Loop** | `PlanHardLoopTrajectory` — geometric templates only; defends against MIND's round-trip shortcut attack | Same as Loop, but not decomposable into Inverse |
| **Hard_Equivalence** | `PlanHardEquivalenceTrajectory` — macro reordering (L vs. zig-zag) with visually distinctive A and B | Same as Equivalence |
| **Hard_Inverse** | `PlanHardInverseTrajectory` — A·A⁻¹ that mixes rotations, providing a non-abelian witness | Same as Inverse |

#### How the two paths of an Equivalence pair are stored

`PlanEquivalenceTrajectory` populates both `CurrentTrajectoryActions` (A) and `PairedTrajectoryActions` (B). The state machine **first walks A to completion**, returns to the root pose, and **then walks B**. The two trajectories receive different global IDs but each metadata records the other in `paired_trajectory.partner_trajectory_id`, which downstream evaluation scripts use to reconstruct the (A,B) pair.

#### Collision routing (`reasoning/` vs. `random_walk/`)

After every step, the "expected end-pose" is compared with the "swept actual end-pose" — if the displacement exceeds `CollisionThreshold = 1 cm`, `bCurrentTrajectoryHasCollision = true` is set, and the hit Actor's name / phys material / normal / impact point are written into the step record. `Traj_FinalizeTrajectory` then chooses the destination directory:

- **`reasoning/`** — 0 collisions, the algebraic relation holds exactly → goes into the test set for consistency evaluation.
- **`random_walk/`** — at least one collision → reduced to a regular action-conditioned video; still trainable, but excluded from Inv/Loop/Equiv metrics.

### 2.4 Hard-tier Scale Ladder (response to MIND)

`EScaleTier` + `HardTierScaleLadder` implement the four-step-size ladder from paper §4.3:

| Tier | trans × | rot × | Aligned with |
|---|---|---|---|
| BaseDefault | 1.0× | 1.0× | Our already-rendered 300 h Easy baseline |
| Fine | 0.5× | 1.0× | Habitat / AI2-THOR / ProcTHOR (25–50 cm) |
| Coarse | 2.0× | 1.0× | Matrix-Game / Minecraft (~1 block) |
| WideRot | 1.0× | 2.0× | AI2-THOR (90°/step) / ProcTHOR (30°/step) |

Each Hard-tier trajectory is randomly assigned a tier by `Weight`. Before planning starts, `ApplyHardTierScaleJitter()` multiplies the `Delta*` fields by the chosen scale; after rendering, `RestoreActionSpaceScale()` puts them back. The tier name is written to `metadata.scale_tier.name` so evaluation scripts can group by tier and report cross-tier robustness `Gap_r` (paper §6.3).

### 2.5 Deterministic rendering: every "looks-irrelevant" switch

To make $g\colon\mathcal{S}\to\mathcal{X}$ truly deterministic, all of the following are explicitly controlled (corresponds to paper §3.1 "deterministic capture pipeline"):

| Item | Switch | Default | Why |
|---|---|---|---|
| Freeze directional light | `bFreezeLighting` | `true` | Counterfactual reasoning requires that pixel changes come only from actions, not from time-of-day |
| Sun pitch / yaw | fixed | -45° / 0° | Moderate shadows, bright frames |
| Motion Blur | `bEnableMotionBlur` | `false` | Each frame is captured independently in offline collection; motion blur introduces artefacts |
| Depth of Field | `bEnableDepthOfField` | `false` | Training data needs the entire frame in focus |
| Lens Flares | `bEnableLensFlares` | `false` | Temporal jitter |
| Ray Tracing | `bEnableRayTracing` | `false` | Temporal jitter + heavy GPU cost |
| Auto exposure | `bUseFixedExposure=false`, but clamped to [0.5, 8.0] | — | Fully fixed exposure burns out scenes with strong indoor/outdoor contrast; clamping keeps consistency reasonable while preserving visibility |
| Teleport settle | `TeleportSettleFrames` | 15 | Lumen GI convergence + auto-exposure stabilisation |
| Warm-up | `WarmupFrames` | 30 | Level stream-in, shader compilation |

### 2.6 Performance and throughput

- **`FramesPerTick = 8`** — run 8 state-machine frames per engine Tick, skipping 7 engine-tick overheads. Too high freezes the editor; too low leaves throughput bottlenecked on the engine tick rate.
- **Async file writes** — when `bAsyncFileWrite=true`, all PNG/JPEG encoding + I/O is dispatched to background `Async(EAsyncExecution::ThreadPool, ...)`, letting the main thread immediately render the next frame; the `PendingAsyncWrites` counter blocks shutdown until all writes flush.
- **JPEG vs. BMP** — `ImageFormat=JPEG` + `JpegQuality=95` is the default. At 512×512, a JPEG frame is ~30–50 KB versus BMP's 768 KB (15–25× smaller); visually indistinguishable, training results match empirically, encoding speed is on par with BMP.
- **Collision whitelist** — `CollisionWhitelistKeywords = [Landscape, Floor, Ground]`. Large ground actors like `Landscape_0` or `LandscapeStreamingProxy_1` have huge collision volumes but cause no visual clipping; whitelisting them lets collision detection ignore them so the camera can move freely across large scenes.

---

## 3. Output directory layout

`SaveRootDirectory` defaults to `F:/A_worldmodel/UE5data/`. Every run automatically creates a timestamped sub-directory:

```
{SaveRoot}/run_YYYYMMDD_HHMMSS/
├── manifest.json                           # Global manifest
├── reasoning/                              # 0 collisions — algebraic identities hold exactly
│   ├── traj_000000/
│   │   ├── metadata.json                   # trajectory_type, paired_trajectory, action_sequence, root_state, collision_mask, scale_tier, ...
│   │   ├── step_00/
│   │   │   ├── frame_0000.jpg ... frame_0008.jpg     # 9 RGB frames
│   │   │   └── depth/
│   │   │       └── frame_0000.png ... frame_0008.png # 16-bit PNG, normalised to DepthMaxDistance
│   │   ├── step_01/  ...
│   │   └── step_39/
│   └── traj_000001/  ...
└── random_walk/                            # at least 1 collision
    └── traj_000XYZ/  ...
```

Key fields of `metadata.json`:

```json
{
  "trajectory_type": "equivalence",
  "algebraic_property": "equivalence: T_A(s₀) = T_B(s₀), A ≠ B (commutative shuffle)",
  "paired_trajectory": {
    "is_paired": true, "role": "path_A",
    "partner_trajectory_id": 60, "primary_trajectory_id": 59,
    "relation": "path_A and path_B are commutative shuffles, should reach same final state"
  },
  "has_collision": false,
  "total_steps": 40, "frames_per_step": 9, "total_frames": 360,
  "render_config": {"resolution": [1280,720], "fov": 79.0, "image_format":"jpeg",
                    "modalities": ["rgb","depth"]},
  "root_state": {"position":[-3990,1700,400], "rotation":[0,60,0]},
  "action_sequence": ["turn_right","look_down",...,"move_left"],
  "collision_mask": [0,0,0,...],
  "scale_tier": {"name":"BaseDefault","trans":1.0,"rot":1.0}
}
```

---

## 4. Usage

### 4.1 Render in UE5

1. Drop `DataCollector.cpp` / `.h` into your UE5 C++ project (the editor will auto-compile). The project's `.Build.cs` needs to depend on `Engine, RenderCore, RHI, ImageWrapper, ImageWriteQueue`.
2. (Only needed for optical flow.) Create `M_VelocityPass` under Content:
   - `Material Domain = Post Process`
   - Add a `SceneTexture` node, set `Scene Texture Id = Velocity`
   - Connect `SceneTexture.RGB → Emissive Color`
   - `Blendable Location = Replacing the Tonemapper`
3. Drag `ADataCollector` into the level. In the Details panel:
   - Set `SaveRootDirectory`, `SamplingRangeX/Y`, `SamplingInterval`, `SamplingZ`
   - Choose `CollectorMode = TrajectoryGraph`
   - Set the weights: for an Easy-tier run, set `WeightInverse / WeightLoop / WeightEquivalence`; to top up Hard data, zero all Easy weights and tune `WeightHard*` instead.
4. Hit Play / Simulate. The state machine will iterate through every sample point and exit on completion.
5. While rendering you can monitor:
   - The UE5 Output Log — key events and plan-retry stats
   - `{SaveRoot}/run_*/log/` — per-collision details (Actor name, phys material, normal)

> ⚠️ Do not let the editor sleep or lock the screen during rendering: UE5 may down-tick or pause when the window is minimised. We recommend running via `RunUAT` or in standalone `-game` mode.

### 4.2 Frames → mp4

```bash
# Default fast-validate: 360p, CRF=28, fps=16, ultrafast, depth at 720p
python frames_to_mp4.py \
    --root_dir   F:/A_worldmodel/UE5data/ \
    --output_dir F:/A_worldmodel/mp4_data/ \
    --workers 8 --skip_existing

# Training-quality: original resolution + near-lossless
python frames_to_mp4.py \
    --root_dir   F:/A_worldmodel/UE5data/ \
    --output_dir F:/A_worldmodel/mp4_data/ \
    --resolution 720p --crf 18 --preset fast \
    --workers 8
```

Main parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `--root_dir` | — | Auto-scan all `run_*` sub-directories underneath |
| `--input_dir` | — | Explicitly specify one or more run directories (mutually exclusive with `--root_dir`) |
| `--resolution` | `360p` | `360p` / `480p` / `720p` / `1080p` / `original` / `WxH` |
| `--depth_resolution` | `720p` | Depth-video resolution (depth carries more information than RGB, default is higher) |
| `--crf` / `--depth_crf` | 28 / 18 | H.264 quality; depth defaults to higher quality |
| `--workers` | 4 | Number of `ProcessPoolExecutor` workers |
| `--skip_existing` | off | Incremental run: trajectories that already have mp4 + meta are skipped |
| `--no_export_depth` | — | Disable depth-video export |

Implementation notes:

- ffmpeg is preferred (auto-detected); falls back to OpenCV `VideoWriter` otherwise.
- The ffmpeg path uses `concat demuxer + libx264`. Depth videos right-shift the 16-bit PNG by 8 bits to obtain 8-bit grayscale (~3× faster than float division), then pipe straight to ffmpeg with `-pix_fmt gray`, avoiding a redundant BGR→YUV copy.
- The output is laid out as `run_*/{reasoning,random_walk}/traj_*.mp4`, copying the original `metadata.json` and appending `video_encoding` / `depth_video_encoding` fields.

---

## 5. Mapping to paper / dataset

| Paper § | Code location |
|---|---|
| §3.1 deterministic capture pipeline | `BeginPlay()` render initialisation + `ApplyRenderQualitySettings()` + `FreezeLighting()` |
| §3.2 Inverse / Loop / Equivalence construction | `PlanInverseTrajectory()` / `PlanLoopTrajectory()` / `PlanEquivalenceTrajectory()` |
| §4.2 trajectory graph (40 actions × 9 frames = 360 frames @ 16 fps) | `TrajectorySteps=40`, `FramesPerAction=9`, `FramesPerSecond=16` |
| §4.3 reasoning/ vs. random_walk/ split (>1 cm threshold) | `CollisionThreshold=1.0`, `Traj_FinalizeTrajectory` |
| §4.3 Scale ladder (BaseDefault / Fine / Coarse / WideRot) | `EScaleTier`, `HardTierScaleLadder`, `ApplyHardTierScaleJitter()` |
| Hard tier (response to MIND non-abelian witness) | `PlanHardLoopTrajectory` / `PlanHardEquivalenceTrajectory` / `PlanHardInverseTrajectory` |
| Equivalence (A,B) pairing | `bHasPairedTrajectory`, `PairedTrajectoryActions`, `partner_trajectory_id` field |

---

## 6. Known limitations / future work

- **Static-scene assumption.** The pipeline only renders single-agent static scenes; dynamic subjects (pedestrians, vehicles, destructibles) would break the determinism of $g$. Those axes are covered by complementary benchmarks such as HM-World / WildWorld.
- **9-action discrete grid.** Same family as Habitat / ProcTHOR but obviously not continuous control; the paper's self-consistency evaluation tier exists precisely to absorb cross-framework per-action-scale differences.
- **Loop pass-rate is markedly lower than Inverse / Equivalence.** Five-stage homing + polygon geometry are extremely sensitive to collision, so even with 20 plan retries the Layer-2 pixel-validator pass rate is only ~5% in many scenes. To top up Loop data, increase `WeightHardLoop` and prefer large rooms / outdoor scenes.
- **Optical flow is disabled by default.** It requires the user to author `M_VelocityPass` in the editor, and the released dataset does not advertise an optical-flow modality — so it is soft-disabled by default and turned on only when needed.
