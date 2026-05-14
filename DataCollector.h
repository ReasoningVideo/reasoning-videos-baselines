#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/CapsuleComponent.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/DirectionalLight.h"
#include "Materials/MaterialInterface.h"
#include "Async/Async.h"
#include "DataCollector.generated.h"

// ============================================================================
// Action Definition
// Aligned with the discrete action space used in world-model / embodied-AI work.
// Each action is represented as a (delta_position, delta_rotation) tuple.
// References: UniPi (Du et al. 2023), GAIA-1 (Hu et al. 2023), RT-2 (Brohan et al. 2023)
// ============================================================================
USTRUCT(BlueprintType)
struct FActionDefinition
{
    GENERATED_BODY()

    // Unique identifier name for the action (e.g., "move_forward", "turn_left")
    UPROPERTY(EditAnywhere, Category = "Action")
    FString ActionName;

    // Translational delta of the action (relative offset in the local frame)
    // Corresponds to delta_position = (dx, dy, dz) in local frame
    UPROPERTY(EditAnywhere, Category = "Action")
    FVector DeltaPosition = FVector::ZeroVector;

    // Rotational delta of the action (Pitch, Yaw, Roll)
    // Corresponds to delta_rotation = (dpitch, dyaw, droll)
    UPROPERTY(EditAnywhere, Category = "Action")
    FRotator DeltaRotation = FRotator::ZeroRotator;

    FActionDefinition() {}

    FActionDefinition(const FString& InName, const FVector& InDeltaPos, const FRotator& InDeltaRot)
        : ActionName(InName), DeltaPosition(InDeltaPos), DeltaRotation(InDeltaRot)
    {}
};

// ============================================================================
// Spawn Point
// One sampling pose = location + orientation
// ============================================================================
USTRUCT()
struct FSamplePoint
{
    GENERATED_BODY()

    FVector Location;
    FRotator Rotation;

    FSamplePoint() : Location(FVector::ZeroVector), Rotation(FRotator::ZeroRotator) {}
    FSamplePoint(const FVector& InLoc, const FRotator& InRot) : Location(InLoc), Rotation(InRot) {}
};

// ============================================================================
// Collector Mode
// ============================================================================
UENUM(BlueprintType)
enum class ECollectorMode : uint8
{
    // Counterfactual Tree mode (original behavior)
    // For each sample point: render one context clip + one branch clip per action
    CounterfactualTree  UMETA(DisplayName = "Counterfactual Tree (Original)"),

    // Trajectory mode (new)
    // For each sample point: randomly sample an N-step action sequence forming
    // a complete path. Execute one action per step and render the matching frames.
    // Paths that collide      -> random_walk/ (random-walk data)
    // Paths without collision -> reasoning/   (reasoning-structured data)
    TrajectoryGraph     UMETA(DisplayName = "Trajectory Graph (New)")
};

// ============================================================================
// Scale Ladder -- the "step-size tiers" for Hard tier trajectories
//
// Paper §4.3 "Scale ladder": every Hard tier trajectory is **randomly assigned**
// a scale tier (weighted sampling by FScaleLadderEntry::Weight). The four
// tiers cover the dominant training step sizes used by world models in the
// literature:
//
//   Fine      (50cm/15°)   -> Habitat / AI2-THOR / ProcTHOR 25-50cm tier
//   Default   (100cm/15°)  -> our baseline (compatible with the 300h Easy data already rendered)
//   Coarse    (200cm/15°)  -> Matrix-Game / Minecraft-scale (~1 block)
//   WideRot   (100cm/30°)  -> AI2-THOR (90°/step) / ProcTHOR (30°/step)
//
// Evaluation side (paper §6.1):
//   Track A -- Pose-conditioned models : step size is irrelevant, pose is fed directly
//   Track B -- Discrete-action models  : pick the ladder tier closest to training
//   Cross-tier primary metric = Gap_r (scale-robust, see §6.3)
// ============================================================================
UENUM(BlueprintType)
enum class EScaleTier : uint8
{
    BaseDefault UMETA(DisplayName = "BaseDefault (1.0x trans, 1.0x rot)"),
    Fine        UMETA(DisplayName = "Fine (0.5x trans, 1.0x rot)"),
    Coarse      UMETA(DisplayName = "Coarse (2.0x trans, 1.0x rot)"),
    WideRot     UMETA(DisplayName = "WideRot (1.0x trans, 2.0x rot)")
};

USTRUCT(BlueprintType)
struct FScaleLadderEntry
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, Category = "ScaleLadder")
    EScaleTier Tier = EScaleTier::BaseDefault;

    // Multiplier relative to TranslationMagnitude / RotationMagnitude
    UPROPERTY(EditAnywhere, Category = "ScaleLadder", meta = (ClampMin = "0.1", ClampMax = "10.0"))
    float TranslationScale = 1.0f;

    UPROPERTY(EditAnywhere, Category = "ScaleLadder", meta = (ClampMin = "0.1", ClampMax = "10.0"))
    float RotationScale = 1.0f;

    // Sampling weight (unnormalized); 0 disables this tier
    UPROPERTY(EditAnywhere, Category = "ScaleLadder", meta = (ClampMin = "0.0", ClampMax = "100.0"))
    float Weight = 0.0f;

    FScaleLadderEntry() {}
    FScaleLadderEntry(EScaleTier InTier, float InTrans, float InRot, float InWeight)
        : Tier(InTier), TranslationScale(InTrans), RotationScale(InRot), Weight(InWeight) {}
};

// ============================================================================
// Trajectory Type -- corresponds to the five algebraic relations from the paper
//
// Core paper claim: standard next-step prediction does not guarantee global
// consistency; path-level relational constraints (Inverse / Loop / Equivalence)
// are required to rule out solutions that are locally correct yet globally
// inconsistent.
//
// vvv Easy tier (legacy version, kept untouched, >200k frames already rendered)
// Seed:        purely random exploration, used as a flat-trajectory control
// Inverse:     out-and-back (A o A^-1 = id), validates inverse consistency
// Loop:        "explore + go home" closure (geometrically near-exact)
// Equivalence: intra-segment shuffle / block-swap (commutative segment permutation)
// ^^^
//
// vvv Hard tier (added in the 2026-04 submission supplement)
// Hard_Loop:         geometric templates (rectangle / triangle / octagon); a real
//                    topological closure that cannot be decomposed into Inverse
// Hard_Equivalence:  macroscopic reorderings (L-shape vs zig-zag) with significant A/B visual divergence
// Hard_Inverse:      A . A^-1 with mixed rotations (non-abelian witness subset --
//                    defends against the "symmetric round-trips" MIND-style shortcut baseline)
// ^^^
// ============================================================================
UENUM(BlueprintType)
enum class ETrajectoryType : uint8
{
    // Easy tier
    Seed                UMETA(DisplayName = "Seed (Random Walk)"),
    Inverse             UMETA(DisplayName = "Inverse (A ∘ A⁻¹ = id)"),
    Loop                UMETA(DisplayName = "Loop (Closure, explore+return)"),
    Equivalence         UMETA(DisplayName = "Equivalence (Segment Shuffle)"),

    // Hard tier
    Hard_Loop           UMETA(DisplayName = "Hard Loop (Geometric Template)"),
    Hard_Equivalence    UMETA(DisplayName = "Hard Equivalence (Macro Reorder)"),
    Hard_Inverse        UMETA(DisplayName = "Hard Inverse (Non-Abelian A·A⁻¹)")
};

// ============================================================================
// Per-step record within a trajectory (one step = the full execution of one action)
// ============================================================================
USTRUCT()
struct FTrajectoryStepRecord
{
    GENERATED_BODY()

    // Index into ActionSpace of the action executed at this step
    int32 ActionIndex = -1;

    // Pose before execution
    FVector StartPosition = FVector::ZeroVector;
    FRotator StartRotation = FRotator::ZeroRotator;

    // Expected end pose (theoretical end pose if no collision)
    FVector ExpectedEndPosition = FVector::ZeroVector;
    FRotator ExpectedEndRotation = FRotator::ZeroRotator;

    // Actual end pose (real end pose after any collisions)
    FVector ActualEndPosition = FVector::ZeroVector;
    FRotator ActualEndRotation = FRotator::ZeroRotator;

    // Whether a collision occurred (|Expected - Actual| > threshold)
    bool bCollisionOccurred = false;

    // Collision displacement magnitude (cm)
    float CollisionDisplacement = 0.0f;

    // Collision detail (only valid when a collision occurred)
    FString CollisionActorName;      // Actor name involved in the collision (e.g., "SM_Wall_01")
    FString CollisionComponentName;  // Component name involved in the collision (e.g., "StaticMeshComponent0")
    FString CollisionPhysMaterial;   // Physical material name (e.g., "PM_Concrete")
    FVector CollisionImpactPoint = FVector::ZeroVector;  // Collision point in world coordinates
    FVector CollisionNormal = FVector::ZeroVector;       // Collision normal

    FTrajectoryStepRecord() {}
};

// ============================================================================
// Output Image Format
// ============================================================================
UENUM(BlueprintType)
enum class EImageOutputFormat : uint8
{
    // JPEG: lossy compression, very small files (about 30-50 KB/frame), speed comparable to BMP.
    // Recommended for large-scale capture; the slight quality loss has almost no effect on training.
    JPEG    UMETA(DisplayName = "JPEG (small, recommended)"),

    // BMP: uncompressed, large files (about 768 KB/frame), fastest to write.
    // Suitable when lossless raw data is required.
    BMP     UMETA(DisplayName = "BMP (uncompressed, lossless)")
};

// ============================================================================
// Collector FSM Phase
// ============================================================================
UENUM()
enum class ECollectorPhase : uint8
{
    // ---- Common phases ----
    // Wait for the scene to finish loading
    WaitForSceneReady,
    // Teleport to the next sample point and wait for the camera to settle
    TeleportToSamplePoint,

    // ---- Counterfactual Tree mode only ----
    // Render the context video for the current state (context / previous state)
    RenderContextVideo,
    // Execute one action and render the corresponding future video (action -> new state)
    RenderActionVideo,
    // Reset to the original state between two actions
    ResetForNextAction,

    // ---- Trajectory Graph mode only ----
    // Begin a new trajectory: sample its action sequence
    Traj_PlanTrajectory,
    // Execute the current step's action and render its frames
    Traj_ExecuteStep,
    // Current step finished; advance to the next step
    Traj_AdvanceStep,
    // Current trajectory finished; save it and decide its split
    Traj_FinalizeTrajectory,

    // ---- Common ----
    // All actions for the current sample point are done; move on to the next
    AdvanceToNextSamplePoint,
    // All capture finished
    Completed
};

// ============================================================================
// Data Collector Actor (main class)
//
// Supports two capture modes:
//
// [Mode 1] CounterfactualTree
//   Automated pipeline:
//     1. Generate a grid of sample points from the user-configured XY range and step
//     2. Each sample point x N orientations = the full set of capture poses
//     3. For each pose: render a context clip + one branch clip per action
//     4. The collision capsule guarantees actions cannot pass through walls
//   Output layout:
//     {SaveRoot}/run_YYYYMMDD_HHMMSS/
//       episode_{EpisodeID}/
//         state_{StateID}/
//           metadata.json
//           S{StateID}_context/
//             frame_0000.jpg ...
//           S{StateID}_A{ActionID}_{Name}/
//             frame_0000.jpg ...
//
// [Mode 2] TrajectoryGraph -- new
//   Automated pipeline:
//     1. Generate the grid of sample points (same as Mode 1)
//     2. Capture N trajectories per sample point (each: 40 steps x 8 frames/step = 320 frames)
//     3. Randomly sample the action sequence (with collision avoidance + consecutive-rotation cap)
//     4. Naturally split by collision: collision-free -> reasoning/, collided -> random_walk/
//   Output layout:
//     {SaveRoot}/run_YYYYMMDD_HHMMSS/
//       manifest.json
//       reasoning/                    (collision-free; algebraic properties hold exactly)
//         traj_000000/
//           metadata.json
//           step_00/frame_0000.jpg ... frame_0007.jpg
//           step_01/ ...
//           step_39/ ...
//       random_walk/                  (collided; random-walk data)
//         traj_000001/
//           metadata.json
//           step_00/ ...
// ============================================================================
UCLASS()
class DATABUILDER_CPP_API ADataCollector : public AActor
{
    GENERATED_BODY()

public:
    ADataCollector();

protected:
    virtual void BeginPlay() override;

public:
    virtual void Tick(float DeltaTime) override;

    // ==================== Components ====================

    // Collision capsule (root component, provides physical collision)
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Collision")
    UCapsuleComponent* CollisionCapsule;

    // Scene capture camera (RGB)
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Capture")
    USceneCaptureComponent2D* SceneCapture;

    // Depth capture camera (Scene Depth -> R16F)
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Capture")
    USceneCaptureComponent2D* DepthCapture;

    // Optical-flow capture camera (Velocity Buffer -> screen-space motion vectors)
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Capture")
    USceneCaptureComponent2D* VelocityCapture;

    // ==================== Multimodal Capture Configuration ====================

    // Whether to capture depth (Scene Depth)
    // Output is a 16-bit grayscale PNG, units of cm: 0 = nearest, 65535 = farthest.
    // Provides geometric-structure information for world-model training.
    UPROPERTY(EditAnywhere, Category = "Modality")
    bool bCaptureDepth = true;

    // Depth clipping range (cm)
    // Pixels beyond this distance are clamped to the maximum value (65535).
    // Default 10000 cm = 100 m, suitable for most indoor + close-range outdoor scenes.
    UPROPERTY(EditAnywhere, Category = "Modality", meta = (EditCondition = "bCaptureDepth", ClampMin = "100.0", ClampMax = "100000.0"))
    float DepthMaxDistance = 10000.0f;

    // Whether to capture optical flow (Velocity Buffer / screen-space motion vectors)
    // Output is a PNG with horizontal/vertical pixel displacement encoded into the RG channels.
    // Provides dynamic-motion information for world-model training.
    //
    // *** Requires the post-process material M_VelocityPass to be created in the UE editor ***
    //   How: see the source comments / README.
    //   If no material is supplied, optical-flow capture is automatically disabled and a warning is printed.
    UPROPERTY(EditAnywhere, Category = "Modality")
    bool bCaptureOpticalFlow = true;

    // Optical-flow post-process material (must be created in the editor and assigned)
    // The material reads UE5's Velocity Buffer via the SceneTexture:Velocity node and
    // encodes the motion vectors into RGB channels for output.
    // If nullptr, optical-flow capture is automatically disabled.
    UPROPERTY(EditAnywhere, Category = "Modality", meta = (EditCondition = "bCaptureOpticalFlow"))
    UMaterialInterface* VelocityPassMaterial = nullptr;

    // ==================== Collision Configuration ====================

    // Capsule radius (cm), modeling the agent's body width
    UPROPERTY(EditAnywhere, Category = "Collision")
    float CapsuleRadius = 34.0f;

    // Capsule half-height (cm), modeling half of the agent's height
    UPROPERTY(EditAnywhere, Category = "Collision")
    float CapsuleHalfHeight = 88.0f;

    // Camera offset relative to the top of the capsule (modeling eye height)
    // Default (0, 0, 60) places the camera 60 cm above the capsule center.
    UPROPERTY(EditAnywhere, Category = "Collision")
    FVector CameraOffset = FVector(0, 0, 60.0f);

    // Whether to force the actor's scale to (1, 1, 1) on BeginPlay.
    // If the blueprint / level accidentally scales the DataCollector, the parent component's
    // scale amplifies CameraOffset and produces "the camera looks high in the sky even though
    // SamplingZ is small".
    UPROPERTY(EditAnywhere, Category = "Collision")
    bool bForceUnitScale = true;

    // ==================== Grid Sampling Configuration ====================

    // Sampling-volume center (world coordinates)
    // If left at (0, 0, 0), the actor's placement location is used as the center.
    UPROPERTY(EditAnywhere, Category = "Sampling")
    bool bUsePlacementAsCenter = true;

    // Sampling-region center (only used when bUsePlacementAsCenter = false)
    UPROPERTY(EditAnywhere, Category = "Sampling", meta = (EditCondition = "!bUsePlacementAsCenter"))
    FVector SamplingCenter = FVector::ZeroVector;

    // Half-range of sampling along X (cm)
    // Sampling range = [Center.X - RangeX, Center.X + RangeX]
    UPROPERTY(EditAnywhere, Category = "Sampling")
    float SamplingRangeX = 2000.0f;

    // Half-range of sampling along Y (cm)
    UPROPERTY(EditAnywhere, Category = "Sampling")
    float SamplingRangeY = 2000.0f;

    // Sampling grid spacing (cm); the same value is used for both X and Y
    UPROPERTY(EditAnywhere, Category = "Sampling")
    float SamplingInterval = 200.0f;

    // Sampling height (cm)
    // Default semantics: this is the final Z height of the DataCollector actor / capsule center.
    // For example, setting this to 250 puts the sample-point actor center at Z = 250.
    UPROPERTY(EditAnywhere, Category = "Sampling")
    float SamplingZ = 200.0f;

    // Kept for backward compatibility with older blueprints; the current sampling logic no
    // longer "drops to ground". The capsule is now placed directly at the fixed SamplingZ
    // height; whether the point is usable is decided purely by collision detection.
    UPROPERTY(EditAnywhere, Category = "Sampling", AdvancedDisplay)
    bool bProjectSamplesToGround = false;

    // Kept for backward compatibility with older blueprints; not used in the current version.
    UPROPERTY(EditAnywhere, Category = "Sampling", AdvancedDisplay, meta = (ClampMin = "100.0"))
    float GroundSearchDistance = 5000.0f;


    // Number of orientation samples (uniformly across 360°)
    // For example, 8 = one orientation every 45°, 12 = one every 30°.
    UPROPERTY(EditAnywhere, Category = "Sampling", meta = (ClampMin = "1", ClampMax = "72"))
    int32 NumYawSamples = 8;

    // Whether to skip sample points that intersect the scene (points inside obstacles)
    UPROPERTY(EditAnywhere, Category = "Sampling")
    bool bSkipOccludedPoints = true;

    // Verbose debug mode: when enabled, every sample point (whether accepted or rejected)
    // logs full collision details.
    // Note: with a large number of sample points this produces a lot of log spam; only
    // enable it during debugging.
    UPROPERTY(EditAnywhere, Category = "Sampling")
    bool bVerboseDebug = false;

    // ==================== Render Configuration ====================

    UPROPERTY(EditAnywhere, Category = "Render")
    int32 ResolutionX = 1280;

    UPROPERTY(EditAnywhere, Category = "Render")
    int32 ResolutionY = 720;

    UPROPERTY(EditAnywhere, Category = "Render")
    float VideoDurationSeconds = 5.0f;

    UPROPERTY(EditAnywhere, Category = "Render")
    int32 FramesPerSecond = 16;

    // Camera horizontal field of view (degrees)
    // 79° = Habitat (Meta AI) standard; the most common in embodied-AI papers
    // 90° = AI2-THOR / Gibson standard (heavier distortion)
    // 60° = roughly the human foveal attention region
    UPROPERTY(EditAnywhere, Category = "Render", meta = (ClampMin = "30.0", ClampMax = "120.0"))
    float CameraFOV = 79.0f;

    // Output image format
    // JPEG: lossy compression, ~30-50 KB/frame at 512x512, 15-25x space savings, speed comparable to BMP
    // BMP:  uncompressed, ~768 KB/frame at 512x512, fastest but uses far more disk
    UPROPERTY(EditAnywhere, Category = "Render")
    EImageOutputFormat ImageFormat = EImageOutputFormat::JPEG;

    // JPEG compression quality (1-100)
    // 95  = nearly lossless, no visible difference (recommended)
    // 85  = slight compression artifacts but smaller files
    // 100 = highest quality, slightly larger files
    UPROPERTY(EditAnywhere, Category = "Render", meta = (EditCondition = "ImageFormat == EImageOutputFormat::JPEG", ClampMin = "1", ClampMax = "100"))
    int32 JpegQuality = 95;

    // ==================== Performance Configuration ====================

    // Number of frames batched per Tick
    // Default 1 = process one frame per Tick (original behavior)
    // 4-16 skips much of the per-Tick engine overhead and improves throughput significantly
    // Note: very large values cause each Tick to block too long and the editor stutters.
    UPROPERTY(EditAnywhere, Category = "Performance", meta = (ClampMin = "1", ClampMax = "64"))
    int32 FramesPerTick = 8;

    // Async file write: when enabled, image saving runs on a background thread and does not block the main thread.
    UPROPERTY(EditAnywhere, Category = "Performance")
    bool bAsyncFileWrite = true;

    // ==================== Render Quality Configuration ====================
    // Disabling unneeded post-process effects can speed up rendering.
    // Each effect has its own toggle, so the configuration can be tuned per training/demo need.

    // Screen-Space Reflection (SSR) -- real-time reflections on smooth surfaces.
    // Disabling saves GPU time, but reflections on metal/water/floor disappear.
    UPROPERTY(EditAnywhere, Category = "Render Quality")
    bool bEnableSSR = true;

    // Ambient Occlusion (AO) -- soft contact shadows where objects meet.
    // When disabled, shading depth in corners and crevices is reduced.
    UPROPERTY(EditAnywhere, Category = "Render Quality")
    bool bEnableAO = true;

    // Motion blur -- smear effect during fast motion.
    // Usually disabled for offline capture (each frame is captured independently and motion blur introduces artifacts).
    UPROPERTY(EditAnywhere, Category = "Render Quality")
    bool bEnableMotionBlur = false;

    // Bloom -- light bleed around bright regions.
    // Disabling produces a "cleaner / sharper" image at the cost of mood.
    UPROPERTY(EditAnywhere, Category = "Render Quality")
    bool bEnableBloom = true;

    // Lens flare -- streaks/spots produced by strong light sources.
    UPROPERTY(EditAnywhere, Category = "Render Quality")
    bool bEnableLensFlares = false;

    // Depth of Field -- foreground/background blur effect.
    // Training data usually wants the entire frame sharp; disable.
    UPROPERTY(EditAnywhere, Category = "Render Quality")
    bool bEnableDepthOfField = false;

    // Volumetric Fog -- light scattering through fog/smoke.
    // GPU-heavy; safe to disable for indoor scenes.
    UPROPERTY(EditAnywhere, Category = "Render Quality")
    bool bEnableVolumetricFog = true;

    // Ray tracing -- RTX-accelerated reflections / shadows / GI.
    // One of the most expensive options; disabled = traditional rasterization.
    UPROPERTY(EditAnywhere, Category = "Render Quality")
    bool bEnableRayTracing = false;

    // ==================== Exposure Configuration ====================
    // SceneCaptureComponent2D does not inherit the editor viewport's auto-exposure settings.
    // Without manual configuration, indoor / backlit areas are extremely underexposed (pure black).
    //
    // Recommended strategies:
    //   - Fixed exposure (bUseFixedExposure = true): consistent inter-frame brightness, more stable training data
    //   - Auto-exposure (bUseFixedExposure = false): mimics human-eye adaptation; both bright and dark scenes remain visible
    //     but inter-frame brightness fluctuates

    // Whether to use fixed exposure (recommended for data capture)
    // true  = use ExposureBias to fix brightness
    // false = enable auto-exposure (Eye Adaptation)
    UPROPERTY(EditAnywhere, Category = "Exposure")
    bool bUseFixedExposure = false;

    // Fixed exposure offset (EV)
    // Effective only when bUseFixedExposure = true
    // 0.0 = default brightness
    // positive = brighter (each +1.0 doubles brightness)
    // negative = darker  (each -1.0 halves brightness)
    // Suggested range: -2.0 to +4.0; indoor scenes usually need +1.0 to +3.0
    UPROPERTY(EditAnywhere, Category = "Exposure", meta = (EditCondition = "bUseFixedExposure", ClampMin = "-10.0", ClampMax = "20.0"))
    float FixedExposureBias = 1.0f;

    // Auto-exposure minimum brightness (only when bUseFixedExposure = false)
    // Raising this prevents excessive underexposure in dark areas.
    // Default 0.03; suggested 0.5-2.0 to avoid overly dark indoor scenes.
    UPROPERTY(EditAnywhere, Category = "Exposure", meta = (EditCondition = "!bUseFixedExposure", ClampMin = "0.001", ClampMax = "10.0"))
    float AutoExposureMinBrightness = 0.5f;

    // Auto-exposure maximum brightness (only when bUseFixedExposure = false)
    // Lowering this prevents excessive overexposure in bright areas.
    UPROPERTY(EditAnywhere, Category = "Exposure", meta = (EditCondition = "!bUseFixedExposure", ClampMin = "1.0", ClampMax = "20.0"))
    float AutoExposureMaxBrightness = 8.0f;

    // Auto-exposure offset (EV; only when bUseFixedExposure = false)
    // Additional brightness offset on top of auto-exposure.
    UPROPERTY(EditAnywhere, Category = "Exposure", meta = (EditCondition = "!bUseFixedExposure", ClampMin = "-5.0", ClampMax = "10.0"))
    float AutoExposureBias = 1.0f;

    // Auto-exposure adaptation speed: dark -> bright
    UPROPERTY(EditAnywhere, Category = "Exposure", meta = (EditCondition = "!bUseFixedExposure", ClampMin = "0.1", ClampMax = "20.0"))
    float AutoExposureSpeedUp = 3.0f;

    // Auto-exposure adaptation speed: bright -> dark
    UPROPERTY(EditAnywhere, Category = "Exposure", meta = (EditCondition = "!bUseFixedExposure", ClampMin = "0.1", ClampMax = "20.0"))
    float AutoExposureSpeedDown = 1.0f;

    // ==================== Lighting Configuration ====================
    // Counterfactual reasoning requires that visual change come solely from action; lighting must be fixed.

    // Whether to freeze the scene's directional light at runtime.
    // When enabled, automatically locates the directional light in the scene and freezes its
    // rotation, preventing day/night cycling.
    UPROPERTY(EditAnywhere, Category = "Lighting")
    bool bFreezeLighting = true;

    // Fixed sun pitch
    // -45 ~ 2-3 PM sun angle (moderate shadows, bright image)
    // -90 = noon (almost no shadows)
    // -20 = late afternoon (long shadows)
    UPROPERTY(EditAnywhere, Category = "Lighting", meta = (EditCondition = "bFreezeLighting", ClampMin = "-90.0", ClampMax = "0.0"))
    float SunPitch = -45.0f;

    // Fixed sun yaw
    // Controls which horizontal direction the sun comes from.
    UPROPERTY(EditAnywhere, Category = "Lighting", meta = (EditCondition = "bFreezeLighting", ClampMin = "-180.0", ClampMax = "180.0"))
    float SunYaw = 0.0f;

    // ==================== Action Space Configuration ====================

    // Translation step size (cm/step)
    // Academic baselines:
    //   Habitat (Meta AI):    25 cm/step (discrete teleport)
    //   AI2-THOR (Allen AI):  25 cm/step (discrete teleport)
    //   Gibson (Stanford):    25 cm/step
    //   ProcTHOR:             25 cm/step
    // This project: each step renders 8 progressive interpolated frames (not teleport), completed in 0.5 s.
    //   100 cm/step -> equivalent speed 2 m/s ~ normal walking speed (academically reasonable)
    //   50  cm/step -> equivalent speed 1 m/s ~ slow walk (recommended for small indoor scenes)
    //   200 cm/step -> equivalent speed 4 m/s ~ jogging (fast; usable for large scenes)
    UPROPERTY(EditAnywhere, Category = "ActionSpace", meta = (ClampMin = "10.0", ClampMax = "500.0"))
    float TranslationMagnitude = 100.0f;

    // Rotation step size (deg/step)
    // Academic baselines:
    //   Habitat:     10°/step (discrete teleport)
    //   AI2-THOR:    90°/step (discrete teleport, very coarse)
    //   Gibson:      free rotation (continuous control)
    //   ProcTHOR:    30°/step
    // This project: each step renders 8 frames with linear interpolation (smooth rotation in 0.5 s).
    //   15°/step -> 30°/s ~ normal head-turn speed (smooth and natural; recommended)
    //   10°/step -> 20°/s ~ slow head turn (stable but slower)
    //   30°/step -> 60°/s ~ fast head turn (feels jittery)
    // Note: look_up / look_down actually use half of this value (pitch changes are usually smaller).
    UPROPERTY(EditAnywhere, Category = "ActionSpace", meta = (ClampMin = "5.0", ClampMax = "90.0"))
    float RotationMagnitude = 15.0f;

    UPROPERTY(EditAnywhere, Category = "ActionSpace")
    TArray<FActionDefinition> ActionSpace;

    // ==================== Dataset Configuration ====================

    // Capture mode: CounterfactualTree (original) or TrajectoryGraph (new)
    UPROPERTY(EditAnywhere, Category = "Dataset")
    ECollectorMode CollectorMode = ECollectorMode::TrajectoryGraph;

    UPROPERTY(EditAnywhere, Category = "Dataset")
    FString SaveRootDirectory = TEXT("F:/A_worldmodel/UE5data/");

    UPROPERTY(EditAnywhere, Category = "Dataset")
    int32 EpisodeID = 0;

    // ==================== Trajectory Mode Configuration ====================

    // Total number of steps per trajectory (one step = one action)
    // Default 40 steps x 0.5 s/step = 20 seconds.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph", ClampMin = "1", ClampMax = "200"))
    int32 TrajectorySteps = 40;

    // Frames per action (= action duration x FPS)
    // Default 9 frames, aligned with the Chunk-AR chunk size.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph", ClampMin = "1", ClampMax = "64"))
    int32 FramesPerAction = 9;

    // Collision detection threshold (cm): if |actual - expected position| exceeds this, declare a collision.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph", ClampMin = "0.1", ClampMax = "50.0"))
    float CollisionThreshold = 1.0f;

    // ==================== Collision Whitelist ====================
    // Some actors have huge collision volumes but never produce visible clipping (e.g. landscape ground).
    // Adding them to the whitelist lets collision detection ignore them so the camera can move freely
    // without being "blocked" by floors / large environment props.
    //
    // Match rule: the actor's name OR class name **contains** any of the whitelist keywords.
    // For example "Landscape" matches "Landscape_0", "LandscapeStreamingProxy_1", etc.
    // Case-insensitive.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    TArray<FString> CollisionWhitelistKeywords = { TEXT("Landscape"), TEXT("Floor"), TEXT("Ground") };

    // Whether to use collision-avoidance sampling (line-trace lookahead + resample)
    // When enabled, a ray cast is performed before sampling each action to try to avoid colliding actions.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    bool bCollisionAvoidanceSampling = true;

    // Maximum number of resamples during collision avoidance
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "bCollisionAvoidanceSampling", ClampMin = "1", ClampMax = "20"))
    int32 MaxResampleAttempts = 5;

    // Number of trajectories to capture per sample point
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph", ClampMin = "1", ClampMax = "100"))
    int32 TrajectoriesPerSamplePoint = 5;

    // Action sampling weights (one weight per action in ActionSpace, controlling sampling probability)
    // If empty, default weights are auto-initialized in BeginPlay.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    TArray<float> ActionSamplingWeights;

    // Cap on consecutive rotation steps; once exceeded, force the next sampled action to be a translation.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph", ClampMin = "1", ClampMax = "10"))
    int32 MaxConsecutiveRotations = 3;

    // ==================== Plan Retry Configuration ====================
    // Key insight: UE capsule-collision simulation is nearly instant (<1 ms), while rendering one
    // trajectory takes ~1-2 minutes. So we retry many times in the pure-simulation phase and only
    // enter the expensive render flow after planning succeeds.

    // Maximum number of plan retries on failure (each retry resamples both type and parameters; pure simulation, no rendering)
    // With 20 retries: Hard_Equivalence success rate goes from ~24% to ~99%, Hard_Loop ~94% -> ~99.99%.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph", ClampMin = "1", ClampMax = "100"))
    int32 MaxPlanRetries = 20;

    // If all plan attempts fail, skip (do not render, do not fall back to Seed) and try the next trajectory.
    // When enabled, no Seed fallback trajectories are produced, which is appropriate when only specific types are needed.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    bool bSkipFailedPlans = true;

    // Rotate the starting yaw every N retries (changing direction can avoid one-sided wall collisions).
    // For example, 5 = rotate 45° every 5 retries; 20 retries cover 4 orientations x 5 parameter sets.
    UPROPERTY(EditAnywhere, Category = "Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph", ClampMin = "0", ClampMax = "50"))
    int32 RetryYawRotateInterval = 5;

    // ==================== Structured Trajectory Configuration ====================
    // Corresponds to the three algebraic relations from the paper:
    //   Easy tier (legacy): Inverse / Loop / Equivalence
    //   Hard tier (new):    Hard_Loop / Hard_Equivalence / Hard_Inverse
    //
    // Use cases:
    //   (A) Build the dataset from scratch: both Easy and Hard weights are non-zero
    //   (B) Augment existing Easy data with Hard samples (2026-04 submission): set Easy weights
    //       to 0, set WeightSeed to a small fallback value, distribute Hard weights as needed.
    //
    // Defaults are tuned for the "augment with Hard" scenario.

    // -- Easy tier weights ----------------------------------------------
    // 2026-04-26 inventory: 30,197 Easy tier trajectories on disk; Layer-2 passed 6,273
    //   inverse: 2585, equivalence: 2900, loop: 182
    // Continuing to render Easy yields no benefit -- set all Easy weights to 0; Seed is also 0 (no flat-control top-up needed).
    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    float WeightSeed = 0.0f;

    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    float WeightInverse = 0.0f;

    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    float WeightLoop = 0.0f;

    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    float WeightEquivalence = 0.0f;

    // -- Easy Equivalence quality control --------------------------------
    // Reject the trajectory if, after shuffle/block-swap, the number of differing steps < MinEquivDivergence x TrajectorySteps.
    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float MinEquivDivergence = 0.25f;

    // The maximum Euclidean distance between intermediate trajectory points of P and P' must be >= this value (cm).
    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (ClampMin = "0.0"))
    float MinEquivSpatialDivergence = 150.0f;

    // -- Hard tier weights (P0 Hard data top-up) -------------------------
    // Goal: ~600 passed pairs per Hard relation per scene.
    // Loop weighting: Easy stock is only 182 (vs Inverse 2585, Equivalence 2900),
    //   and Hard_Loop's geometric closure construction has a slightly lower success
    //   rate, so it gets more sampling chances.
    //
    //   Hard_Loop=45  Hard_Equivalence=30  Hard_Inverse=25
    //   Seed=0        (fallback off, to avoid polluting the target distribution)
    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    float WeightHardLoop = 45.0f;

    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    float WeightHardEquivalence = 30.0f;

    // Hard_Inverse: A . A^-1 with mixed rotations (non-abelian witness subset, directly
    // addressing the MIND / symmetric round-trip baseline shortcut).
    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (EditCondition = "CollectorMode == ECollectorMode::TrajectoryGraph"))
    float WeightHardInverse = 25.0f;

    // ==================== Hard tier scale ladder (Option B) =================
    // Enables the "4-tier named scale ladder" for Hard tier trajectories, directly addressing
    // MIND's action-space generalisation pitch. Every Hard tier trajectory gets a randomly
    // assigned EScaleTier (weighted by Weight); metadata writes the tier into the scale_tier.name
    // field so evaluation scripts can group by tier.
    //
    // Effective only on Hard tier (Hard_Loop / Hard_Equivalence / Hard_Inverse); Easy tier
    // and Seed always use 1.0x (BaseDefault).
    //
    // Default weights: BaseDefault dominates (aligned with the 300h Easy already rendered),
    // the other three tiers act as supplements.
    //   BaseDefault 50  Fine 20  Coarse 15  WideRot 15
    // P0 stage (2026-04-26): lock the default ladder first to fully top up the 3 Hard relations.
    // Once Hard x default reaches ~600 passed pairs per cell, enable ladder jitter for
    // P1 stage (fine / coarse / wide_rot).
    // To re-enable the ladder: switch back to true and adjust HardTierScaleLadder Weight per P1 batch.
    UPROPERTY(EditAnywhere, Category = "Structured Trajectory")
    bool bEnableHardTierScaleJitter = false;

    UPROPERTY(EditAnywhere, Category = "Structured Trajectory", meta = (EditCondition = "bEnableHardTierScaleJitter"))
    TArray<FScaleLadderEntry> HardTierScaleLadder = {
        FScaleLadderEntry(EScaleTier::BaseDefault, 1.0f, 1.0f, 50.0f),
        FScaleLadderEntry(EScaleTier::Fine,        0.5f, 1.0f, 20.0f),
        FScaleLadderEntry(EScaleTier::Coarse,      2.0f, 1.0f, 15.0f),
        FScaleLadderEntry(EScaleTier::WideRot,     1.0f, 2.0f, 15.0f)
    };

private:
    // ==================== Internal State ====================

    // Effective save root used at runtime (SaveRootDirectory + timestamp subfolder)
    // Auto-generated in BeginPlay; format: {SaveRootDirectory}/run_YYYYMMDD_HHMMSS/
    FString RuntimeSaveRoot;

    // Collision log file path (initialized in BeginPlay; appended dynamically per collision)
    FString CollisionLogPath;
    int32 TotalCollisionCount = 0;

    ECollectorPhase CurrentPhase;

    // Pre-generated queue of all sample points
    TArray<FSamplePoint> SamplePoints;
    int32 CurrentSampleIndex;

    // StateID assigned to the current sample point (auto-incrementing)
    int32 CurrentStateID;

    // Wait counters (scene loading / post-teleport settle)
    int32 WaitFrameCounter;
    static constexpr int32 WarmupFrames = 30;

    // Frames to wait after teleport -- Lumen GI needs 10-20 frames to converge
    // indirect lighting, and auto-exposure also needs a few frames of history to stabilize.
    // Setting this too low produces dark first frames (GI not converged) or inaccurate exposure.
    static constexpr int32 TeleportSettleFrames = 15;

    // Video frame counters
    int32 CurrentFrameIndex;
    int32 TotalFramesPerVideo;

    // Action index
    int32 CurrentActionIndex;

    // Original pose at the current sample point
    FVector OriginalLocation;
    FRotator OriginalRotation;

    // Action interpolation targets
    FVector ActionStartLocation;
    FRotator ActionStartRotation;
    FVector ActionTargetLocation;
    FRotator ActionTargetRotation;

    // ==================== Trajectory Mode Internal State ====================

    // Action sequence for the current trajectory (indices into ActionSpace)
    TArray<int32> CurrentTrajectoryActions;

    // Per-step records for the current trajectory
    TArray<FTrajectoryStepRecord> CurrentTrajectoryRecords;

    // Index of the step currently being executed (0-based)
    int32 CurrentTrajectoryStep;

    // Frame index within the current step
    int32 CurrentStepFrameIndex;

    // Whether the current trajectory has had any collision (used for the split)
    bool bCurrentTrajectoryHasCollision;

    // Whether any frame within the current step had a collision (used to record collision detail in AdvanceStep)
    bool bCurrentStepHadFrameCollision;
    FHitResult LastStepHitResult;  // Cache of the most recent collision HitResult within the current step

    // Trajectory counter (globally increasing, used for file naming)
    int32 TrajectoryGlobalID;

    // Number of trajectories already captured at the current sample point
    int32 CurrentTrajectoryIndexAtPoint;

    // Split statistics
    int32 TotalReasoningTrajectories;
    int32 TotalRandomWalkTrajectories;

    // ==================== Structured Trajectory Internal State ====================

    // Type of the current trajectory
    ETrajectoryType CurrentTrajectoryType;

    // Paired-trajectory support (Equivalence produces a paired path)
    // When bHasPairedTrajectory = true, PairedTrajectoryActions holds the second path.
    bool bHasPairedTrajectory;
    TArray<int32> PairedTrajectoryActions;
    TArray<FTrajectoryStepRecord> PairedTrajectoryRecords;
    bool bCurrentlyRenderingPaired;  // Whether the paired path is currently being rendered
    bool bPairedTrajectoryHasCollision;
    int32 PairedTrajectoryGlobalID;

    // Per-type trajectory counts
    TMap<ETrajectoryType, int32> TrajectoryTypeCount;

    // ==================== Option B: Hard tier scale jitter state =========
    // ApplyHardTierScaleJitter saves the original ActionSpace; FinalizeTrajectory restores it.
    // Used only when the current trajectory is Hard tier and bEnableHardTierScaleJitter = true.
    TArray<FActionDefinition> SavedActionSpace;
    bool bActionSpaceIsJittered = false;

    // Scale multipliers of the current Hard tier trajectory (written into metadata)
    float CurrentTrajectoryTransScale = 1.0f;
    float CurrentTrajectoryRotScale = 1.0f;
    EScaleTier CurrentScaleTier = EScaleTier::BaseDefault;

    // ==================== Plan Retry Internal State ====================
    // Flag indicating planning has fully failed (used by the FSM to skip rendering and not enter Traj_ExecuteStep)
    bool bPlanningFailed = false;

    // Plan statistics: attempts vs successes (per trajectory type; an efficiency report is emitted on completion)
    TMap<ETrajectoryType, int32> PlanAttemptCount;
    TMap<ETrajectoryType, int32> PlanSuccessCount;
    int32 TotalPlanSkips = 0;  // Number of trajectories skipped because all plan attempts failed

    // ==================== Internal Methods ====================

    // Generate the grid of sample points
    void GenerateSamplePoints();

    // Check whether a location is occluded by an obstacle (collision check)
    bool IsLocationOccluded(const FVector& Location) const;

    // Check whether the actor in a hit result is in the whitelist (should be ignored)
    bool IsActorWhitelisted(const AActor* HitActor) const;

    // Move with collision (sweep)
    // Move from the current location to the target location, stopping on collision.
    // Returns whether a collision occurred.
    // OutHit: if non-null and a collision occurred, filled with collision details.
    bool MoveWithCollision(const FVector& TargetLocation, FHitResult* OutHit = nullptr);

    // Initialize the default action space
    void InitializeDefaultActionSpace();

    // Save one frame (RGB + optional depth / optical flow)
    void SaveFrame(const FString& SubDirectory, int32 FrameIndex);

    // Save a depth frame as 16-bit PNG (normalized to DepthMaxDistance)
    void SaveDepthFrame(const FString& SubDirectory, int32 FrameIndex);

    // Save an optical-flow frame as 16-bit PNG (RG dual-channel encoded)
    void SaveVelocityFrame(const FString& SubDirectory, int32 FrameIndex);

    // Generic helper for asynchronous file writing
    void AsyncSaveFile(TArray<uint8>&& FileData, const FString& FullPath);

    // Path utilities
    FString GetEpisodePath() const;
    FString GetContextSubDir() const;
    FString GetActionSubDir(int32 ActionIdx) const;

    // Metadata
    void SaveMetadataJSON();
    void SaveGlobalManifest();

    // Filesystem
    void EnsureDirectoryExists(const FString& Path);

    // Freeze scene lighting (locate the directional light and freeze it)
    void FreezeLighting();

    // Apply render-quality settings (configure each effect according to its Render Quality toggle)
    void ApplyRenderQualitySettings();

    // Process a single FSM tick (called repeatedly by Tick for batching)
    // Returns true if more frames can be processed, false if the current Tick should yield.
    bool ProcessOneFrame();

    // ==================== Trajectory Mode Internal Methods ====================

    // Sample a trajectory's action sequence (with collision avoidance)
    void PlanTrajectoryActions();

    // Sample a single action (with collision avoidance + consecutive-rotation cap)
    int32 SampleSingleAction(const FVector& CurrentPos, const FRotator& CurrentRot);

    // Check whether an action is purely a rotation (no translation component)
    bool IsRotationAction(int32 ActionIdx) const;

    // Use a line trace to predict whether an action will collide
    bool WouldActionCollide(const FVector& FromPos, const FRotator& FromRot, int32 ActionIdx) const;

    // Initialize the default action sampling weights
    void InitializeDefaultActionWeights();

    // Get the trajectory save path (split into reasoning/ or random_walk/ based on collision)
    FString GetTrajectoryPath(bool bHasCollision) const;

    // Get the per-step frame save subdirectory within a trajectory
    FString GetTrajectoryStepSubDir(int32 StepIndex) const;

    // Save the trajectory metadata JSON
    void SaveTrajectoryMetadataJSON(const FString& TrajectoryDir);

    // ==================== Structured Trajectory Planning Methods ====================

    // Pick a trajectory type at random by weight
    ETrajectoryType SampleTrajectoryType() const;

    // Per-type trajectory planners (output goes into CurrentTrajectoryActions)
    // Returns true on success, false on failure (caller should fall back to Seed)
    void PlanSeedTrajectory();
    bool PlanInverseTrajectory();
    bool PlanLoopTrajectory();
    bool PlanEquivalenceTrajectory();  // Also fills PairedTrajectoryActions

    // -- Hard tier (2026-04 submission supplement) ----------------------
    bool PlanHardLoopTrajectory();        // Geometric templates (rectangle / triangle / octagon)
    bool PlanHardEquivalenceTrajectory(); // L-shape vs zig-zag macro reordering
    bool PlanHardInverseTrajectory();     // A . A^-1 with mixed rotations (non-abelian witness)

    // -- Option B: Hard tier scale jittering ----------------------------
    // Before planning a Hard tier trajectory, pick one tier from HardTierScaleLadder by weight
    // and apply its (TranslationScale, RotationScale) to the Delta* of ActionSpace.
    // Automatically restored on trajectory completion (FinalizeTrajectory).
    void ApplyHardTierScaleJitter(float& OutTransScale, float& OutRotScale);
    void RestoreActionSpaceScale();

    // Helper: swept line trace along the entire candidate action sequence; returns true if collision-free throughout.
    bool IsPathCollisionFree(
        const FVector& StartPos, const FRotator& StartRot,
        const TArray<int32>& Actions) const;

    // Sample a natural exploration path (forward-biased with inertia)
    // Starting from (StartPos, StartRot), sample NumSteps steps.
    // Output: OutActions = action sequence, OutEndPos / Rot = end pose
    // bForbidPitch: forbid look_up / look_down (used by Loop to avoid Z drift)
    void SampleNaturalPath(
        const FVector& StartPos, const FRotator& StartRot,
        int32 NumSteps,
        TArray<int32>& OutActions,
        FVector& OutEndPos, FRotator& OutEndRot,
        bool bForbidPitch = false);

    // Get the inverse-action index of an action (forward<->backward, left<->right, turn_left<->turn_right, etc.)
    int32 GetInverseAction(int32 ActionIdx) const;

    // Check whether an action is purely a translation (has translation, no rotation)
    bool IsTranslationAction(int32 ActionIdx) const;

    // Compute an action sequence from the current pose back to the target pose (used by Loop's go-home leg)
    // Returns true on success, false if it cannot complete within MaxSteps.
    bool ComputeGoHomePath(
        const FVector& CurrentPos, const FRotator& CurrentRot,
        const FVector& TargetPos, const FRotator& TargetRot,
        int32 MaxSteps,
        TArray<int32>& OutActions) const;

    // Randomly shuffle within commutative segments of an action sequence (used by Equivalence)
    void ShuffleCommutativeSegments(TArray<int32>& Actions) const;

    // Compute the maximum spatial divergence (cm) between the intermediate trajectories of two action sequences
    float ComputeMaxSpatialDivergence(
        const FVector& StartPos, const FRotator& StartRot,
        const TArray<int32>& PathA, const TArray<int32>& PathB) const;

    // Get the trajectory type's string name (used in metadata)
    static FString TrajectoryTypeToString(ETrajectoryType Type);

    // Scale tier -> string (used in metadata; evaluation scripts group by this field)
    static FString ScaleTierToString(EScaleTier Tier);

    // Counter for outstanding async file writes (used to wait for all async tasks to finish)
    FThreadSafeCounter PendingAsyncWrites;
};
