#include "DataCollector.h"
#include "ImageUtils.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformFileManager.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Misc/FileHelper.h"
#include "CollisionQueryParams.h"
#include "Engine/World.h"
#include "Engine/OverlapResult.h"
#include "Engine/DirectionalLight.h"
#include "Components/LightComponent.h"
#include "EngineUtils.h"
#include "IImageWrapperModule.h"
#include "IImageWrapper.h"
#include "Engine/PostProcessVolume.h"

// ============================================================================
// Constructor
// ============================================================================
ADataCollector::ADataCollector()
{
    PrimaryActorTick.bCanEverTick = true;

    // ---- 1. Create the collision capsule as the root component ----
    // This capsule models the agent's physical volume.
    // It collides with the scene's static meshes (walls, furniture, etc.).
    CollisionCapsule = CreateDefaultSubobject<UCapsuleComponent>(TEXT("CollisionCapsule"));
    RootComponent = CollisionCapsule;

    // Configure collision properties
    CollisionCapsule->SetCapsuleRadius(34.0f);
    CollisionCapsule->SetCapsuleHalfHeight(88.0f);
    CollisionCapsule->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    CollisionCapsule->SetCollisionObjectType(ECollisionChannel::ECC_Pawn);
    CollisionCapsule->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
    // Don't simulate physics (we drive movement manually) but still participate in collision queries.
    CollisionCapsule->SetSimulatePhysics(false);

    // ---- 2. Create the scene capture camera (RGB) ----
    SceneCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("SceneCapture"));
    SceneCapture->SetupAttachment(CollisionCapsule);
    SceneCapture->FOVAngle = 90.0f;

    // [Key] Use FinalColorHDR instead of FinalColorLDR.
    // In LDR mode, SceneCapture's tonemapping / exposure is inconsistent with the editor viewport;
    // in particular auto-exposure has no history-frame accumulation and frequently causes severe
    // underexposure (a fully black image).
    // HDR mode keeps the full post-process chain, and with an RGBA16F RenderTarget produces correct output.
    SceneCapture->CaptureSource = SCS_FinalColorHDR;

    // Disable auto-capture -- only render when CaptureScene() is called manually from code.
    // Otherwise the engine auto-renders once per frame + we render once = double cost.
    SceneCapture->bCaptureEveryFrame = false;
    SceneCapture->bCaptureOnMovement = false;

    // ---- 3. Create the depth capture camera ----
    DepthCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("DepthCapture"));
    DepthCapture->SetupAttachment(CollisionCapsule);
    DepthCapture->FOVAngle = 90.0f;
    // SceneDepth: per-pixel distance from camera to the scene (cm)
    DepthCapture->CaptureSource = SCS_SceneDepth;
    DepthCapture->bCaptureEveryFrame = false;
    DepthCapture->bCaptureOnMovement = false;

    // ---- 4. Create the optical-flow (velocity) capture camera ----
    // Uses a PostProcess Material (M_VelocityPass) to extract motion vectors from the Velocity Buffer.
    // CaptureSource = FinalColorHDR; the post-process material writes velocity data into the final color.
    VelocityCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("VelocityCapture"));
    VelocityCapture->SetupAttachment(CollisionCapsule);
    VelocityCapture->FOVAngle = 90.0f;
    VelocityCapture->CaptureSource = SCS_FinalColorHDR;
    VelocityCapture->bCaptureEveryFrame = false;
    VelocityCapture->bCaptureOnMovement = false;
}

// ============================================================================
// BeginPlay - run the full initialization pipeline
// ============================================================================
void ADataCollector::BeginPlay()
{
    Super::BeginPlay();

    // If the blueprint / level scaled the DataCollector, the capsule and camera offset are
    // amplified, making the rendered viewpoint look much higher than SamplingZ. So by default
    // we force the scale back to 1.
    const FVector InitialActorScale = GetActorScale3D();
    if (bForceUnitScale && !InitialActorScale.Equals(FVector::OneVector, 0.01f))
    {
        UE_LOG(LogTemp, Warning, TEXT("[Transform] DataCollector scale was (%.3f, %.3f, %.3f), forcing to (1,1,1)"),
            InitialActorScale.X, InitialActorScale.Y, InitialActorScale.Z);
        SetActorScale3D(FVector::OneVector);
    }

    // Update capsule dimensions (using the values configured in the editor)
    CollisionCapsule->SetCapsuleRadius(CapsuleRadius);
    CollisionCapsule->SetCapsuleHalfHeight(CapsuleHalfHeight);

    // Apply the camera offset (modeling eyes at the top of the body)
    SceneCapture->SetRelativeLocation(CameraOffset);
    SceneCapture->FOVAngle = CameraFOV;

    // Create the Render Target
    // [Key] Use RGBA16F (HDR) format combined with SCS_FinalColorHDR:
    //   - RGBA8 + LDR mode  : insufficient precision + inconsistent tonemapping -> shadows clipped
    //   - RGBA16F + HDR mode: full post-process chain + correct tonemapping -> matches the editor viewport
    // bForceLinearGamma = true: prevents the RenderTarget from applying gamma correction a second time
    // (the engine has already done it during post-processing).
    UTextureRenderTarget2D* RenderTarget = NewObject<UTextureRenderTarget2D>();
    RenderTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA16f;
    RenderTarget->InitCustomFormat(ResolutionX, ResolutionY, PF_FloatRGBA, true);
    RenderTarget->bForceLinearGamma = true;
    RenderTarget->TargetGamma = 0.0f;  // 0 = engine handles gamma automatically, no extra correction
    RenderTarget->UpdateResource();

    SceneCapture->TextureTarget = RenderTarget;
    SceneCapture->CaptureSource = SCS_FinalColorHDR;

    // ---- Depth RenderTarget initialization ----
    if (bCaptureDepth)
    {
        DepthCapture->SetRelativeLocation(CameraOffset);
        DepthCapture->FOVAngle = CameraFOV;

        // R32F single-channel float texture, stores precise depth values (cm)
        UTextureRenderTarget2D* DepthRT = NewObject<UTextureRenderTarget2D>();
        DepthRT->RenderTargetFormat = ETextureRenderTargetFormat::RTF_R32f;
        DepthRT->InitCustomFormat(ResolutionX, ResolutionY, PF_R32_FLOAT, false);
        DepthRT->bForceLinearGamma = true; // depth values are linear; no gamma correction
        DepthRT->UpdateResource();

        DepthCapture->TextureTarget = DepthRT;
        DepthCapture->CaptureSource = SCS_SceneDepth;

        UE_LOG(LogTemp, Warning, TEXT("[Modality] Depth capture enabled (MaxDist=%.0f cm, R32F → 16bit PNG)"), DepthMaxDistance);
    }

    // ---- Optical-flow / velocity RenderTarget initialization ----
    if (bCaptureOpticalFlow)
    {
        if (!VelocityPassMaterial)
        {
            UE_LOG(LogTemp, Error, TEXT("[Modality] Optical flow DISABLED: VelocityPassMaterial is not set!"));
            UE_LOG(LogTemp, Error, TEXT("[Modality] Please create a Post-Process Material (M_VelocityPass) in the editor:"));
            UE_LOG(LogTemp, Error, TEXT("[Modality]   1. Create Material, set Material Domain = Post Process"));
            UE_LOG(LogTemp, Error, TEXT("[Modality]   2. Add SceneTexture node, set to 'Velocity'"));
            UE_LOG(LogTemp, Error, TEXT("[Modality]   3. Connect SceneTexture RGB to Emissive Color"));
            UE_LOG(LogTemp, Error, TEXT("[Modality]   4. Set Blendable Location = Before Tonemapping"));
            UE_LOG(LogTemp, Error, TEXT("[Modality]   5. Assign the material to DataCollector's VelocityPassMaterial property"));
            bCaptureOpticalFlow = false;
        }
        else
        {
            VelocityCapture->SetRelativeLocation(CameraOffset);
            VelocityCapture->FOVAngle = CameraFOV;

            // RGBA16F RenderTarget for storing high-precision velocity vectors
            UTextureRenderTarget2D* VelocityRT = NewObject<UTextureRenderTarget2D>();
            VelocityRT->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA16f;
            VelocityRT->InitCustomFormat(ResolutionX, ResolutionY, PF_FloatRGBA, false);
            VelocityRT->bForceLinearGamma = true;
            VelocityRT->UpdateResource();

            VelocityCapture->TextureTarget = VelocityRT;
            VelocityCapture->CaptureSource = SCS_FinalColorHDR;

            // ============================================================
            // ShowFlags configuration -- key points:
            //   1. PostProcessing must be enabled (so Blendable materials run)
            //   2. Tonemapper must stay enabled! Because the material is set to
            //      "Replacing the Tonemapper"; if the Tonemapper stage is disabled,
            //      the material has no chance to execute.
            //   3. Disable everything else unnecessary to avoid interference.
            // ============================================================
            FEngineShowFlags& VelFlags = VelocityCapture->ShowFlags;
            VelFlags.SetPostProcessing(true);   // must be on
            VelFlags.SetTonemapper(true);        // must be on! material executes during the Tonemapper stage
            VelFlags.SetBloom(false);
            VelFlags.SetLensFlares(false);
            VelFlags.SetDepthOfField(false);
            VelFlags.SetVolumetricFog(false);
            VelFlags.SetScreenSpaceReflections(false);
            VelFlags.SetAmbientOcclusion(false);
            VelFlags.SetMotionBlur(false);
            VelFlags.SetEyeAdaptation(false);    // disable auto-exposure to prevent brightness fluctuations

            // Add the post-process material to SceneCapture's Blendables.
            // The material's Blendable Location = "Replacing the Tonemapper",
            // so it replaces the default tonemapper during that stage,
            // reads motion vectors from SceneTexture:Velocity, and writes them out as the final color.
            VelocityCapture->PostProcessSettings.WeightedBlendables.Array.Empty();
            FWeightedBlendable Blendable;
            Blendable.Weight = 1.0f;
            Blendable.Object = VelocityPassMaterial;
            VelocityCapture->PostProcessSettings.WeightedBlendables.Array.Add(Blendable);

            // Override auto-exposure so the values written by the material are not modified by exposure.
            VelocityCapture->PostProcessSettings.bOverride_AutoExposureBias = true;
            VelocityCapture->PostProcessSettings.AutoExposureBias = 0.0f;
            VelocityCapture->PostProcessSettings.bOverride_AutoExposureMinBrightness = true;
            VelocityCapture->PostProcessSettings.AutoExposureMinBrightness = 1.0f;
            VelocityCapture->PostProcessSettings.bOverride_AutoExposureMaxBrightness = true;
            VelocityCapture->PostProcessSettings.AutoExposureMaxBrightness = 1.0f;

            UE_LOG(LogTemp, Warning, TEXT("[Modality] Optical flow capture enabled (PostProcess Material: %s → RGBA16F → PNG)"),
                *VelocityPassMaterial->GetName());
        }
    }

    // Freeze scene lighting (prevents day/night cycling from breaking counterfactual reasoning)
    if (bFreezeLighting)
    {
        FreezeLighting();
    }

    // Apply render-quality settings (configure each effect according to its Render Quality toggle)
    ApplyRenderQualitySettings();

    // Initialize the action space
    if (ActionSpace.Num() == 0)
    {
        InitializeDefaultActionSpace();
    }

    // Compute the total frames per video (Counterfactual Tree mode only)
    TotalFramesPerVideo = FMath::RoundToInt(VideoDurationSeconds * FramesPerSecond);

    // Trajectory mode: initialize action sampling weights
    if (CollectorMode == ECollectorMode::TrajectoryGraph)
    {
        if (ActionSamplingWeights.Num() != ActionSpace.Num())
        {
            InitializeDefaultActionWeights();
        }
        // Initialize the trajectory counters
        TrajectoryGlobalID = 0;
        TotalReasoningTrajectories = 0;
        TotalRandomWalkTrajectories = 0;

        // Initialize the collision log (appended dynamically as collisions occur)
        TotalCollisionCount = 0;
    }

    // ---- Generate all sample points ----
    GenerateSamplePoints();

    if (SamplePoints.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("No valid sample points generated! Check your Sampling settings."));
        SetActorTickEnabled(false);
        return;
    }

    // Initialize the FSM
    CurrentSampleIndex = 0;
    CurrentStateID = 0;
    CurrentPhase = ECollectorPhase::WaitForSceneReady;
    WaitFrameCounter = 0;

    // Generate a timestamped subfolder based on launch time
    FDateTime Now = FDateTime::Now();
    FString TimestampFolder = FString::Printf(TEXT("run_%04d%02d%02d_%02d%02d%02d"),
        Now.GetYear(), Now.GetMonth(), Now.GetDay(),
        Now.GetHour(), Now.GetMinute(), Now.GetSecond());

    // Make sure SaveRootDirectory ends with a separator
    if (!SaveRootDirectory.EndsWith(TEXT("/")) && !SaveRootDirectory.EndsWith(TEXT("\\")))
    {
        SaveRootDirectory += TEXT("/");
    }
    RuntimeSaveRoot = SaveRootDirectory + TimestampFolder + TEXT("/");

    // Make sure the root directory exists
    EnsureDirectoryExists(RuntimeSaveRoot);

    // Initialize the collision log file (entries are appended in real time, so a mid-run crash
    // does not lose already-recorded collisions).
    CollisionLogPath = RuntimeSaveRoot + TEXT("collision_log.txt");
    {
        FString Header;
        Header += TEXT("================================================================\r\n");
        Header += TEXT("  DataCollector Collision Log\r\n");
        Header += FString::Printf(TEXT("  Started: %s\r\n"), *FDateTime::Now().ToString());
        Header += FString::Printf(TEXT("  Collision Threshold: %.1f cm\r\n"), CollisionThreshold);
        Header += FString::Printf(TEXT("  Translation Magnitude: %.1f cm\r\n"), TranslationMagnitude);
        Header += FString::Printf(TEXT("  Rotation Magnitude: %.1f deg\r\n"), RotationMagnitude);
        Header += FString::Printf(TEXT("  Trajectory Steps: %d, Frames/Action: %d\r\n"), TrajectorySteps, FramesPerAction);
        Header += TEXT("  (Entries are appended in real-time as collisions occur)\r\n");
        Header += TEXT("================================================================\r\n\r\n");
        FFileHelper::SaveStringToFile(Header, *CollisionLogPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
    }

    // Trajectory mode: create the reasoning/, random_walk/ and staging/ subdirectories
    if (CollectorMode == ECollectorMode::TrajectoryGraph)
    {
        EnsureDirectoryExists(RuntimeSaveRoot + TEXT("reasoning/"));
        EnsureDirectoryExists(RuntimeSaveRoot + TEXT("random_walk/"));
        EnsureDirectoryExists(RuntimeSaveRoot + TEXT("staging/"));
    }

    UE_LOG(LogTemp, Warning, TEXT("============================================================"));
    if (CollectorMode == ECollectorMode::CounterfactualTree)
    {
        UE_LOG(LogTemp, Warning, TEXT("=== Counterfactual Data Collector - Tree Mode ==="));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("=== Trajectory Data Collector - Graph Mode ==="));
        UE_LOG(LogTemp, Warning, TEXT("  Trajectory: %d steps × %d frames/step = %d frames/trajectory"),
            TrajectorySteps, FramesPerAction, TrajectorySteps * FramesPerAction);
        UE_LOG(LogTemp, Warning, TEXT("  Trajectories per sample point: %d"), TrajectoriesPerSamplePoint);
        UE_LOG(LogTemp, Warning, TEXT("  Collision avoidance: %s (threshold=%.1f cm, max resample=%d)"),
            bCollisionAvoidanceSampling ? TEXT("ON") : TEXT("OFF"), CollisionThreshold, MaxResampleAttempts);
        UE_LOG(LogTemp, Warning, TEXT("  Max consecutive rotations: %d"), MaxConsecutiveRotations);
        UE_LOG(LogTemp, Warning, TEXT("  Output split: reasoning/ + random_walk/"));
    }
    UE_LOG(LogTemp, Warning, TEXT("  Episode: %d"), EpisodeID);
    UE_LOG(LogTemp, Warning, TEXT("  Sample Points: %d"), SamplePoints.Num());
    UE_LOG(LogTemp, Warning, TEXT("  Action Space: %d actions"), ActionSpace.Num());
    if (CollectorMode == ECollectorMode::CounterfactualTree)
    {
        UE_LOG(LogTemp, Warning, TEXT("  Video: %.1fs @ %d FPS = %d frames/video"), VideoDurationSeconds, FramesPerSecond, TotalFramesPerVideo);
    }
    UE_LOG(LogTemp, Warning, TEXT("  Resolution: %dx%d | FOV: %.0f | Format: %s%s"),
        ResolutionX, ResolutionY, CameraFOV,
        ImageFormat == EImageOutputFormat::JPEG ? TEXT("JPEG") : TEXT("BMP"),
        ImageFormat == EImageOutputFormat::JPEG ? *FString::Printf(TEXT(" (Q=%d)"), JpegQuality) : TEXT(""));
    UE_LOG(LogTemp, Warning, TEXT("  Performance: FramesPerTick=%d | AsyncIO=%s"),
        FramesPerTick,
        bAsyncFileWrite ? TEXT("ON") : TEXT("OFF"));
    UE_LOG(LogTemp, Warning, TEXT("  Modality: RGB=ON | Depth=%s | OpticalFlow=%s"),
        bCaptureDepth ? TEXT("ON") : TEXT("OFF"),
        bCaptureOpticalFlow ? TEXT("ON") : TEXT("OFF"));
    const FVector FinalActorScale = GetActorScale3D();
    const FVector InitialCameraWorldLoc = SceneCapture->GetComponentLocation();
    UE_LOG(LogTemp, Warning, TEXT("  Capsule: R=%.0f HH=%.0f | Camera Offset: (%.0f, %.0f, %.0f)"),
        CapsuleRadius, CapsuleHalfHeight, CameraOffset.X, CameraOffset.Y, CameraOffset.Z);
    UE_LOG(LogTemp, Warning, TEXT("  Transform: ActorScale=(%.3f, %.3f, %.3f) | InitialActorZ=%.1f | InitialCameraZ=%.1f"),
        FinalActorScale.X, FinalActorScale.Y, FinalActorScale.Z,
        GetActorLocation().Z, InitialCameraWorldLoc.Z);
    if (CollectorMode == ECollectorMode::CounterfactualTree)
    {
        UE_LOG(LogTemp, Warning, TEXT("  Total videos to render: %d"), SamplePoints.Num() * (1 + ActionSpace.Num()));
        UE_LOG(LogTemp, Warning, TEXT("  Total frames to render: %d"), SamplePoints.Num() * (1 + ActionSpace.Num()) * TotalFramesPerVideo);
    }
    else
    {
        int32 TotalTrajectories = SamplePoints.Num() * TrajectoriesPerSamplePoint;
        int32 TotalFrames = TotalTrajectories * TrajectorySteps * FramesPerAction;
        UE_LOG(LogTemp, Warning, TEXT("  Total trajectories to render: %d"), TotalTrajectories);
        UE_LOG(LogTemp, Warning, TEXT("  Total frames to render: %d"), TotalFrames);
    }
    UE_LOG(LogTemp, Warning, TEXT("  Output: %s"), *RuntimeSaveRoot);
    UE_LOG(LogTemp, Warning, TEXT("============================================================"));
}

// ============================================================================
// Generate the grid of sample points.
//   On the XY plane, generate a uniform grid spaced by SamplingInterval.
//   For each grid point, try to place the capsule directly at the fixed SamplingZ height.
//   If the capsule overlaps the scene with a blocking response, the point is rejected.
//   Each accepted grid point x NumYawSamples orientations = one set of sample points.
//
//   Debug info:
//     - The first MaxDetailedLogs rejected points print the full list of overlap objects.
//     - After the full sweep, prints how many times each blocker name was encountered.
// ============================================================================
void ADataCollector::GenerateSamplePoints()
{
    SamplePoints.Empty();

    // Determine the center (XY)
    FVector Center;
    if (bUsePlacementAsCenter)
    {
        Center = GetActorLocation();
    }
    else
    {
        Center = SamplingCenter;
    }

    if (bProjectSamplesToGround)
    {
        UE_LOG(LogTemp, Warning, TEXT("[Sampling] bProjectSamplesToGround is deprecated and ignored. Using direct capsule placement at fixed SamplingZ."));
    }

    const float TestRadius = CapsuleRadius;
    const float TestHalfHeight = CapsuleHalfHeight;

    UE_LOG(LogTemp, Warning, TEXT("[Sampling] Center XY=(%.1f, %.1f), Fixed SamplingZ=%.1f"),
        Center.X, Center.Y, SamplingZ);
    UE_LOG(LogTemp, Warning, TEXT("[Sampling] Test capsule: Radius=%.1f, HalfHeight=%.1f  (capsule top Z=%.1f, bottom Z=%.1f)"),
        TestRadius, TestHalfHeight, SamplingZ + TestHalfHeight, SamplingZ - TestHalfHeight);

    // Range
    const float StartX = Center.X - SamplingRangeX;
    const float EndX   = Center.X + SamplingRangeX;
    const float StartY = Center.Y - SamplingRangeY;
    const float EndY   = Center.Y + SamplingRangeY;

    int32 TotalPositions = 0;
    int32 SkippedBlocked = 0;
    int32 AcceptedPositions = 0;

    // blocker name -> hit count
    TMap<FString, int32> BlockerStats;

    FCollisionShape Shape = FCollisionShape::MakeCapsule(TestRadius, TestHalfHeight);

    if (bVerboseDebug)
    {
        UE_LOG(LogTemp, Warning, TEXT("[Sampling] *** VERBOSE DEBUG MODE ON — will log EVERY sample point ***"));
    }

    for (float X = StartX; X <= EndX + 0.1f; X += SamplingInterval)
    {
        for (float Y = StartY; Y <= EndY + 0.1f; Y += SamplingInterval)
        {
            TotalPositions++;

            const FVector TestLocation(X, Y, SamplingZ);

            if (!bSkipOccludedPoints)
            {
                // No collision check; accept directly.
                if (bVerboseDebug)
                {
                    UE_LOG(LogTemp, Log,
                        TEXT("[Sampling #%d] ACCEPTED (no collision check) at (%.1f, %.1f, %.1f)"),
                        TotalPositions, TestLocation.X, TestLocation.Y, TestLocation.Z);
                }
            }
            else
            {
                // ---------- Collision check ----------
                FCollisionQueryParams Params;
                Params.AddIgnoredActor(this);

                TArray<FOverlapResult> Overlaps;
                const bool bOverlap = GetWorld()->OverlapMultiByChannel(
                    Overlaps,
                    TestLocation,
                    FQuat::Identity,
                    ECollisionChannel::ECC_Pawn,
                    Shape,
                    Params
                );

                if (bOverlap && Overlaps.Num() > 0)
                {
                    SkippedBlocked++;

                    // --- Verbose: print full collision info for every rejected point ---
                    if (bVerboseDebug)
                    {
                        UE_LOG(LogTemp, Warning,
                            TEXT("[Sampling #%d] BLOCKED at (%.1f, %.1f, %.1f)  capsule bottom=%.1f top=%.1f  overlaps=%d:"),
                            TotalPositions,
                            TestLocation.X, TestLocation.Y, TestLocation.Z,
                            TestLocation.Z - TestHalfHeight, TestLocation.Z + TestHalfHeight,
                            Overlaps.Num());

                        for (const FOverlapResult& Overlap : Overlaps)
                        {
                            const AActor* OActor = Overlap.GetActor();
                            const UPrimitiveComponent* OComp = Overlap.GetComponent();
                            FString ActorName = OActor ? OActor->GetName() : TEXT("(null)");
                            FString CompName  = OComp  ? OComp->GetName()  : TEXT("(null)");
                            FString ClassName = OActor ? OActor->GetClass()->GetName() : TEXT("(null)");

                            UE_LOG(LogTemp, Warning,
                                TEXT("[Sampling #%d]   Actor=%s  Class=%s  Component=%s"),
                                TotalPositions, *ActorName, *ClassName, *CompName);
                        }
                    }

                    // --- Stats (collected regardless of verbose) ---
                    for (const FOverlapResult& Overlap : Overlaps)
                    {
                        const AActor* OActor = Overlap.GetActor();
                        FString Key = OActor ? FString::Printf(TEXT("%s (%s)"), *OActor->GetName(), *OActor->GetClass()->GetName()) : TEXT("(null)");
                        BlockerStats.FindOrAdd(Key) += 1;
                    }

                    continue; // skip this position
                }
                else
                {
                    // --- Verbose: print accepted points ---
                    if (bVerboseDebug)
                    {
                        UE_LOG(LogTemp, Log,
                            TEXT("[Sampling #%d] ACCEPTED at (%.1f, %.1f, %.1f)  capsule bottom=%.1f top=%.1f  (no overlap)"),
                            TotalPositions,
                            TestLocation.X, TestLocation.Y, TestLocation.Z,
                            TestLocation.Z - TestHalfHeight, TestLocation.Z + TestHalfHeight);
                    }
                }
            }

            // ---------- Position is usable ----------
            AcceptedPositions++;
            const float YawStep = 360.0f / static_cast<float>(NumYawSamples);
            for (int32 YawIdx = 0; YawIdx < NumYawSamples; YawIdx++)
            {
                const float Yaw = YawIdx * YawStep;
                const FRotator Rot(0, Yaw, 0);
                SamplePoints.Add(FSamplePoint(TestLocation, Rot));
            }
        }
    }

    // ---- Summary ----
    UE_LOG(LogTemp, Warning, TEXT("[Sampling] ============ SUMMARY ============"));
    UE_LOG(LogTemp, Warning, TEXT("[Sampling] Grid: %.0f x %.0f cm, Interval: %.0f cm"),
        SamplingRangeX * 2, SamplingRangeY * 2, SamplingInterval);
    UE_LOG(LogTemp, Warning, TEXT("[Sampling] Positions: %d total, %d blocked, %d valid"),
        TotalPositions, SkippedBlocked, TotalPositions - SkippedBlocked);
    UE_LOG(LogTemp, Warning, TEXT("[Sampling] Yaw samples per position: %d (every %.1f degrees)"),
        NumYawSamples, 360.0f / NumYawSamples);
    UE_LOG(LogTemp, Warning, TEXT("[Sampling] Total sample points: %d"), SamplePoints.Num());

    // ---- Blocker frequency ranking ----
    if (BlockerStats.Num() > 0)
    {
        // Sort by hit count, descending
        BlockerStats.ValueSort([](int32 A, int32 B) { return A > B; });

        UE_LOG(LogTemp, Warning, TEXT("[Sampling] ---- Blocker frequency (top blockers) ----"));
        int32 PrintCount = 0;
        for (const auto& Pair : BlockerStats)
        {
            UE_LOG(LogTemp, Warning, TEXT("[Sampling]   %6d hits: %s"), Pair.Value, *Pair.Key);
            PrintCount++;
            if (PrintCount >= 20) // print at most 20
            {
                UE_LOG(LogTemp, Warning, TEXT("[Sampling]   ... and %d more unique blockers"), BlockerStats.Num() - PrintCount);
                break;
            }
        }
    }

    if (SamplePoints.Num() > 0)
    {
        const FVector& FirstLoc = SamplePoints[0].Location;
        const FVector FirstCameraLoc = FirstLoc + CameraOffset;
        UE_LOG(LogTemp, Warning, TEXT("[Sampling] First sample actor Z=%.1f, camera Z=%.1f (CameraOffset.Z=%.1f)"),
            FirstLoc.Z, FirstCameraLoc.Z, CameraOffset.Z);
    }
    else if (SkippedBlocked == TotalPositions)
    {
        UE_LOG(LogTemp, Error, TEXT("[Sampling] ALL %d positions were blocked! Possible causes:"), TotalPositions);
        UE_LOG(LogTemp, Error, TEXT("[Sampling]   1. SamplingZ=%.1f is too low -> capsule bottom (%.1f) penetrates ground"),
            SamplingZ, SamplingZ - TestHalfHeight);
        UE_LOG(LogTemp, Error, TEXT("[Sampling]   2. SamplingZ=%.1f is too high -> capsule top (%.1f) hits ceiling"),
            SamplingZ, SamplingZ + TestHalfHeight);
        UE_LOG(LogTemp, Error, TEXT("[Sampling]   3. Landscape/floor blocks ECC_Pawn at this height range"));
        UE_LOG(LogTemp, Error, TEXT("[Sampling]   4. Check the blocker list above to identify what is being hit"));
    }
}

// ============================================================================
// Check whether a location is occluded by an obstacle.
//
// Performs an Overlap test using a capsule with the same dimensions as the collector.
// Note: detailed logging has been moved into GenerateSamplePoints for centralized handling.
// ============================================================================
bool ADataCollector::IsLocationOccluded(const FVector& Location) const
{
    if (!GetWorld()) return false;

    FCollisionShape Shape = FCollisionShape::MakeCapsule(CapsuleRadius, CapsuleHalfHeight);

    FCollisionQueryParams Params;
    Params.AddIgnoredActor(this);

    // If a whitelist exists, use OverlapMulti and check overlaps individually.
    if (CollisionWhitelistKeywords.Num() > 0)
    {
        TArray<FOverlapResult> Overlaps;
        bool bOverlap = GetWorld()->OverlapMultiByChannel(
            Overlaps,
            Location,
            FQuat::Identity,
            ECollisionChannel::ECC_Pawn,
            Shape,
            Params
        );

        if (!bOverlap) return false;

        // Filter out whitelisted actors.
        for (const FOverlapResult& Overlap : Overlaps)
        {
            AActor* HitActor = Overlap.GetActor();
            if (HitActor && !IsActorWhitelisted(HitActor))
            {
                return true; // a non-whitelisted blocker exists
            }
        }
        return false; // all overlaps are in the whitelist
    }

    // Without a whitelist, use the original fast test.
    return GetWorld()->OverlapBlockingTestByChannel(
        Location,
        FQuat::Identity,
        ECollisionChannel::ECC_Pawn,
        Shape,
        Params
    );
}

// ============================================================================
// Collision-whitelist check.
// Returns true if the actor's name or class name contains any whitelist keyword (case-insensitive).
// Whitelisted actors are not treated as collision obstacles.
// ============================================================================
bool ADataCollector::IsActorWhitelisted(const AActor* HitActor) const
{
    if (!HitActor || CollisionWhitelistKeywords.Num() == 0) return false;

    FString ActorName = HitActor->GetName();
    FString ClassName = HitActor->GetClass()->GetName();

    for (const FString& Keyword : CollisionWhitelistKeywords)
    {
        if (Keyword.IsEmpty()) continue;

        if (ActorName.Contains(Keyword, ESearchCase::IgnoreCase) ||
            ClassName.Contains(Keyword, ESearchCase::IgnoreCase))
        {
            return true;
        }
    }
    return false;
}

// ============================================================================
// Move with collision (Sweep Test).
// Sweeps from the current location toward the target location.
// If an obstacle is hit along the way, stop at the impact point.
// Returns: true = collision occurred, false = no collision.
// ============================================================================
bool ADataCollector::MoveWithCollision(const FVector& TargetLocation, FHitResult* OutHit)
{
    if (!GetWorld()) return false;

    FVector CurrentLocation = GetActorLocation();
    FVector MoveDirection = TargetLocation - CurrentLocation;
    float MoveDistance = MoveDirection.Size();

    if (MoveDistance < 0.01f)
    {
        return false; // no movement needed
    }

    FCollisionShape Shape = FCollisionShape::MakeCapsule(CapsuleRadius, CapsuleHalfHeight);

    FCollisionQueryParams Params;
    Params.AddIgnoredActor(this);
    Params.bReturnPhysicalMaterial = true; // request physical-material info

    // If a whitelist exists, use SweepMulti and filter individually.
    if (CollisionWhitelistKeywords.Num() > 0)
    {
        TArray<FHitResult> Hits;
        bool bHit = GetWorld()->SweepMultiByChannel(
            Hits,
            CurrentLocation,
            TargetLocation,
            FQuat::Identity,
            ECollisionChannel::ECC_Pawn,
            Shape,
            Params
        );

        if (bHit)
        {
            // Find the first non-whitelisted collision.
            for (const FHitResult& Hit : Hits)
            {
                AActor* HitActor = Hit.GetActor();
                if (HitActor && IsActorWhitelisted(HitActor))
                {
                    continue; // whitelisted actor, skip
                }

                // Non-whitelisted collision -- stop at the impact point.
                FVector SafeLocation = Hit.Location + Hit.Normal * 1.0f;
                SetActorLocation(SafeLocation);

                if (OutHit)
                {
                    *OutHit = Hit;
                }
                return true;
            }
        }

        // No valid collision (all whitelisted or none); move normally.
        SetActorLocation(TargetLocation);
        return false;
    }

    // Without a whitelist, use the original logic.
    FHitResult Hit;
    bool bHit = GetWorld()->SweepSingleByChannel(
        Hit,
        CurrentLocation,
        TargetLocation,
        FQuat::Identity,
        ECollisionChannel::ECC_Pawn,
        Shape,
        Params
    );

    if (bHit)
    {
        // Hit an obstacle: move just before the impact point (small gap to avoid getting stuck).
        FVector SafeLocation = Hit.Location + Hit.Normal * 1.0f;
        SetActorLocation(SafeLocation);

        // Forward collision details to the caller.
        if (OutHit)
        {
            *OutHit = Hit;
        }
        return true; // collision occurred
    }
    else
    {
        // No collision; move to the target normally.
        SetActorLocation(TargetLocation);
        return false; // no collision
    }
}

// ============================================================================
// Initialize the default action space
// ============================================================================
void ADataCollector::InitializeDefaultActionSpace()
{
    // Action space rationale (assuming 0.5 s / 8 frames of progressive interpolation per step):
    //
    // Translation D (cm/step):
    //   D=100 -> 2.0 m/s (normal walking, recommended)
    //   D=50  -> 1.0 m/s (slow walk, small indoor scenes)
    //   D=200 -> 4.0 m/s (jogging, large scenes)
    //   Reference: Habitat / AI2-THOR use 25 cm teleport, but this project uses 8-frame interpolation.
    //
    // Yaw rotation R (deg/step):
    //   R=15 -> 30°/s (normal head turn, recommended)
    //   R=10 -> 20°/s (stable but slow)
    //   R=30 -> 60°/s (fast head turn, looks shaky)
    //
    // Pitch rotation (deg/step):
    //   Use R*0.5: in everyday motion, pitch changes are far smaller than yaw.
    //   With R=15, pitch = 7.5°/step -> 15°/s (smooth and natural)

    const float D = TranslationMagnitude;
    const float R = RotationMagnitude;
    const float PitchR = R * 0.5f; // pitch uses half the rotation magnitude

    ActionSpace.Empty();

    // --- Translation actions (4, in local frame) ---
    ActionSpace.Add(FActionDefinition(TEXT("move_forward"),  FVector(D, 0, 0),   FRotator::ZeroRotator));
    ActionSpace.Add(FActionDefinition(TEXT("move_backward"), FVector(-D, 0, 0),  FRotator::ZeroRotator));
    ActionSpace.Add(FActionDefinition(TEXT("move_left"),     FVector(0, -D, 0),  FRotator::ZeroRotator));
    ActionSpace.Add(FActionDefinition(TEXT("move_right"),    FVector(0, D, 0),   FRotator::ZeroRotator));

    // --- Yaw rotation actions (2) ---
    ActionSpace.Add(FActionDefinition(TEXT("turn_left"),  FVector::ZeroVector, FRotator(0, -R, 0)));
    ActionSpace.Add(FActionDefinition(TEXT("turn_right"), FVector::ZeroVector, FRotator(0, R, 0)));

    // --- Pitch rotation actions (2; uses half magnitude) ---
    ActionSpace.Add(FActionDefinition(TEXT("look_up"),    FVector::ZeroVector, FRotator(PitchR, 0, 0)));
    ActionSpace.Add(FActionDefinition(TEXT("look_down"),  FVector::ZeroVector, FRotator(-PitchR, 0, 0)));

    // --- No-op (1) ---
    ActionSpace.Add(FActionDefinition(TEXT("no_op"), FVector::ZeroVector, FRotator::ZeroRotator));

    UE_LOG(LogTemp, Log, TEXT("[DataCollector] Action space initialized: D=%.1f cm (%.1f m/s), R=%.1f° (%.1f°/s), Pitch=%.1f° (%.1f°/s)"),
        D, D * 2.0f / 100.0f,  // 2 = 1/0.5s
        R, R * 2.0f,
        PitchR, PitchR * 2.0f);
}

// ============================================================================
// Tick - the core FSM (batch-processing mode)
//
// Each Tick calls ProcessOneFrame() at most FramesPerTick times.
// This skips the engine overhead between Ticks (physics, UI, GC, etc.) and
// significantly improves offline data-capture throughput.
//
// Flow:
//   WaitForSceneReady ->
//   [TeleportToSamplePoint -> RenderContextVideo ->
//    [RenderActionVideo -> ResetForNextAction] x N_actions ->
//    AdvanceToNextSamplePoint] x N_samplePoints ->
//   Completed
// ============================================================================
void ADataCollector::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    for (int32 i = 0; i < FramesPerTick; i++)
    {
        if (!ProcessOneFrame())
        {
            break; // need to wait for the next Tick (e.g. WaitForSceneReady / Completed)
        }
    }
}

// ============================================================================
// ProcessOneFrame - process one frame of FSM logic
// Returns true:  can process the next frame (within the same Tick)
// Returns false: must yield from the current Tick (waiting for the engine, or finished)
// ============================================================================
bool ADataCollector::ProcessOneFrame()
{
    switch (CurrentPhase)
    {
    // ================================================================
    // Phase 0: wait for the initial scene load
    // ================================================================
    case ECollectorPhase::WaitForSceneReady:
    {
        WaitFrameCounter++;
        if (WaitFrameCounter >= WarmupFrames)
        {
            UE_LOG(LogTemp, Warning, TEXT("[Phase] Scene ready. Starting batch collection of %d sample points..."), SamplePoints.Num());
            CurrentSampleIndex = 0;
            CurrentStateID = 0;
            CurrentPhase = ECollectorPhase::TeleportToSamplePoint;
            return true; // can begin processing immediately
        }
        return false; // wait for the next Tick
    }

    // ================================================================
    // Phase 1: teleport to the current sample point
    // ================================================================
    case ECollectorPhase::TeleportToSamplePoint:
    {
        if (CurrentSampleIndex >= SamplePoints.Num())
        {
            CurrentPhase = ECollectorPhase::Completed;
            return false;
        }

        const FSamplePoint& SP = SamplePoints[CurrentSampleIndex];

        // Teleport directly
        SetActorLocation(SP.Location);
        SetActorRotation(SP.Rotation);

        // Record the original pose
        OriginalLocation = SP.Location;
        OriginalRotation = SP.Rotation;

        const FVector CameraWorldLoc = SceneCapture->GetComponentLocation();
        const FVector ActorScale = GetActorScale3D();

        UE_LOG(LogTemp, Warning, TEXT("========================================"));
        UE_LOG(LogTemp, Warning, TEXT("[Sample %d/%d] State=%d  Pos=(%.0f, %.0f, %.0f)  Yaw=%.0f"),
            CurrentSampleIndex + 1, SamplePoints.Num(), CurrentStateID,
            SP.Location.X, SP.Location.Y, SP.Location.Z,
            SP.Rotation.Yaw);
        UE_LOG(LogTemp, Warning, TEXT("[Sample %d/%d] ActorScale=(%.3f, %.3f, %.3f) | CameraWorld=(%.1f, %.1f, %.1f)"),
            CurrentSampleIndex + 1, SamplePoints.Num(),
            ActorScale.X, ActorScale.Y, ActorScale.Z,
            CameraWorldLoc.X, CameraWorldLoc.Y, CameraWorldLoc.Z);

        // Branch on the capture mode.
        if (CollectorMode == ECollectorMode::TrajectoryGraph)
        {
            // Trajectory mode: initialize the per-sample-point trajectory counter.
            CurrentTrajectoryIndexAtPoint = 0;
            WaitFrameCounter = 0;
            CurrentPhase = ECollectorPhase::Traj_PlanTrajectory;
        }
        else
        {
            // Counterfactual Tree mode: create the episode directory, enter context rendering.
            EnsureDirectoryExists(GetEpisodePath());
            WaitFrameCounter = 0;
            CurrentPhase = ECollectorPhase::RenderContextVideo;
            CurrentFrameIndex = 0;

            // Create the context directory.
            FString ContextDir = GetEpisodePath() / GetContextSubDir();
            EnsureDirectoryExists(ContextDir);
        }
        return true; // continue
    }

    // ================================================================
    // Phase 2: render the context video (static observation)
    // ================================================================
    case ECollectorPhase::RenderContextVideo:
    {
        // Wait for the image to settle for several frames before saving.
        // During this time we still call CaptureScene() so that auto-exposure can warm up
        // (Eye Adaptation needs history frames).
        if (WaitFrameCounter < TeleportSettleFrames)
        {
            // Warm-up capture: do not save; let auto-exposure and Lumen GI converge.
            SceneCapture->CaptureScene();
            WaitFrameCounter++;
            return false; // wait for it to settle; we need real frame intervals
        }

        if (CurrentFrameIndex < TotalFramesPerVideo)
        {
            SaveFrame(GetContextSubDir(), CurrentFrameIndex);
            CurrentFrameIndex++;
            return true; // continue with the next frame
        }
        else
        {
            UE_LOG(LogTemp, Log, TEXT("  Context video captured (%d frames)"), TotalFramesPerVideo);
            CurrentActionIndex = 0;
            CurrentPhase = ECollectorPhase::ResetForNextAction;
            return true;
        }
    }

    // ================================================================
    // Phase 3: reset to the original pose; prepare for the next action
    // ================================================================
    case ECollectorPhase::ResetForNextAction:
    {
        if (CurrentActionIndex >= ActionSpace.Num())
        {
            // All actions for the current sample point are done.
            CurrentPhase = ECollectorPhase::AdvanceToNextSamplePoint;
            return true;
        }

        // Reset
        SetActorLocation(OriginalLocation);
        SetActorRotation(OriginalRotation);

        // Compute the target pose
        const FActionDefinition& Action = ActionSpace[CurrentActionIndex];
        ActionStartLocation = OriginalLocation;
        ActionStartRotation = OriginalRotation;

        FVector WorldDelta = OriginalRotation.RotateVector(Action.DeltaPosition);
        ActionTargetLocation = OriginalLocation + WorldDelta;
        ActionTargetRotation = OriginalRotation + Action.DeltaRotation;

        // Create the action directory
        FString ActionDir = GetEpisodePath() / GetActionSubDir(CurrentActionIndex);
        EnsureDirectoryExists(ActionDir);

        CurrentFrameIndex = 0;

        UE_LOG(LogTemp, Log, TEXT("  Rendering action [%d/%d]: '%s'"),
            CurrentActionIndex + 1, ActionSpace.Num(), *Action.ActionName);

        CurrentPhase = ECollectorPhase::RenderActionVideo;
        return true;
    }

    // ================================================================
    // Phase 4: render the action branch video (smooth movement with collision)
    // ================================================================
    case ECollectorPhase::RenderActionVideo:
    {
        if (CurrentFrameIndex < TotalFramesPerVideo)
        {
            // Compute interpolation alpha in [0, 1]
            float Alpha = static_cast<float>(CurrentFrameIndex) / static_cast<float>(FMath::Max(TotalFramesPerVideo - 1, 1));

            // Linear interpolation toward the target
            FVector DesiredLocation = FMath::Lerp(ActionStartLocation, ActionTargetLocation, Alpha);
            FRotator DesiredRotation = FMath::Lerp(ActionStartRotation, ActionTargetRotation, Alpha);

            // Move with sweep collision (stops at walls).
            MoveWithCollision(DesiredLocation);
            // Rotation is unaffected by collision.
            SetActorRotation(DesiredRotation);

            // Save the frame.
            SaveFrame(GetActionSubDir(CurrentActionIndex), CurrentFrameIndex);
            CurrentFrameIndex++;
            return true; // continue with the next frame
        }
        else
        {
            UE_LOG(LogTemp, Log, TEXT("    Action '%s' captured (%d frames)"),
                *ActionSpace[CurrentActionIndex].ActionName, TotalFramesPerVideo);

            CurrentActionIndex++;
            CurrentPhase = ECollectorPhase::ResetForNextAction;
            return true;
        }
    }

    // ================================================================
    // ================================================================
    // Trajectory mode - Phase T1: plan the action sequence for the trajectory
    // ================================================================
    case ECollectorPhase::Traj_PlanTrajectory:
    {
        // Wait for the image to settle (warm up GI and auto-exposure)
        if (WaitFrameCounter < TeleportSettleFrames)
        {
            // Warm-up capture: do not save; let auto-exposure and Lumen GI converge.
            SceneCapture->CaptureScene();
            WaitFrameCounter++;
            return false;
        }

        // Return to root state
        SetActorLocation(OriginalLocation);
        SetActorRotation(OriginalRotation);

        // Plan the action sequence (may produce a paired trajectory)
        PlanTrajectoryActions();

        // === Handle planning failure (skip rendering, no output) ===
        // When bSkipFailedPlans = true and all retries fail, bPlanningFailed = true.
        // We skip rendering, do not allocate a TrajectoryGlobalID, and try the next trajectory directly.
        if (bPlanningFailed)
        {
            bPlanningFailed = false;
            CurrentTrajectoryIndexAtPoint++;

            if (CurrentTrajectoryIndexAtPoint < TrajectoriesPerSamplePoint)
            {
                // The current sample point still has quota; return to root state and re-plan.
                SetActorLocation(OriginalLocation);
                SetActorRotation(OriginalRotation);
                WaitFrameCounter = 0;
                CurrentPhase = ECollectorPhase::Traj_PlanTrajectory;
            }
            else
            {
                // The current sample point's quota is used up; move on to the next sample point.
                CurrentPhase = ECollectorPhase::AdvanceToNextSamplePoint;
            }
            return true;
        }

        // If a paired trajectory exists (Equivalence), pre-allocate its ID.
        if (bHasPairedTrajectory)
        {
            PairedTrajectoryGlobalID = TrajectoryGlobalID + 1;
        }

        // Initialize trajectory state -- render the primary path P first.
        bCurrentlyRenderingPaired = false;
        CurrentTrajectoryStep = 0;
        CurrentStepFrameIndex = 0;
        bCurrentTrajectoryHasCollision = false;
        bPairedTrajectoryHasCollision = false;
        CurrentTrajectoryRecords.Empty();
        CurrentTrajectoryRecords.SetNum(TrajectorySteps);
        PairedTrajectoryRecords.Empty();
        if (bHasPairedTrajectory)
        {
            PairedTrajectoryRecords.SetNum(TrajectorySteps);
        }

        UE_LOG(LogTemp, Warning, TEXT("  [Trajectory %d] Type=%s, Planned %d actions for sample point %d (traj %d/%d)%s"),
            TrajectoryGlobalID, *TrajectoryTypeToString(CurrentTrajectoryType),
            TrajectorySteps, CurrentSampleIndex,
            CurrentTrajectoryIndexAtPoint + 1, TrajectoriesPerSamplePoint,
            bHasPairedTrajectory ? *FString::Printf(TEXT(", paired=%d"), PairedTrajectoryGlobalID) : TEXT(""));

        CurrentPhase = ECollectorPhase::Traj_ExecuteStep;
        return true;
    }

    // ================================================================
    // Trajectory mode - Phase T2: execute the current step and render its frames
    // ================================================================
    case ECollectorPhase::Traj_ExecuteStep:
    {
        if (CurrentTrajectoryStep >= TrajectorySteps)
        {
            // All steps done.
            CurrentPhase = ECollectorPhase::Traj_FinalizeTrajectory;
            return true;
        }

        // Choose the currently active action sequence and record array (primary or paired path).
        TArray<int32>& ActiveActions = bCurrentlyRenderingPaired ? PairedTrajectoryActions : CurrentTrajectoryActions;
        TArray<FTrajectoryStepRecord>& ActiveRecords = bCurrentlyRenderingPaired ? PairedTrajectoryRecords : CurrentTrajectoryRecords;

        // On the first frame of the current step, set up the interpolation start/end.
        if (CurrentStepFrameIndex == 0)
        {
            const int32 ActionIdx = ActiveActions[CurrentTrajectoryStep];
            const FActionDefinition& Action = ActionSpace[ActionIdx];

            ActionStartLocation = GetActorLocation();
            ActionStartRotation = GetActorRotation();

            FVector WorldDelta = ActionStartRotation.RotateVector(Action.DeltaPosition);
            ActionTargetLocation = ActionStartLocation + WorldDelta;
            ActionTargetRotation = ActionStartRotation + Action.DeltaRotation;

            // Record the step's starting state.
            FTrajectoryStepRecord& Record = ActiveRecords[CurrentTrajectoryStep];
            Record.ActionIndex = ActionIdx;
            Record.StartPosition = ActionStartLocation;
            Record.StartRotation = ActionStartRotation;
            Record.ExpectedEndPosition = ActionTargetLocation;
            Record.ExpectedEndRotation = ActionTargetRotation;

            // Reset the per-step frame-level collision tracking.
            bCurrentStepHadFrameCollision = false;
            LastStepHitResult = FHitResult();
        }

        // Compute interpolation alpha in [0, 1]
        float Alpha = static_cast<float>(CurrentStepFrameIndex) / static_cast<float>(FMath::Max(FramesPerAction - 1, 1));

        // Linear interpolation toward the target
        FVector DesiredLocation = FMath::Lerp(ActionStartLocation, ActionTargetLocation, Alpha);
        FRotator DesiredRotation = FMath::Lerp(ActionStartRotation, ActionTargetRotation, Alpha);

        // Move with sweep collision; pass a HitResult pointer to capture collision details.
        FHitResult FrameHit;
        bool bFrameCollision = MoveWithCollision(DesiredLocation, &FrameHit);
        // Rotation is unaffected by collision.
        SetActorRotation(DesiredRotation);

        // Cache collision info (keep the most recent HitResult within this step).
        if (bFrameCollision)
        {
            bCurrentStepHadFrameCollision = true;
            LastStepHitResult = FrameHit;
        }

        // Save the frame -- determine the save directory first (use a temporary path; the final
        // location is decided after the trajectory finishes).
        // Direct save path: reasoning/ or random_walk/.
        // But the final split is not yet known, so save to a staging path.
        FString StepSubDir = GetTrajectoryStepSubDir(CurrentTrajectoryStep);
        SaveFrame(StepSubDir, CurrentStepFrameIndex);

        CurrentStepFrameIndex++;

        // Last frame of the current step.
        if (CurrentStepFrameIndex >= FramesPerAction)
        {
            CurrentPhase = ECollectorPhase::Traj_AdvanceStep;
            return true;
        }

        return true; // continue rendering the next frame
    }

    // ================================================================
    // Trajectory mode - Phase T3: current step done; record results and advance
    // ================================================================
    case ECollectorPhase::Traj_AdvanceStep:
    {
        // Choose the currently active record array.
        TArray<FTrajectoryStepRecord>& ActiveRecords = bCurrentlyRenderingPaired ? PairedTrajectoryRecords : CurrentTrajectoryRecords;
        bool& bActiveHasCollision = bCurrentlyRenderingPaired ? bPairedTrajectoryHasCollision : bCurrentTrajectoryHasCollision;

        // Record the actual end pose.
        FTrajectoryStepRecord& Record = ActiveRecords[CurrentTrajectoryStep];
        Record.ActualEndPosition = GetActorLocation();
        Record.ActualEndRotation = GetActorRotation();

        // Collision detection: compare expected vs actual displacement.
        float Displacement = FVector::Dist(Record.ExpectedEndPosition, Record.ActualEndPosition);
        Record.CollisionDisplacement = Displacement;
        Record.bCollisionOccurred = (Displacement > CollisionThreshold);

        if (Record.bCollisionOccurred)
        {
            bActiveHasCollision = true;

            // Extract collision details from the cached HitResult.
            if (bCurrentStepHadFrameCollision)
            {
                AActor* HitActor = LastStepHitResult.GetActor();
                UPrimitiveComponent* HitComp = LastStepHitResult.GetComponent();

                Record.CollisionActorName = HitActor ? HitActor->GetName() : TEXT("(unknown)");
                Record.CollisionComponentName = HitComp ? HitComp->GetName() : TEXT("(unknown)");
                Record.CollisionImpactPoint = LastStepHitResult.ImpactPoint;
                Record.CollisionNormal = LastStepHitResult.ImpactNormal;

                // Try to get the physical-material name.
                Record.CollisionPhysMaterial = TEXT("(none)");
                if (LastStepHitResult.PhysMaterial.IsValid())
                {
                    Record.CollisionPhysMaterial = LastStepHitResult.PhysMaterial->GetName();
                }

                // ---- Detailed collision log (Output Log + file append) ----
                FString PairedTag = bCurrentlyRenderingPaired ? TEXT(" [paired]") : TEXT("");
                FString TrajectoryTag = FString::Printf(TEXT("Traj %d%s"), TrajectoryGlobalID, *PairedTag);

                UE_LOG(LogTemp, Warning, TEXT("    ┌── COLLISION at Step %d/%d%s ──"),
                    CurrentTrajectoryStep + 1, TrajectorySteps, *PairedTag);
                UE_LOG(LogTemp, Warning, TEXT("    │ Action:       %s (idx=%d)"),
                    *ActionSpace[Record.ActionIndex].ActionName, Record.ActionIndex);
                UE_LOG(LogTemp, Warning, TEXT("    │ Displacement: %.2f cm (threshold=%.1f)"),
                    Displacement, CollisionThreshold);
                UE_LOG(LogTemp, Warning, TEXT("    │ Hit Actor:    %s"), *Record.CollisionActorName);
                UE_LOG(LogTemp, Warning, TEXT("    │ Hit Comp:     %s"), *Record.CollisionComponentName);
                UE_LOG(LogTemp, Warning, TEXT("    │ Phys Mat:     %s"), *Record.CollisionPhysMaterial);
                UE_LOG(LogTemp, Warning, TEXT("    │ Impact Point: (%.1f, %.1f, %.1f)"),
                    Record.CollisionImpactPoint.X, Record.CollisionImpactPoint.Y, Record.CollisionImpactPoint.Z);
                UE_LOG(LogTemp, Warning, TEXT("    │ Normal:       (%.2f, %.2f, %.2f)"),
                    Record.CollisionNormal.X, Record.CollisionNormal.Y, Record.CollisionNormal.Z);
                UE_LOG(LogTemp, Warning, TEXT("    │ Agent Pos:    (%.1f, %.1f, %.1f) → expected (%.1f, %.1f, %.1f) → actual (%.1f, %.1f, %.1f)"),
                    Record.StartPosition.X, Record.StartPosition.Y, Record.StartPosition.Z,
                    Record.ExpectedEndPosition.X, Record.ExpectedEndPosition.Y, Record.ExpectedEndPosition.Z,
                    Record.ActualEndPosition.X, Record.ActualEndPosition.Y, Record.ActualEndPosition.Z);
                UE_LOG(LogTemp, Warning, TEXT("    └────────────────────────────"));

                // ---- Append the collision entry to collision_log.txt ----
                TotalCollisionCount++;
                FString Entry;
                Entry += FString::Printf(TEXT("===== Collision #%d =====\r\n"), TotalCollisionCount);
                Entry += FString::Printf(TEXT("  Trajectory:    %s\r\n"), *TrajectoryTag);
                Entry += FString::Printf(TEXT("  Step:          %d / %d\r\n"), CurrentTrajectoryStep + 1, TrajectorySteps);
                Entry += FString::Printf(TEXT("  Action:        %s (idx=%d)\r\n"), *ActionSpace[Record.ActionIndex].ActionName, Record.ActionIndex);
                Entry += FString::Printf(TEXT("  Displacement:  %.2f cm (threshold=%.1f)\r\n"), Displacement, CollisionThreshold);
                Entry += FString::Printf(TEXT("  Hit Actor:     %s\r\n"), *Record.CollisionActorName);
                Entry += FString::Printf(TEXT("  Hit Component: %s\r\n"), *Record.CollisionComponentName);
                Entry += FString::Printf(TEXT("  Phys Material: %s\r\n"), *Record.CollisionPhysMaterial);
                Entry += FString::Printf(TEXT("  Impact Point:  (%.1f, %.1f, %.1f)\r\n"),
                    Record.CollisionImpactPoint.X, Record.CollisionImpactPoint.Y, Record.CollisionImpactPoint.Z);
                Entry += FString::Printf(TEXT("  Normal:        (%.2f, %.2f, %.2f)\r\n"),
                    Record.CollisionNormal.X, Record.CollisionNormal.Y, Record.CollisionNormal.Z);
                Entry += FString::Printf(TEXT("  Agent Start:   (%.1f, %.1f, %.1f)\r\n"),
                    Record.StartPosition.X, Record.StartPosition.Y, Record.StartPosition.Z);
                Entry += FString::Printf(TEXT("  Agent Expected:(%.1f, %.1f, %.1f)\r\n"),
                    Record.ExpectedEndPosition.X, Record.ExpectedEndPosition.Y, Record.ExpectedEndPosition.Z);
                Entry += FString::Printf(TEXT("  Agent Actual:  (%.1f, %.1f, %.1f)\r\n"),
                    Record.ActualEndPosition.X, Record.ActualEndPosition.Y, Record.ActualEndPosition.Z);
                Entry += TEXT("\r\n");
                FFileHelper::SaveStringToFile(Entry, *CollisionLogPath,
                    FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append);
            }
            else
            {
                // The frame-level sweep did not report a collision but the displacement exceeds the
                // threshold (rare: e.g. push-back after wall penetration).
                Record.CollisionActorName = TEXT("(no sweep hit)");
                UE_LOG(LogTemp, Warning, TEXT("    Step %d/%d%s: COLLISION (action=%s, displacement=%.2f cm) — no sweep hit detected (possible push-back)"),
                    CurrentTrajectoryStep + 1, TrajectorySteps,
                    bCurrentlyRenderingPaired ? TEXT(" [paired]") : TEXT(""),
                    *ActionSpace[Record.ActionIndex].ActionName,
                    Displacement);

                // Append to the log file dynamically.
                TotalCollisionCount++;
                FString Entry;
                Entry += FString::Printf(TEXT("===== Collision #%d =====\r\n"), TotalCollisionCount);
                Entry += FString::Printf(TEXT("  Trajectory:    Traj %d%s\r\n"), TrajectoryGlobalID,
                    bCurrentlyRenderingPaired ? TEXT(" [paired]") : TEXT(""));
                Entry += FString::Printf(TEXT("  Step:          %d / %d\r\n"), CurrentTrajectoryStep + 1, TrajectorySteps);
                Entry += FString::Printf(TEXT("  Action:        %s (idx=%d)\r\n"), *ActionSpace[Record.ActionIndex].ActionName, Record.ActionIndex);
                Entry += FString::Printf(TEXT("  Displacement:  %.2f cm (threshold=%.1f)\r\n"), Displacement, CollisionThreshold);
                Entry += FString::Printf(TEXT("  NOTE:          No sweep hit detected (possible push-back)\r\n"));
                Entry += TEXT("\r\n");
                FFileHelper::SaveStringToFile(Entry, *CollisionLogPath,
                    FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append);
            }
        }

        // Advance to the next step.
        CurrentTrajectoryStep++;
        CurrentStepFrameIndex = 0;
        CurrentPhase = ECollectorPhase::Traj_ExecuteStep;
        return true;
    }

    // ================================================================
    // Trajectory mode - Phase T4: trajectory done; save metadata and split
    // ================================================================
    case ECollectorPhase::Traj_FinalizeTrajectory:
    {
        // Wait for the current trajectory's async writes to complete before moving the directory,
        // otherwise files may not be fully written yet.
        if (PendingAsyncWrites.GetValue() > 0)
        {
            return false; // re-check on the next Tick
        }

        // ----------------------------------------------------------------
        // Paired-trajectory flow control:
        //   Primary path P done -> if a paired path P' exists -> reset to root, render P'
        //   Paired path P' done -> save metadata for both and split together
        // ----------------------------------------------------------------
        if (!bCurrentlyRenderingPaired && bHasPairedTrajectory)
        {
            // ---- Primary path P just finished; need to render the paired path P' next ----
            // Note: do NOT move the primary's staging directory here!
            // We do not yet know whether the paired path will collide; both must complete before splitting together.

            int32 PrimaryCollisionCount = 0;
            for (const FTrajectoryStepRecord& R : CurrentTrajectoryRecords)
            {
                if (R.bCollisionOccurred) PrimaryCollisionCount++;
            }

            UE_LOG(LogTemp, Warning, TEXT("  [Trajectory %d] Primary path done (collisions: %d/%d), now rendering paired path %d..."),
                TrajectoryGlobalID,
                PrimaryCollisionCount, TrajectorySteps, PairedTrajectoryGlobalID);

            // Reset to root state and start rendering the paired path.
            SetActorLocation(OriginalLocation);
            SetActorRotation(OriginalRotation);

            // Switch to paired-path mode.
            bCurrentlyRenderingPaired = true;
            CurrentTrajectoryStep = 0;
            CurrentStepFrameIndex = 0;

            // Temporarily swap TrajectoryGlobalID so SaveFrame uses the paired path's staging directory.
            // (SaveFrame relies on TrajectoryGlobalID to determine the staging path.)
            TrajectoryGlobalID = PairedTrajectoryGlobalID;

            WaitFrameCounter = 0;
            CurrentPhase = ECollectorPhase::Traj_ExecuteStep;
            return true;
        }

        // ---- Trajectory is fully done (no paired path, or paired path also rendered) ----

        // Paired-trajectory split consistency: if a paired path exists, either collision sends both to random_walk.
        // This guarantees that paired trajectories in reasoning/ are always complete.
        if (bHasPairedTrajectory)
        {
            if (bCurrentTrajectoryHasCollision || bPairedTrajectoryHasCollision)
            {
                bCurrentTrajectoryHasCollision = true;
                bPairedTrajectoryHasCollision = true;
            }
        }

        // Restore TrajectoryGlobalID to the primary's ID (it was temporarily swapped during paired rendering).
        int32 PrimaryTrajectoryGlobalID;
        if (bCurrentlyRenderingPaired)
        {
            PrimaryTrajectoryGlobalID = PairedTrajectoryGlobalID - 1;
            TrajectoryGlobalID = PrimaryTrajectoryGlobalID;
        }
        else
        {
            PrimaryTrajectoryGlobalID = TrajectoryGlobalID;
        }

        // Move primary path: staging -> final
        {
            FString PrimaryStagingDir = FString::Printf(TEXT("%sstaging/traj_%06d"), *RuntimeSaveRoot, PrimaryTrajectoryGlobalID);
            FString PrimaryFinalDir = GetTrajectoryPath(bCurrentTrajectoryHasCollision);

            IPlatformFile& PlatformFile0 = FPlatformFileManager::Get().GetPlatformFile();
            EnsureDirectoryExists(PrimaryFinalDir);
            PlatformFile0.CopyDirectoryTree(*PrimaryFinalDir, *PrimaryStagingDir, true);
            PlatformFile0.DeleteDirectoryRecursively(*PrimaryStagingDir);

            FString Category = bCurrentTrajectoryHasCollision ? TEXT("random_walk") : TEXT("reasoning");
            UE_LOG(LogTemp, Warning, TEXT("  [Trajectory %d] Primary path → %s"),
                PrimaryTrajectoryGlobalID, *Category);
        }

        // Move paired path: staging -> final (if any)
        if (bCurrentlyRenderingPaired)
        {
            int32 PairedCollisionCount = 0;
            for (const FTrajectoryStepRecord& R : PairedTrajectoryRecords)
            {
                if (R.bCollisionOccurred) PairedCollisionCount++;
            }

            FString PairedStagingDir = FString::Printf(TEXT("%sstaging/traj_%06d"), *RuntimeSaveRoot, PairedTrajectoryGlobalID);

            // Temporarily set TrajectoryGlobalID to the paired ID so GetTrajectoryPath builds the correct path.
            TrajectoryGlobalID = PairedTrajectoryGlobalID;
            FString PairedFinalDir = GetTrajectoryPath(bPairedTrajectoryHasCollision);

            IPlatformFile& PlatformFile2 = FPlatformFileManager::Get().GetPlatformFile();
            EnsureDirectoryExists(PairedFinalDir);
            PlatformFile2.CopyDirectoryTree(*PairedFinalDir, *PairedStagingDir, true);
            PlatformFile2.DeleteDirectoryRecursively(*PairedStagingDir);

            FString PairedCategory = bPairedTrajectoryHasCollision ? TEXT("random_walk") : TEXT("reasoning");
            UE_LOG(LogTemp, Warning, TEXT("  [Trajectory %d] Paired path → %s (collisions: %d/%d)"),
                PairedTrajectoryGlobalID, *PairedCategory, PairedCollisionCount, TrajectorySteps);

            // Restore to the primary's ID.
            TrajectoryGlobalID = PrimaryTrajectoryGlobalID;
        }

        // ---- Save metadata for both ----

        // Count primary-path collisions.
        int32 CollisionCount = 0;
        for (const FTrajectoryStepRecord& R : CurrentTrajectoryRecords)
        {
            if (R.bCollisionOccurred) CollisionCount++;
        }

        // Primary metadata
        FString PrimaryMetaDir = GetTrajectoryPath(bCurrentTrajectoryHasCollision);
        SaveTrajectoryMetadataJSON(PrimaryMetaDir);

        // Update primary stats.
        if (bCurrentTrajectoryHasCollision)
        {
            TotalRandomWalkTrajectories++;
        }
        else
        {
            TotalReasoningTrajectories++;
        }

        FString Category = bCurrentTrajectoryHasCollision ? TEXT("random_walk") : TEXT("reasoning");
        UE_LOG(LogTemp, Warning, TEXT("  [Trajectory %d] DONE → %s (collisions: %d/%d, type=%s)"),
            TrajectoryGlobalID, *Category, CollisionCount, TrajectorySteps,
            *TrajectoryTypeToString(CurrentTrajectoryType));

        TrajectoryGlobalID++;

        // Paired-path metadata (if any).
        if (bHasPairedTrajectory)
        {
            // Set TrajectoryGlobalID = PairedTrajectoryGlobalID so GetTrajectoryPath is correct.
            TrajectoryGlobalID = PairedTrajectoryGlobalID;

            FString PairedMetaDir = GetTrajectoryPath(bPairedTrajectoryHasCollision);
            SaveTrajectoryMetadataJSON(PairedMetaDir);

            // Update paired stats.
            if (bPairedTrajectoryHasCollision)
            {
                TotalRandomWalkTrajectories++;
            }
            else
            {
                TotalReasoningTrajectories++;
            }

            FString PairedCategory = bPairedTrajectoryHasCollision ? TEXT("random_walk") : TEXT("reasoning");
            UE_LOG(LogTemp, Warning, TEXT("  [Trajectory %d] DONE (paired) → %s (type=%s)"),
                PairedTrajectoryGlobalID, *PairedCategory,
                *TrajectoryTypeToString(CurrentTrajectoryType));

            // Restore and skip past the paired ID.
            TrajectoryGlobalID = PairedTrajectoryGlobalID + 1;
        }

        // Option B: restore Hard tier scale jitter (so the next trajectory starts from the default scale)
        if (bActionSpaceIsJittered)
        {
            RestoreActionSpaceScale();
        }

        CurrentTrajectoryIndexAtPoint++;

        // Check whether the current sample point still needs more trajectories.
        if (CurrentTrajectoryIndexAtPoint < TrajectoriesPerSamplePoint)
        {
            // Reset to root state and start the next trajectory.
            SetActorLocation(OriginalLocation);
            SetActorRotation(OriginalRotation);
            WaitFrameCounter = 0;
            CurrentPhase = ECollectorPhase::Traj_PlanTrajectory;
        }
        else
        {
            // Current sample point done.
            CurrentPhase = ECollectorPhase::AdvanceToNextSamplePoint;
        }
        return true;
    }

    // ================================================================
    // Phase 5: current sample point done; advance to the next
    // ================================================================
    case ECollectorPhase::AdvanceToNextSamplePoint:
    {
        // Reset
        SetActorLocation(OriginalLocation);
        SetActorRotation(OriginalRotation);

        // Save the current sample point's metadata (Counterfactual Tree mode only).
        if (CollectorMode == ECollectorMode::CounterfactualTree)
        {
            SaveMetadataJSON();
        }

        UE_LOG(LogTemp, Warning, TEXT("[Sample %d/%d] DONE. State=%d"),
            CurrentSampleIndex + 1, SamplePoints.Num(), CurrentStateID);

        // Advance
        CurrentSampleIndex++;
        CurrentStateID++;
        CurrentPhase = ECollectorPhase::TeleportToSamplePoint;
        return true;
    }

    // ================================================================
    // Phase 6: all done
    // ================================================================
    case ECollectorPhase::Completed:
    {
        // Wait for all async writes to complete.
        if (PendingAsyncWrites.GetValue() > 0)
        {
            UE_LOG(LogTemp, Warning, TEXT("[Completion] Waiting for %d async writes to finish..."), PendingAsyncWrites.GetValue());
            return false; // re-check on the next Tick
        }

        // Save the global manifest.
        SaveGlobalManifest();

        // Append the trailing summary to the collision log.
        {
            FString Footer;
            Footer += TEXT("================================================================\r\n");
            Footer += FString::Printf(TEXT("  Collection Finished: %s\r\n"), *FDateTime::Now().ToString());
            Footer += FString::Printf(TEXT("  Total Collisions: %d\r\n"), TotalCollisionCount);
            Footer += FString::Printf(TEXT("  Total Trajectories: %d\r\n"), TrajectoryGlobalID);
            Footer += FString::Printf(TEXT("  Reasoning: %d | Random Walk: %d\r\n"), TotalReasoningTrajectories, TotalRandomWalkTrajectories);
            if (TrajectoryGlobalID > 0)
            {
                Footer += FString::Printf(TEXT("  Collision Rate: %.1f%% of trajectories had collisions\r\n"),
                    100.0f * TotalRandomWalkTrajectories / TrajectoryGlobalID);
            }
            Footer += TEXT("================================================================\r\n");
            FFileHelper::SaveStringToFile(Footer, *CollisionLogPath,
                FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append);

            UE_LOG(LogTemp, Warning, TEXT("Collision log finalized: %s (%d collisions)"), *CollisionLogPath, TotalCollisionCount);
        }

        // Trajectory mode: clean up the staging directory.
        if (CollectorMode == ECollectorMode::TrajectoryGraph)
        {
            IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
            FString StagingRoot = RuntimeSaveRoot + TEXT("staging/");
            PlatformFile.DeleteDirectoryRecursively(*StagingRoot);
        }

        UE_LOG(LogTemp, Warning, TEXT("============================================================"));
        UE_LOG(LogTemp, Warning, TEXT("=== ALL DATA COLLECTION COMPLETED ==="));
        UE_LOG(LogTemp, Warning, TEXT("  Episode: %d"), EpisodeID);
        UE_LOG(LogTemp, Warning, TEXT("  Sample points processed: %d"), SamplePoints.Num());
        UE_LOG(LogTemp, Warning, TEXT("  States collected: %d"), CurrentStateID);
        if (CollectorMode == ECollectorMode::CounterfactualTree)
        {
            UE_LOG(LogTemp, Warning, TEXT("  Total videos: %d"), CurrentStateID * (1 + ActionSpace.Num()));
            UE_LOG(LogTemp, Warning, TEXT("  Total frames: %d"), CurrentStateID * (1 + ActionSpace.Num()) * TotalFramesPerVideo);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("  Total trajectories: %d"), TrajectoryGlobalID);
            UE_LOG(LogTemp, Warning, TEXT("    → reasoning/: %d (collision-free)"), TotalReasoningTrajectories);
            UE_LOG(LogTemp, Warning, TEXT("    → random_walk/: %d (with collisions)"), TotalRandomWalkTrajectories);
            UE_LOG(LogTemp, Warning, TEXT("  Total frames: %d"), TrajectoryGlobalID * TrajectorySteps * FramesPerAction);
            if (TrajectoryGlobalID > 0)
            {
                float ReasoningPct = 100.0f * TotalReasoningTrajectories / TrajectoryGlobalID;
                UE_LOG(LogTemp, Warning, TEXT("  Reasoning ratio: %.1f%%"), ReasoningPct);
            }

            // Per-trajectory-type stats.
            UE_LOG(LogTemp, Warning, TEXT("  Trajectory type breakdown:"));
            ETrajectoryType AllTypes[] = {
                ETrajectoryType::Seed, ETrajectoryType::Inverse,
                ETrajectoryType::Loop, ETrajectoryType::Equivalence,
                ETrajectoryType::Hard_Loop,
                ETrajectoryType::Hard_Equivalence, ETrajectoryType::Hard_Inverse
            };
            constexpr int32 NumAllTypes = sizeof(AllTypes) / sizeof(AllTypes[0]);
            for (int32 ti = 0; ti < NumAllTypes; ti++)
            {
                int32 TypeCount = TrajectoryTypeCount.Contains(AllTypes[ti]) ? TrajectoryTypeCount[AllTypes[ti]] : 0;
                UE_LOG(LogTemp, Warning, TEXT("    → %s: %d"),
                    *TrajectoryTypeToString(AllTypes[ti]), TypeCount);
            }

            // ===== Plan retry efficiency report =====
            UE_LOG(LogTemp, Warning, TEXT(""));
            UE_LOG(LogTemp, Warning, TEXT("  ===== Planning Retry Report (MaxRetries=%d, YawInterval=%d) ====="), MaxPlanRetries, RetryYawRotateInterval);
            UE_LOG(LogTemp, Warning, TEXT("  SkipFailedPlans=%s | Total skips (no render): %d"),
                bSkipFailedPlans ? TEXT("true") : TEXT("false"), TotalPlanSkips);
            for (int32 ti = 0; ti < NumAllTypes; ti++)
            {
                int32 Attempts = PlanAttemptCount.Contains(AllTypes[ti]) ? PlanAttemptCount[AllTypes[ti]] : 0;
                int32 Successes = PlanSuccessCount.Contains(AllTypes[ti]) ? PlanSuccessCount[AllTypes[ti]] : 0;
                if (Attempts > 0)
                {
                    float Rate = (float)Successes / (float)Attempts * 100.0f;
                    UE_LOG(LogTemp, Warning, TEXT("    %s: %d attempts -> %d successes (%.1f%% per-attempt, avg %.1f retries/success)"),
                        *TrajectoryTypeToString(AllTypes[ti]), Attempts, Successes, Rate,
                        Successes > 0 ? (float)Attempts / (float)Successes : 0.0f);
                }
            }
        }
        UE_LOG(LogTemp, Warning, TEXT("  Output directory: %s"), *RuntimeSaveRoot);
        UE_LOG(LogTemp, Warning, TEXT("============================================================"));

        SetActorTickEnabled(false);
        return false;
    }
    }

    return false;
}

// ============================================================================
// Save a single frame (RGB + optional depth + optional optical flow)
//
// Performance strategy:
//   1. Trigger CaptureScene() for everything first so the GPU pipeline can render in parallel.
//   2. Then ReadPixels in batch (GPU->CPU transfer stalls; concentrating it reduces overhead).
//   3. Encoding and file writing run on background threads.
//
// SubDirectory is relative to BasePath.
// In Counterfactual Tree mode  BasePath = GetEpisodePath()
// In Trajectory mode           BasePath = trajectory directory (reasoning/traj_xxx/ or random_walk/traj_xxx/)
// ============================================================================
void ADataCollector::SaveFrame(const FString& SubDirectory, int32 FrameIndex)
{
    if (!SceneCapture || !SceneCapture->TextureTarget) return;

    // Determine the base path.
    FString BasePath;
    if (CollectorMode == ECollectorMode::TrajectoryGraph)
    {
        // Trajectory mode: the final split is not yet known, so use a staging path.
        // Frames are saved as they are rendered, before all steps complete.
        // Strategy: always save to reasoning/ first, and move to random_walk/ in FinalizeTrajectory if needed.
        // Better strategy: use a fixed staging path.
        BasePath = FString::Printf(TEXT("%sstaging/traj_%06d/"), *RuntimeSaveRoot, TrajectoryGlobalID);
    }
    else
    {
        BasePath = GetEpisodePath();
    }

    FString FullSubDir = BasePath / SubDirectory;
    EnsureDirectoryExists(FullSubDir);

    // ---- Step 1: trigger GPU rendering for all captures ----
    // The three SceneCaptures are attached to the same actor and share the same pose.
    // Calling CaptureScene() consecutively lets the GPU pipeline overlap them.

    SceneCapture->CaptureScene();

    if (bCaptureDepth && DepthCapture && DepthCapture->TextureTarget)
    {
        DepthCapture->CaptureScene();
    }

    if (bCaptureOpticalFlow && VelocityCapture && VelocityCapture->TextureTarget)
    {
        VelocityCapture->CaptureScene();
    }

    // ---- Step 2: read back RGB pixels and save ----
    {
        UTextureRenderTarget2D* RT = SceneCapture->TextureTarget;
        FTextureRenderTargetResource* RTResource = RT->GameThread_GetRenderTargetResource();
        if (!RTResource) return;

        const int32 Width = ResolutionX;
        const int32 Height = ResolutionY;

        // Read linear color values from the RGBA16F HDR RenderTarget.
        // SCS_FinalColorHDR has already done tonemapping inside the rendering pipeline,
        // so the output is linear-space LDR (range 0..1) and needs sRGB conversion.
        TArray<FLinearColor> LinearPixels;
        LinearPixels.SetNumUninitialized(Width * Height);
        RTResource->ReadLinearColorPixels(LinearPixels);

        // Convert LinearColor -> FColor (automatically applies sRGB gamma correction).
        TArray<FColor> Pixels;
        Pixels.SetNumUninitialized(Width * Height);
        for (int32 i = 0; i < Width * Height; i++)
        {
            Pixels[i] = LinearPixels[i].ToFColor(true); // true = apply sRGB gamma
        }

        if (ImageFormat == EImageOutputFormat::JPEG)
        {
            FString FileName = FString::Printf(TEXT("frame_%04d.jpg"), FrameIndex);
            FString FullPath = BasePath / SubDirectory / FileName;

            IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
            TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::JPEG);

            if (ImageWrapper.IsValid())
            {
                ImageWrapper->SetRaw(
                    Pixels.GetData(),
                    Pixels.Num() * sizeof(FColor),
                    Width, Height,
                    ERGBFormat::BGRA, 8
                );

                const TArray64<uint8>& CompressedData = ImageWrapper->GetCompressed(JpegQuality);

                TArray<uint8> FileData;
                FileData.SetNumUninitialized(CompressedData.Num());
                FMemory::Memcpy(FileData.GetData(), CompressedData.GetData(), CompressedData.Num());

                AsyncSaveFile(MoveTemp(FileData), FullPath);
            }
        }
        else
        {
            FString FileName = FString::Printf(TEXT("frame_%04d.bmp"), FrameIndex);
            FString FullPath = BasePath / SubDirectory / FileName;

            const int32 RowBytes = Width * 3;
            const int32 PaddedRowBytes = (RowBytes + 3) & ~3;
            const int32 PixelDataSize = PaddedRowBytes * Height;
            const int32 FileSize = 14 + 40 + PixelDataSize;

            TArray<uint8> BMPData;
            BMPData.SetNumZeroed(FileSize);
            uint8* Data = BMPData.GetData();

            Data[0] = 'B'; Data[1] = 'M';
            *reinterpret_cast<int32*>(&Data[2]) = FileSize;
            *reinterpret_cast<int32*>(&Data[10]) = 14 + 40;
            *reinterpret_cast<int32*>(&Data[14]) = 40;
            *reinterpret_cast<int32*>(&Data[18]) = Width;
            *reinterpret_cast<int32*>(&Data[22]) = Height;
            *reinterpret_cast<int16*>(&Data[26]) = 1;
            *reinterpret_cast<int16*>(&Data[28]) = 24;
            *reinterpret_cast<int32*>(&Data[34]) = PixelDataSize;

            uint8* PixelStart = Data + 14 + 40;
            for (int32 Y = 0; Y < Height; Y++)
            {
                const FColor* SrcRow = &Pixels[(Height - 1 - Y) * Width];
                uint8* DstRow = PixelStart + Y * PaddedRowBytes;
                for (int32 X = 0; X < Width; X++)
                {
                    DstRow[X * 3 + 0] = SrcRow[X].B;
                    DstRow[X * 3 + 1] = SrcRow[X].G;
                    DstRow[X * 3 + 2] = SrcRow[X].R;
                }
            }

            AsyncSaveFile(MoveTemp(BMPData), FullPath);
        }
    }

    // ---- Step 3: read back depth and save ----
    if (bCaptureDepth && DepthCapture && DepthCapture->TextureTarget)
    {
        SaveDepthFrame(BasePath / SubDirectory, FrameIndex);
    }

    // ---- Step 4: read back optical flow and save ----
    if (bCaptureOpticalFlow && VelocityCapture && VelocityCapture->TextureTarget)
    {
        SaveVelocityFrame(BasePath / SubDirectory, FrameIndex);
    }
}

// ============================================================================
// Save a depth frame as a 16-bit PNG grayscale image.
//
// Encoding:
//   depth_cm = scene depth (in cm, from SceneDepth output)
//   pixel_value = clamp(depth_cm / DepthMaxDistance, 0, 1) * 65535
//   -> 16-bit single-channel PNG
//
// Precision: with DepthMaxDistance = 10000 cm,
//   65535 levels / 10000 cm = ~0.15 cm per gradation -- plenty for training.
// ============================================================================
void ADataCollector::SaveDepthFrame(const FString& SubDirectory, int32 FrameIndex)
{
    UTextureRenderTarget2D* RT = DepthCapture->TextureTarget;
    FTextureRenderTargetResource* RTResource = RT->GameThread_GetRenderTargetResource();
    if (!RTResource) return;

    const int32 Width = ResolutionX;
    const int32 Height = ResolutionY;

    // R32F RenderTarget -> read back as FLinearColor (4 floats per pixel)
    TArray<FLinearColor> FloatPixels;
    FloatPixels.SetNumUninitialized(Width * Height);
    RTResource->ReadLinearColorPixels(FloatPixels);

    // Normalize float depth to 16-bit and encode as PNG.
    // Allocate 16-bit grayscale data (2 bytes per pixel).
    TArray<uint8> RawData;
    RawData.SetNumUninitialized(Width * Height * 2);

    const float InvMaxDist = 1.0f / DepthMaxDistance;
    for (int32 i = 0; i < Width * Height; i++)
    {
        // SceneDepth outputs depth (cm) in the R channel.
        float DepthCm = FloatPixels[i].R;
        float Normalized = FMath::Clamp(DepthCm * InvMaxDist, 0.0f, 1.0f);
        uint16 Value = static_cast<uint16>(Normalized * 65535.0f);

        // Little-endian storage (PNG handles this correctly).
        RawData[i * 2 + 0] = static_cast<uint8>(Value & 0xFF);
        RawData[i * 2 + 1] = static_cast<uint8>((Value >> 8) & 0xFF);
    }

    // Encode as 16-bit PNG via IImageWrapper.
    IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
    TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);

    if (ImageWrapper.IsValid())
    {
        ImageWrapper->SetRaw(
            RawData.GetData(),
            RawData.Num(),
            Width, Height,
            ERGBFormat::Gray, 16
        );

        const TArray64<uint8>& CompressedData = ImageWrapper->GetCompressed(0); // PNG lossless

        TArray<uint8> FileData;
        FileData.SetNumUninitialized(CompressedData.Num());
        FMemory::Memcpy(FileData.GetData(), CompressedData.GetData(), CompressedData.Num());

        // Save into the depth/ subdirectory.
        FString DepthDir = SubDirectory / TEXT("depth");
        EnsureDirectoryExists(DepthDir);
        FString FileName = FString::Printf(TEXT("frame_%04d.png"), FrameIndex);
        FString FullPath = DepthDir / FileName;

        AsyncSaveFile(MoveTemp(FileData), FullPath);
    }
}

// ============================================================================
// Save the optical-flow / velocity buffer as a PNG.
//
// Source:
//   VelocityCapture is captured via the post-process material (M_VelocityPass).
//   The material reads UE5's Velocity Buffer through the SceneTexture:Velocity node.
//   The values in the Velocity Buffer are screen-space motion vectors.
//
// Encoding:
//   The post-process material outputs RGB: R = horizontal motion (U), G = vertical motion (V), B = optional.
//   The range of FloatPixels read back depends on the material's transform.
//   Default: the material outputs SceneTexture:Velocity's RG channels directly.
//   We map them to [0, 255] and save as RGBA PNG:
//     encoded = clamp((velocity + 1.0) * 0.5 * 255, 0, 255)
//
// Decoding: velocity = encoded / 255.0 * 2.0 - 1.0
//   Multiply by resolution to recover pixel displacement.
// ============================================================================
void ADataCollector::SaveVelocityFrame(const FString& SubDirectory, int32 FrameIndex)
{
    UTextureRenderTarget2D* RT = VelocityCapture->TextureTarget;
    FTextureRenderTargetResource* RTResource = RT->GameThread_GetRenderTargetResource();
    if (!RTResource) return;

    const int32 Width = ResolutionX;
    const int32 Height = ResolutionY;

    // RGBA16F -> read back as FLinearColor.
    TArray<FLinearColor> FloatPixels;
    FloatPixels.SetNumUninitialized(Width * Height);
    RTResource->ReadLinearColorPixels(FloatPixels);

    // Encode as 8-bit RGBA PNG (R = U, G = V, B = magnitude).
    // Map velocity [-1, 1] -> [0, 255].
    TArray<FColor> EncodedPixels;
    EncodedPixels.SetNumUninitialized(Width * Height);

    for (int32 i = 0; i < Width * Height; i++)
    {
        // The motion vector lives in the RG channels.
        float U = FloatPixels[i].R; // horizontal motion
        float V = FloatPixels[i].G; // vertical motion

        // Map [-1, 1] -> [0, 255]
        uint8 EncU = static_cast<uint8>(FMath::Clamp((U + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));
        uint8 EncV = static_cast<uint8>(FMath::Clamp((V + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));

        // The B channel stores motion magnitude (for visualization).
        float Magnitude = FMath::Sqrt(U * U + V * V);
        uint8 EncMag = static_cast<uint8>(FMath::Clamp(Magnitude * 127.5f, 0.0f, 255.0f));

        EncodedPixels[i] = FColor(EncU, EncV, EncMag, 255);
    }

    // Save losslessly as PNG (optical flow needs exact values).
    IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
    TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);

    if (ImageWrapper.IsValid())
    {
        ImageWrapper->SetRaw(
            EncodedPixels.GetData(),
            EncodedPixels.Num() * sizeof(FColor),
            Width, Height,
            ERGBFormat::BGRA, 8
        );

        const TArray64<uint8>& CompressedData = ImageWrapper->GetCompressed(0);

        TArray<uint8> FileData;
        FileData.SetNumUninitialized(CompressedData.Num());
        FMemory::Memcpy(FileData.GetData(), CompressedData.GetData(), CompressedData.Num());

        // Save into the flow/ subdirectory.
        FString FlowDir = SubDirectory / TEXT("flow");
        EnsureDirectoryExists(FlowDir);
        FString FileName = FString::Printf(TEXT("frame_%04d.png"), FrameIndex);
        FString FullPath = FlowDir / FileName;

        AsyncSaveFile(MoveTemp(FileData), FullPath);
    }
}

// ============================================================================
// Generic helper for asynchronous file writing
// ============================================================================
void ADataCollector::AsyncSaveFile(TArray<uint8>&& FileData, const FString& FullPath)
{
    if (bAsyncFileWrite)
    {
        PendingAsyncWrites.Increment();
        AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask,
            [Data = MoveTemp(FileData), Path = FullPath, this]()
            {
                FFileHelper::SaveArrayToFile(Data, *Path);
                PendingAsyncWrites.Decrement();
            });
    }
    else
    {
        FFileHelper::SaveArrayToFile(FileData, *FullPath);
    }
}

// ============================================================================
// Path utilities
// ============================================================================
FString ADataCollector::GetEpisodePath() const
{
    return FString::Printf(TEXT("%sepisode_%04d/state_%04d"), *RuntimeSaveRoot, EpisodeID, CurrentStateID);
}

FString ADataCollector::GetContextSubDir() const
{
    return FString::Printf(TEXT("S%04d_context"), CurrentStateID);
}

FString ADataCollector::GetActionSubDir(int32 ActionIdx) const
{
    const FString& ActionName = ActionSpace[ActionIdx].ActionName;
    return FString::Printf(TEXT("S%04d_A%02d_%s"), CurrentStateID, ActionIdx, *ActionName);
}

void ADataCollector::EnsureDirectoryExists(const FString& Path)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*Path))
    {
        PlatformFile.CreateDirectoryTree(*Path);
    }
}

// ============================================================================
// Save the metadata JSON for the current sample point.
// ============================================================================
void ADataCollector::SaveMetadataJSON()
{
    FString JSON;
    JSON += TEXT("{\n");

    JSON += FString::Printf(TEXT("  \"episode_id\": %d,\n"), EpisodeID);
    JSON += FString::Printf(TEXT("  \"state_id\": %d,\n"), CurrentStateID);
    JSON += TEXT("  \"data_type\": \"counterfactual_tree\",\n\n");

    // Render config
    JSON += TEXT("  \"render_config\": {\n");
    JSON += FString::Printf(TEXT("    \"resolution\": [%d, %d],\n"), ResolutionX, ResolutionY);
    JSON += FString::Printf(TEXT("    \"fps\": %d,\n"), FramesPerSecond);
    JSON += FString::Printf(TEXT("    \"duration_seconds\": %.2f,\n"), VideoDurationSeconds);
    JSON += FString::Printf(TEXT("    \"total_frames_per_video\": %d,\n"), TotalFramesPerVideo);
    JSON += FString::Printf(TEXT("    \"fov\": %.1f,\n"), CameraFOV);
    JSON += FString::Printf(TEXT("    \"image_format\": \"%s\",\n"),
        ImageFormat == EImageOutputFormat::JPEG ? TEXT("jpeg") : TEXT("bmp"));
    JSON += FString::Printf(TEXT("    \"modalities\": [\"rgb\"%s%s]\n"),
        bCaptureDepth ? TEXT(", \"depth\"") : TEXT(""),
        bCaptureOpticalFlow ? TEXT(", \"optical_flow\"") : TEXT(""));
    JSON += TEXT("  },\n\n");

    // Agent config
    JSON += TEXT("  \"agent_config\": {\n");
    JSON += FString::Printf(TEXT("    \"capsule_radius\": %.1f,\n"), CapsuleRadius);
    JSON += FString::Printf(TEXT("    \"capsule_half_height\": %.1f,\n"), CapsuleHalfHeight);
    JSON += FString::Printf(TEXT("    \"camera_offset\": [%.1f, %.1f, %.1f],\n"),
        CameraOffset.X, CameraOffset.Y, CameraOffset.Z);
    JSON += TEXT("    \"collision_enabled\": true\n");
    JSON += TEXT("  },\n\n");

    // Environment
    JSON += TEXT("  \"environment\": {\n");
    JSON += FString::Printf(TEXT("    \"lighting_frozen\": %s,\n"), bFreezeLighting ? TEXT("true") : TEXT("false"));
    JSON += FString::Printf(TEXT("    \"sun_pitch\": %.1f,\n"), SunPitch);
    JSON += FString::Printf(TEXT("    \"sun_yaw\": %.1f\n"), SunYaw);
    JSON += TEXT("  },\n\n");

    // State
    JSON += TEXT("  \"state\": {\n");
    JSON += FString::Printf(TEXT("    \"position\": [%.4f, %.4f, %.4f],\n"),
        OriginalLocation.X, OriginalLocation.Y, OriginalLocation.Z);
    JSON += FString::Printf(TEXT("    \"rotation\": [%.4f, %.4f, %.4f],\n"),
        OriginalRotation.Pitch, OriginalRotation.Yaw, OriginalRotation.Roll);
    JSON += FString::Printf(TEXT("    \"context_video_dir\": \"%s\"\n"), *GetContextSubDir());
    JSON += TEXT("  },\n\n");

    // Action space
    JSON += TEXT("  \"action_space\": [\n");
    for (int32 i = 0; i < ActionSpace.Num(); i++)
    {
        const FActionDefinition& A = ActionSpace[i];
        JSON += TEXT("    {\n");
        JSON += FString::Printf(TEXT("      \"action_id\": %d,\n"), i);
        JSON += FString::Printf(TEXT("      \"action_name\": \"%s\",\n"), *A.ActionName);
        JSON += FString::Printf(TEXT("      \"delta_position\": [%.4f, %.4f, %.4f],\n"),
            A.DeltaPosition.X, A.DeltaPosition.Y, A.DeltaPosition.Z);
        JSON += FString::Printf(TEXT("      \"delta_rotation\": [%.4f, %.4f, %.4f],\n"),
            A.DeltaRotation.Pitch, A.DeltaRotation.Yaw, A.DeltaRotation.Roll);
        JSON += FString::Printf(TEXT("      \"video_dir\": \"%s\"\n"), *GetActionSubDir(i));
        JSON += (i < ActionSpace.Num() - 1) ? TEXT("    },\n") : TEXT("    }\n");
    }
    JSON += TEXT("  ],\n\n");

    // Tree structure
    JSON += TEXT("  \"tree_structure\": {\n");
    JSON += FString::Printf(TEXT("    \"root_node\": \"S%04d\",\n"), CurrentStateID);
    JSON += TEXT("    \"branches\": [\n");
    for (int32 i = 0; i < ActionSpace.Num(); i++)
    {
        JSON += FString::Printf(TEXT("      {\"parent\": \"S%04d\", \"action\": \"%s\", \"child_dir\": \"%s\"}"),
            CurrentStateID, *ActionSpace[i].ActionName, *GetActionSubDir(i));
        JSON += (i < ActionSpace.Num() - 1) ? TEXT(",\n") : TEXT("\n");
    }
    JSON += TEXT("    ]\n");
    JSON += TEXT("  }\n");

    JSON += TEXT("}\n");

    FString MetadataPath = GetEpisodePath() / TEXT("metadata.json");
    FFileHelper::SaveStringToFile(JSON, *MetadataPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}

// ============================================================================
// Save the global dataset manifest.
// Records all state info from this run so a dataloader can load everything at once.
// ============================================================================
void ADataCollector::SaveGlobalManifest()
{
    FString JSON;
    JSON += TEXT("{\n");
    JSON += FString::Printf(TEXT("  \"episode_id\": %d,\n"), EpisodeID);

    if (CollectorMode == ECollectorMode::CounterfactualTree)
    {
        // ---- Counterfactual Tree manifest ----
        JSON += FString::Printf(TEXT("  \"mode\": \"counterfactual_tree\",\n"));
        JSON += FString::Printf(TEXT("  \"total_states\": %d,\n"), CurrentStateID);
        JSON += FString::Printf(TEXT("  \"total_actions_per_state\": %d,\n"), ActionSpace.Num());
        JSON += FString::Printf(TEXT("  \"frames_per_video\": %d,\n"), TotalFramesPerVideo);

        // Sampling config
        JSON += TEXT("  \"sampling_config\": {\n");
        JSON += FString::Printf(TEXT("    \"range_x\": %.1f,\n"), SamplingRangeX);
        JSON += FString::Printf(TEXT("    \"range_y\": %.1f,\n"), SamplingRangeY);
        JSON += FString::Printf(TEXT("    \"interval\": %.1f,\n"), SamplingInterval);
        JSON += FString::Printf(TEXT("    \"num_yaw_samples\": %d,\n"), NumYawSamples);
        JSON += FString::Printf(TEXT("    \"skip_occluded\": %s\n"), bSkipOccludedPoints ? TEXT("true") : TEXT("false"));
        JSON += TEXT("  },\n\n");

        // All state indices
        JSON += TEXT("  \"states\": [\n");
        for (int32 i = 0; i < CurrentStateID; i++)
        {
            const FSamplePoint& SP = SamplePoints[i];
            JSON += FString::Printf(TEXT("    {\"state_id\": %d, \"position\": [%.4f, %.4f, %.4f], \"yaw\": %.4f, \"dir\": \"state_%04d\"}"),
                i, SP.Location.X, SP.Location.Y, SP.Location.Z, SP.Rotation.Yaw, i);
            JSON += (i < CurrentStateID - 1) ? TEXT(",\n") : TEXT("\n");
        }
        JSON += TEXT("  ]\n");
    }
    else
    {
        // ---- Trajectory mode manifest ----
        JSON += FString::Printf(TEXT("  \"mode\": \"trajectory_graph\",\n"));
        JSON += FString::Printf(TEXT("  \"total_sample_points\": %d,\n"), SamplePoints.Num());
        JSON += FString::Printf(TEXT("  \"trajectories_per_point\": %d,\n"), TrajectoriesPerSamplePoint);
        JSON += FString::Printf(TEXT("  \"total_trajectories\": %d,\n"), TrajectoryGlobalID);
        JSON += FString::Printf(TEXT("  \"reasoning_trajectories\": %d,\n"), TotalReasoningTrajectories);
        JSON += FString::Printf(TEXT("  \"random_walk_trajectories\": %d,\n"), TotalRandomWalkTrajectories);
        if (TrajectoryGlobalID > 0)
        {
            JSON += FString::Printf(TEXT("  \"reasoning_ratio\": %.4f,\n"),
                static_cast<float>(TotalReasoningTrajectories) / TrajectoryGlobalID);
        }

        // Trajectory parameters
        JSON += TEXT("  \"trajectory_config\": {\n");
        JSON += FString::Printf(TEXT("    \"steps_per_trajectory\": %d,\n"), TrajectorySteps);
        JSON += FString::Printf(TEXT("    \"frames_per_step\": %d,\n"), FramesPerAction);
        JSON += FString::Printf(TEXT("    \"total_frames_per_trajectory\": %d,\n"), TrajectorySteps * FramesPerAction);
        JSON += FString::Printf(TEXT("    \"collision_threshold\": %.1f,\n"), CollisionThreshold);
        JSON += FString::Printf(TEXT("    \"collision_avoidance\": %s,\n"), bCollisionAvoidanceSampling ? TEXT("true") : TEXT("false"));
        JSON += FString::Printf(TEXT("    \"max_consecutive_rotations\": %d\n"), MaxConsecutiveRotations);
        JSON += TEXT("  },\n\n");

        // Render config
        JSON += TEXT("  \"render_config\": {\n");
        JSON += FString::Printf(TEXT("    \"resolution\": [%d, %d],\n"), ResolutionX, ResolutionY);
        JSON += FString::Printf(TEXT("    \"fov\": %.1f,\n"), CameraFOV);
        JSON += FString::Printf(TEXT("    \"image_format\": \"%s\",\n"),
            ImageFormat == EImageOutputFormat::JPEG ? TEXT("jpeg") : TEXT("bmp"));
        JSON += FString::Printf(TEXT("    \"modalities\": [\"rgb\"%s%s]\n"),
            bCaptureDepth ? TEXT(", \"depth\"") : TEXT(""),
            bCaptureOpticalFlow ? TEXT(", \"optical_flow\"") : TEXT(""));
        JSON += TEXT("  },\n\n");

        // Action space
        JSON += TEXT("  \"action_space\": [\n");
        for (int32 i = 0; i < ActionSpace.Num(); i++)
        {
            const FActionDefinition& A = ActionSpace[i];
            JSON += FString::Printf(TEXT("    {\"id\": %d, \"name\": \"%s\", \"delta_pos\": [%.2f, %.2f, %.2f], \"delta_rot\": [%.2f, %.2f, %.2f]}"),
                i, *A.ActionName, A.DeltaPosition.X, A.DeltaPosition.Y, A.DeltaPosition.Z,
                A.DeltaRotation.Pitch, A.DeltaRotation.Yaw, A.DeltaRotation.Roll);
            JSON += (i < ActionSpace.Num() - 1) ? TEXT(",\n") : TEXT("\n");
        }
        JSON += TEXT("  ],\n\n");

        // Action sampling weights
        JSON += TEXT("  \"action_sampling_weights\": [");
        for (int32 i = 0; i < ActionSamplingWeights.Num(); i++)
        {
            JSON += FString::Printf(TEXT("%.2f"), ActionSamplingWeights[i]);
            if (i < ActionSamplingWeights.Num() - 1) JSON += TEXT(", ");
        }
        JSON += TEXT("],\n\n");

        // ==================== Structured trajectory stats ====================

        // Per-type sampling weight configuration
        JSON += TEXT("  \"structured_trajectory_config\": {\n");
        JSON += FString::Printf(TEXT("    \"weight_seed\": %.1f,\n"), WeightSeed);
        JSON += FString::Printf(TEXT("    \"weight_inverse\": %.1f,\n"), WeightInverse);
        JSON += FString::Printf(TEXT("    \"weight_loop\": %.1f,\n"), WeightLoop);
        JSON += FString::Printf(TEXT("    \"weight_equivalence\": %.1f,\n"), WeightEquivalence);
        JSON += FString::Printf(TEXT("    \"weight_hard_loop\": %.1f,\n"), WeightHardLoop);
        JSON += FString::Printf(TEXT("    \"weight_hard_equivalence\": %.1f,\n"), WeightHardEquivalence);
        JSON += FString::Printf(TEXT("    \"weight_hard_inverse\": %.1f\n"), WeightHardInverse);
        JSON += TEXT("  },\n\n");

        // Option B: scale ladder configuration (4 named tiers)
        JSON += TEXT("  \"hard_tier_scale_ladder\": {\n");
        JSON += FString::Printf(TEXT("    \"enabled\": %s,\n"),
            bEnableHardTierScaleJitter ? TEXT("true") : TEXT("false"));
        JSON += FString::Printf(TEXT("    \"base_translation_cm\": %.2f,\n"), TranslationMagnitude);
        JSON += FString::Printf(TEXT("    \"base_rotation_deg\": %.2f,\n"), RotationMagnitude);
        JSON += TEXT("    \"entries\": [\n");
        for (int32 i = 0; i < HardTierScaleLadder.Num(); i++)
        {
            const FScaleLadderEntry& E = HardTierScaleLadder[i];
            JSON += TEXT("      {");
            JSON += FString::Printf(TEXT("\"name\": \"%s\", "), *ScaleTierToString(E.Tier));
            JSON += FString::Printf(TEXT("\"translation_scale\": %.3f, "), E.TranslationScale);
            JSON += FString::Printf(TEXT("\"rotation_scale\": %.3f, "), E.RotationScale);
            JSON += FString::Printf(TEXT("\"weight\": %.2f, "), E.Weight);
            JSON += FString::Printf(TEXT("\"effective_translation_cm\": %.2f, "),
                E.TranslationScale * TranslationMagnitude);
            JSON += FString::Printf(TEXT("\"effective_rotation_deg\": %.2f"),
                E.RotationScale * RotationMagnitude);
            JSON += (i < HardTierScaleLadder.Num() - 1) ? TEXT("},\n") : TEXT("}\n");
        }
        JSON += TEXT("    ]\n");
        JSON += TEXT("  },\n\n");

        // Actual count of trajectories captured per type
        JSON += TEXT("  \"trajectory_type_counts\": {\n");
        ETrajectoryType AllTypes[] = {
            ETrajectoryType::Seed, ETrajectoryType::Inverse,
            ETrajectoryType::Loop, ETrajectoryType::Equivalence,
            ETrajectoryType::Hard_Loop,
            ETrajectoryType::Hard_Equivalence, ETrajectoryType::Hard_Inverse
        };
        constexpr int32 NumAllTypes = sizeof(AllTypes) / sizeof(AllTypes[0]);
        for (int32 i = 0; i < NumAllTypes; i++)
        {
            int32 Count = TrajectoryTypeCount.Contains(AllTypes[i]) ? TrajectoryTypeCount[AllTypes[i]] : 0;
            JSON += FString::Printf(TEXT("    \"%s\": %d"),
                *TrajectoryTypeToString(AllTypes[i]), Count);
            JSON += (i < NumAllTypes - 1) ? TEXT(",\n") : TEXT("\n");
        }
        JSON += TEXT("  },\n\n");

        // Output directory layout
        JSON += TEXT("  \"output_structure\": {\n");
        JSON += TEXT("    \"reasoning\": \"reasoning/traj_XXXXXX/ — collision-free trajectories, algebraic identities hold exactly\",\n");
        JSON += TEXT("    \"random_walk\": \"random_walk/traj_XXXXXX/ — trajectories with collisions, realistic interaction data\"\n");
        JSON += TEXT("  }\n");
    }

    JSON += TEXT("}\n");

    FString ManifestPath = RuntimeSaveRoot + TEXT("manifest.json");
    FFileHelper::SaveStringToFile(JSON, *ManifestPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);

    UE_LOG(LogTemp, Warning, TEXT("Global manifest saved: %s"), *ManifestPath);
}

// ============================================================================
// Freeze scene lighting.
//
// Counterfactual reasoning requires that visual change come solely from action,
// so the environment lighting must be fixed. This function locates every
// DirectionalLight in the scene, sets its rotation to a fixed value, and disables
// Mobility = Movable so that no blueprint or Timeline can modify it at runtime.
// ============================================================================
void ADataCollector::FreezeLighting()
{
    UWorld* World = GetWorld();
    if (!World) return;

    int32 FrozenCount = 0;
    FRotator FixedSunRotation(SunPitch, SunYaw, 0.0f);

    // Iterate over every DirectionalLight actor in the scene.
    for (TActorIterator<ADirectionalLight> It(World); It; ++It)
    {
        ADirectionalLight* DirLight = *It;
        if (!DirLight) continue;

        // Set the fixed rotation.
        DirLight->SetActorRotation(FixedSunRotation);

        // Switch Mobility to Stationary (cannot be moved at runtime, but still casts dynamic shadows).
        // Stationary is more flexible than Static (Static cannot be modified at runtime),
        // and safer than Movable (prevents accidental blueprint modifications).
        ULightComponent* LightComp = DirLight->GetLightComponent();
        if (LightComp)
        {
            LightComp->SetMobility(EComponentMobility::Stationary);
        }

        FrozenCount++;

        UE_LOG(LogTemp, Warning, TEXT("[Lighting] Frozen DirectionalLight '%s' → Pitch=%.1f, Yaw=%.1f"),
            *DirLight->GetName(), SunPitch, SunYaw);
    }

    if (FrozenCount == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("[Lighting] No DirectionalLight found in scene. Lighting may still change!"));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("[Lighting] Frozen %d DirectionalLight(s). Lighting is now fixed."), FrozenCount);
    }
}

// ============================================================================
// Apply render-quality settings.
//
// Configures SceneCapture's ShowFlags and PostProcess settings according to the
// individual toggles under Details -> Render Quality.
// Each effect can be toggled independently, allowing flexible trade-offs between
// speed and image quality.
// ============================================================================
void ADataCollector::ApplyRenderQualitySettings()
{
    if (!SceneCapture) return;

    FEngineShowFlags& ShowFlags = SceneCapture->ShowFlags;

    // ==================== Core lighting ShowFlags ====================
    // SceneCaptureComponent2D's ShowFlags defaults differ from the editor viewport!
    // The editor viewport defaults to enabling everything; SceneCapture may be missing
    // some critical flags. We must enable these core rendering features explicitly,
    // otherwise the image will be much darker than the editor:
    //   - missing GI -> no indirect light -> indoor scenes very dark
    //   - missing SkyLighting -> SkyLight has no effect -> shadowed regions pure black
    //   - missing ReflectionEnvironment -> reflection probes do not work

    ShowFlags.SetLighting(true);              // master lighting toggle; off = pure black
    ShowFlags.SetPostProcessing(true);        // master post-process toggle (tonemapping/exposure/etc.)
    ShowFlags.SetTonemapper(true);            // tonemapping
    ShowFlags.SetGlobalIllumination(true);    // global illumination (Lumen GI / SSGI / LPV)
    ShowFlags.SetReflectionEnvironment(true); // reflection environment (reflection probes / SSR fallback)
    ShowFlags.SetSkyLighting(true);           // SkyLight (sky diffuse)
    ShowFlags.SetAmbientCubemap(true);        // ambient cubemap (fallback ambient lighting)
    ShowFlags.SetDynamicShadows(true);        // dynamic shadows
    ShowFlags.SetPointLights(true);           // point lights (indoor lamps)
    ShowFlags.SetSpotLights(true);            // spot lights
    ShowFlags.SetRectLights(true);            // rect lights
    ShowFlags.SetDirectionalLights(true);     // directional lights (sun)
    ShowFlags.SetTexturedLightProfiles(true); // IES light profiles
    ShowFlags.SetAtmosphere(true);            // atmospheric scattering (sky color)
    ShowFlags.SetFog(true);                   // distance fog / exponential height fog

    // ==================== ShowFlags often missed by SceneCapture ====================
    // Enabled by default in the editor viewport, but possibly off (or not inherited) in SceneCapture:
    ShowFlags.SetTranslucency(true);          // translucent objects (glass / water / vegetation); off = transparent objects vanish
    ShowFlags.SetDecals(true);                // decals (wall stains / cracks / road markings); near-zero cost
    ShowFlags.SetDeferredLighting(true);      // deferred lighting pass (UE5's main lighting path; required)
    ShowFlags.SetParticles(true);             // particle systems (smoke / fire / dust / falling leaves)
    ShowFlags.SetInstancedFoliage(true);      // instanced foliage rendering (grass / leaves / shrubs)
    ShowFlags.SetInstancedStaticMeshes(true); // instanced static meshes (large numbers of repeated objects)
    ShowFlags.SetInstancedGrass(true);        // instanced grass
    ShowFlags.SetPaper2DSprites(false);       // 2D sprites (not needed in UE5 3D scenes)
    // Note: UE 5.5 removed the Tessellation ShowFlag (Nanite replaces traditional tessellation).
    ShowFlags.SetSubsurfaceScattering(true);  // subsurface scattering (skin / wax / marble translucency)

    // ==================== Critical indirect-lighting ShowFlags ====================
    // Enabled by default in the editor viewport but possibly missing in SceneCapture:
    ShowFlags.SetIndirectLightingCache(true);  // baked indirect lighting cache (for static / stationary lights)
    ShowFlags.SetLightFunctions(true);         // light functions
    ShowFlags.SetLightShafts(true);            // volumetric light shafts (god rays)
    ShowFlags.SetDistanceFieldAO(true);        // distance field ambient occlusion
    // Note: DistanceFieldGI and ScreenSpaceGI are not standalone ShowFlags in UE 5.5; managed internally by Lumen.
    ShowFlags.SetCapsuleShadows(true);         // capsule shadows
    ShowFlags.SetContactShadows(true);         // contact shadows (sharpens small-object shadows)
    ShowFlags.SetVolumetricLightmap(true);     // volumetric lightmap (baked GI for dynamic objects)

    // ==================== Optional quality effects ====================
    // These are "nice to have"; toggle as needed.
    ShowFlags.SetScreenSpaceReflections(bEnableSSR);
    ShowFlags.SetAmbientOcclusion(bEnableAO);
    ShowFlags.SetMotionBlur(bEnableMotionBlur);
    ShowFlags.SetBloom(bEnableBloom);
    ShowFlags.SetLensFlares(bEnableLensFlares);
    ShowFlags.SetDepthOfField(bEnableDepthOfField);
    ShowFlags.SetVolumetricFog(bEnableVolumetricFog);

    // ==================== Anti-Aliasing ====================
    // SceneCaptureComponent2D does not inherit the editor viewport's AA settings by default!
    // We must enable TAA both in ShowFlags and PostProcessSettings.
    // TAA (Temporal Anti-Aliasing) is the highest-quality real-time AA in UE5
    // and is a dependency of Lumen GI / Reflections and Virtual Shadow Maps (VSM).

    ShowFlags.SetAntiAliasing(true);          // master AA toggle
    ShowFlags.SetTemporalAA(true);            // enable Temporal AA

    // In UE 5.5 the AA method is controlled directly via ShowFlags
    // (SetTemporalAA + SetAntiAliasing); FPostProcessSettings no longer exposes AntiAliasingMethod.
    // ShowFlags.SetTemporalAA(true) above is sufficient to force TAA.

    // ==================== Sharpening ====================
    // Note: in UE 5.5, FPostProcessSettings no longer exposes Sharpness / SharpenFilterIntensity.
    // TAA sharpening is controlled internally by the engine's TSR/TAA pipeline.
    // To tweak sharpening manually, use Project Settings -> Rendering -> TSR Sharpness,
    // or the r.TemporalAA.Sharpen console variable (0.0..1.0).

    // ==================== Color accuracy ====================
    // Make sure color output is consistent across scenes; no random color-temperature shifts.
    SceneCapture->PostProcessSettings.bOverride_WhiteTemp = true;
    SceneCapture->PostProcessSettings.WhiteTemp = 6500.0f;  // D65 standard daylight white point

    SceneCapture->PostProcessSettings.bOverride_WhiteTint = true;
    SceneCapture->PostProcessSettings.WhiteTint = 0.0f;     // no green/magenta shift

    // ==================== Chromatic aberration ====================
    // SceneCapture may default to a slight chromatic aberration; for offline capture, disable to keep the image clean.
    SceneCapture->PostProcessSettings.bOverride_SceneFringeIntensity = true;
    SceneCapture->PostProcessSettings.SceneFringeIntensity = 0.0f;  // disable chromatic aberration

    // ==================== Vignette ====================
    // The editor defaults to a slight vignette; disable for training data so brightness stays uniform.
    SceneCapture->PostProcessSettings.bOverride_VignetteIntensity = true;
    SceneCapture->PostProcessSettings.VignetteIntensity = 0.0f;  // disable vignette

    // ==================== Film grain ====================
    // UE5 may default to slight film-grain noise; disable for training data.
    SceneCapture->PostProcessSettings.bOverride_FilmGrainIntensity = true;
    SceneCapture->PostProcessSettings.FilmGrainIntensity = 0.0f;  // disable film grain

    // Motion blur must also be overridden in PostProcess to be fully disabled.
    if (!bEnableMotionBlur)
    {
        SceneCapture->PostProcessSettings.bOverride_MotionBlurAmount = true;
        SceneCapture->PostProcessSettings.MotionBlurAmount = 0.0f;
        SceneCapture->PostProcessSettings.bOverride_MotionBlurMax = true;
        SceneCapture->PostProcessSettings.MotionBlurMax = 0.0f;
    }

    // Ray tracing
    SceneCapture->bUseRayTracingIfEnabled = bEnableRayTracing;

    // ==================== Exposure ====================
    // SceneCaptureComponent2D does not inherit auto-exposure from the editor viewport.
    // Without manual configuration, indoor / backlit areas are extremely underexposed (pure black).
    //
    // Important: with SCS_FinalColorHDR, exposure and tonemapping are done inside the engine,
    // and the output is the HDR linear value after tonemapping.
    // Auto-exposure may be inaccurate in SceneCapture's single-frame mode (no history),
    // so prefer fixed exposure or pair it with bAlwaysPersistRenderingState.

    // Enable rendering-state persistence so auto-exposure has history-frame data.
    SceneCapture->bAlwaysPersistRenderingState = true;

    if (bUseFixedExposure)
    {
        // ---- Fixed exposure mode ----
        ShowFlags.SetEyeAdaptation(false);

        SceneCapture->PostProcessSettings.bOverride_AutoExposureMethod = true;
        SceneCapture->PostProcessSettings.AutoExposureMethod = EAutoExposureMethod::AEM_Manual;

        SceneCapture->PostProcessSettings.bOverride_AutoExposureBias = true;
        SceneCapture->PostProcessSettings.AutoExposureBias = FixedExposureBias;

        UE_LOG(LogTemp, Warning, TEXT("[Exposure] Fixed exposure mode: EV bias = %.1f"), FixedExposureBias);
    }
    else
    {
        // ---- Auto-exposure mode ----
        ShowFlags.SetEyeAdaptation(true);

        SceneCapture->PostProcessSettings.bOverride_AutoExposureMethod = true;
        SceneCapture->PostProcessSettings.AutoExposureMethod = EAutoExposureMethod::AEM_Histogram;

        SceneCapture->PostProcessSettings.bOverride_AutoExposureMinBrightness = true;
        SceneCapture->PostProcessSettings.AutoExposureMinBrightness = AutoExposureMinBrightness;

        SceneCapture->PostProcessSettings.bOverride_AutoExposureMaxBrightness = true;
        SceneCapture->PostProcessSettings.AutoExposureMaxBrightness = AutoExposureMaxBrightness;

        SceneCapture->PostProcessSettings.bOverride_AutoExposureBias = true;
        SceneCapture->PostProcessSettings.AutoExposureBias = AutoExposureBias;

        SceneCapture->PostProcessSettings.bOverride_AutoExposureSpeedUp = true;
        SceneCapture->PostProcessSettings.AutoExposureSpeedUp = AutoExposureSpeedUp;

        SceneCapture->PostProcessSettings.bOverride_AutoExposureSpeedDown = true;
        SceneCapture->PostProcessSettings.AutoExposureSpeedDown = AutoExposureSpeedDown;

        // Tweak the low end of the exposure-compensation curve to prevent excessive underexposure in dark areas.
        SceneCapture->PostProcessSettings.bOverride_AutoExposureLowPercent = true;
        SceneCapture->PostProcessSettings.AutoExposureLowPercent = 70.0f;  // sample more dark pixels

        SceneCapture->PostProcessSettings.bOverride_AutoExposureHighPercent = true;
        SceneCapture->PostProcessSettings.AutoExposureHighPercent = 98.0f;

        UE_LOG(LogTemp, Warning, TEXT("[Exposure] Auto exposure mode: MinBright=%.2f  MaxBright=%.2f  Bias=%.1f  SpeedUp=%.1f  SpeedDown=%.1f"),
            AutoExposureMinBrightness, AutoExposureMaxBrightness, AutoExposureBias, AutoExposureSpeedUp, AutoExposureSpeedDown);
    }

    // --- Print summary ---
    UE_LOG(LogTemp, Warning, TEXT("[Render Quality] AA=TAA  SSR=%s  AO=%s  MotionBlur=%s  Bloom=%s  LensFlare=%s  DoF=%s  VolumetricFog=%s  RayTracing=%s  GI=ON  SkyLight=ON"),
        bEnableSSR ? TEXT("ON") : TEXT("OFF"),
        bEnableAO ? TEXT("ON") : TEXT("OFF"),
        bEnableMotionBlur ? TEXT("ON") : TEXT("OFF"),
        bEnableBloom ? TEXT("ON") : TEXT("OFF"),
        bEnableLensFlares ? TEXT("ON") : TEXT("OFF"),
        bEnableDepthOfField ? TEXT("ON") : TEXT("OFF"),
        bEnableVolumetricFog ? TEXT("ON") : TEXT("OFF"),
        bEnableRayTracing ? TEXT("ON") : TEXT("OFF"));
    UE_LOG(LogTemp, Warning, TEXT("[Render Quality] Translucency=ON  Decals=ON  Particles=ON  Foliage=ON  SSS=ON  Vignette=OFF  FilmGrain=OFF  ChromaticAberr=OFF  WhiteTemp=6500K"));
}

// ============================================================================
// ==================== Trajectory mode method implementations ====================
// ============================================================================

// ============================================================================
// Initialize the default action sampling weights.
//
// Per the paper's design: translation : rotation : no_op ~ 60% : 30% : 10%.
// Concrete weights:
//   forward=2.0, backward=1.0, left=1.5, right=1.5            (translation: total 6.0)
//   turn_left=1.0, turn_right=1.0, look_up=0.5, look_down=0.5 (rotation:    total 3.0)
//   no_op=0.5
// ============================================================================
void ADataCollector::InitializeDefaultActionWeights()
{
    ActionSamplingWeights.Empty();
    ActionSamplingWeights.SetNum(ActionSpace.Num());

    for (int32 i = 0; i < ActionSpace.Num(); i++)
    {
        const FString& Name = ActionSpace[i].ActionName;

        if (Name == TEXT("move_forward"))         ActionSamplingWeights[i] = 2.0f;
        else if (Name == TEXT("move_backward"))    ActionSamplingWeights[i] = 1.0f;
        else if (Name == TEXT("move_left"))        ActionSamplingWeights[i] = 1.5f;
        else if (Name == TEXT("move_right"))       ActionSamplingWeights[i] = 1.5f;
        else if (Name == TEXT("turn_left"))        ActionSamplingWeights[i] = 1.0f;
        else if (Name == TEXT("turn_right"))       ActionSamplingWeights[i] = 1.0f;
        else if (Name == TEXT("look_up"))          ActionSamplingWeights[i] = 0.5f;
        else if (Name == TEXT("look_down"))        ActionSamplingWeights[i] = 0.5f;
        else if (Name == TEXT("no_op"))            ActionSamplingWeights[i] = 0.5f;
        else                                       ActionSamplingWeights[i] = 1.0f; // unknown action -> default 1.0
    }

    UE_LOG(LogTemp, Warning, TEXT("[Trajectory] Initialized default action sampling weights (%d actions)"), ActionSpace.Num());
}

// ============================================================================
// Check whether an action is purely a rotation (no translation component).
// ============================================================================
bool ADataCollector::IsRotationAction(int32 ActionIdx) const
{
    if (ActionIdx < 0 || ActionIdx >= ActionSpace.Num()) return false;
    const FActionDefinition& A = ActionSpace[ActionIdx];
    return A.DeltaPosition.IsNearlyZero(0.01f) && !A.DeltaRotation.IsNearlyZero(0.01f);
}

// ============================================================================
// Use a line trace to predict whether an action will collide.
//
// From a given pose, compute the action's expected target location, then run
// a sweep test to predict whether it will hit an obstacle.
// ============================================================================
bool ADataCollector::WouldActionCollide(const FVector& FromPos, const FRotator& FromRot, int32 ActionIdx) const
{
    if (!GetWorld()) return false;
    if (ActionIdx < 0 || ActionIdx >= ActionSpace.Num()) return false;

    const FActionDefinition& Action = ActionSpace[ActionIdx];

    // Pure rotation actions never collide.
    if (Action.DeltaPosition.IsNearlyZero(0.01f))
    {
        return false;
    }

    // Compute the target location.
    FVector WorldDelta = FromRot.RotateVector(Action.DeltaPosition);
    FVector TargetPos = FromPos + WorldDelta;

    // Sweep test
    FCollisionShape Shape = FCollisionShape::MakeCapsule(CapsuleRadius, CapsuleHalfHeight);
    FCollisionQueryParams Params;
    Params.AddIgnoredActor(this);

    // If a whitelist exists, use SweepMulti and filter individually.
    if (CollisionWhitelistKeywords.Num() > 0)
    {
        TArray<FHitResult> Hits;
        bool bHit = GetWorld()->SweepMultiByChannel(
            Hits,
            FromPos,
            TargetPos,
            FQuat::Identity,
            ECollisionChannel::ECC_Pawn,
            Shape,
            Params
        );

        if (!bHit) return false;

        // Check whether any non-whitelisted collisions remain.
        for (const FHitResult& Hit : Hits)
        {
            AActor* HitActor = Hit.GetActor();
            if (HitActor && !IsActorWhitelisted(HitActor))
            {
                return true; // a real collision exists
            }
        }
        return false; // every collision is whitelisted
    }

    // Without a whitelist, use the original logic.
    FHitResult Hit;
    bool bHit = GetWorld()->SweepSingleByChannel(
        Hit,
        FromPos,
        TargetPos,
        FQuat::Identity,
        ECollisionChannel::ECC_Pawn,
        Shape,
        Params
    );

    return bHit;
}

// ============================================================================
// Sample a single action (with collision avoidance + consecutive-rotation cap).
//
// Strategy:
//   1. Check whether the most recent N steps are all rotations -> force a translation.
//   2. Sample one candidate action by weight.
//   3. If collision avoidance is on, predict using WouldActionCollide.
//   4. If predicted collision, resample (up to MaxResampleAttempts).
//   5. If everything collides, fall back to no_op or a rotation.
// ============================================================================
int32 ADataCollector::SampleSingleAction(const FVector& CurrentPos, const FRotator& CurrentRot)
{
    // ---- Check the consecutive-rotation cap ----
    bool bForceTranslation = false;
    int32 RecentCount = CurrentTrajectoryActions.Num();
    if (RecentCount >= MaxConsecutiveRotations)
    {
        bForceTranslation = true;
        for (int32 k = RecentCount - MaxConsecutiveRotations; k < RecentCount; k++)
        {
            if (!IsRotationAction(CurrentTrajectoryActions[k]))
            {
                bForceTranslation = false;
                break;
            }
        }
    }

    // ---- Build effective weights ----
    TArray<float> EffectiveWeights;
    EffectiveWeights.SetNum(ActionSpace.Num());
    float TotalWeight = 0.0f;

    for (int32 i = 0; i < ActionSpace.Num(); i++)
    {
        if (bForceTranslation && (IsRotationAction(i) || ActionSpace[i].ActionName == TEXT("no_op")))
        {
            EffectiveWeights[i] = 0.0f; // forcibly exclude rotations and no_op
        }
        else
        {
            EffectiveWeights[i] = ActionSamplingWeights[i];
        }
        TotalWeight += EffectiveWeights[i];
    }

    // Safety check: if all weights are 0 (extreme case), fall back to a uniform distribution.
    if (TotalWeight < 0.001f)
    {
        for (int32 i = 0; i < ActionSpace.Num(); i++)
        {
            EffectiveWeights[i] = 1.0f;
        }
        TotalWeight = ActionSpace.Num();
    }

    // ---- Weighted random sample + collision avoidance ----
    auto WeightedSample = [&]() -> int32
    {
        float Roll = FMath::FRand() * TotalWeight;
        float Cumulative = 0.0f;
        for (int32 i = 0; i < ActionSpace.Num(); i++)
        {
            Cumulative += EffectiveWeights[i];
            if (Roll <= Cumulative)
            {
                return i;
            }
        }
        return ActionSpace.Num() - 1; // fallback
    };

    int32 Candidate = WeightedSample();

    if (bCollisionAvoidanceSampling)
    {
        for (int32 Attempt = 0; Attempt < MaxResampleAttempts; Attempt++)
        {
            if (!WouldActionCollide(CurrentPos, CurrentRot, Candidate))
            {
                return Candidate; // no collision, use this action
            }
            Candidate = WeightedSample(); // resample
        }

        // Every attempt collided; try no_op or rotations.
        for (int32 i = 0; i < ActionSpace.Num(); i++)
        {
            if (ActionSpace[i].ActionName == TEXT("no_op") || IsRotationAction(i))
            {
                if (!WouldActionCollide(CurrentPos, CurrentRot, i))
                {
                    return i;
                }
            }
        }
    }

    return Candidate; // unable to avoid; return the last candidate
}

// ============================================================================
// Plan one trajectory's action sequence -- the dispatch entry point (with retries + skip).
//
// Picks a trajectory type at random based on weights and calls the corresponding planner.
// In the pure-simulation phase, retry up to MaxPlanRetries times (each retry resamples
// parameters and optionally rotates the start yaw); only enter the render flow on success.
//
// If every retry fails:
//   bSkipFailedPlans = true  -> set bPlanningFailed; the FSM skips this trajectory (no rendering)
//   bSkipFailedPlans = false -> fall back to Seed (legacy behavior)
//
// For Equivalence, also fills the paired path (PairedTrajectoryActions).
// ============================================================================
void ADataCollector::PlanTrajectoryActions()
{
    CurrentTrajectoryActions.Empty();
    CurrentTrajectoryActions.Reserve(TrajectorySteps);

    // Reset paired-trajectory state.
    bHasPairedTrajectory = false;
    bCurrentlyRenderingPaired = false;
    bPairedTrajectoryHasCollision = false;
    PairedTrajectoryActions.Empty();
    PairedTrajectoryRecords.Empty();

    // Reset the planning-failed flag.
    bPlanningFailed = false;

    // Save the original orientation; used to rotate the starting yaw across retries.
    const FRotator OrigRot = GetActorRotation();
    bool bPlanSuccess = false;

    // ========== Retry loop ==========
    // Each retry: reset state -> (optional) rotate yaw -> sample type -> (optional) scale jitter -> attempt to plan.
    // Capsule collision simulation is <1 ms per call, so 20 retries are essentially free.
    for (int32 Retry = 0; Retry < MaxPlanRetries && !bPlanSuccess; ++Retry)
    {
        // ---- Reset state for this attempt ----
        CurrentTrajectoryActions.Empty();
        CurrentTrajectoryActions.Reserve(TrajectorySteps);
        bHasPairedTrajectory = false;
        PairedTrajectoryActions.Empty();
        PairedTrajectoryRecords.Empty();

        // If the previous attempt jittered ActionSpace, restore it first.
        if (bActionSpaceIsJittered)
        {
            RestoreActionSpaceScale();
        }

        // ---- Every RetryYawRotateInterval retries, rotate the starting yaw once ----
        // Changing direction can avoid one-sided wall collisions (e.g. sample point facing the edge of a pool;
        // a 45° rotation may unblock it).
        if (Retry > 0 && RetryYawRotateInterval > 0 && (Retry % RetryYawRotateInterval) == 0)
        {
            FRotator NewRot = OrigRot;
            NewRot.Yaw += (360.0f / FMath::Max(1, NumYawSamples)) * (Retry / RetryYawRotateInterval);
            SetActorRotation(NewRot);
        }

        // ---- Pick a trajectory type ----
        CurrentTrajectoryType = SampleTrajectoryType();

        // ---- Option B: apply step-size jitter for Hard tier ----
        CurrentTrajectoryTransScale = 1.0f;
        CurrentTrajectoryRotScale = 1.0f;
        const bool bIsHardTier =
            CurrentTrajectoryType == ETrajectoryType::Hard_Loop       ||
            CurrentTrajectoryType == ETrajectoryType::Hard_Equivalence||
            CurrentTrajectoryType == ETrajectoryType::Hard_Inverse;
        if (bIsHardTier && bEnableHardTierScaleJitter)
        {
            ApplyHardTierScaleJitter(CurrentTrajectoryTransScale, CurrentTrajectoryRotScale);
        }

        // Track attempt count.
        PlanAttemptCount.FindOrAdd(CurrentTrajectoryType)++;

        // ---- Dispatch to the corresponding planner ----
        switch (CurrentTrajectoryType)
        {
        case ETrajectoryType::Seed:
            PlanSeedTrajectory();
            bPlanSuccess = true;
            break;

        case ETrajectoryType::Inverse:
            bPlanSuccess = PlanInverseTrajectory();
            break;

        case ETrajectoryType::Loop:
            bPlanSuccess = PlanLoopTrajectory();
            break;

        case ETrajectoryType::Equivalence:
            bPlanSuccess = PlanEquivalenceTrajectory();
            break;

        // -- Hard tier ---------------------------------------------------------
        case ETrajectoryType::Hard_Loop:
            bPlanSuccess = PlanHardLoopTrajectory();
            break;

        case ETrajectoryType::Hard_Equivalence:
            bPlanSuccess = PlanHardEquivalenceTrajectory();
            break;

        case ETrajectoryType::Hard_Inverse:
            bPlanSuccess = PlanHardInverseTrajectory();
            break;
        }

        // On planning success, log retry count (Retry=0 means it succeeded on the first try).
        if (bPlanSuccess)
        {
            PlanSuccessCount.FindOrAdd(CurrentTrajectoryType)++;
            if (Retry > 0)
            {
                UE_LOG(LogTemp, Log, TEXT("    [Plan] %s succeeded on retry %d/%d"),
                    *TrajectoryTypeToString(CurrentTrajectoryType), Retry + 1, MaxPlanRetries);
            }
        }
    }
    // ========== End of retry loop ==========

    // Restore orientation (regardless of success, keep the sample point's original state consistent).
    SetActorRotation(OrigRot);

    if (bPlanSuccess)
    {
        // If ActionSpace was jittered and planning succeeded, keep the jittered state until rendering finishes
        // (RestoreActionSpaceScale is called in FinalizeTrajectory).

        // Update stats.
        TrajectoryTypeCount.FindOrAdd(CurrentTrajectoryType)++;

        UE_LOG(LogTemp, Warning, TEXT("    [Plan] Type=%s ScaleTier=%s (t%.2fx r%.2fx) Actions=%d%s"),
            *TrajectoryTypeToString(CurrentTrajectoryType),
            *ScaleTierToString(CurrentScaleTier),
            CurrentTrajectoryTransScale, CurrentTrajectoryRotScale,
            CurrentTrajectoryActions.Num(),
            bHasPairedTrajectory ? TEXT(", +paired path") : TEXT(""));
    }
    else
    {
        // Planning failed: restore any potential scale jitter.
        if (bActionSpaceIsJittered)
        {
            RestoreActionSpaceScale();
        }

        if (bSkipFailedPlans)
        {
            // Do not render and do not fall back to Seed; signal the FSM to skip this trajectory.
            bPlanningFailed = true;
            TotalPlanSkips++;
            UE_LOG(LogTemp, Warning, TEXT("    [Plan] All %d retries exhausted for this sample point, skipping (no Seed fallback)"),
                MaxPlanRetries);
        }
        else
        {
            // Legacy behavior: fall back to Seed.
            UE_LOG(LogTemp, Warning, TEXT("    [Plan] All %d retries exhausted, falling back to Seed"),
                MaxPlanRetries);
            CurrentTrajectoryType = ETrajectoryType::Seed;
            CurrentTrajectoryTransScale = 1.0f;
            CurrentTrajectoryRotScale = 1.0f;
            CurrentScaleTier = EScaleTier::BaseDefault;
            bHasPairedTrajectory = false;
            PairedTrajectoryActions.Empty();
            PlanSeedTrajectory();

            TrajectoryTypeCount.FindOrAdd(CurrentTrajectoryType)++;

            UE_LOG(LogTemp, Warning, TEXT("    [Plan] Type=%s ScaleTier=%s (t%.2fx r%.2fx) Actions=%d"),
                *TrajectoryTypeToString(CurrentTrajectoryType),
                *ScaleTierToString(CurrentScaleTier),
                CurrentTrajectoryTransScale, CurrentTrajectoryRotScale,
                CurrentTrajectoryActions.Num());
        }
    }
}

// ============================================================================
// Get the trajectory save path (split by collision).
//
// Output layout:
//   {RuntimeSaveRoot}/reasoning/traj_{GlobalID}/     (no collision)
//   {RuntimeSaveRoot}/random_walk/traj_{GlobalID}/   (with collisions)
// ============================================================================
FString ADataCollector::GetTrajectoryPath(bool bHasCollision) const
{
    FString Category = bHasCollision ? TEXT("random_walk") : TEXT("reasoning");
    return FString::Printf(TEXT("%s%s/traj_%06d/"), *RuntimeSaveRoot, *Category, TrajectoryGlobalID);
}

// ============================================================================
// Get the per-step frame save subdirectory within a trajectory.
//
// Output: step_00/ ~ step_39/
// Each subdirectory contains frame_0000.jpg ~ frame_0007.jpg (8 frames).
// ============================================================================
FString ADataCollector::GetTrajectoryStepSubDir(int32 StepIndex) const
{
    return FString::Printf(TEXT("step_%02d"), StepIndex);
}

// ============================================================================
// Save the trajectory metadata JSON.
//
// Contents: action sequence, per-step pose record, collision info, split result.
// ============================================================================
void ADataCollector::SaveTrajectoryMetadataJSON(const FString& TrajectoryDir)
{
    EnsureDirectoryExists(TrajectoryDir);

    // Determine whether we are saving the primary path or the paired path.
    // Logic: if TrajectoryGlobalID == PairedTrajectoryGlobalID and bHasPairedTrajectory,
    //        we are saving the paired path.
    bool bSavingPairedPath = bHasPairedTrajectory && (TrajectoryGlobalID == PairedTrajectoryGlobalID);

    const TArray<int32>& SaveActions = bSavingPairedPath ? PairedTrajectoryActions : CurrentTrajectoryActions;
    const TArray<FTrajectoryStepRecord>& SaveRecords = bSavingPairedPath ? PairedTrajectoryRecords : CurrentTrajectoryRecords;
    bool bSaveHasCollision = bSavingPairedPath ? bPairedTrajectoryHasCollision : bCurrentTrajectoryHasCollision;

    FString JSON;
    JSON += TEXT("{\n");

    // Basic info
    JSON += FString::Printf(TEXT("  \"trajectory_id\": %d,\n"), TrajectoryGlobalID);
    JSON += FString::Printf(TEXT("  \"episode_id\": %d,\n"), EpisodeID);
    JSON += FString::Printf(TEXT("  \"state_id\": %d,\n"), CurrentStateID);
    JSON += FString::Printf(TEXT("  \"sample_index\": %d,\n"), CurrentSampleIndex);
    JSON += FString::Printf(TEXT("  \"trajectory_index_at_point\": %d,\n"), CurrentTrajectoryIndexAtPoint);
    JSON += FString::Printf(TEXT("  \"data_type\": \"%s\",\n"),
        bSaveHasCollision ? TEXT("random_walk") : TEXT("reasoning"));
    JSON += FString::Printf(TEXT("  \"has_collision\": %s,\n"),
        bSaveHasCollision ? TEXT("true") : TEXT("false"));

    // ==================== Structured trajectory info ====================

    // Trajectory type (paper's four algebraic relations + Seed control)
    JSON += FString::Printf(TEXT("  \"trajectory_type\": \"%s\",\n"),
        *TrajectoryTypeToString(CurrentTrajectoryType));

    // Algebraic property description (helps training scripts understand the semantics).
    FString AlgebraicProperty;
    switch (CurrentTrajectoryType)
    {
    case ETrajectoryType::Seed:
        AlgebraicProperty = TEXT("none (flat random walk, control group)");
        break;
    case ETrajectoryType::Inverse:
        AlgebraicProperty = TEXT("inverse: A ∘ A⁻¹ = id (explore then reverse)");
        break;
    case ETrajectoryType::Loop:
        AlgebraicProperty = TEXT("loop: T_A(s₀) = s₀ (closed-loop closure, explore+return)");
        break;
    case ETrajectoryType::Equivalence:
        AlgebraicProperty = TEXT("equivalence: T_A(s₀) = T_B(s₀), A ≠ B (commutative segment shuffle)");
        break;
    case ETrajectoryType::Hard_Loop:
        AlgebraicProperty = TEXT("hard-loop: T_A(s₀) = s₀ via topological template (rectangle / triangle / hexagon)");
        break;
    case ETrajectoryType::Hard_Equivalence:
        AlgebraicProperty = TEXT("hard-equivalence: L-shape vs zig-zag in the translation subgroup, identical endpoint");
        break;
    case ETrajectoryType::Hard_Inverse:
        AlgebraicProperty = TEXT("hard-inverse: A ∘ A⁻¹ = id with rotation-mixed explore segment (non-abelian, not reducible to symmetric round-trips)");
        break;
    }
    JSON += FString::Printf(TEXT("  \"algebraic_property\": \"%s\",\n"), *AlgebraicProperty);

    // Tier label (Easy / Hard) for easy filtering in the manifest.
    FString TierName;
    switch (CurrentTrajectoryType)
    {
    case ETrajectoryType::Seed:
        TierName = TEXT("control"); break;
    case ETrajectoryType::Inverse:
    case ETrajectoryType::Loop:
    case ETrajectoryType::Equivalence:
        TierName = TEXT("easy"); break;
    default:
        TierName = TEXT("hard"); break;
    }
    JSON += FString::Printf(TEXT("  \"tier\": \"%s\",\n"), *TierName);

    // Option B: step-size scale tier
    //   Named ladder entry: default / fine / coarse / wide_rot (see paper §4.3)
    //   translation_scale x base TranslationMagnitude = effective cm/step
    //   rotation_scale    x base RotationMagnitude    = effective deg/step
    JSON += TEXT("  \"scale_tier\": {\n");
    JSON += FString::Printf(TEXT("    \"name\": \"%s\",\n"),
        *ScaleTierToString(CurrentScaleTier));
    JSON += FString::Printf(TEXT("    \"translation_scale\": %.3f,\n"), CurrentTrajectoryTransScale);
    JSON += FString::Printf(TEXT("    \"rotation_scale\": %.3f,\n"), CurrentTrajectoryRotScale);
    JSON += FString::Printf(TEXT("    \"base_translation_cm\": %.2f,\n"), TranslationMagnitude);
    JSON += FString::Printf(TEXT("    \"base_rotation_deg\": %.2f,\n"), RotationMagnitude);
    JSON += FString::Printf(TEXT("    \"effective_translation_cm\": %.2f,\n"),
        CurrentTrajectoryTransScale * TranslationMagnitude);
    JSON += FString::Printf(TEXT("    \"effective_rotation_deg\": %.2f\n"),
        CurrentTrajectoryRotScale * RotationMagnitude);
    JSON += TEXT("  },\n");

    // Paired-trajectory relation (present for Equivalence / Hard_Equivalence).
    if (bHasPairedTrajectory)
    {
        int32 PrimaryID = PairedTrajectoryGlobalID - 1;
        int32 PairedID = PairedTrajectoryGlobalID;

        JSON += TEXT("  \"paired_trajectory\": {\n");
        JSON += FString::Printf(TEXT("    \"is_paired\": true,\n"));
        JSON += FString::Printf(TEXT("    \"role\": \"%s\",\n"),
            bSavingPairedPath ? TEXT("path_B") : TEXT("path_A"));
        JSON += FString::Printf(TEXT("    \"partner_trajectory_id\": %d,\n"),
            bSavingPairedPath ? PrimaryID : PairedID);
        JSON += FString::Printf(TEXT("    \"primary_trajectory_id\": %d,\n"), PrimaryID);
        JSON += FString::Printf(TEXT("    \"paired_trajectory_id\": %d,\n"), PairedID);

        // Equivalence: both paths should reach the same final state (commutative equivalence).
        if (CurrentTrajectoryType == ETrajectoryType::Equivalence)
        {
            JSON += TEXT("    \"relation\": \"path_A and path_B are commutative shuffles, should reach same final state\"\n");
        }
        else if (CurrentTrajectoryType == ETrajectoryType::Hard_Equivalence)
        {
            JSON += TEXT("    \"relation\": \"L-shape vs zig-zag in translation subgroup; identical endpoint, disjoint trajectories\"\n");
        }
        JSON += TEXT("  },\n");
    }
    else
    {
        JSON += TEXT("  \"paired_trajectory\": {\n");
        JSON += TEXT("    \"is_paired\": false\n");
        JSON += TEXT("  },\n");
    }

    // Collision stats
    int32 CollisionCount = 0;
    for (const FTrajectoryStepRecord& R : SaveRecords)
    {
        if (R.bCollisionOccurred) CollisionCount++;
    }
    JSON += FString::Printf(TEXT("  \"collision_count\": %d,\n"), CollisionCount);
    JSON += FString::Printf(TEXT("  \"total_steps\": %d,\n"), TrajectorySteps);
    JSON += FString::Printf(TEXT("  \"frames_per_step\": %d,\n"), FramesPerAction);
    JSON += FString::Printf(TEXT("  \"total_frames\": %d,\n\n"), TrajectorySteps * FramesPerAction);

    // Render config
    JSON += TEXT("  \"render_config\": {\n");
    JSON += FString::Printf(TEXT("    \"resolution\": [%d, %d],\n"), ResolutionX, ResolutionY);
    JSON += FString::Printf(TEXT("    \"fov\": %.1f,\n"), CameraFOV);
    JSON += FString::Printf(TEXT("    \"image_format\": \"%s\",\n"),
        ImageFormat == EImageOutputFormat::JPEG ? TEXT("jpeg") : TEXT("bmp"));
    JSON += FString::Printf(TEXT("    \"modalities\": [\"rgb\"%s%s]\n"),
        bCaptureDepth ? TEXT(", \"depth\"") : TEXT(""),
        bCaptureOpticalFlow ? TEXT(", \"optical_flow\"") : TEXT(""));
    JSON += TEXT("  },\n\n");

    // Root state
    JSON += TEXT("  \"root_state\": {\n");
    JSON += FString::Printf(TEXT("    \"position\": [%.4f, %.4f, %.4f],\n"),
        OriginalLocation.X, OriginalLocation.Y, OriginalLocation.Z);
    JSON += FString::Printf(TEXT("    \"rotation\": [%.4f, %.4f, %.4f]\n"),
        OriginalRotation.Pitch, OriginalRotation.Yaw, OriginalRotation.Roll);
    JSON += TEXT("  },\n\n");

    // Action sequence
    JSON += TEXT("  \"action_sequence\": [");
    for (int32 i = 0; i < SaveActions.Num(); i++)
    {
        JSON += FString::Printf(TEXT("\"%s\""), *ActionSpace[SaveActions[i]].ActionName);
        if (i < SaveActions.Num() - 1) JSON += TEXT(", ");
    }
    JSON += TEXT("],\n\n");

    // Collision mask
    JSON += TEXT("  \"collision_mask\": [");
    for (int32 i = 0; i < SaveRecords.Num(); i++)
    {
        JSON += SaveRecords[i].bCollisionOccurred ? TEXT("1") : TEXT("0");
        if (i < SaveRecords.Num() - 1) JSON += TEXT(", ");
    }
    JSON += TEXT("],\n\n");

    // Per-step detailed records
    JSON += TEXT("  \"steps\": [\n");
    for (int32 i = 0; i < SaveRecords.Num(); i++)
    {
        const FTrajectoryStepRecord& R = SaveRecords[i];
        JSON += TEXT("    {\n");
        JSON += FString::Printf(TEXT("      \"step\": %d,\n"), i);
        JSON += FString::Printf(TEXT("      \"action\": \"%s\",\n"), *ActionSpace[R.ActionIndex].ActionName);
        JSON += FString::Printf(TEXT("      \"action_id\": %d,\n"), R.ActionIndex);
        JSON += FString::Printf(TEXT("      \"start_pos\": [%.4f, %.4f, %.4f],\n"),
            R.StartPosition.X, R.StartPosition.Y, R.StartPosition.Z);
        JSON += FString::Printf(TEXT("      \"start_rot\": [%.4f, %.4f, %.4f],\n"),
            R.StartRotation.Pitch, R.StartRotation.Yaw, R.StartRotation.Roll);
        JSON += FString::Printf(TEXT("      \"expected_end_pos\": [%.4f, %.4f, %.4f],\n"),
            R.ExpectedEndPosition.X, R.ExpectedEndPosition.Y, R.ExpectedEndPosition.Z);
        JSON += FString::Printf(TEXT("      \"actual_end_pos\": [%.4f, %.4f, %.4f],\n"),
            R.ActualEndPosition.X, R.ActualEndPosition.Y, R.ActualEndPosition.Z);
        JSON += FString::Printf(TEXT("      \"actual_end_rot\": [%.4f, %.4f, %.4f],\n"),
            R.ActualEndRotation.Pitch, R.ActualEndRotation.Yaw, R.ActualEndRotation.Roll);
        JSON += FString::Printf(TEXT("      \"collision\": %s,\n"), R.bCollisionOccurred ? TEXT("true") : TEXT("false"));
        JSON += FString::Printf(TEXT("      \"collision_displacement\": %.4f,\n"), R.CollisionDisplacement);
        if (R.bCollisionOccurred && !R.CollisionActorName.IsEmpty())
        {
            JSON += FString::Printf(TEXT("      \"collision_actor\": \"%s\",\n"), *R.CollisionActorName);
            JSON += FString::Printf(TEXT("      \"collision_component\": \"%s\",\n"), *R.CollisionComponentName);
            JSON += FString::Printf(TEXT("      \"collision_phys_material\": \"%s\",\n"), *R.CollisionPhysMaterial);
            JSON += FString::Printf(TEXT("      \"collision_impact_point\": [%.4f, %.4f, %.4f],\n"),
                R.CollisionImpactPoint.X, R.CollisionImpactPoint.Y, R.CollisionImpactPoint.Z);
            JSON += FString::Printf(TEXT("      \"collision_normal\": [%.4f, %.4f, %.4f],\n"),
                R.CollisionNormal.X, R.CollisionNormal.Y, R.CollisionNormal.Z);
        }
        JSON += FString::Printf(TEXT("      \"frame_dir\": \"%s\"\n"), *GetTrajectoryStepSubDir(i));
        JSON += (i < SaveRecords.Num() - 1) ? TEXT("    },\n") : TEXT("    }\n");
    }
    JSON += TEXT("  ]\n");

    JSON += TEXT("}\n");

    FString MetadataPath = TrajectoryDir / TEXT("metadata.json");
    FFileHelper::SaveStringToFile(JSON, *MetadataPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}

// ============================================================================
// ==================== Structured trajectory planning implementations ====================
// ============================================================================

// ============================================================================
// Trajectory type enum -> string
// ============================================================================
FString ADataCollector::TrajectoryTypeToString(ETrajectoryType Type)
{
    switch (Type)
    {
    case ETrajectoryType::Seed:             return TEXT("seed");
    case ETrajectoryType::Inverse:          return TEXT("inverse");
    case ETrajectoryType::Loop:             return TEXT("loop");
    case ETrajectoryType::Equivalence:      return TEXT("equivalence");
    case ETrajectoryType::Hard_Loop:        return TEXT("hard_loop");
    case ETrajectoryType::Hard_Equivalence: return TEXT("hard_equivalence");
    case ETrajectoryType::Hard_Inverse:     return TEXT("hard_inverse");
    default:                                return TEXT("unknown");
    }
}

// ============================================================================
// Pick a trajectory type at random by weight.
//
// Supports 7 types (3 Easy + 3 Hard + Seed); weights come from UPROPERTY config.
// If the total weight is 0, falls back to Seed.
// ============================================================================
ETrajectoryType ADataCollector::SampleTrajectoryType() const
{
    constexpr int32 N = 7;
    float Weights[N] = {
        WeightSeed, WeightInverse, WeightLoop, WeightEquivalence,
        WeightHardLoop, WeightHardEquivalence, WeightHardInverse
    };
    ETrajectoryType Types[N] = {
        ETrajectoryType::Seed, ETrajectoryType::Inverse, ETrajectoryType::Loop,
        ETrajectoryType::Equivalence,
        ETrajectoryType::Hard_Loop, ETrajectoryType::Hard_Equivalence,
        ETrajectoryType::Hard_Inverse
    };

    float Total = 0.0f;
    for (int32 i = 0; i < N; i++) Total += FMath::Max(0.0f, Weights[i]);
    if (Total < 0.001f) return ETrajectoryType::Seed;

    float Roll = FMath::FRand() * Total;
    float Cumulative = 0.0f;
    for (int32 i = 0; i < N; i++)
    {
        Cumulative += FMath::Max(0.0f, Weights[i]);
        if (Roll <= Cumulative) return Types[i];
    }
    return ETrajectoryType::Seed;
}

// ============================================================================
// Get the inverse-action index of an action.
//
// Mapping:
//   move_forward  <-> move_backward
//   move_left     <-> move_right
//   turn_left     <-> turn_right
//   look_up       <-> look_down
//   no_op         <-> no_op
// ============================================================================
int32 ADataCollector::GetInverseAction(int32 ActionIdx) const
{
    if (ActionIdx < 0 || ActionIdx >= ActionSpace.Num()) return ActionIdx;

    const FString& Name = ActionSpace[ActionIdx].ActionName;

    // Look up the inverse action's name.
    FString InverseName;
    if (Name == TEXT("move_forward"))       InverseName = TEXT("move_backward");
    else if (Name == TEXT("move_backward")) InverseName = TEXT("move_forward");
    else if (Name == TEXT("move_left"))     InverseName = TEXT("move_right");
    else if (Name == TEXT("move_right"))    InverseName = TEXT("move_left");
    else if (Name == TEXT("turn_left"))     InverseName = TEXT("turn_right");
    else if (Name == TEXT("turn_right"))    InverseName = TEXT("turn_left");
    else if (Name == TEXT("look_up"))       InverseName = TEXT("look_down");
    else if (Name == TEXT("look_down"))     InverseName = TEXT("look_up");
    else if (Name == TEXT("no_op"))         InverseName = TEXT("no_op");
    else return ActionIdx; // unknown action returns itself

    // Find the inverse action in ActionSpace.
    for (int32 i = 0; i < ActionSpace.Num(); i++)
    {
        if (ActionSpace[i].ActionName == InverseName)
        {
            return i;
        }
    }
    return ActionIdx; // not found -> return itself
}

// ============================================================================
// Check whether an action is purely a translation (has translation, no rotation).
// ============================================================================
bool ADataCollector::IsTranslationAction(int32 ActionIdx) const
{
    if (ActionIdx < 0 || ActionIdx >= ActionSpace.Num()) return false;
    const FActionDefinition& A = ActionSpace[ActionIdx];
    return !A.DeltaPosition.IsNearlyZero(0.01f) && A.DeltaRotation.IsNearlyZero(0.01f);
}

// ============================================================================
// Sample a natural exploration path.
//
// Design (matches the paper's Section 5.2 "exploration phase"):
//   1. Forward bias: real-world walking is "forward" ~50% of the time.
//   2. Inertia: probability of repeating the previous action is +40%.
//   3. Rotation/translation alternation: avoid more than MaxConsecutiveRotations rotations in a row.
//   4. Avoid sharp oscillation: forbid two opposite actions back-to-back (forward -> backward).
//   5. Sparse pitch: trigger look_up / look_down only occasionally.
//
// Parameters:
//   StartPos / StartRot: starting pose
//   NumSteps: how many steps to sample
//   OutActions: output action-index sequence
//   OutEndPos / OutEndRot: output end pose
// ============================================================================
void ADataCollector::SampleNaturalPath(
    const FVector& StartPos, const FRotator& StartRot,
    int32 NumSteps,
    TArray<int32>& OutActions,
    FVector& OutEndPos, FRotator& OutEndRot,
    bool bForbidPitch)
{
    OutActions.Empty();
    OutActions.Reserve(NumSteps);

    FVector SimPos = StartPos;
    FRotator SimRot = StartRot;
    int32 LastAction = -1;
    int32 ConsecutiveRotations = 0;

    for (int32 Step = 0; Step < NumSteps; Step++)
    {
        // Build effective weights.
        TArray<float> StepWeights;
        StepWeights.SetNum(ActionSpace.Num());
        float TotalWeight = 0.0f;

        for (int32 i = 0; i < ActionSpace.Num(); i++)
        {
            float W = ActionSamplingWeights[i];

            // Rule 1: too many consecutive rotations -> forbid rotations and no_op.
            if (ConsecutiveRotations >= MaxConsecutiveRotations &&
                (IsRotationAction(i) || ActionSpace[i].ActionName == TEXT("no_op")))
            {
                W = 0.0f;
            }

            // Rule 2: forbid back-to-back inverse actions (avoid forward->backward oscillation).
            if (LastAction >= 0 && i == GetInverseAction(LastAction) && i != LastAction)
            {
                W *= 0.05f; // greatly reduce but do not completely forbid
            }

            // Rule 3: inertia -- repeating the previous action gets +40%.
            if (LastAction >= 0 && i == LastAction)
            {
                W *= 1.4f;
            }

            // Rule 4: collision avoidance.
            if (bCollisionAvoidanceSampling && IsTranslationAction(i))
            {
                if (WouldActionCollide(SimPos, SimRot, i))
                {
                    W *= 0.01f; // almost exclude, but not completely 0 (avoid deadlock)
                }
            }

            // Rule 5: forbid pitch changes (used by Loop to avoid Z drift).
            if (bForbidPitch &&
                (ActionSpace[i].ActionName == TEXT("look_up") || ActionSpace[i].ActionName == TEXT("look_down")))
            {
                W = 0.0f;
            }

            StepWeights[i] = W;
            TotalWeight += W;
        }

        // Safety check
        if (TotalWeight < 0.001f)
        {
            for (int32 i = 0; i < ActionSpace.Num(); i++) StepWeights[i] = 1.0f;
            TotalWeight = ActionSpace.Num();
        }

        // Weighted random sample
        float Roll = FMath::FRand() * TotalWeight;
        float Cumulative = 0.0f;
        int32 ChosenAction = ActionSpace.Num() - 1;
        for (int32 i = 0; i < ActionSpace.Num(); i++)
        {
            Cumulative += StepWeights[i];
            if (Roll <= Cumulative)
            {
                ChosenAction = i;
                break;
            }
        }

        OutActions.Add(ChosenAction);
        LastAction = ChosenAction;

        // Update consecutive-rotation counter.
        if (IsRotationAction(ChosenAction))
        {
            ConsecutiveRotations++;
        }
        else
        {
            ConsecutiveRotations = 0;
        }

        // Advance the simulated pose.
        const FActionDefinition& Action = ActionSpace[ChosenAction];
        FVector WorldDelta = SimRot.RotateVector(Action.DeltaPosition);
        SimPos += WorldDelta;
        SimRot = SimRot + Action.DeltaRotation;
    }

    OutEndPos = SimPos;
    OutEndRot = SimRot;
}

// ============================================================================
// Seed trajectory: pure random exploration (control group).
//
// Uses SampleNaturalPath to sample TrajectorySteps natural-path steps.
// This is the original flat trajectory and serves as the non-structured baseline.
// ============================================================================
void ADataCollector::PlanSeedTrajectory()
{
    FVector StartPos = GetActorLocation();
    FRotator StartRot = GetActorRotation();
    FVector EndPos;
    FRotator EndRot;

    SampleNaturalPath(StartPos, StartRot, TrajectorySteps,
        CurrentTrajectoryActions, EndPos, EndRot);
}

// ============================================================================
// Inverse trajectory: "out and back" (A o A^-1 = id).
//
// Corresponds to the paper's Proposition 1 (inverse consistency).
// Construction: P_explore (k steps) + [no_op padding] + P_return (k steps)
//   where P_return = the per-step inverse of reverse(P_explore)
//   Theoretical guarantee: T_{P_return o P_explore}(s_0) = s_0 (in the absence of collisions).
//
// Naturalism: simulates "walk into an area, look around, then retrace your steps".
//   The middle no_op block is like "looking around once you arrive".
//
// Evaluation metric: Inv-Err = LPIPS(frame_0, frame_last)
// ============================================================================
bool ADataCollector::PlanInverseTrajectory()
{
    // k = exploration step count, randomly chosen in [12, 20].
    // Ensure 2k <= TrajectorySteps and leave room for no_op padding.
    int32 MaxK = TrajectorySteps / 2;
    int32 MinK = FMath::Max(8, MaxK / 2);
    int32 K = FMath::RandRange(MinK, MaxK);

    // Sample the exploration path.
    FVector StartPos = GetActorLocation();
    FRotator StartRot = GetActorRotation();
    TArray<int32> ExploreActions;
    FVector ExploreEndPos;
    FRotator ExploreEndRot;

    SampleNaturalPath(StartPos, StartRot, K, ExploreActions, ExploreEndPos, ExploreEndRot);

    // Build the inverse path: take the inverse of each action in reverse(ExploreActions).
    TArray<int32> ReturnActions;
    ReturnActions.Reserve(K);
    for (int32 i = K - 1; i >= 0; i--)
    {
        ReturnActions.Add(GetInverseAction(ExploreActions[i]));
    }

    // Compute the no_op padding length.
    int32 PaddingSteps = TrajectorySteps - 2 * K;

    // Find the no_op index.
    int32 NoOpIdx = -1;
    for (int32 i = 0; i < ActionSpace.Num(); i++)
    {
        if (ActionSpace[i].ActionName == TEXT("no_op"))
        {
            NoOpIdx = i;
            break;
        }
    }
    if (NoOpIdx < 0) NoOpIdx = ActionSpace.Num() - 1; // fallback

    // Assemble the full path: Explore + Padding + Return
    CurrentTrajectoryActions.Empty();
    CurrentTrajectoryActions.Reserve(TrajectorySteps);

    // Exploration segment
    for (int32 a : ExploreActions)
    {
        CurrentTrajectoryActions.Add(a);
    }

    // no_op pause segment (simulates "after arriving, look around")
    for (int32 i = 0; i < PaddingSteps; i++)
    {
        CurrentTrajectoryActions.Add(NoOpIdx);
    }

    // Return segment
    for (int32 a : ReturnActions)
    {
        CurrentTrajectoryActions.Add(a);
    }

    // Validate length
    check(CurrentTrajectoryActions.Num() == TrajectorySteps);

    UE_LOG(LogTemp, Log, TEXT("    [Inverse] Explore=%d steps + Padding=%d + Return=%d steps"),
        K, PaddingSteps, K);

    return true;
}

// ============================================================================
// Compute the "go-home" path from the current pose to a target pose.
//
// Strategy:
//   Step 1: rotate yaw to align (turn_left / turn_right)
//   Step 2: rotate pitch to align (look_up / look_down)
//   Step 3: compute the position delta in the now-aligned target's local frame
//   Step 4: use move_forward / backward + move_left / right to reposition
//
// Note: every action increment is RotationMagnitude (default 30°) and
//       TranslationMagnitude (default 200 cm), so the go-home path is exact only when
//       the deltas are integer multiples of these increments.
// ============================================================================
bool ADataCollector::ComputeGoHomePath(
    const FVector& CurrentPos, const FRotator& CurrentRot,
    const FVector& TargetPos, const FRotator& TargetRot,
    int32 MaxSteps,
    TArray<int32>& OutActions) const
{
    OutActions.Empty();

    // Look up each action's index.
    int32 IdxTurnLeft = -1, IdxTurnRight = -1;
    int32 IdxLookUp = -1, IdxLookDown = -1;
    int32 IdxForward = -1, IdxBackward = -1;
    int32 IdxLeft = -1, IdxRight = -1;
    int32 IdxNoOp = -1;

    for (int32 i = 0; i < ActionSpace.Num(); i++)
    {
        const FString& N = ActionSpace[i].ActionName;
        if (N == TEXT("turn_left"))        IdxTurnLeft = i;
        else if (N == TEXT("turn_right"))  IdxTurnRight = i;
        else if (N == TEXT("look_up"))     IdxLookUp = i;
        else if (N == TEXT("look_down"))   IdxLookDown = i;
        else if (N == TEXT("move_forward"))  IdxForward = i;
        else if (N == TEXT("move_backward")) IdxBackward = i;
        else if (N == TEXT("move_left"))     IdxLeft = i;
        else if (N == TEXT("move_right"))    IdxRight = i;
        else if (N == TEXT("no_op"))         IdxNoOp = i;
    }

    // Safety check
    if (IdxTurnLeft < 0 || IdxTurnRight < 0 || IdxForward < 0 || IdxBackward < 0)
    {
        return false;
    }

    // ---- Step 1: align yaw ----
    float DeltaYaw = TargetRot.Yaw - CurrentRot.Yaw;
    // Normalize to [-180, 180]
    while (DeltaYaw > 180.0f) DeltaYaw -= 360.0f;
    while (DeltaYaw < -180.0f) DeltaYaw += 360.0f;

    int32 YawSteps = FMath::RoundToInt(FMath::Abs(DeltaYaw) / RotationMagnitude);
    int32 YawAction = (DeltaYaw < 0) ? IdxTurnLeft : IdxTurnRight;
    for (int32 i = 0; i < YawSteps; i++)
    {
        OutActions.Add(YawAction);
    }

    // ---- Step 2: align pitch ----
    if (IdxLookUp >= 0 && IdxLookDown >= 0)
    {
        float DeltaPitch = TargetRot.Pitch - CurrentRot.Pitch;
        while (DeltaPitch > 180.0f) DeltaPitch -= 360.0f;
        while (DeltaPitch < -180.0f) DeltaPitch += 360.0f;

        int32 PitchSteps = FMath::RoundToInt(FMath::Abs(DeltaPitch) / RotationMagnitude);
        // look_up has DeltaRotation.Pitch > 0, look_down has < 0.
        int32 PitchAction = (DeltaPitch > 0) ? IdxLookUp : IdxLookDown;
        for (int32 i = 0; i < PitchSteps; i++)
        {
            OutActions.Add(PitchAction);
        }
    }

    // ---- Step 3: compute the positional delta ----
    // After turning around, the heading is TargetRot.
    FVector PosDiff = TargetPos - CurrentPos;
    // Convert the world-space delta into TargetRot's local frame.
    FVector LocalDiff = TargetRot.UnrotateVector(PosDiff);

    // LocalDiff.X = forward-direction offset
    // LocalDiff.Y = right-direction offset
    // Use RoundToInt to quantize to the nearest integer step.
    // Note: when bForbidPitch = true (Loop only), pitch is always 0,
    // so there is no Z drift; quantization error is only in the rotated XY plane.
    int32 ForwardSteps = FMath::RoundToInt(LocalDiff.X / TranslationMagnitude);
    int32 RightSteps = FMath::RoundToInt(LocalDiff.Y / TranslationMagnitude);

    // Compute the residual after quantization. If too large (> TranslationMagnitude * 0.45),
    // we are right between two grid points; this loop will not close exactly -- abort.
    float ResidualX = FMath::Abs(LocalDiff.X - ForwardSteps * TranslationMagnitude);
    float ResidualY = FMath::Abs(LocalDiff.Y - RightSteps * TranslationMagnitude);
    float MaxResidual = FMath::Max(ResidualX, ResidualY);

    // If the Z offset is too large (we have no move_up / move_down), abort as well.
    float ResidualZ = FMath::Abs(LocalDiff.Z);
    if (ResidualZ > TranslationMagnitude * 0.3f)
    {
        UE_LOG(LogTemp, Log, TEXT("    [GoHome] Z residual %.1f cm too large, abort"), ResidualZ);
        return false;
    }

    // ---- Step 4: Forward / Backward ----
    if (ForwardSteps > 0 && IdxForward >= 0)
    {
        for (int32 i = 0; i < ForwardSteps; i++) OutActions.Add(IdxForward);
    }
    else if (ForwardSteps < 0 && IdxBackward >= 0)
    {
        for (int32 i = 0; i < -ForwardSteps; i++) OutActions.Add(IdxBackward);
    }

    // ---- Step 5: Left / Right ----
    if (RightSteps > 0 && IdxRight >= 0)
    {
        for (int32 i = 0; i < RightSteps; i++) OutActions.Add(IdxRight);
    }
    else if (RightSteps < 0 && IdxLeft >= 0)
    {
        for (int32 i = 0; i < -RightSteps; i++) OutActions.Add(IdxLeft);
    }

    // Check whether we exceed the step limit.
    if (OutActions.Num() > MaxSteps)
    {
        OutActions.Empty();
        return false;
    }

    return true;
}

// ============================================================================
// Loop trajectory: "explore + go home" (T_A(s_0) = s_0).
//
// Implements the paper's loop closure property.
// Construction: P_explore (k steps) + P_home (computed) + no_op padding
//   P_explore = natural exploration (with look_up / look_down disabled, keeping pitch = 0)
//   P_home    = the shortest go-home path computed by ComputeGoHomePath
//   Theoretical guarantee: returns exactly to the initial pose (collision-free).
//
// Improvement: with pitch disabled, translation is confined to the XY plane, eliminating Z drift.
//   Go-home accuracy is then limited only by RoundToInt quantization (<= ±50 cm),
//   and most loops can in practice close exactly (rotated translations stay on the grid).
//
// Naturalism: "wander a bit -> head home".
//   The "turn around -> walk straight -> adjust" pattern of the go-home leg looks very natural.
//
// Evaluation metric: Loop-Err = LPIPS(frame_0, frame_last)
// ============================================================================
bool ADataCollector::PlanLoopTrajectory()
{
    FVector StartPos = GetActorLocation();
    FRotator StartRot = GetActorRotation();

    // Try several different exploration lengths.
    for (int32 Attempt = 0; Attempt < 5; Attempt++)
    {
        // Exploration step count: random in [15, 28] (leave enough steps to go home).
        int32 K = FMath::RandRange(15, FMath::Min(28, TrajectorySteps - 10));

        // Sample the exploration path (forbid look_up / look_down to avoid Z drift).
        TArray<int32> ExploreActions;
        FVector ExploreEndPos;
        FRotator ExploreEndRot;
        SampleNaturalPath(StartPos, StartRot, K, ExploreActions, ExploreEndPos, ExploreEndRot,
            /*bForbidPitch=*/ true);

        // Compute the go-home path.
        int32 MaxHomeSteps = TrajectorySteps - K;
        TArray<int32> HomeActions;
        bool bHomeOK = ComputeGoHomePath(ExploreEndPos, ExploreEndRot,
            StartPos, StartRot, MaxHomeSteps, HomeActions);

        if (bHomeOK)
        {
            // Success! Assemble the full path.
            int32 PaddingSteps = TrajectorySteps - K - HomeActions.Num();

            int32 NoOpIdx = -1;
            for (int32 i = 0; i < ActionSpace.Num(); i++)
            {
                if (ActionSpace[i].ActionName == TEXT("no_op")) { NoOpIdx = i; break; }
            }
            if (NoOpIdx < 0) NoOpIdx = ActionSpace.Num() - 1;

            CurrentTrajectoryActions.Empty();
            CurrentTrajectoryActions.Reserve(TrajectorySteps);

            // Exploration segment
            for (int32 a : ExploreActions) CurrentTrajectoryActions.Add(a);

            // no_op pause segment (pause at the farthest point and look around before heading back)
            for (int32 i = 0; i < PaddingSteps; i++) CurrentTrajectoryActions.Add(NoOpIdx);

            // Go-home segment
            for (int32 a : HomeActions) CurrentTrajectoryActions.Add(a);

            check(CurrentTrajectoryActions.Num() == TrajectorySteps);

            UE_LOG(LogTemp, Log, TEXT("    [Loop] Explore=%d + Pause=%d + GoHome=%d (attempt %d)"),
                K, PaddingSteps, HomeActions.Num(), Attempt + 1);

            return true;
        }
    }

    // All 5 attempts failed.
    UE_LOG(LogTemp, Warning, TEXT("    [Loop] Failed to find valid loop path in 5 attempts"));
    return false;
}

// ============================================================================
// Randomly shuffle within commutative segments of an action sequence.
//
// Rationale: in this discrete action space:
//   - Same-type translation actions commute with each other (collision-free):
//     T_forward o T_left = T_left o T_forward (both are translations in the local frame).
//   - Same-type rotation actions commute with each other:
//     yaw and pitch are on different axes.
//
// Note: translation and rotation do NOT commute (rotation changes the translation direction).
//
// Algorithm:
//   1. Scan the sequence and identify each maximal pure-translation or pure-rotation segment.
//   2. Apply a Fisher-Yates shuffle within each segment.
// ============================================================================
void ADataCollector::ShuffleCommutativeSegments(TArray<int32>& Actions) const
{
    int32 N = Actions.Num();
    if (N < 2) return;

    int32 SegStart = 0;
    while (SegStart < N)
    {
        // Determine the type of the current segment.
        bool bIsTranslation = IsTranslationAction(Actions[SegStart]);
        bool bIsRotation = IsRotationAction(Actions[SegStart]);

        if (!bIsTranslation && !bIsRotation)
        {
            // no_op, skip
            SegStart++;
            continue;
        }

        // Find the end of the segment.
        int32 SegEnd = SegStart + 1;
        while (SegEnd < N)
        {
            if (bIsTranslation && IsTranslationAction(Actions[SegEnd]))
            {
                SegEnd++;
            }
            else if (bIsRotation && IsRotationAction(Actions[SegEnd]))
            {
                SegEnd++;
            }
            else
            {
                break;
            }
        }

        // Only shuffle if the segment length >= 2.
        int32 SegLen = SegEnd - SegStart;
        if (SegLen >= 2)
        {
            // Fisher-Yates shuffle
            for (int32 i = SegLen - 1; i > 0; i--)
            {
                int32 j = FMath::RandRange(0, i);
                if (i != j)
                {
                    int32 Temp = Actions[SegStart + i];
                    Actions[SegStart + i] = Actions[SegStart + j];
                    Actions[SegStart + j] = Temp;
                }
            }
        }

        SegStart = SegEnd;
    }
}

// -----------------------------------------------------------------------------
// Helper: compute the maximum spatial divergence (cm) between two action sequences
// starting from the same pose.
//
// Step-by-step simulates each step's position for PathA and PathB, returning
// max ||PosA_i - PosB_i||. Used by Equivalence's spatial-divergence check.
// -----------------------------------------------------------------------------
float ADataCollector::ComputeMaxSpatialDivergence(
    const FVector& StartPos, const FRotator& StartRot,
    const TArray<int32>& PathA, const TArray<int32>& PathB) const
{
    FVector PosA = StartPos, PosB = StartPos;
    FRotator RotA = StartRot, RotB = StartRot;

    float MaxDiv = 0.0f;
    const int32 Steps = FMath::Min(PathA.Num(), PathB.Num());

    for (int32 i = 0; i < Steps; i++)
    {
        // Advance A
        if (PathA[i] >= 0 && PathA[i] < ActionSpace.Num())
        {
            const FActionDefinition& ActA = ActionSpace[PathA[i]];
            PosA += RotA.RotateVector(ActA.DeltaPosition);
            RotA = RotA + ActA.DeltaRotation;
        }
        // Advance B
        if (PathB[i] >= 0 && PathB[i] < ActionSpace.Num())
        {
            const FActionDefinition& ActB = ActionSpace[PathB[i]];
            PosB += RotB.RotateVector(ActB.DeltaPosition);
            RotB = RotB + ActB.DeltaRotation;
        }
        float Dist = (PosA - PosB).Size();
        if (Dist > MaxDiv) MaxDiv = Dist;
    }
    return MaxDiv;
}

// -----------------------------------------------------------------------------
// Helper: look up an action's index in ActionSpace by name.
// -----------------------------------------------------------------------------
static int32 FindActionIndexByName(const TArray<FActionDefinition>& Actions, const FString& Name)
{
    for (int32 i = 0; i < Actions.Num(); i++)
    {
        if (Actions[i].ActionName == Name) return i;
    }
    return -1;
}

// -----------------------------------------------------------------------------
// Helper: simulate the end pose of an action sequence (no rendering, no collision check).
// -----------------------------------------------------------------------------
static void SimulatePathEndpoint(
    const TArray<FActionDefinition>& ActionSpace,
    const FVector& StartPos, const FRotator& StartRot,
    const TArray<int32>& Actions,
    FVector& OutEndPos, FRotator& OutEndRot)
{
    FVector Pos = StartPos;
    FRotator Rot = StartRot;
    for (int32 idx : Actions)
    {
        if (idx < 0 || idx >= ActionSpace.Num()) continue;
        const FActionDefinition& A = ActionSpace[idx];
        Pos += Rot.RotateVector(A.DeltaPosition);
        Rot = Rot + A.DeltaRotation;
    }
    OutEndPos = Pos;
    OutEndRot = Rot;
}

// ============================================================================
// Equivalence trajectory: "different paths, same destination" (T_A(s_0) = T_B(s_0), A != B).
//
// Implements the paper's path-equivalence property.
// Two-phase construction:
//   1. Primary strategy: generate a natural path P, then shuffle commutative segments to obtain P'.
//      Validate (a) the action-difference ratio >= MinEquivDivergence,
//               (b) the intermediate spatial divergence >= MinEquivSpatialDivergence.
//   2. Fallback strategy: if shuffle cannot reach sufficient divergence (the natural path's
//      commutative segments are too short), use mini L-shape vs zig-zag (same construction
//      as Hard_Equivalence, but with m, n in [3, 6]).
//
// Output: P -> CurrentTrajectoryActions, P' -> PairedTrajectoryActions.
//
// Evaluation metric: Equiv-Err = LPIPS(frame_P_last, frame_P'_last)
// ============================================================================
bool ADataCollector::PlanEquivalenceTrajectory()
{
    const FVector  StartPos = GetActorLocation();
    const FRotator StartRot = GetActorRotation();

    // ---------------- Primary strategy: shuffle commutative segments ----------------
    FVector EndPos;
    FRotator EndRot;

    SampleNaturalPath(StartPos, StartRot, TrajectorySteps,
        CurrentTrajectoryActions, EndPos, EndRot);

    PairedTrajectoryActions = CurrentTrajectoryActions;
    ShuffleCommutativeSegments(PairedTrajectoryActions);

    // Tally the action differences.
    int32 DiffCount = 0;
    for (int32 i = 0; i < TrajectorySteps; i++)
    {
        if (CurrentTrajectoryActions[i] != PairedTrajectoryActions[i]) DiffCount++;
    }
    float DivRatio = (float)DiffCount / FMath::Max(TrajectorySteps, 1);

    // If the shuffle reaches the action-divergence threshold, also check spatial divergence.
    if (DivRatio >= MinEquivDivergence)
    {
        float MaxSpatialDiv = ComputeMaxSpatialDivergence(
            StartPos, StartRot, CurrentTrajectoryActions, PairedTrajectoryActions);

        if (MaxSpatialDiv >= MinEquivSpatialDivergence)
        {
            bHasPairedTrajectory = true;
            UE_LOG(LogTemp, Log, TEXT("    [Equivalence] Shuffle OK: %d/%d steps differ (%.0f%%), max spatial div %.1f cm"),
                DiffCount, TrajectorySteps, DivRatio * 100.0f, MaxSpatialDiv);
            return true;
        }

        UE_LOG(LogTemp, Log, TEXT("    [Equivalence] Shuffle spatial div %.1f cm < min %.1f cm, trying mini L-shape fallback"),
            MaxSpatialDiv, MinEquivSpatialDivergence);
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("    [Equivalence] Shuffle divergence %.0f%% < min %.0f%%, trying mini L-shape fallback"),
            DivRatio * 100.0f, MinEquivDivergence * 100.0f);
    }

    // ---------------- Fallback strategy: mini L-shape vs zig-zag ----------------
    //
    // Same construction as PlanHardEquivalenceTrajectory but with m, n in [3, 6] (Easy scale).
    // Translation-subgroup commutativity guarantees the endpoints match exactly.

    const int32 IdxForward  = FindActionIndexByName(ActionSpace, TEXT("move_forward"));
    const int32 IdxBackward = FindActionIndexByName(ActionSpace, TEXT("move_backward"));
    const int32 IdxLeft     = FindActionIndexByName(ActionSpace, TEXT("move_left"));
    const int32 IdxRight    = FindActionIndexByName(ActionSpace, TEXT("move_right"));
    const int32 IdxNoOp     = FindActionIndexByName(ActionSpace, TEXT("no_op"));
    const int32 IdxTurnL    = FindActionIndexByName(ActionSpace, TEXT("turn_left"));
    const int32 IdxTurnR    = FindActionIndexByName(ActionSpace, TEXT("turn_right"));
    const int32 IdxLookUp   = FindActionIndexByName(ActionSpace, TEXT("look_up"));
    const int32 IdxLookDown = FindActionIndexByName(ActionSpace, TEXT("look_down"));

    if (IdxForward < 0 || IdxBackward < 0 || IdxLeft < 0 || IdxRight < 0 || IdxNoOp < 0)
    {
        return false;
    }

    // Self-inverse rotation pairs available: look_up <-> look_down, turn_left <-> turn_right.
    const bool bHasRotations = (IdxTurnL >= 0 && IdxTurnR >= 0 && IdxLookUp >= 0 && IdxLookDown >= 0);

    struct FAxisPair { int32 U, V; const TCHAR* Tag; };
    FAxisPair Pairs[] = {
        { IdxForward,  IdxLeft,     TEXT("F-L") },
        { IdxForward,  IdxRight,    TEXT("F-R") },
        { IdxBackward, IdxLeft,     TEXT("B-L") },
        { IdxBackward, IdxRight,    TEXT("B-R") },
    };
    const FAxisPair& P = Pairs[FMath::RandRange(0, 3)];

    // Easy scale: m, n in [3, 6] (Hard tier uses [6, 12])
    const int32 m = FMath::RandRange(3, 6);
    const int32 n = FMath::RandRange(3, 6);
    if (m + n > TrajectorySteps - 4) return false;

    // A: L-shape  U^m . V^n
    TArray<int32> PathA;
    PathA.Reserve(TrajectorySteps);
    for (int32 i = 0; i < m; i++) PathA.Add(P.U);
    for (int32 i = 0; i < n; i++) PathA.Add(P.V);

    // B: zig-zag  (UV)^k . U^(m-k) . V^(n-k)  with k = min(m, n)
    const int32 k = FMath::Min(m, n);
    TArray<int32> PathB;
    PathB.Reserve(TrajectorySteps);
    for (int32 i = 0; i < k; i++) { PathB.Add(P.U); PathB.Add(P.V); }
    for (int32 i = 0; i < m - k; i++) PathB.Add(P.U);
    for (int32 i = 0; i < n - k; i++) PathB.Add(P.V);

    check(PathA.Num() == PathB.Num());

    // Scatter self-inverse rotation pairs evenly through the path's middle, replacing no_op padding.
    // PathA uses look_up + look_down (pitch look-around), PathB uses turn_left + turn_right (yaw look-around).
    // Each rotation pair takes 2 slots and cancels itself, so it does not affect the final pose;
    // an odd remainder of 1 slot is filled with no_op.
    auto ScatterSelfInversePairs = [&](TArray<int32>& Path, int32 TargetLen,
                                       int32 ActionFwd, int32 ActionRev, int32 FallbackNoOp)
    {
        const int32 Padding = TargetLen - Path.Num();
        if (Padding <= 0) return;
        const int32 NumPairs  = Padding / 2;   // number of complete rotation pairs
        const int32 Remainder = Padding % 2;   // odd remainder of 1 -> use no_op
        const int32 TotalInsertions = NumPairs + Remainder;
        const int32 ActionLen = Path.Num();
        TArray<int32> Result;
        Result.Reserve(TargetLen);
        int32 Inserted = 0;
        for (int32 i = 0; i < ActionLen; i++)
        {
            Result.Add(Path[i]);
            const int32 ShouldHaveInserted = ((i + 1) * TotalInsertions) / ActionLen;
            while (Inserted < ShouldHaveInserted)
            {
                if (Inserted < NumPairs)
                {
                    Result.Add(ActionFwd);
                    Result.Add(ActionRev);
                }
                else
                {
                    Result.Add(FallbackNoOp);
                }
                Inserted++;
            }
        }
        Path = Result;
    };

    // Fallback: scatter pure no_op (when rotation actions are unavailable).
    auto ScatterNoOps = [&](TArray<int32>& Path, int32 TargetLen, int32 NoOpAction) {
        const int32 Padding = TargetLen - Path.Num();
        if (Padding <= 0) return;
        const int32 ActionLen = Path.Num();
        TArray<int32> Result;
        Result.Reserve(TargetLen);
        int32 Inserted = 0;
        for (int32 i = 0; i < ActionLen; i++)
        {
            Result.Add(Path[i]);
            const int32 ShouldHaveInserted = ((i + 1) * Padding) / ActionLen;
            while (Inserted < ShouldHaveInserted)
            {
                Result.Add(NoOpAction);
                Inserted++;
            }
        }
        Path = Result;
    };

    if (bHasRotations)
    {
        // PathA: pitch look-around (look_up + look_down)
        ScatterSelfInversePairs(PathA, TrajectorySteps, IdxLookUp, IdxLookDown, IdxNoOp);
        // PathB: yaw look-around (turn_left + turn_right)
        ScatterSelfInversePairs(PathB, TrajectorySteps, IdxTurnL, IdxTurnR, IdxNoOp);
    }
    else
    {
        ScatterNoOps(PathA, TrajectorySteps, IdxNoOp);
        ScatterNoOps(PathB, TrajectorySteps, IdxNoOp);
    }

    // Collision check
    if (!IsPathCollisionFree(StartPos, StartRot, PathA)) return false;
    if (!IsPathCollisionFree(StartPos, StartRot, PathB)) return false;

    // Make sure A != B.
    bool bDifferent = false;
    for (int32 i = 0; i < PathA.Num(); i++) if (PathA[i] != PathB[i]) { bDifferent = true; break; }
    if (!bDifferent) return false;

    CurrentTrajectoryActions = PathA;
    PairedTrajectoryActions = PathB;
    bHasPairedTrajectory = true;

    UE_LOG(LogTemp, Log, TEXT("    [Equivalence] Mini L-shape fallback: axis=%s m=%d n=%d (A=L-shape, B=zigzag)"),
        P.Tag, m, n);

    return true;
}


// ============================================================================
// ============================================================================
//                           HARD TIER  (2026-04 submission)
// ============================================================================
// ============================================================================

// -----------------------------------------------------------------------------
// Helper: swept line trace along a candidate action sequence (collision pre-check).
// Returns true if the entire path is collision-free.
// -----------------------------------------------------------------------------
bool ADataCollector::IsPathCollisionFree(
    const FVector& StartPos, const FRotator& StartRot,
    const TArray<int32>& Actions) const
{
    FVector Pos = StartPos;
    FRotator Rot = StartRot;
    for (int32 idx : Actions)
    {
        if (WouldActionCollide(Pos, Rot, idx)) return false;
        if (idx < 0 || idx >= ActionSpace.Num()) continue;
        const FActionDefinition& A = ActionSpace[idx];
        Pos += Rot.RotateVector(A.DeltaPosition);
        Rot = Rot + A.DeltaRotation;
    }
    return true;
}


// ============================================================================
// Hard Loop: geometric-template closure.
//
// Design rationale: unlike Easy Loop's "explore + GoHome", Hard Loop directly uses
// topologically closed geometric templates. Templates close **exactly** on the
// discrete action grid and do not depend on quantization residuals.
//
// Template library (no_op padded out to TrajectorySteps):
//
//   TEMPLATE_RECTANGLE_4      Quadrilateral: (F^n . L^6)^4   ->  4 x (n + 6) steps
//       - L^6 = 6 x 15° = 90° left turn
//       - default n=3, length = 4x9 = 36 steps; padding = 4 steps
//
//   TEMPLATE_RECTANGLE_4R     Reverse rectangle: (F^n . R^6)^4 (right-turn variant)
//
//   TEMPLATE_TRIANGLE_3       Equilateral triangle: (F^n . L^8)^3  ->  3 x (n + 8) steps
//       - L^8 = 8 x 15° = 120° left turn
//       - default n=3, length = 3x11 = 33 steps
//
//   TEMPLATE_HEXAGON_6        Hexagon: (F^n . L^4)^6  ->  6 x (n + 4) steps
//       - L^4 = 4 x 15° = 60° left turn
//       - default n=2, length = 6x6 = 36 steps
//
//   TEMPLATE_FIGURE8          Figure-8: a left loop followed by a right loop
//       - (F^2 . L^6)^4 . (F^2 . R^6)^4 = 64 steps total; for a 40-step trajectory it
//         would shrink to (F^2 . L^3)^2 . (F^2 . R^3)^2... 40 steps is too short to
//         produce a clean 8-shape; not enabled for now.
//
// Evaluation metric: Loop-Err = LPIPS(frame_0, frame_last)
// ============================================================================
bool ADataCollector::PlanHardLoopTrajectory()
{
    // Look up the key actions.
    const int32 IdxForward  = FindActionIndexByName(ActionSpace, TEXT("move_forward"));
    const int32 IdxBackward = FindActionIndexByName(ActionSpace, TEXT("move_backward"));
    const int32 IdxLeft     = FindActionIndexByName(ActionSpace, TEXT("move_left"));
    const int32 IdxRight    = FindActionIndexByName(ActionSpace, TEXT("move_right"));
    const int32 IdxTurnL    = FindActionIndexByName(ActionSpace, TEXT("turn_left"));
    const int32 IdxTurnR    = FindActionIndexByName(ActionSpace, TEXT("turn_right"));
    const int32 IdxNoOp     = FindActionIndexByName(ActionSpace, TEXT("no_op"));

    if (IdxForward < 0 || IdxTurnL < 0 || IdxTurnR < 0 || IdxNoOp < 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("    [Hard_Loop] required actions missing"));
        return false;
    }

    const FVector  StartPos = GetActorLocation();
    const FRotator StartRot = GetActorRotation();

    // Compute the per-step rotation magnitude using the (possibly jittered) ActionSpace.
    // Note: after Option B's ApplyHardTierScaleJitter modifies ActionSpace, turn_left's Yaw may
    // become 30° or 7.5° instead of 15°. The "steps per corner" geometry must use the runtime
    // value, otherwise closure error blows up.
    float PerStepYaw = FMath::Abs(ActionSpace[IdxTurnL].DeltaRotation.Yaw);
    if (PerStepYaw < 1e-3f) PerStepYaw = 15.0f;  // fallback

    auto StepsForAngle = [&](float TargetDeg) -> int32
    {
        return FMath::RoundToInt(TargetDeg / PerStepYaw);
    };
    const int32 K90  = StepsForAngle(90.0f);     // rectangle corner
    const int32 K120 = StepsForAngle(120.0f);    // triangle corner
    const int32 K60  = StepsForAngle(60.0f);     // hexagon corner

    // Closure check: if the quantized degrees per corner deviate by >1° from target, the template is unusable.
    auto AngleClosesExactly = [&](int32 Steps, float TargetDeg) -> bool
    {
        return FMath::Abs(Steps * PerStepYaw - TargetDeg) < 1.0f;
    };
    const bool bRectOK = K90  > 0 && AngleClosesExactly(K90,  90.0f);
    const bool bTriOK  = K120 > 0 && AngleClosesExactly(K120, 120.0f);
    const bool bHexOK  = K60  > 0 && AngleClosesExactly(K60,  60.0f);

    // Template enumeration + parameter randomization.
    // Structure: TArray<TArray<int32>> -- each top-level entry is a "segment (n moves + k turns)" template.
    // For simplicity (and to align with the paper, which features the 3 most common closures),
    // we list 4 candidates and try them in order of success.
    struct FLoopTemplate
    {
        FString Name;
        TArray<int32> Pattern;   // full path (without padding)
        int32 NumSides;          // number of sides (used to scatter no_op after corners)
    };

    TArray<FLoopTemplate> Candidates;

    auto BuildRectangle = [&](int32 n, int32 TurnIdx, const FString& Name) -> FLoopTemplate
    {
        FLoopTemplate T; T.Name = Name; T.NumSides = 4;
        for (int32 side = 0; side < 4; side++)
        {
            for (int32 i = 0; i < n; i++) T.Pattern.Add(IdxForward);
            for (int32 i = 0; i < K90; i++) T.Pattern.Add(TurnIdx);   // 90°
        }
        return T;
    };
    auto BuildTriangle = [&](int32 n, int32 TurnIdx, const FString& Name) -> FLoopTemplate
    {
        FLoopTemplate T; T.Name = Name; T.NumSides = 3;
        for (int32 side = 0; side < 3; side++)
        {
            for (int32 i = 0; i < n; i++) T.Pattern.Add(IdxForward);
            for (int32 i = 0; i < K120; i++) T.Pattern.Add(TurnIdx);  // 120°
        }
        return T;
    };
    auto BuildHexagon = [&](int32 n, int32 TurnIdx, const FString& Name) -> FLoopTemplate
    {
        FLoopTemplate T; T.Name = Name; T.NumSides = 6;
        for (int32 side = 0; side < 6; side++)
        {
            for (int32 i = 0; i < n; i++) T.Pattern.Add(IdxForward);
            for (int32 i = 0; i < K60; i++) T.Pattern.Add(TurnIdx);   // 60°
        }
        return T;
    };

    // Choose n such that total <= TrajectorySteps (leaves room for padding):
    //   rectangle: 4n + 4*K90  <= TrajectorySteps
    //   triangle:  3n + 3*K120 <= TrajectorySteps
    //   hexagon:   6n + 6*K60  <= TrajectorySteps
    const int32 RectN = bRectOK ? FMath::RandRange(
        2, FMath::Max(2, (TrajectorySteps - 4 * K90) / 4)) : 0;
    const int32 TriN  = bTriOK ? FMath::RandRange(
        2, FMath::Max(2, (TrajectorySteps - 3 * K120) / 3)) : 0;
    const int32 HexN  = bHexOK ? FMath::RandRange(
        1, FMath::Max(1, (TrajectorySteps - 6 * K60) / 6)) : 0;

    // Randomize left vs right turn direction.
    const bool bTurnLeft = FMath::RandBool();
    const int32 TurnIdx = bTurnLeft ? IdxTurnL : IdxTurnR;
    const FString TurnTag = bTurnLeft ? TEXT("CCW") : TEXT("CW");

    if (bRectOK)
        Candidates.Add(BuildRectangle(RectN, TurnIdx,
            FString::Printf(TEXT("rect_n%d_k%d_%s"), RectN, K90, *TurnTag)));
    if (bTriOK)
        Candidates.Add(BuildTriangle (TriN,  TurnIdx,
            FString::Printf(TEXT("tri_n%d_k%d_%s"),  TriN, K120, *TurnTag)));
    if (bHexOK)
        Candidates.Add(BuildHexagon  (HexN,  TurnIdx,
            FString::Printf(TEXT("hex_n%d_k%d_%s"),  HexN, K60,  *TurnTag)));

    if (Candidates.Num() == 0)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("    [Hard_Loop] no template closes under PerStepYaw=%.2f°"), PerStepYaw);
        return false;
    }

    // Shuffle the try order to get more type variety.
    for (int32 i = Candidates.Num() - 1; i > 0; i--)
    {
        const int32 j = FMath::RandRange(0, i);
        if (i != j) Candidates.Swap(i, j);
    }

    for (const FLoopTemplate& T : Candidates)
    {
        if (T.Pattern.Num() > TrajectorySteps) continue;

        // Line-trace pre-check: must be collision-free along the full path.
        if (!IsPathCollisionFree(StartPos, StartRot, T.Pattern)) continue;

        // Assembly: scatter no_op after each corner of the template (pause at each corner to take in the view).
        const int32 TotalPadding = TrajectorySteps - T.Pattern.Num();
        const int32 S = T.NumSides;                          // number of corners
        const int32 StepsPerSide = T.Pattern.Num() / S;      // steps per side (including the turn)

        // Distribute TotalPadding evenly across the S corners.
        // base = no_ops per corner, extra = remainder (assigned to the first few corners).
        const int32 PadBase  = TotalPadding / S;
        const int32 PadExtra = TotalPadding % S;

        CurrentTrajectoryActions.Empty();
        CurrentTrajectoryActions.Reserve(TrajectorySteps);
        for (int32 side = 0; side < S; side++)
        {
            // Copy this side's actions (forward + turn).
            const int32 SegStart = side * StepsPerSide;
            const int32 SegEnd   = SegStart + StepsPerSide;
            for (int32 j = SegStart; j < SegEnd; j++)
            {
                CurrentTrajectoryActions.Add(T.Pattern[j]);
            }
            // After the corner, insert no_op (the first PadExtra corners get one extra step).
            const int32 PadHere = PadBase + (side < PadExtra ? 1 : 0);
            for (int32 p = 0; p < PadHere; p++)
            {
                CurrentTrajectoryActions.Add(IdxNoOp);
            }
        }

        check(CurrentTrajectoryActions.Num() == TrajectorySteps);

        UE_LOG(LogTemp, Log, TEXT("    [Hard_Loop] template=%s len=%d + padding=%d (scattered to %d corners)"),
            *T.Name, T.Pattern.Num(), TotalPadding, S);
        return true;
    }

    UE_LOG(LogTemp, Warning, TEXT("    [Hard_Loop] all templates blocked by collision"));
    return false;
}


// ============================================================================
// Hard Equivalence: macroscopic reordering (L-shape vs zig-zag).
//
// The legacy Equivalence relied on intra-segment shuffles, with weak visual
// divergence. Hard Equivalence instead uses two templates that are multiset-equal
// in the **pure-translation** (abelian) subgroup but whose sequences differ
// **dramatically**:
//
//   A_LSHAPE:   U^m . V^n        (L-shape: all U first, then all V)
//   A_ZIGZAG:   (U . V)^k . U^(m-k) . V^(n-k)    (zig-zag: alternate, then finish)
//     where k = min(m, n).
//
// As long as we stay in pure translation (no rotation), SE(2)'s translation subgroup
// is abelian and:
//   T_{A_LSHAPE} = T_{A_ZIGZAG}  (same m x U + n x V)
//
// But the intermediate trajectory shapes differ entirely: L-shape is a single big
// corner; zig-zag is a sequence of small steps. In paper figures the visual
// distinction is obvious -- a reviewer immediately understands "these two paths
// reach the same place but in clearly different ways".
//
// Evaluation metric: Equiv-Err = LPIPS(A_last, B_last)
// ============================================================================
bool ADataCollector::PlanHardEquivalenceTrajectory()
{
    const int32 IdxForward  = FindActionIndexByName(ActionSpace, TEXT("move_forward"));
    const int32 IdxBackward = FindActionIndexByName(ActionSpace, TEXT("move_backward"));
    const int32 IdxLeft     = FindActionIndexByName(ActionSpace, TEXT("move_left"));
    const int32 IdxRight    = FindActionIndexByName(ActionSpace, TEXT("move_right"));
    const int32 IdxNoOp     = FindActionIndexByName(ActionSpace, TEXT("no_op"));
    const int32 IdxTurnL    = FindActionIndexByName(ActionSpace, TEXT("turn_left"));
    const int32 IdxTurnR    = FindActionIndexByName(ActionSpace, TEXT("turn_right"));
    const int32 IdxLookUp   = FindActionIndexByName(ActionSpace, TEXT("look_up"));
    const int32 IdxLookDown = FindActionIndexByName(ActionSpace, TEXT("look_down"));

    if (IdxForward < 0 || IdxBackward < 0 || IdxLeft < 0 || IdxRight < 0 || IdxNoOp < 0)
    {
        return false;
    }

    // Self-inverse rotation pairs available: look_up <-> look_down, turn_left <-> turn_right.
    const bool bHasRotations = (IdxTurnL >= 0 && IdxTurnR >= 0 && IdxLookUp >= 0 && IdxLookDown >= 0);

    struct FAxisPair { int32 U, V; const TCHAR* Tag; };
    FAxisPair Pairs[] = {
        { IdxForward,  IdxLeft,     TEXT("F-L") },
        { IdxForward,  IdxRight,    TEXT("F-R") },
        { IdxBackward, IdxLeft,     TEXT("B-L") },
        { IdxBackward, IdxRight,    TEXT("B-R") },
    };
    const FAxisPair& P = Pairs[FMath::RandRange(0, 3)];

    // m + n <= TrajectorySteps - PaddingMin (leave room for padding fallback)
    const int32 m = FMath::RandRange(6, 12);
    const int32 n = FMath::RandRange(6, 12);
    if (m + n > TrajectorySteps - 4) return false;

    const FVector  StartPos = GetActorLocation();
    const FRotator StartRot = GetActorRotation();

    // A: L-shape   U^m . V^n
    TArray<int32> PathA;
    PathA.Reserve(TrajectorySteps);
    for (int32 i = 0; i < m; i++) PathA.Add(P.U);
    for (int32 i = 0; i < n; i++) PathA.Add(P.V);

    // B: zig-zag  (UV)^k . U^(m-k) . V^(n-k)  with k = min(m, n)
    const int32 k = FMath::Min(m, n);
    TArray<int32> PathB;
    PathB.Reserve(TrajectorySteps);
    for (int32 i = 0; i < k; i++) { PathB.Add(P.U); PathB.Add(P.V); }
    for (int32 i = 0; i < m - k; i++) PathB.Add(P.U);
    for (int32 i = 0; i < n - k; i++) PathB.Add(P.V);

    check(PathA.Num() == PathB.Num());

    // Scatter self-inverse rotation pairs evenly through the path's middle, replacing no_op padding.
    // PathA uses look_up + look_down (pitch look-around), PathB uses turn_left + turn_right (yaw look-around).
    auto ScatterSelfInversePairs = [&](TArray<int32>& Path, int32 TargetLen,
                                       int32 ActionFwd, int32 ActionRev, int32 FallbackNoOp)
    {
        const int32 Padding = TargetLen - Path.Num();
        if (Padding <= 0) return;
        const int32 NumPairs  = Padding / 2;
        const int32 Remainder = Padding % 2;
        const int32 TotalInsertions = NumPairs + Remainder;
        const int32 ActionLen = Path.Num();
        TArray<int32> Result;
        Result.Reserve(TargetLen);
        int32 Inserted = 0;
        for (int32 i = 0; i < ActionLen; i++)
        {
            Result.Add(Path[i]);
            const int32 ShouldHaveInserted = ((i + 1) * TotalInsertions) / ActionLen;
            while (Inserted < ShouldHaveInserted)
            {
                if (Inserted < NumPairs)
                {
                    Result.Add(ActionFwd);
                    Result.Add(ActionRev);
                }
                else
                {
                    Result.Add(FallbackNoOp);
                }
                Inserted++;
            }
        }
        Path = Result;
    };

    // Fallback: scatter pure no_op (when rotation actions are unavailable).
    auto ScatterNoOps = [&](TArray<int32>& Path, int32 TargetLen, int32 NoOpAction) {
        const int32 Padding = TargetLen - Path.Num();
        if (Padding <= 0) return;
        const int32 ActionLen = Path.Num();
        TArray<int32> Result;
        Result.Reserve(TargetLen);
        int32 Inserted = 0;
        for (int32 i = 0; i < ActionLen; i++)
        {
            Result.Add(Path[i]);
            const int32 ShouldHaveInserted = ((i + 1) * Padding) / ActionLen;
            while (Inserted < ShouldHaveInserted)
            {
                Result.Add(NoOpAction);
                Inserted++;
            }
        }
        Path = Result;
    };

    if (bHasRotations)
    {
        // PathA: pitch look-around (look_up + look_down)
        ScatterSelfInversePairs(PathA, TrajectorySteps, IdxLookUp, IdxLookDown, IdxNoOp);
        // PathB: yaw look-around (turn_left + turn_right)
        ScatterSelfInversePairs(PathB, TrajectorySteps, IdxTurnL, IdxTurnR, IdxNoOp);
    }
    else
    {
        ScatterNoOps(PathA, TrajectorySteps, IdxNoOp);
        ScatterNoOps(PathB, TrajectorySteps, IdxNoOp);
    }

    // Collision check
    if (!IsPathCollisionFree(StartPos, StartRot, PathA)) return false;
    if (!IsPathCollisionFree(StartPos, StartRot, PathB)) return false;

    // Make sure A != B (handles the degenerate m=0 or n=0 case).
    bool bDifferent = false;
    for (int32 i = 0; i < PathA.Num(); i++) if (PathA[i] != PathB[i]) { bDifferent = true; break; }
    if (!bDifferent) return false;

    CurrentTrajectoryActions = PathA;
    PairedTrajectoryActions = PathB;
    bHasPairedTrajectory = true;

    UE_LOG(LogTemp, Log, TEXT("    [Hard_Equivalence] axis=%s m=%d n=%d  (A=L-shape, B=zigzag)"),
        P.Tag, m, n);

    return true;
}




// ============================================================================
// Hard Inverse: A . A^-1 with mixed rotations.
//
// Design motivation (2026-04 submission supplement):
//   In Easy Inverse, the explore segment is sampled by SampleNaturalPath, which is
//   only ~30% rotations on average; the rest is pure translation. Translations
//   commute in SE(2) / SE(3), so most Easy Inverse instances are essentially
//   instances of MIND / symmetric round-trip baseline's abelian subset
//   (A . A^-1 where A in the translation subgroup).
//
//   Hard Inverse forces the explore segment to **interleave** at least K_rot
//   rotation steps, so A . A^-1's geometric trace is no longer a straight
//   out-and-back but instead "L-shape forward + 90° turn + continue forward + retrace".
//   This actually tests whether the model can perform path inversion on a
//   non-commutative subset.
//
// Construction:
//   1. Random explore: forward*m1, turn*r1, forward*m2, turn*r2, ...
//      Constraints: total rotation steps >= K_rot, total steps K in [8, 16]
//   2. Return = reverse(explore), each step replaced by GetInverseAction
//   3. Pad with no_op up to TrajectorySteps
//
// Validation: simulated end pose ~ start pose (tolerate |delta_theta| < 1° quantization residual)
// Evaluation: Inv-Err = LPIPS(frame_0, frame_last); ideal = 0
// ============================================================================
bool ADataCollector::PlanHardInverseTrajectory()
{
    const int32 IdxForward  = FindActionIndexByName(ActionSpace, TEXT("move_forward"));
    const int32 IdxBackward = FindActionIndexByName(ActionSpace, TEXT("move_backward"));
    const int32 IdxLeft     = FindActionIndexByName(ActionSpace, TEXT("move_left"));
    const int32 IdxRight    = FindActionIndexByName(ActionSpace, TEXT("move_right"));
    const int32 IdxTurnL    = FindActionIndexByName(ActionSpace, TEXT("turn_left"));
    const int32 IdxTurnR    = FindActionIndexByName(ActionSpace, TEXT("turn_right"));
    const int32 IdxNoOp     = FindActionIndexByName(ActionSpace, TEXT("no_op"));

    if (IdxForward < 0 || IdxBackward < 0 || IdxLeft < 0 || IdxRight < 0 ||
        IdxTurnL < 0 || IdxTurnR < 0 || IdxNoOp < 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("    [Hard_Inverse] required actions missing"));
        return false;
    }

    const FVector  StartPos = GetActorLocation();
    const FRotator StartRot = GetActorRotation();

    // ---- Parameters: explore length K in [8, 16], force K_rot >= 2 rotation blocks ----
    // Ensure 2*K <= TrajectorySteps - 2 to leave room for padding.
    const int32 MaxK = FMath::Clamp((TrajectorySteps - 2) / 2, 8, 16);
    const int32 MinK = FMath::Max(8, MaxK - 6);

    // ---- Resample until we get a path that (a) has >= 2 rotation steps, (b) is collision-free,
    //      (c) closes under A . A^-1. ----
    constexpr int32 MaxAttempts = 12;
    const int32 TransPool[4] = { IdxForward, IdxBackward, IdxLeft, IdxRight };
    const int32 RotPool[2]   = { IdxTurnL,   IdxTurnR };

    for (int32 attempt = 0; attempt < MaxAttempts; attempt++)
    {
        const int32 K = FMath::RandRange(MinK, MaxK);

        // Generate explore: random concatenation of "translation block + rotation block + ...".
        // Each translation block m in [2, 4] steps; each rotation block r in [1, 3] steps.
        TArray<int32> ExploreActions;
        ExploreActions.Reserve(K);
        int32 NumRotSteps = 0;
        bool bCurrentIsRot = false;   // start with translation

        while (ExploreActions.Num() < K)
        {
            if (!bCurrentIsRot)
            {
                // Translation block
                const int32 m = FMath::Min(FMath::RandRange(2, 4), K - ExploreActions.Num());
                const int32 TransIdx = TransPool[FMath::RandRange(0, 3)];
                for (int32 i = 0; i < m; i++) ExploreActions.Add(TransIdx);
            }
            else
            {
                // Rotation block
                const int32 r = FMath::Min(FMath::RandRange(1, 3), K - ExploreActions.Num());
                const int32 RotIdx = RotPool[FMath::RandRange(0, 1)];
                for (int32 i = 0; i < r; i++) { ExploreActions.Add(RotIdx); NumRotSteps++; }
            }
            bCurrentIsRot = !bCurrentIsRot;
        }

        // Require at least 2 rotation steps (threshold per the paper: "K_rot >= 2 ensures A is not in the abelian subset").
        if (NumRotSteps < 2) continue;

        // Build Return = reverse(explore), each step replaced by its inverse.
        TArray<int32> ReturnActions;
        ReturnActions.Reserve(K);
        for (int32 i = ExploreActions.Num() - 1; i >= 0; i--)
        {
            ReturnActions.Add(GetInverseAction(ExploreActions[i]));
        }

        // Assemble the full path: Explore + no_op (mid pause) + Return.
        // Semantics: pause and look around at the farthest point, then retrace (consistent with Easy Inverse).
        const int32 PaddingSteps = TrajectorySteps - ExploreActions.Num() - ReturnActions.Num();
        TArray<int32> FullPath;
        FullPath.Reserve(TrajectorySteps);
        FullPath.Append(ExploreActions);
        for (int32 i = 0; i < PaddingSteps; i++) FullPath.Add(IdxNoOp);
        FullPath.Append(ReturnActions);

        // Collision pre-check
        if (!IsPathCollisionFree(StartPos, StartRot, FullPath)) continue;

        // Validate closure: end pose should return to start (translation < 0.5 x step, rotation < 2°).
        FVector EndPos; FRotator EndRot;
        SimulatePathEndpoint(ActionSpace, StartPos, StartRot, FullPath, EndPos, EndRot);
        const float PosErr = (EndPos - StartPos).Size();
        // Yaw difference: take modulo 360, accounting for quantization residual.
        float YawErr = FMath::Abs(FRotator::NormalizeAxis(EndRot.Yaw - StartRot.Yaw));
        // Threshold uses the current (possibly jittered) effective translation step, not the UPROPERTY base.
        const float EffectiveTransCm = FMath::Max(
            ActionSpace[IdxForward].DeltaPosition.Size(), 1.0f);
        if (PosErr > 0.5f * EffectiveTransCm) continue;
        if (YawErr > 2.0f) continue;

        CurrentTrajectoryActions = FullPath;
        UE_LOG(LogTemp, Log, TEXT("    [Hard_Inverse] K=%d rot_steps=%d pos_err=%.2fcm yaw_err=%.2f°"),
            K, NumRotSteps, PosErr, YawErr);
        return true;
    }

    UE_LOG(LogTemp, Warning, TEXT("    [Hard_Inverse] all %d attempts failed"), MaxAttempts);
    return false;
}


// ============================================================================
// Option B: Hard tier scale jittering (4 named tiers).
//
// Before planning a Hard tier trajectory, pick one tier from HardTierScaleLadder
// by weight, and apply its (TranslationScale, RotationScale) to ActionSpace's Delta*.
// On trajectory completion (Traj_FinalizeTrajectory), call RestoreActionSpaceScale to revert.
//
// Implementation: instead of changing TranslationMagnitude / RotationMagnitude (those are
// the UPROPERTY base values), we modify each entry's Delta* in the ActionSpace array directly,
// so SampleNaturalPath / MoveWithCollision / WouldActionCollide all pick it up automatically.
//
// Corresponds to paper §4.3 "Scale ladder":
//   Fine      (50cm/15°)  -- Habitat / AI2-THOR tier
//   Default   (100cm/15°) -- our baseline
//   Coarse    (200cm/15°) -- Matrix-Game tier
//   WideRot   (100cm/30°) -- AI2-THOR / ProcTHOR coarse-rotation tier
// ============================================================================
void ADataCollector::ApplyHardTierScaleJitter(float& OutTransScale, float& OutRotScale)
{
    // Defaults: no jitter
    OutTransScale = 1.0f;
    OutRotScale = 1.0f;
    CurrentScaleTier = EScaleTier::BaseDefault;

    if (!bEnableHardTierScaleJitter || HardTierScaleLadder.Num() == 0)
    {
        return;
    }

    // Idempotent: if already jittered without restore (should not normally happen), restore first.
    if (bActionSpaceIsJittered)
    {
        RestoreActionSpaceScale();
    }

    // Weighted sample of one tier.
    float TotalWeight = 0.0f;
    for (const FScaleLadderEntry& E : HardTierScaleLadder)
    {
        TotalWeight += FMath::Max(0.0f, E.Weight);
    }
    if (TotalWeight < 1e-3f)
    {
        // All weights are zero -- equivalent to disabling jitter.
        return;
    }

    float Roll = FMath::FRand() * TotalWeight;
    float Cumulative = 0.0f;
    const FScaleLadderEntry* Chosen = &HardTierScaleLadder[0];
    for (const FScaleLadderEntry& E : HardTierScaleLadder)
    {
        Cumulative += FMath::Max(0.0f, E.Weight);
        if (Roll <= Cumulative) { Chosen = &E; break; }
    }

    const float TransScale = Chosen->TranslationScale;
    const float RotScale   = Chosen->RotationScale;

    OutTransScale = TransScale;
    OutRotScale = RotScale;
    CurrentScaleTier = Chosen->Tier;

    // BaseDefault means 1.0x / 1.0x -- no need to actually jitter ActionSpace,
    // but still write into metadata so evaluation scripts can group by tier.
    const bool bNeedJitter = !(FMath::IsNearlyEqual(TransScale, 1.0f, 1e-3f) &&
                               FMath::IsNearlyEqual(RotScale,   1.0f, 1e-3f));
    if (!bNeedJitter) return;

    // Back up ActionSpace.
    SavedActionSpace = ActionSpace;
    bActionSpaceIsJittered = true;

    // Apply to every action.
    for (FActionDefinition& A : ActionSpace)
    {
        A.DeltaPosition = A.DeltaPosition * TransScale;
        A.DeltaRotation = FRotator(
            A.DeltaRotation.Pitch * RotScale,
            A.DeltaRotation.Yaw   * RotScale,
            A.DeltaRotation.Roll  * RotScale);
    }

    UE_LOG(LogTemp, Log, TEXT("    [ScaleJitter] tier=%s applied t%.2fx r%.2fx → eff=(%.1fcm, %.1f°)"),
        *ScaleTierToString(Chosen->Tier),
        TransScale, RotScale,
        TransScale * TranslationMagnitude, RotScale * RotationMagnitude);
}

void ADataCollector::RestoreActionSpaceScale()
{
    if (!bActionSpaceIsJittered) return;
    ActionSpace = SavedActionSpace;
    SavedActionSpace.Empty();
    bActionSpaceIsJittered = false;
    UE_LOG(LogTemp, Log, TEXT("    [ScaleJitter] restored ActionSpace to base scale"));
}

FString ADataCollector::ScaleTierToString(EScaleTier Tier)
{
    switch (Tier)
    {
    case EScaleTier::BaseDefault: return TEXT("default");
    case EScaleTier::Fine:        return TEXT("fine");
    case EScaleTier::Coarse:      return TEXT("coarse");
    case EScaleTier::WideRot:     return TEXT("wide_rot");
    default:                      return TEXT("unknown");
    }
}
