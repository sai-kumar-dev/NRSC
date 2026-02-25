# Ocean Heat Flux PINN Algorithm Flowchart

## Overview
This flowchart represents the complete algorithm for training a Physics-Informed Neural Network (PINN) to predict ocean heat fluxes.

```mermaid
flowchart TD
    %% Starting point
    Start([START]) --> Init[Initialize System]
    
    %% Initialization Phase
    Init --> Memory[Set Memory Limits & Logging]
    Memory --> Seeds[Set Random Seeds for Reproducibility]
    Seeds --> Config[Load Configuration Parameters]
    Config --> DeviceCheck{CUDA Available?}
    DeviceCheck -->|Yes| SetCuda[Set Device = CUDA]
    DeviceCheck -->|No| SetCPU[Set Device = CPU]
    SetCuda --> DataLoad
    SetCPU --> DataLoad
    
    %% Data Loading Phase
    DataLoad[Load NetCDF Data File] --> FileCheck{File Exists?}
    FileCheck -->|No| Error1[Raise FileNotFoundError]
    FileCheck -->|Yes| ExtractData[Extract Variables: u10, v10, d2m, t2m, sst, sp, rho]
    ExtractData --> CreateMask[Create Ocean Mask]
    
    %% Ocean Mask Creation
    CreateMask --> MaskSST[Check SST valid & 271K ≤ SST ≤ 308K]
    MaskSST --> MaskPressure[Check Surface Pressure: 95000 ≤ sp ≤ 105000 Pa]
    MaskPressure --> CombineMasks[Combine Masks: ocean_mask = sst_mask & pressure_mask]
    CombineMasks --> LogMaskStats[Log Ocean Percentage]
    
    %% Data Interpolation
    LogMaskStats --> InterpolateLoop{For each variable}
    InterpolateLoop --> TimeLoop{For each time step}
    TimeLoop --> CheckOcean{Ocean points exist?}
    CheckOcean -->|Yes| InterpolateNaN[Interpolate NaN values using griddata]
    CheckOcean -->|No| KeepOriginal[Keep original data]
    InterpolateNaN --> NextTime{More time steps?}
    KeepOriginal --> NextTime
    NextTime -->|Yes| TimeLoop
    NextTime -->|No| NextVar{More variables?}
    NextVar -->|Yes| InterpolateLoop
    NextVar -->|No| PrepareData
    
    %% Data Preparation Phase
    PrepareData[Prepare Data in Chunks] --> CheckDataSize{Total ocean points > max_samples?}
    CheckDataSize -->|Yes| SetSubsample[Set effective_subsample_ratio = max_samples / total_points]
    CheckDataSize -->|No| UseConfigRatio[Use configured subsample_ratio]
    SetSubsample --> ReshapeData
    UseConfigRatio --> ReshapeData
    
    ReshapeData[Reshape data to 2D arrays] --> InitChunks[Initialize X_chunks, y_chunks, samples_collected = 0]
    InitChunks --> ProcessLoop{For t = 0 to time_dim}
    
    %% Data Processing Loop
    ProcessLoop --> LogProgress{t % 500 == 0?}
    LogProgress -->|Yes| MonitorMem[Monitor Memory Usage]
    LogProgress -->|No| CheckMaxSamples
    MonitorMem --> CheckMaxSamples
    CheckMaxSamples{samples_collected ≥ max_samples?}
    CheckMaxSamples -->|Yes| StopCollection[Stop data collection]
    CheckMaxSamples -->|No| SubsampleCheck{Random() > subsample_ratio?}
    SubsampleCheck -->|Yes| NextTimeStep{More time steps?}
    SubsampleCheck -->|No| ExtractOceanData[Extract ocean data for time step]
    
    ExtractOceanData --> ValidDataCheck{Valid ocean data exists?}
    ValidDataCheck -->|No| NextTimeStep
    ValidDataCheck -->|Yes| StackFeatures[Stack features: [u10, v10, d2m, t2m, sst, sp, rho]]
    StackFeatures --> RemoveNaN[Remove NaN values]
    RemoveNaN --> ChunkCheck{len(X_t) > chunk_size?}
    ChunkCheck -->|Yes| RandomSubsample[Random subsample to chunk_size]
    ChunkCheck -->|No| CalcWindSpeed
    RandomSubsample --> CalcWindSpeed
    
    %% Bulk Formula Calculations
    CalcWindSpeed[Calculate wind_speed = sqrt(u10² + v10²)] --> CalcHumidity[Calculate specific humidity from dewpoint]
    CalcHumidity --> CalcSSTHumidity[Calculate specific humidity at sea surface]
    CalcSSTHumidity --> CalcSensibleFlux[Calculate sensible_heat_flux = ρ * cp * C_H * wind_speed * (sst - t2m)]
    CalcSensibleFlux --> CalcLatentFlux[Calculate latent_heat_flux = ρ * L * C_E * wind_speed * (q_s - q_a)]
    CalcLatentFlux --> CombineTargets[Combine into y_t = [sensible_flux, latent_flux]]
    CombineTargets --> AppendChunks[Append X_t, y_t to chunks]
    AppendChunks --> UpdateCount[samples_collected += len(X_t)]
    UpdateCount --> ClearVars[Clear intermediate variables]
    ClearVars --> MemCleanup{t % 1000 == 0?}
    MemCleanup -->|Yes| ClearMemory[clear_memory()]
    MemCleanup -->|No| NextTimeStep
    ClearMemory --> NextTimeStep
    NextTimeStep -->|Yes| ProcessLoop
    NextTimeStep -->|No| ConcatenateChunks
    StopCollection --> ConcatenateChunks
    
    %% Data Finalization
    ConcatenateChunks[Concatenate all chunks: X = vstack(X_chunks), y = vstack(y_chunks)] --> CheckEmpty{Chunks empty?}
    CheckEmpty -->|Yes| Error2[Raise ValueError: No valid data]
    CheckEmpty -->|No| QualityCheck[Check for NaN/Inf values]
    QualityCheck --> CleanData[Remove problematic values]
    CleanData --> LogFinalShapes[Log final data shapes]
    
    %% Data Normalization and Splitting
    LogFinalShapes --> NormalizeData[Normalize features with StandardScaler]
    NormalizeData --> CreateTensors[Convert to PyTorch tensors]
    CreateTensors --> SplitData[Split: 80% train, 10% validation, 10% test]
    SplitData --> CreateLoaders[Create DataLoaders with batch_size]
    
    %% Model Initialization
    CreateLoaders --> InitModel[Initialize OceanHeatFluxPINN]
    InitModel --> ModelConfig[input_dim=7, hidden_dim=128, output_dim=2, num_layers=4]
    ModelConfig --> SetDevice[Move model to device]
    SetDevice --> InitOptimizer[Initialize Adam optimizer with learning_rate]
    InitOptimizer --> InitScheduler[Initialize ReduceLROnPlateau scheduler]
    
    %% Training Loop
    InitScheduler --> TrainingLoop{For epoch = 0 to num_epochs}
    TrainingLoop --> SetTrainMode[model.train()]
    SetTrainMode --> InitEpochLoss[train_loss_epoch = 0.0]
    InitEpochLoss --> BatchLoop{For each batch in train_loader}
    
    %% Training Batch Processing
    BatchLoop --> MoveTensors[Move X_batch, y_batch to device]
    MoveTensors --> ZeroGrad[optimizer.zero_grad()]
    ZeroGrad --> ForwardPass[y_pred = model(X_batch)]
    ForwardPass --> CalcDataLoss[data_loss = MSE(y_pred, y_true)]
    
    %% Physics Constraints
    CalcDataLoss --> ExtractVars[Extract variables from X_batch]
    ExtractVars --> CalcWindSpeedPhys[wind_speed = sqrt(u10² + v10²)]
    CalcWindSpeedPhys --> CalcTempDiff[temp_diff = sst - t2m]
    CalcTempDiff --> SignConstraint[sign_constraint = mean((sign(sensible_pred) - sign(temp_diff))²)]
    SignConstraint --> WindConstraint[wind_constraint = mean((|sensible_pred|/wind_speed - |latent_pred|/wind_speed)²)]
    WindConstraint --> CombinePhysics[physics_loss = sign_constraint + wind_constraint]
    CombinePhysics --> TotalLoss[total_loss = data_loss + λ_physics * physics_loss]
    
    %% Backpropagation
    TotalLoss --> Backward[total_loss.backward()]
    Backward --> OptimizerStep[optimizer.step()]
    OptimizerStep --> UpdateLoss[train_loss_epoch += total_loss.item()]
    UpdateLoss --> MoreBatches{More batches?}
    MoreBatches -->|Yes| BatchLoop
    MoreBatches -->|No| ValidationPhase
    
    %% Validation Phase
    ValidationPhase --> SetEvalMode[model.eval()]
    SetEvalMode --> InitValLoss[val_loss_epoch = 0.0]
    InitValLoss --> ValBatchLoop{For each batch in val_loader}
    ValBatchLoop --> NoGradContext[with torch.no_grad()]
    NoGradContext --> ValForward[Forward pass for validation]
    ValForward --> ValLossCalc[Calculate validation loss]
    ValLossCalc --> UpdateValLoss[val_loss_epoch += loss]
    UpdateValLoss --> MoreValBatches{More val batches?}
    MoreValBatches -->|Yes| ValBatchLoop
    MoreValBatches -->|No| CalcAvgLosses
    
    %% Epoch Finalization
    CalcAvgLosses[Calculate average losses] --> AppendLosses[Append to train_losses, val_losses]
    AppendLosses --> SchedulerStep[scheduler.step(val_loss)]
    SchedulerStep --> LogProgress2{epoch % 10 == 0?}
    LogProgress2 -->|Yes| LogEpoch[Log epoch progress]
    LogProgress2 -->|No| ClearMemEpoch
    LogEpoch --> ClearMemEpoch[clear_memory()]
    ClearMemEpoch --> MoreEpochs{More epochs?}
    MoreEpochs -->|Yes| TrainingLoop
    MoreEpochs -->|No| TestPhase
    
    %% Testing Phase
    TestPhase --> SetTestEval[model.eval()]
    SetTestEval --> TestLoop{For each batch in test_loader}
    TestLoop --> TestForward[Forward pass on test data]
    TestForward --> InverseTransform[Inverse transform predictions & targets]
    InverseTransform --> CollectPredictions[Collect all predictions & targets]
    CollectPredictions --> MoreTestBatches{More test batches?}
    MoreTestBatches -->|Yes| TestLoop
    MoreTestBatches -->|No| CalcMetrics
    
    %% Metrics Calculation
    CalcMetrics --> SensibleMetrics[Calculate sensible heat flux metrics]
    SensibleMetrics --> LatentMetrics[Calculate latent heat flux metrics]
    LatentMetrics --> ComprehensiveMetrics{Calculate MSE, MAE, RMSE, R², Accuracy, Precision, Recall, F1}
    ComprehensiveMetrics --> LogMetrics[Log all metrics]
    
    %% Results Saving
    LogMetrics --> CreateResultsDir[Create results directory]
    CreateResultsDir --> SaveModel[Save model state & config]
    SaveModel --> SaveScalers[Save StandardScalers]
    SaveScalers --> SaveHistory[Save training history]
    SaveHistory --> SaveMetrics[Save comprehensive metrics]
    SaveMetrics --> CreateViz[Create visualizations]
    
    %% Visualization Creation
    CreateViz --> PlotTrainingHistory[Plot training & validation losses]
    PlotTrainingHistory --> PlotScatter[Plot predicted vs true scatter plots]
    PlotScatter --> PlotResiduals[Plot residual plots]
    PlotResiduals --> PlotDistributions[Plot flux distributions]
    PlotDistributions --> SavePlots[Save all plots]
    
    %% Inference Example
    SavePlots --> InferenceExample[Run inference example]
    InferenceExample --> LoadTrainedModel[Load trained model & scalers]
    LoadTrainedModel --> ExampleInput[Create example input data]
    ExampleInput --> MakePredictions[Make predictions on examples]
    MakePredictions --> LogResults[Log inference results]
    
    %% Completion
    LogResults --> Success[Training completed successfully!]
    Success --> End([END])
    
    %% Error Handling
    Error1 --> End
    Error2 --> End
    
    %% Memory Management
    subgraph MemoryManagement [Memory Management]
        MonitorMem --> HighMemory{Memory > 75%?}
        HighMemory -->|Yes| ClearMem[clear_memory(): gc.collect(), torch.cuda.empty_cache()]
        HighMemory -->|No| ContinueProcessing[Continue Processing]
        ClearMem --> ContinueProcessing
    end
    
    %% Physics Constraints Details
    subgraph PhysicsConstraints [Physics Constraints]
        SignConstraint --> PhysicsDetail1[Sensible heat flux sign should match temperature difference sign]
        WindConstraint --> PhysicsDetail2[Flux magnitude should increase with wind speed]
    end
    
    %% Bulk Formulas Details
    subgraph BulkFormulas [Bulk Formulas]
        CalcSensibleFlux --> BulkDetail1[SHF = ρ * cp * C_H * |V| * (SST - T2m)]
        CalcLatentFlux --> BulkDetail2[LHF = ρ * L * C_E * |V| * (q_s - q_a)]
    end
    
    %% Model Architecture Details
    subgraph ModelArch [Model Architecture]
        InitModel --> Layer1[Linear(7, 128) + LeakyReLU]
        Layer1 --> HiddenLayers[3 Hidden Layers: Linear(128, 128) + LeakyReLU]
        HiddenLayers --> OutputLayer[Linear(128, 2)]
        OutputLayer --> WeightInit[Xavier Normal Initialization]
    end
    
    %% Decision Points Color Coding
    classDef decisionNode fill:#FFE135,stroke:#333,stroke-width:2px,color:#000
    classDef processNode fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    classDef errorNode fill:#FF6B6B,stroke:#333,stroke-width:2px,color:#fff
    classDef startEndNode fill:#90EE90,stroke:#333,stroke-width:3px,color:#000
    
    class FileCheck,DeviceCheck,CheckMaxSamples,SubsampleCheck,ValidDataCheck,ChunkCheck,CheckEmpty,HighMemory,MoreBatches,MoreEpochs,MoreTestBatches decisionNode
    class Error1,Error2 errorNode
    class Start,End,Success startEndNode
```

## Key Algorithm Components

### 1. **Initialization Phase**
- Set memory limits (80% of available memory)
- Configure logging system
- Set random seeds for reproducibility
- Load configuration parameters
- Detect and set compute device (CUDA/CPU)

### 2. **Data Loading and Preprocessing**
- Load NetCDF file containing ERA5 data
- Extract variables: u10, v10, d2m, t2m, sst, sp, rho
- Create ocean mask based on SST and surface pressure criteria
- Interpolate NaN values in ocean areas using scipy griddata

### 3. **Data Preparation Loop**
- **Decision**: Check if dataset size exceeds maximum samples
- **Loop**: Process each time step
- **Decision**: Apply subsampling based on probability
- **Decision**: Check for valid ocean data
- Calculate bulk formula physics for heat flux targets

### 4. **Neural Network Architecture**
- Input: 7 variables (u10, v10, d2m, t2m, sst, sp, rho)
- Hidden layers: 4 layers with 128 neurons each
- Activation: LeakyReLU
- Output: 2 variables (sensible and latent heat flux)
- Weight initialization: Xavier Normal

### 5. **Training Loop Structure**
- **Outer Loop**: Epochs (default: 70)
- **Inner Loop**: Batches
- **Decision**: Memory monitoring every 500 steps
- **Decision**: Validation every epoch
- **Decision**: Learning rate scheduling based on validation loss

### 6. **Physics-Informed Loss Function**
- Data loss: MSE between predictions and bulk formula targets
- Physics constraints:
  - Sign constraint: Sensible heat flux sign matches temperature difference
  - Wind constraint: Flux magnitude increases with wind speed
- Total loss: data_loss + λ_physics × physics_loss

### 7. **Decision Points in Algorithm**
- File existence check
- CUDA availability
- Dataset size vs memory limits
- Subsampling probability
- Valid data availability
- Chunk size limits
- Memory usage thresholds
- Training progress (batch/epoch completion)
- Convergence criteria

### 8. **Memory Management**
- **Loop**: Monitor memory usage every 1000 time steps
- **Decision**: If memory > 75%, trigger garbage collection
- Clear intermediate variables after each time step
- Use float32 instead of float64 to save memory

### 9. **Evaluation and Metrics**
- Regression metrics: MSE, MAE, RMSE, R²
- Classification metrics: Accuracy, Precision, Recall, F1-Score
- Physics constraint validation
- Comprehensive visualizations

### 10. **Output Generation**
- Model saving with configuration
- Scaler saving for inference
- Training history and metrics
- Visualization plots (9 subplots)
- NetCDF export of predictions
- Performance summary CSV

## Loop Structures

1. **Variable Interpolation Loop**: For each variable → For each time step
2. **Data Preparation Loop**: For each time step with early termination
3. **Training Loop**: For each epoch → For each batch
4. **Validation Loop**: For each validation batch
5. **Testing Loop**: For each test batch
6. **Memory Monitoring Loop**: Periodic checks during processing

## Memory Management Strategy
The algorithm implements comprehensive memory management with periodic cleanup, monitoring, and early termination to prevent out-of-memory errors while processing large oceanographic datasets.
