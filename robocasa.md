#单任务
run.sh，备份修改
data.config 修改成单任务，备份修改
base_vlm 记得采用绝对路径
单次点火测试
2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   communication_data_type ...... None
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   compile_config ............... deepcompile=False free_activation=False offload_activation=False offload_opt_states=False double_buffer=True symmetric_memory=False debug_log=False offload_parameters=False sync_before_reduce=False sync_after_reduce=False sync_before_allgather=False sync_after_allgather=False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   curriculum_enabled_legacy .... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   curriculum_params_legacy ..... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'pin_memory': False, 'curriculum_learning': {'enabled': False}, 'dynamic_batching': {'enabled': False, 'lr_scaling_method': 'linear', 'min_batch_size': 1, 'max_batch_size': None, 'sequence_picking_order': 'dataloader', 'verbose': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   data_efficiency_enabled ...... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   dataloader_drop_last ......... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   disable_allgather ............ False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   dump_state ................... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   dynamic_loss_scale_args ...... None
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   eigenvalue_enabled ........... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   eigenvalue_gas_boundary_resolution  1
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   eigenvalue_layer_num ......... 0
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   eigenvalue_max_iter .......... 100
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   eigenvalue_stability ......... 1e-06
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   eigenvalue_tol ............... 0.01
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   eigenvalue_verbose ........... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   elasticity_enabled ........... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   flops_profiler_config ........ {
    "enabled": false, 
    "recompute_fwd_factor": 0.0, 
    "profile_step": 1, 
    "module_depth": -1, 
    "top_modules": 1, 
    "detailed": true, 
    "output_file": null
}
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   fp16_auto_cast ............... None
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   fp16_enabled ................. False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   fp16_master_weights_and_gradients  False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   global_rank .................. 0
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   grad_accum_dtype ............. None
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   gradient_accumulation_steps .. 1
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   gradient_clipping ............ 1.0
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   gradient_predivide_factor .... 1.0
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   graph_harvesting ............. False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   initial_dynamic_scale ........ 1
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   load_universal_checkpoint .... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   loss_scale ................... 1.0
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   memory_breakdown ............. False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   mics_hierarchial_params_gather  False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   mics_shard_size .............. -1
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') comet=CometConfig(enabled=False, samples_log_interval=100, project=None, workspace=None, api_key=None, experiment_name=None, experiment_key=None, online=None, mode=None) wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName')
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   nebula_config ................ {
    "enabled": false, 
    "persistent_storage_path": null, 
    "persistent_time_interval": 100, 
    "num_of_version_in_retention": 2, 
    "enable_nebula_load": true, 
    "load_path": null
}
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   optimizer_legacy_fusion ...... False
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   optimizer_name ............... None
[2026-05-25 10:21:14,278] [INFO] [config.py:1007:print]   optimizer_params ............. None
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   pld_enabled .................. False
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   pld_params ................... False
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   prescale_gradients ........... False
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   scheduler_name ............... None
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   scheduler_params ............. None
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   seq_parallel_communication_data_type  torch.float32
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   sparse_attention ............. None
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   sparse_gradients_enabled ..... False
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   steps_per_print .............. inf
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   tensor_parallel_config ....... dtype=torch.float16 autotp_size=0 tp_overlap_comm=False tensor_parallel=TPConfig(tp_size=1, tp_grain_size=1, mpu=None, tp_group=None) injection_policy_tuple=None keep_module_on_host=False replace_with_kernel_inject=False
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   timers_config ................ enabled=True synchronized=True
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   train_batch_size ............. 1
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   train_micro_batch_size_per_gpu  1
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   use_data_before_expert_parallel_  False
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   use_node_local_storage ....... False
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   wall_clock_breakdown ......... False
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   weight_quantization_config ... None
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   world_size ................... 1
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   zero_allow_untested_optimizer  True
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   zero_config .................. stage=2 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500000000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500000000 overlap_comm=True load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=None sub_group_size=1000000000 cpu_offload_param=None cpu_offload_use_pin_memory=None prefetch_bucket_size=50000000 param_persistence_threshold=100000 model_persistence_threshold=9223372036854775807 max_live_parameters=1000000000 max_reuse_distance=1000000000 gather_16bit_weights_on_model_save=False module_granularity_threshold=0 use_all_reduce_for_fetch_params=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False zeropp_loco_param=None mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True log_trace_cache_warnings=False
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   zero_enabled ................. True
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   zero_force_ds_cpu_optimizer .. True
[2026-05-25 10:21:14,279] [INFO] [config.py:1007:print]   zero_optimization_stage ...... 2
[2026-05-25 10:21:14,279] [INFO] [config.py:993:print_user_config]   json = {
    "fp16": {
        "enabled": false
    }, 
    "bf16": {
        "enabled": true
    }, 
    "train_micro_batch_size_per_gpu": 1, 
    "train_batch_size": 1, 
    "gradient_accumulation_steps": 1, 
    "zero_optimization": {
        "stage": 2, 
        "allgather_partitions": true, 
        "allgather_bucket_size": 5.000000e+08, 
        "reduce_scatter": true, 
        "reduce_bucket_size": 5.000000e+08, 
        "overlap_comm": true, 
        "contiguous_gradients": true, 
        "cpu_offload": false
    }, 
    "gradient_clipping": 1.0, 
    "steps_per_print": inf, 
    "zero_allow_untested_optimizer": true
}
05/25 [10:21:21] INFO     | >> ***** Training Configuration *****                                                                                               train_starvla.py:362
                 INFO     | >>   Total optimization steps = 5                                                                                                   train_starvla.py:363
                 INFO     | >>   Per device batch size = 1                                                                                                      train_starvla.py:364
                 INFO     | >>   Gradient accumulation steps = 1                                                                                                train_starvla.py:365
                 INFO     | >>   Total batch size = 1                                                                                                           train_starvla.py:366
 20%|█████████████████████                                                                                    | 1/5 [02:05<08:23, 125.91s/it, data_times=2.521, model_times=123.388]05/25 [10:23:27] INFO     | >> Step 1, Loss: {'action_dit_loss': 1.6883128881454468, 'timing/data': 2.5213822219520807, 'timing/model': 123.3882260620594,      train_starvla.py:274
                          'learning_rate/qwen_vl_interface': 2e-09, 'learning_rate/action_model': 2e-08, 'epoch': 0.0})                                                             
 40%|███████████████████████████████████████████▏                                                                | 2/5 [02:06<02:36, 52.04s/it, data_times=0.000, model_times=0.326]                 INFO     | >> Step 2, Loss: {'action_dit_loss': 1.6019517183303833, 'timing/data': 0.00037179701030254364, 'timing/model': 0.3255851212888956, train_starvla.py:274
                          'learning_rate/qwen_vl_interface': 4e-09, 'learning_rate/action_model': 4e-08, 'epoch': 0.0})                                                             
 60%|████████████████████████████████████████████████████████████████▊                                           | 3/5 [02:06<00:56, 28.41s/it, data_times=0.000, model_times=0.287]                 INFO     | >> Step 3, Loss: {'action_dit_loss': 1.7637126445770264, 'timing/data': 0.00017510168254375458, 'timing/model': 0.2871099393814802, train_starvla.py:274
                          'learning_rate/qwen_vl_interface': 6e-09, 'learning_rate/action_model': 6e-08, 'epoch': 0.0})                                                             
 80%|██████████████████████████████████████████████████████████████████████████████████████▍                     | 4/5 [02:06<00:17, 17.31s/it, data_times=0.001, model_times=0.285]05/25 [10:23:28] INFO     | >> Step 4, Loss: {'action_dit_loss': 1.5695278644561768, 'timing/data': 0.0011049974709749222, 'timing/model': 0.28523675724864006, train_starvla.py:274
                          'learning_rate/qwen_vl_interface': 8e-09, 'learning_rate/action_model': 8e-08, 'epoch': 0.0})                                                             
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [02:07<00:00, 11.17s/it, data_times=0.000, model_times=0.284]                 INFO     | >> Step 5, Loss: {'action_dit_loss': 1.5972522497177124, 'timing/data': 0.00016439706087112427, 'timing/model':                     train_starvla.py:274
                          0.28398409858345985, 'learning_rate/qwen_vl_interface': 1e-08, 'learning_rate/action_model': 1.0000000000000001e-07, 'epoch': 0.0})                       
05/25 [10:23:36] INFO     | >> Training complete. Final model saved at ./playground/Checkpoints/debug_pi_milk_5step/final_model                                 train_starvla.py:411
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [02:14<00:00, 26.94s/it, data_times=0.000, model_times=0.284]
                 INFO     | >> ... and that's all, folks!                                                                                                       train_starvla.py:442
dsw-754811-64c49475dd-ghkm8:3613671:3613671 [0] NCCL INFO comm 0x556b3d181b80 rank 0 nranks 1 cudaDev 0 busId 10c000 - Destroy COMPLETE
dsw-754811-64c49475dd-ghkm8:3613671:3613671 [0] NCCL INFO comm 0x556b93981ca0 rank 0 nranks 1 cudaDev 0 busId 10c000 - Destroy COMPLETE
点火测试成功
#4卡/8卡训练
pip install numpydantic
from decord import VideoReader