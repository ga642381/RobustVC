# Scripts usage examples

### Preprocessing for VCTK dataset

```
# copying VCTK test dataset to this repo
python dataset_copy.py test [path_to_VCTK]/VCTK-Corpus ../vctk_test_clean

# preprocessing (resample and vad) for VCTK
python dataset_resample_vad.py ../vctk_test_clean ../vctk_test_vad
```

### Creating training dataset for VC models

```
python dataset_resample_vad.py [path_to_VCTK]/VCTK-Corpus [path_to_training_dir]/vctk_all_vad --out_sample_rate 16000
python dataset_demand.py [path_to_training_dir]/vctk_all_vad [path_to_demand]/demand [path_to_training_dir]/vctk_all_vad_demand
```

### Creating datasets with artificial noises (additive, reverb, band rejection)

```
python dataset_noise.py --p_clean 0 --p_add 0 --p_reverb 1 --p_band 0 ../vctk_test_vad ../vctk_test_vad_reverb
python dataset_noise.py --p_clean 0 --p_add 0 --p_reverb 0 --p_band 1 ../vctk_test_vad ../vctk_test_vad_band
python dataset_noise.py --p_clean 0 --p_add 0 --p_reverb 1 --p_band 1 ../vctk_test_vad_demand ../vctk_test_vad_demand_reverb_band
python dataset_noise.py --p_clean 0 --p_add 0 --p_reverb 1 --p_band 1 ../vctk_test_vad_wham ../vctk_test_vad_wham_reverb_band
```

### Creating datasets with environmental noises (WHAM!, DEMAND)

```
python dataset_wham.py ../vctk_test_vad [path_to_wham]/high_res_wham ../vctk_test_wham
python dataset_demand.py ../vctk_test_vad [path_to_demand]/demand ../vctk_test_vad_demand --mode test
```

### Creating datasets with DEMUCS speech enhancement

```
python dataset_demucs.py ../vctk_test_vad_demand ../vctk_test_vad_demand_demucs
python dataset_demucs.py ../vctk_test_vad_wham ../vctk_test_vad_wham_demucs
python dataset_demucs.py ../vctk_test_vad_reverb ../vctk_test_vad_reverb_demucs
python dataset_demucs.py ../vctk_test_vad_band ../vctk_test_vad_band_demucs
python dataset_demucs.py ../vctk_test_vad_demand_reverb_band ../vctk_test_vad_demand_reverb_band_demucs
python dataset_demucs.py ../vctk_test_vad_wham_reverb_band ../vctk_test_vad_wham_reverb_band_demucs
```

### Creating datasets with metricgan+ speech enhancement

```
python dataset_metricgan.py ../vctk_test_vad_demand ../vctk_test_vad_demand_metricgan
python dataset_metricgan.py ../vctk_test_vad_wham ../vctk_test_vad_wham_metricgan
python dataset_metricgan.py ../vctk_test_vad_demand_reverb_band ../vctk_test_vad_demand_reverb_band_metricgan
python dataset_metricgan.py ../vctk_test_vad_wham_reverb_band ../vctk_test_vad_wham_reverb_band_metricgan
```

### Creating datasets with DCCRNet speech enhancement

```
python dataset_dccrnet.py ../vctk_test_vad_demand ../vctk_test_vad_demand_dccrnet
python dataset_dccrnet.py ../vctk_test_vad_wham ../vctk_test_vad_wham_dccrnet
```

### evaluate PESQ

```
python metric_pesq.py ../vctk_test_vad ../vctk_test_vad_demand_demucs
python metric_pesq.py ../vctk_test_vad ../vctk_test_vad_demand_metricgan
python metric_pesq.py ../vctk_test_vad ../vctk_test_vad_demand_dccrnet
python metric_pesq.py ../vctk_test_vad ../vctk_test_vad_demand_reverb_band_demucs
python metric_pesq.py ../vctk_test_vad ../vctk_test_vad_wham_demucs
python metric_pesq.py ../vctk_test_vad ../vctk_test_vad_wham_metricgan
python metric_pesq.py ../vctk_test_vad ../vctk_test_vad_demand_reverb_band_metricgan
```

# attack

```
python dataset_adain_attack.py --save_dir ../vctk_test_vad_AdaIN-VC-robust-adv_attack --model_path ../../Voice-conversion-evaluation/models/any2any/AdaIN-VC-robust/checkpoints/[model_name.ckpt] --metadata_path ../../Voice-conversion-evaluation/metadata/VCTK_to_VCTK.json
python dataset_adain_attack.py --save_dir ../vctk_test_vad_AdaIN-VC-robust_attack --model_path ../../Voice-conversion-evaluation/models/any2any/AdaIN-VC-robust/checkpoints/[model_name.ckpt] --metadata_path ../../Voice-conversion-evaluation/metadata/VCTK_to_VCTK.json
python dataset_adain_attack.py --save_dir ../vctk_test_vad_AdaIN-VC_attack --model_path ../../Voice-conversion-evaluation/models/any2any/AdaIN-VC/checkpoints/[model_name.ckpt] --metadata_path ../../Voice-conversion-evaluation/metadata/VCTK_to_VCTK.json

python dataset_s2vc_attack.py --save_dir ../vctk_test_vad_S2VC_attack --model_path ../../Voice-conversion-evaluation/models/any2any/S2VC-robust/checkpoints/model_0927_b12_c10_50k.pt --feat_type cpc --metadata_path ../../Voice-conversion-evaluation/metadata/VCTK_to_VCTK.json
python dataset_s2vc_attack.py --save_dir ../vctk_test_vad_S2VC_attack --model_path ../../Voice-conversion-evaluation/models/any2any/S2VC-robust/checkpoints/model_0927_b12_c04_50k.pt --feat_type cpc --metadata_path ../../Voice-conversion-evaluation/metadata/VCTK_to_VCTK.json
```
