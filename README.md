# robust-vc
Study of robustness in voice conversion

``` bash
AdaIN-VC
├── AdaIN-VC (original code)
│   └──data
├── AdaIN-VC-robust (training with wav augmentation)
│   ├── data
│   ├── output
│   └── processed
└── vocoder (pretrained vocoder)

S2VC
├── S2VC  (original code)
│   ├── data
│   └── models
├── S2VC-robust (training with wav augmentation)
│   ├── data
│   ├── models
│   ├── output
│   └── processed
│       └── cpc
└── vocoder (pretrained vocoder)

assets
├── jupyter_lab (containing some code to play with)
├── script      
│   ├── command.txt (example commands to use the scripts below)
│   ├── dataset_copy.py 
│   ├── dataset_noise.py
│   └── dataset_resample_vad.py
├── VCTK_split.py (defining the train, valid and test dataset)
├── vctk_test_clean (copyed test dataset)
├── vctk_test_noisy (added noise to vad dataset)
└── vctk_test_vad   (resampled, normalized to -3dB, and sox vad)
```




