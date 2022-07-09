# Roubst-VC

## Toward Degradation-Robust Voice Conversion

- Paper Title: Toward Degradation-Robust Voice Conversion
- Authors: Chien-yu Huang*, Kai-Wei Chang*, Hung-yi Lee
- Paper Link: https://arxiv.org/abs/2110.07537

To appear in the proceedings of ICASSP 2022, equal contribution from first two authors

## Proposed Approaches

Both **Speech Enhancement Concatenation** and **End-to-End Denoising Training** can effectively imporve state-of-the-art VC models' **degradation robustness** and **adversarial robustness**.

### Approach1: Speech Enhancement Concatenation

![](https://i.imgur.com/QSQoK0O.png)

- Pros: Any-off-the-shelf model applies.
- Cons: More computations are required for inference.

### Approach2: End-to-End Denoising Training

![](https://i.imgur.com/uE4WZwx.png)

- Pros: Combine Voice conversion and speech enhancement in a single model.
- Cons: Need more resouces for training.

## Code
Please refer to [cyhuang-tw/robust-vc](https://github.com/cyhuang-tw/robust-vc)

## Demo Page

https://cyhuang-tw.github.io/robust-vc-demo/

## Citation

```
@inproceedings{huang2022toward,
  title={Toward Degradation-Robust Voice Conversion},
  author={Huang, Chien-yu and Chang, Kai-Wei and Lee, Hung-yi},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6777--6781},
  year={2022},
  organization={IEEE}
}
```
