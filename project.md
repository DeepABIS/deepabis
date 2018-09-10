# Project Log

## Runs

| Run | Dataset       | Model     | LR Decay | Transformation   | Class weights | Description | F1 (Species)  |
| --- | ------------- |---------- | -------- | ---------------- | ------------- | ----------- | ---:|
| 1   | genus8        | Baseline  | yes      | Per Channel      |               | Baseline Run.       |  0.715   |
| 2   | genus8        | Baseline  | yes      | Per Channel      |               | Rerun of 1, but with generator. Also, 8 instead of 9 classes. The 9th class was mistakenly added as a zero class. |  0.699   |
| 3   | genus8        | Baseline  | yes      | Per Channel      | yes           |             | 0.758 |
| 4   | genus8_aug    | Baseline  | yes      | Mean Subtraction | yes           |             | 0.6312 |
| 5   | genus8_2      | Baseline  | yes      | Mean Subtraction | yes           |             | 0.739 |
| 6   | genus8_2_aug  | Baseline  | yes      | Mean Subtraction | yes           |             | 0.672
| 7   | genus8_2      | Baseline  | yes      | Per Channel      | yes           |             | 0.686
| 8   | genus8_2_aug2 | Baseline  | no       | Mean Subtraction | yes           |             | 0.664
| 9   | genus8_2      | Blocks4   | no       | Mean Subtraction | yes           |             | 0.792
| 10  | genus8_2_aug2 | Blocks4   | no       | Mean Subtraction | yes           |             | 0.788
| 11  | genus8_2_aug4 | Blocks4   | no       | Mean Subtraction | yes           |             | 0.841
| 12  | genus8_2_aug4 | Oneloss   | no       | Mean Subtraction | yes           |             | 0.791
| 13  | genus8_2_aug4 | Mobilenet | no       | Mean Subtraction | yes           |             | 0.854
| 14  | genus8_2_aug5 | Mobilenet | no       | Mean Subtraction | yes           |             | 0.874
| 15  | genus8_2_aug5 | Mobilenet | no       | Mean Subtraction | yes           |             | 0.874
| 16  | genus8_2_aug5 | MobilenetV2 | no       | Per Channel | yes           |             | 0.878
| 17  | genus8_2_aug5 | MobilenetV2 | no       | Per Channel | yes           | Properly normalize data            | 0.888
| 18  | genus8_3 | MobilenetV2 | no       | Per Channel | yes           |             | 


## Datasets

| Name       | Train | Test | Ratio | Augmentation | Description
| ---------- | ----- | ---- | ----- | ------------ | -----------
| genus8     | 6835  | 760  | 10%   | None         |
| genus8_aug | 10112 | 1519 | 20%   | Balanced     | Rotate, Zoom, Noise, Flip.
| genus8_2   | 6835  | 760  | 10%   | None         | Same as genus8, but (probably) different composition and in train/test folder format.
| genus8_2_aug   | 12075  | 760  | 10%   | Balanced  | Rotate (-30, 30), Zoom (1.2), Noise, Flip. 
| genus8_2_aug2   | 11315  | 760  | 10%   | Balanced  | Rotate (-30, 30), Zoom (1.2), Noise. 
| genus8_2_aug3   | 11315  | 760  | 10%   | Balanced  | Rotate (-15, 15), Zoom (1.1), Noise. 
| genus8_2_aug4   | 13670  | 760  | 10%   | 2x  | Rotate (-15, 15), Zoom (1.1), Noise.
| genus8_2_aug5   | 13670  | 760  | 10%   | 2x  | Rotate (-15, 15), Zoom (1.1), Shear (30), FlipUD (0.25), Noise.
| genus8_2_stratify   | 13670  | 760  | 10%   | 2x  | Rotate (-15, 15), Zoom (1.1), Shear (30), FlipUD (0.25), Noise.
| genus8_3   | 20505  | 2280  | 10%   | 3x  | Add(-50, 50), Affine(-15, 15), Affine(1.1), FlipLR(0.3), FlipUD(0.25), Affine(Shear=30), Crop(0, 16)

## Models
| Name       | Classes  | Description
| ---------- | -------- | -----------
| Baseline   | 8, 124   | 
| Blocks4    | 8, 124   | Baseline with 4 blocks.


## Transformations
| Name                      | Formula               |
| ------------------------- | --------------------- |
| Per Channel Normalization | (X - mean(X)) / std(X)
| Mean Subtraction          | (X - mean(X)) / 255 