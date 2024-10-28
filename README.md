# [Ergonomic Risk Prediction for Awkward Postures From 3D Keypoints Using Deep Learning](https://ieeexplore.ieee.org/document/10286039)

## REBA Dataset
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) dataset is used for preparing this large ground truth REBA score dataset. We have computed REBA score for individual body segments, score A, score B, score C, Risk and action label for overall posture for both left side and right side of human body posture separately. As we are only concerned about the highest score of a body posture, final Risk and Action scores are calculated by taking the highest value from the left and right scores. This dataset can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1_THSvQVy3KW8yWwOO0a9oS4qlZ_6APdo/view?usp=sharing) 

## Merge REBA dataset with Human3.6m dataset:
Our REBA dataset can be merged with Human3.6m dataset's 3D body joint keypoints. Go to [Human3.6m](http://vision.imar.ro/human3.6m/description.php) dataset *Download* tab. From *TRAINING DATA* go to *By subjet* and download *D3 Positions mono*  Poses for subject S1, S5, S6, S7, S8, S8, S9, S10 & S11. Extract the tar files and concat. Finally these keypoints data can be merged with our REBA Dataset.

## Citation
If you find this repo useful in your work or research, please cite:
```
@ARTICLE{10286039,
  author={Hossain, Md. Shakhaout and Azam, Sami and Karim, Asif and Montaha, Sidratul and Quadir, Ryana and De Boer, Friso and Altaf-Ul-Amin, Md.},
  journal={IEEE Access}, 
  title={Ergonomic Risk Prediction for Awkward Postures From 3D Keypoints Using Deep Learning}, 
  year={2023},
  volume={11},
  number={},
  pages={114497-114508},
  doi={10.1109/ACCESS.2023.3324659}}
```
