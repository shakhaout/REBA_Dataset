# REBA Dataset
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) dataset is used for preparing this large ground truth REBA score dataset. We have computed REBA score for individual body segments, score A, score B, score C, Risk and action label for overall posture for both left side and right side of human body posture separately. As we are only concerned about the highest score of a body posture, final Risk and Action scores are calculated by taking the highest value from the left and right scores. This dataset can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1_THSvQVy3KW8yWwOO0a9oS4qlZ_6APdo/view?usp=sharing) 

## Merge REBA dataset with Human3.6m dataset:
Our REBA dataset can be merged with Human3.6m dataset's 3D body joint keypoints.
