## Introduction
- <b>INRIA Holidays IR</b> is the image retrieval task on INRIA Holidays dataset. The Holidays dataset is a set of images which mainly contains some of our personal holidays photos. The remaining ones were taken on purpose to test the robustness to various attacks: rotations, viewpoint and illumination changes, blurring, etc. The dataset includes a very large variety of scene types (natural, man-made, water and fire effects, etc) and images are in high resolution. The dataset contains 500 image groups, each of which represents a distinct scene or object. The first image of each group is the query image and the correct retrieval results are the other images of the group.

- <b>Methodoloy</b>: we used <b>VGG16</b> for extracting features from images, and than we built a <b>SiameseNet</b> to extract higer features that work well for image retrieval tasks. Addition to boost matching, we used <b>KDTree</b> to descrease time query.

<p align="center">
    <img src="./demo.png" width=600>
 
