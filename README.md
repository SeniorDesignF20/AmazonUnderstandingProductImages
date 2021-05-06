# Honors Senior Design - Fall 2020 to Spring 2021
# The University of Texas at Austin at Zoom
# Electrical and Computer Engineering Dept.
# Team H19

Team Members:
* Ryed Ahmed
* Matt MacDonald
* Soroush Famili
* Jessica Pham
* William Gu

Mentors:
* Dr. Zhangyang "Atlas" Wang (UT)
* Dr. Sumeet Katariya (Amazon)

# Amazon - Understanding Product Images
Given the large size of Amazon’s catalog, there are many products that look very similar but contain small yet important differences that can only be found upon close inspection. A model that only analyzes trivial pixel-level differences would be inadequate, considering the variety of images that could be encountered. Thus, we developed a robust object detection model that detects big picture differences in product image pairs, if they exist. Utilizing a convolution neural network followed by GradCAM, our model determines if there are differences across images and places bounding boxes around the areas of difference for visualization. Additionally, we built a web-based UI that allows for humans to seamlessly use the model to analyze their own pictures.

# Inspiration
Project was largely inspired by J. Wu, Y. Ye, Y. Che, and Z. Weng. Spot the Difference by Object Detection. 2018. https://arxiv.org/abs/1801.01051v1.

# Usage

## GradCAM Demo
To demo the project with the existing models, go into the folder 'model' and run 'python3 test_gradcam.py'. If you want to use a specific model, edit the dataset_size parameter in test_gradcam.py. If you are using Windows Subsystem for Linux make sure you can display images using a tool such as XLaunch. Feel free to experiment with different GradCAM settings in test_gradcam.py such as using XGradCAM or experiment with the smooth settings. 

## Example
The demo will run the gradcam model on all image pairs in the Test_GradCAM folder. If you want to test your own images, place your image pair in a folder and place that folder inside Test_GradCAM. The following is a result we get when running test_gradcam.py using the default small model, GradCAMPlusPlus, aug_smooth=True, eigen_smooth=False.
![Cereal](https://user-images.githubusercontent.com/31623958/117377762-9e0fd880-ae99-11eb-84ac-ed37a706a467.png)


## UI Demo
To run the UI, go to the UserInterface folder and run 'python3 manage.py runserver' and in your browser go to localhost (hhtp://128.0.0.1:8000/).

## Example
![UI1](https://user-images.githubusercontent.com/31623958/117378342-d5cb5000-ae9a-11eb-9f46-2dfa2ab1f4fe.JPG)
![UI2](https://user-images.githubusercontent.com/31623958/117378348-d8c64080-ae9a-11eb-92ba-8e66870a6ab7.JPG)
