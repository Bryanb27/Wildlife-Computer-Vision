# Model Performance Overview
This first version of the custom animal species classifier, built using the MobileNetV2 architecture and fine-tuned for multi-class classification, achieves the following metrics:

* Training Accuracy: 88,56%
* Training Loss: 0.8706
* Validation Accuracy: 84,17%
* Validation Loss: 1.1304

The dataset is relatively small (5,400 images), and achieving close to 85% accuracy with new data and having relatively low training and validation losses is a pretty good result.

For this new version i've applied the following:

* Added **batch normalization** layers around the dense layer to speed up training and improve stability by normalizing activations.
* Added **learning rate schedule**, allowing the learning rate to dynamically adjust throughout the training process instead of using a fixed value.
* Took down the **batch size** to improve loss performance in both training and validation.