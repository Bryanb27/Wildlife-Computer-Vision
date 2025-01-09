# Model Performance Overview
This first version of the custom animal species classifier, built using the MobileNetV2 architecture and fine-tuned for multi-class classification, achieves the following metrics:

* Training Accuracy: 75,91%
* Training Loss: 1.7239
* Validation Accuracy: 72.76%
* Validation Loss: 1.8994

This new version achieves a training accuracy of 75.91% and a validation accuracy of 72.76%. The close accuracy values suggest the model is generalizing moderately well now, but the higher validation loss shows that the model still have some problems with overfitting.

I've applied, to fix some of the overfitting from the previous version, the following:

* Made the **input shape bigger** to preserve more details.
* Added **L2 regularization** to the weights of the dense layer. The regularization helps prevent overfitting by penalizing large weight values.
* Added a **dropout** of 0.5 to the model, meaning that 50% of the neurons in that layer will be randomly set to zero (ignored) during each training iteration. This forces the model to learn more robust and generalizable features by not relying too heavily on any specific set of neurons.
* **Took down** some of the values in the image data generator because it was losing its ability to adapt to new data (it was fluctuating heavily).
* Changed the **validation split** to 20%.