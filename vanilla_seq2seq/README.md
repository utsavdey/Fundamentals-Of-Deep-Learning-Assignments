# Vanilla Seq-to-Seq
----------------------------------------------------
 For Vanilla Seq-to-Seq have implemented the Encoder-Decoder model without the attention. We have built stacked encoder-decoder model here.
# Set up and Installation: #
----------------------------------------------------
Easiest way to try this code out is :-
1. Download the .pynb file
2. Upload it to Google Colab
3. Use the "Run All Cells" option to begin training the best model.

# Best Model #
----------------------------------------------------
![model](https://user-images.githubusercontent.com/12824938/119309723-4b516180-bc8c-11eb-9d5e-4a8781bf6e82.png)

Accuracy on test dataset= 34.00%

# Methods #
|   | Method name     | Description                                                                                                          |
|---|-----------------|----------------------------------------------------------------------------------------------------------------------|
| 1 | data            | Prepare the data by padding the output and then tokenizing it .                                                      |
| 2 | build_model     | Create a model without compiling , as required by parameters.                                                        |
| 3 | build_inference | Modifies our trained model to build a model that is capable of decoding given input.                                          |
| 4 | decode_batch    | It is used to decode batch of inputs using inference model                                                           |
| 5 | test_accuracy   | Returns the accuracy of the model using test data. Also generates file containing succesful prediciton and failures.  |
| 6 | batch_validate  | Returns the accuracy of the model using validation data                                                              |
| 7 | train           | Train using configs sent by wandb.                                                                                   |
| 8 | manual_train    | Train using our custom configs.                                                                                      |


# Ackonledgement #
1. Course slides of CS6910 course by Prof. Mithesh Khapra
