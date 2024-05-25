# neuralimagemodel
Train a neural model to classify images

First you run the code below to download the dataset to your local machine
```python
python load_dataset.py
```

Then, train and save your model
```python
python train.py
```

Finally, load the saved model and start to make predictions
```python
python load_saved_model.py
```

The model is saved as `model_image_classifier.h5`.

Enjoy!