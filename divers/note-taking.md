# Image segmentation

Three types

- semantic segmentation: identify all the categories in the image. All the humans get the human label, all the cars get the label defined for the car category... And the background is also detected.

- instance segmentation: the idea is to detect all the objects that are not the background. If two objects belong to the same category, they still get a different label (two different cars will have two different labels).

- panoptic segmentation: Mix of the two others. It detects every categories of the image (every pixel is assigned to a category) and also the different instances.

# U-net implementation

## Double Conv class

Since at every stage of the model, some input gets passed through a double convolutional layer, with a two times repetition of a 2D-convolutional layer, a batch normalization layer, and a ReLU activation function. 

- Les convolutions extraient les features de l'image en appliquant un filtre 3 par 3
- La batch norm normalise les sorties de la couche de convolution. C'est une opération effectuée pour chaque batch.
- La fonction d'activation ReLU est appliquée ensuite pour introduire de la non-linéarité dans le modèle.

The parameter `bias=False` means that the convolution layer doesn't add bias to the output. Normally, it adds a learned additional term, which is useful to include a translation to the extracted feature (?). In any case, the `BatchNorm` layer removes any bias (ajusts the means and std), so having bias on True here doesn't change anything.

The U-Net model implemented is as the following:

![U-NET image](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)