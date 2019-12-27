from whitebox.e2e import Encoding2Encoding
from whitebox.nlp import Embedding
from sklearn.neural_network import MLPRegressor
import torchvision.models as models


# Setting up embedding to embedding translation model
word_encoder = Embedding()
image_encoder = models.resnext50_32x4d(pretrained=True)

model = Encoding2Encoding(
        input_encoder=image_encoder,
        output_encoder=word_encoder,
        translator=MLPRegressor()
)
