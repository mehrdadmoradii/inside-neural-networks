import keras
from manim import *

from dataset import get_dataset


X_train, X_test, y_train, y_test = get_dataset()

model = keras.models.load_model('model.keras')


class InsideNeuralNetwork(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ax = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=7,
            y_length=7,
            tips=False,
        )

        self.dataset = X_test[:100]
        self.y = y_test[:100]

        self.dots = VGroup(*self.c2p(self.dataset))

    def construct(self):

        self.add(self.ax, self.dots)

        for idx, l in enumerate(model.layers[:-1]):
            if hasattr(l, 'get_weights') and len(l.get_weights()) > 0:
                w, b = l.get_weights()
                self.display_title(f'x @ w{idx+1}')
                self.mm(w)
                self.display_title(f'x + b{idx+1}')
                self.vector_addition(b)
                activation_function = keras.activations.get(l.activation)
                self.display_title(l.activation.__name__.replace('_', ' ').title())
                self.dataset_transform(lambda dataset: activation_function(dataset).numpy())

        last_layer = model.layers[-1]
        w, b = last_layer.get_weights()

        def last_layer_weights(dataset):
            y = dataset @ w
            zeros = np.zeros_like(y)
            y = np.hstack((y, zeros))   # transforming data to 1 dimensional space but plotting it on 2d plane
            return y

        self.display_title(f'x @ w{len(model.layers)}')
        self.dataset_transform(last_layer_weights)

        def last_layer_bias(dataset):
            y = dataset.copy()
            y[:, 0] = y[:, 0] + b
            return y

        self.display_title(f'x + b{len(model.layers)}')
        self.dataset_transform(last_layer_bias)

        def last_layer_activation(dataset):
            activation_function = keras.activations.get(last_layer.activation)
            ys = activation_function(dataset[:, 0]).numpy()
            output = np.hstack((dataset[:, 0].reshape(-1, 1), ys.reshape(-1, 1)))
            return output

        self.display_title(last_layer.activation.__name__.replace('_', ' ').title())
        self.dataset_transform(last_layer_activation)

        self.wait(1)

    def dataset_transform(self, f):
        transformed_dataset = f(self.dataset)
        transformed_dots = VGroup(*self.c2p(transformed_dataset))
        self.play(Transform(self.dots, transformed_dots))
        self.dataset = transformed_dataset

    def vector_addition(self, vector):
        if isinstance(vector, list):
            vector = np.array(vector)
        self.dataset_transform(lambda dataset: dataset + vector)

    def mm(self, matrix):
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        self.dataset_transform(lambda dataset: dataset @ matrix)

    def display_title(self, title_text):
        title = Text(title_text, font_size=24, font='Roboto').to_corner(UP + LEFT)
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        self.play(FadeOut(title), run_time=1)

    def c2p(self, points):
        def get_color(idx):
            return ManimColor('#3476cd') if self.y[idx] == 0 else ManimColor('#ab482a')
        return [Dot(self.ax.c2p(p[0], p[1]), color=get_color(idx), fill_opacity=0.8, radius=0.04) for idx, p in enumerate(points)]
     