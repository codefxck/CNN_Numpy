import numpy as np

def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2. / shape[1])

class SimpleCNN_FashionMNIST:
    def __init__(self, input_channels=1, output_classes=10):
        self.conv1 = ConvLayer(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, padding=1)
        self.fc1_input_dim = 128 * 3 * 3  # 确定全连接层的输入维度
        self.fc1 = FullyConnectedLayer(self.fc1_input_dim, 256)
        self.fc2 = FullyConnectedLayer(256, output_classes)
        self.dropout = DropoutLayer(0.5)
        self.shapes = {}  # 用于记录每层的输出形状

    def forward(self, x):
        self.shapes['input'] = x.shape
        x = self.conv1.forward(x)
        x = self.pool(x)
        x = self.conv2.forward(x)
        x = self.pool(x)
        x = self.conv3.forward(x)
        x = self.pool(x)
        self.shapes['pool3'] = x.shape  # 记录pool3的输出形状
        x = x.reshape(x.shape[0], -1)
        x = self.fc1.forward(x)
        x = self.dropout.forward(x)
        x = self.fc2.forward(x)
        return x

    def pool(self, x, size=2, stride=2):
        n, c, h, w = x.shape
        h_out = (h - size) // stride + 1
        w_out = (w - size) // stride + 1
        pooled = np.zeros((n, c, h_out, w_out))
        for i in range(0, h - size + 1, stride):
            for j in range(0, w - size + 1, stride):
                pooled[:, :, i//stride, j//stride] = np.max(x[:, :, i:i+size, j:j+size], axis=(2, 3))
        return pooled

    def backward(self, grad_output, learning_rate, clip_value=1.0):
        grad_output = self.fc2.backward(grad_output, learning_rate)
        grad_output = self.dropout.backward(grad_output)
        grad_output = self.fc1.backward(grad_output, learning_rate)
        grad_output = grad_output.reshape(self.shapes['pool3'])  # 使用记录的形状
        grad_output = self.conv3.backward(grad_output, learning_rate, clip_value)
        grad_output = self.conv2.backward(grad_output, learning_rate, clip_value)
        grad_output = self.conv1.backward(grad_output, learning_rate, clip_value)
        return grad_output

    def save(self, path):
        np.savez(path,
                 conv1_kernels=self.conv1.kernels, conv1_biases=self.conv1.biases,
                 conv2_kernels=self.conv2.kernels, conv2_biases=self.conv2.biases,
                 conv3_kernels=self.conv3.kernels, conv3_biases=self.conv3.biases,
                 fc1_weights=self.fc1.weights, fc1_biases=self.fc1.biases,
                 fc2_weights=self.fc2.weights, fc2_biases=self.fc2.biases)

    def load(self, path):
        data = np.load(path)
        self.conv1.kernels, self.conv1.biases = data['conv1_kernels'], data['conv1_biases']
        self.conv2.kernels, self.conv2.biases = data['conv2_kernels'], data['conv2_biases']
        self.conv3.kernels, self.conv3.biases = data['conv3_kernels'], data['conv3_biases']
        self.fc1.weights, self.fc1.biases = data['fc1_weights'], data['fc1_biases']
        self.fc2.weights, self.fc2.biases = data['fc2_weights'], data['fc2_biases']

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernels = he_initialization((out_channels, in_channels, kernel_size, kernel_size))
        self.biases = np.zeros(out_channels)
        self.grad_kernels = np.zeros_like(self.kernels)
        self.grad_biases = np.zeros_like(self.biases)
        self.input = None

    def forward(self, x):
        self.input = x
        n, c, h, w = x.shape
        h_out = h - self.kernel_size + 1 + 2 * self.padding
        w_out = w - self.kernel_size + 1 + 2 * self.padding
        out = np.zeros((n, self.out_channels, h_out, w_out))
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        for i in range(h_out):
            for j in range(w_out):
                x_slice = x_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                out[:, :, i, j] = np.tensordot(x_slice, self.kernels, axes=([1, 2, 3], [1, 2, 3])) + self.biases
        return out

    def backward(self, grad_output, learning_rate, clip_value=1.0):
        n, c, h, w = grad_output.shape
        grad_input = np.zeros_like(self.input)
        x_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        grad_x_padded = np.pad(grad_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        for i in range(h):
            for j in range(w):
                x_slice = x_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                for k in range(self.out_channels):
                    self.grad_kernels[k] += np.sum(x_slice * (grad_output[:, k, i, j])[:, None, None, None], axis=0)
                grad_x_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += np.sum(self.kernels * (grad_output[:, :, i, j])[:, :, None, None, None], axis=1)
        self.grad_biases = np.sum(grad_output, axis=(0, 2, 3))
        grad_input = grad_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # Clip gradients to prevent overflow
        np.clip(self.grad_kernels, -clip_value, clip_value, out=self.grad_kernels)
        np.clip(self.grad_biases, -clip_value, clip_value, out=self.grad_biases)

        self.kernels -= learning_rate * self.grad_kernels
        self.biases -= learning_rate * self.grad_biases
        return grad_input

class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = he_initialization((input_dim, output_dim))
        self.biases = np.zeros(output_dim)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)

        # Clip gradients to prevent overflow
        np.clip(self.grad_weights, -1.0, 1.0, out=self.grad_weights)
        np.clip(self.grad_biases, -1.0, 1.0, out=self.grad_biases)

        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases
        return grad_input

class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x):
        self.mask = np.random.rand(*x.shape) > self.dropout_rate
        return x * self.mask / (1.0 - self.dropout_rate)

    def backward(self, grad_output):
        return grad_output * self.mask / (1.0 - self.dropout_rate)