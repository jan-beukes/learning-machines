package nn

import "core:fmt"
import "core:slice"
import "core:math"
import "core:math/rand"

Neuron :: struct {
    weights: []f32,
    weight_grads: []f32, // local derivatives
    bias: f32,
    bias_grad: f32,
}

Layer :: struct {
    num_in: int,
    num_out: int,
    neurons: []Neuron,
}

Neural_Network :: struct {
    config: Config,
    layers: []Layer,
    input_size, output_size: int,
    largest_layer_size: int,
}

Data_Point :: struct {
    input: []f32,
    expected: []f32,
    label: int,
}

// NOTE: This does not clone the inputs
batch_create :: proc(inputs: [][]f32, labels: []int, num_labels: int) -> []Data_Point {
    assert(len(inputs) == len(labels))
    batch := make([]Data_Point, len(inputs))
    for i in 0..<len(batch) {
        batch[i].input = inputs[i]

        label := labels[i]
        batch[i].label = label

        // one hot
        assert(label < num_labels)
        batch[i].expected = make([]f32, num_labels)
        batch[i].expected[label] = 1.0
    }

    return batch
}

batch_destroy :: proc(batch: []Data_Point) {
    for dp in batch {
        delete(dp.expected)
    }
    delete(batch)
}

neuron_init :: proc(self: ^Neuron, num_in: int) {
    self.weights = make([]f32, num_in)
    self.weight_grads = make([]f32, num_in)

    for &weight in self.weights {
        weight = rand.float32_range(-1.0, 1.0)
    }
    self.bias = rand.float32_range(-1.0, 1.0)
}

// So we do half of the gradient updating here instead of storing z values
neuron_activate :: proc(self: ^Neuron, x: []f32, activation: Activation, update_grad := true) -> f32 {
    assert(len(self.weights) == len(x))
    z: f32 = 0.0
    for i in 0..<len(x) {
        z += self.weights[i] * x[i]
    }
    z += self.bias

    if update_grad {
        self.bias_grad = activation.derivative(z)
        for i in 0..<len(x) {
            self.weight_grads[i] = activation.derivative(z) * x[i]
        }
    }

    a := activation.function(z)
    return a
}

forward :: proc(self: Neural_Network, input: []f32, output: []f32, update_grad := false) {
    defer free_all(context.temp_allocator)
    layer_inputs := make([]f32, self.largest_layer_size, context.temp_allocator)
    layer_outputs := make([]f32, self.largest_layer_size, context.temp_allocator)

    copy(layer_inputs, input)
    for layer in self.layers {
        for i in 0..<len(layer.neurons) {
            inputs := layer_inputs[:layer.num_in]
            layer_outputs[i] = neuron_activate(&layer.neurons[i], inputs,
                self.config.activation, update_grad)
        }
        copy(layer_inputs, layer_outputs)
    }
    copy(output, layer_outputs[:len(output)])
}

learn :: proc(self: Neural_Network, training_data: []Data_Point, learn_rate: f32) {
    output := make([]f32, self.output_size)
    defer delete(output)
    for data in training_data {
        assert(len(data.input) == self.input_size)
        assert(len(data.expected) == self.output_size)

        forward(self, data.input, output, update_grad=true)
        fmt.println(output)
    }
}

// Initialize a neural network with the given layer sizes and config
// The first size will be of the input
init :: proc(self: ^Neural_Network, layer_sizes: []int, config: Config = DEFAULT_CONFIG) {
    assert(self != nil)

    self.config = config
    self.largest_layer_size = slice.max(layer_sizes)
    self.input_size = layer_sizes[0]
    self.output_size = layer_sizes[len(layer_sizes)-1]

    layers: [dynamic]Layer
    // Create layers for all but first size since that is the input
    for i in 1..<len(layer_sizes) {
        num_in := layer_sizes[i-1]
        num_out := layer_sizes[i]
        layer := Layer{ num_in, num_out, make([]Neuron, layer_sizes[i]) }
        for &neuron in layer.neurons {
            neuron_init(&neuron, num_in)
        }
        append(&layers, layer)
    }

    self.layers = layers[:]
}

deinit :: proc(self: Neural_Network) {
    unimplemented("deinit")
}
