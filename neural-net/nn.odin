package nn

import "core:fmt"
import "core:slice"
import "core:math"
import "core:math/rand"
import "core:mem"
import vmem "core:mem/virtual"

Neuron :: struct {
    weights: []f32,
    bias: f32,
    // local derivatives
    weight_grads: []f32,
    activation_grad: f32,
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
    arena: vmem.Arena,
}

Data_Point :: struct {
    input: []f32,
    expected: []f32,
    label: int,
}

// NOTE: This does not clone the inputs
data_create :: proc(inputs: [][]f32, labels: []int, num_labels: int) -> []Data_Point {
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

    rand.shuffle(batch[:])
    return batch
}

data_destroy :: proc(batch: []Data_Point) {
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
neuron_activate :: proc(self: ^Neuron, x: []f32, activation: Activation, update_grad := false) -> f32 {
    assert(len(self.weights) == len(x))
    z: f32 = 0.0
    for i in 0..<len(x) {
        z += self.weights[i] * x[i]
    }
    z += self.bias

    if update_grad {
        self.activation_grad = activation.derivative(z)
        self.bias_grad = self.activation_grad
        for i in 0..<len(x) {
            self.weight_grads[i] = self.activation_grad * x[i]
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
    copy(output, layer_outputs)
}

backward :: proc(self: Neural_Network, cost_grads: []f32) {
    // output layer
    output_layer := self.layers[len(self.layers)-1]
    assert(len(cost_grads) == len(output_layer.neurons))
    for i in 0..<len(output_layer.neurons) {
        neuron := &output_layer.neurons[i]
        for &wg in neuron.weight_grads {
            wg *= cost_grads[i]
        }
        neuron.activation_grad *= cost_grads[i]
        neuron.bias_grad *= cost_grads[i]
    }

    // hidden layers
    for i := len(self.layers) - 2; i > 0; i -= 1 {
        layer := self.layers[i]
        for i in 0..<len(layer.neurons) {
            neuron := &layer.neurons[i]
            a_cost_grad: f32
            for next in self.layers[i+1].neurons {
                // i is the input corresponding the the current neuron
                a_cost_grad += next.weights[i] * next.activation_grad
            }
            neuron.activation_grad *= a_cost_grad
            neuron.bias_grad *= a_cost_grad
            for &wg in neuron.weight_grads {
                wg *= a_cost_grad
            }
        }
    }
}

learn :: proc(self: Neural_Network, training_data: []Data_Point, learn_rate: f32) {

    output := make([]f32, self.output_size)
    cost_grads := make([]f32, self.output_size)
    defer { delete(output); delete(cost_grads) }
    cost: f32
    for data in training_data {
        assert(len(data.input) == self.input_size)
        assert(len(data.expected) == self.output_size)
        forward(self, data.input, output, update_grad=true)
        cost += self.config.cost.function(data.expected, output)
        for i in 0..<len(output) {
            cost_grads[i] += self.config.cost.derivative(data.expected[i], output[i])
        }
    }
    fmt.println("Cost:", cost)
    // backpropogate
    backward(self, cost_grads)
    // gradient decent and reset grads
    for layer in self.layers {
        for &neuron in layer.neurons {
            neuron.bias += -learn_rate*neuron.bias_grad
            for i in 0..<len(neuron.weights) {
                neuron.weights[i] += -learn_rate*neuron.weight_grads[i]
                neuron.weight_grads[i] = 0.0
            }
            neuron.bias_grad = 0.0
            neuron.activation_grad = 0.0
        }
    }
}

evaluate :: proc(self: Neural_Network, testing_data: []Data_Point) -> f32 {
    num_correct := 0
    num_inputs := len(testing_data)
    output := make([]f32, self.output_size)
    defer delete(output)
    for data in testing_data {
        forward(self, data.input, output)
        predicted := slice.max_index(output)
        if predicted == data.label {
            num_correct += 1
        }
    }
    return f32(num_correct) / f32(num_inputs)
}

// Initialize a neural network with the given layer sizes and config
// The first size will be of the input
init :: proc(self: ^Neural_Network, layer_sizes: []int, config: Config = DEFAULT_CONFIG) {
    assert(self != nil)
    context.allocator = vmem.arena_allocator(&self.arena)

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

deinit :: proc(self: ^Neural_Network) {
    vmem.arena_destroy(&self.arena)
}
