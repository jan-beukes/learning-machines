package nn

import "core:fmt"
import "core:slice"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:mem"
import "core:encoding/cbor"
import "core:thread"
import "core:sync"
import vmem "core:mem/virtual"

Layer :: struct {
    num_in: int,
    num_out: int, // This is equal to the number of neurons on this layer
    // weights[i] is weights for neuron i and weights[i][j] is the jth incomming weight
    weights: [][]f32,
    weight_grads: [][]f32,
    biases: []f32,
    bias_grads: []f32,

    activation: Activation,
}

Neural_Network :: struct {
    layers: []Layer,
    input_size, output_size: int,
    largest_layer_size: int,
    random: Random,
    cost: Cost,
    arena: vmem.Arena `cbor:"-"`,
}

// This is used to store values during training forward pass that are needed when we perform
// backprop to update the gradients
Learn_Data :: struct {
    inputs: []f32, // a(i-1)
    activations: []f32, // a(i)
    weighted_inputs: []f32, // z(i)
    // This name is just what Sebastian Lague called it idk what else to call it
    node_values: []f32, // da(i)/dz(i) * dc/da(i) * ... da(L)/dz(L) * dc/da(L)
}


Data_Point :: struct {
    input: []f32,
    expected: []f32,
    label: int,
}

Learn_Task :: struct {
    network: Neural_Network,
    data_point: Data_Point,
    cost: f32,
    arena: vmem.Arena,
}

// shared nil is important if we want to check if the err is nil otherwise we would have to switch
// the type and check if it is the nil of that type
Error :: union #shared_nil {
    os.Error,
    cbor.Marshal_Error,
    cbor.Unmarshal_Error,
}

// This should probably be used with an arena allocator
learn_data_create :: proc(layers: []Layer, allocator := context.allocator) -> []Learn_Data {
    learn_data := make([]Learn_Data, len(layers), allocator)
    for i in 0..<len(layers) {
        learn_data[i].inputs = make([]f32, layers[i].num_in, allocator)
        learn_data[i].activations = make([]f32, layers[i].num_out, allocator)
        learn_data[i].weighted_inputs = make([]f32, layers[i].num_out, allocator)
        learn_data[i].node_values = make([]f32, layers[i].num_out, allocator)
    }
    return learn_data
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

layer_init :: proc(self: ^Layer, num_in: int, num_out: int,
    activation: Activation, random: Random) {
    self.num_in = num_in
    self.num_out = num_out
    self.activation = activation

    self.weights = make([][]f32, num_out)
    self.weight_grads = make([][]f32, num_out)
    self.biases = make([]f32, num_out)
    self.bias_grads = make([]f32, num_out)
    for i in 0..<num_out {
        self.weights[i] = make([]f32, num_in)
        self.weight_grads[i] = make([]f32, num_in)

        for &weight in self.weights[i] {
            weight = random.function(num_in, num_out, context.random_generator)
        }
    }
}

layer_calculate_output :: proc{
    layer_calculate_output_learn,
    layer_calculate_output_no_learn,
}

// no output since we are filling learn data activations
layer_calculate_output_learn :: proc(self: Layer, input: []f32, learn_data: Learn_Data) {
    assert(len(learn_data.inputs) == len(input), "This layers learn data does not have the same number of inputs")
    copy(learn_data.inputs, input)

    for neuron in 0..<self.num_out {
        weighted_input := self.biases[neuron]
        for i in 0..<len(input) {
            weighted_input += input[i] * self.weights[neuron][i]
        }
        learn_data.weighted_inputs[neuron] = weighted_input
    }
    for i in 0..<self.num_out {
        learn_data.activations[i] = self.activation.function(learn_data.weighted_inputs, i)
    }
}

layer_calculate_output_no_learn :: proc(self: Layer, input, output: []f32) {
    // forward_no_learn calls free_all at the end
    weighted_inputs := make([]f32, self.num_out, context.temp_allocator)
    for neuron in 0..<self.num_out {
        weighted_input := self.biases[neuron]
        for i in 0..<len(input) {
            weighted_input += input[i] * self.weights[neuron][i]
        }
        weighted_inputs[neuron] = weighted_input
    }
    for i in 0..<self.num_out {
        output[i] = self.activation.function(weighted_inputs, i)
    }
}

// update the layer's gradients
layer_update_gradients :: proc(self: ^Layer, neuron: int, layer_learn: Learn_Data) {
    // XXX: MAYBE race conditions are fine?
    for j in 0..<self.num_in {
        dcost_dweight := layer_learn.inputs[j] * layer_learn.node_values[neuron]
        self.weight_grads[neuron][j] += dcost_dweight
    }
    self.bias_grads[neuron] += layer_learn.node_values[neuron]
}

apply_gradients :: proc(self: Neural_Network, learn_rate: f32) {
    // apply gradients and reset to zero
    for layer in self.layers {
        for i in 0..<layer.num_out {
            for j in 0..<layer.num_in {
                layer.weights[i][j] += -learn_rate * layer.weight_grads[i][j]
                layer.weight_grads[i][j] = 0.0
            }
            layer.biases[i] += -learn_rate * layer.bias_grads[i]
            layer.bias_grads[i] = 0.0
        }
    }
}
 
update_gradients :: proc(self: Neural_Network, expected: []f32, learn_data: []Learn_Data) -> f32 {
    // Last layer
    last_layer_learn := learn_data[len(learn_data)-1]
    last_layer := &self.layers[len(self.layers)-1]
    cost := self.cost.function(last_layer_learn.activations, expected)
    for i in 0..<last_layer.num_out {
        cost_derivative := self.cost.derivative(last_layer_learn.activations[i], expected[i])
        activation_derivative := last_layer.activation.derivative(last_layer_learn.weighted_inputs, i)
        last_layer_learn.node_values[i] = activation_derivative * cost_derivative 

        layer_update_gradients(last_layer, i, last_layer_learn)
    }

    // Hidden Layers
    old_node_values := last_layer_learn.node_values
    for i := len(self.layers) - 2; i > 0; i -= 1 {
        layer := &self.layers[i]
        next_layer := self.layers[i+1]
        layer_learn := learn_data[i]

        for i in 0..<layer.num_out {
            node_value: f32
            // propogate node values
            for j in 0..<len(old_node_values) {
                // from current layer i into next layer neuron j
                weighted_input_derivative := next_layer.weights[j][i]
                node_value += weighted_input_derivative * old_node_values[j]
            }
            node_value *= layer.activation.derivative(layer_learn.weighted_inputs, i)
            layer_learn.node_values[i] = node_value

            // update the layer's gradients
            layer_update_gradients(layer, i, layer_learn)
        }
        old_node_values = layer_learn.node_values
    }

    // zero all the node values so that we could potentially reuse learn_data 
    for data in learn_data {
        slice.zero(data.node_values)
    }
    return cost
}

forward :: proc{
    forward_learn,
    forward_no_learn,
}

// runs the inputs throught the neural network and updates the learn data to be used for backprop
forward_learn :: proc(self: Neural_Network, input: []f32, learn_data: []Learn_Data) {
    layer_input := input
    for i in 0..<len(self.layers) {
        layer := self.layers[i]
        layer_calculate_output(layer, layer_input, learn_data[i])
        layer_input = learn_data[i].activations
    }
}

// runs the inputs throught the neural network and writes into the outputs
forward_no_learn :: proc(self: Neural_Network, input: []f32, output: []f32) {
    assert(len(output) == self.output_size)
    assert(len(input) == self.input_size)

    layer_input := make([]f32, self.largest_layer_size, context.temp_allocator)
    layer_output := make([]f32, self.largest_layer_size, context.temp_allocator)
    copy(layer_input, input)
    for i in 0..<len(self.layers) {
        layer := self.layers[i]
        layer_calculate_output(layer, layer_input[:layer.num_in], layer_output[:layer.num_out])
        copy(layer_input, layer_output[:layer.num_out])
    }
    copy(output, layer_output)

    free_all(context.temp_allocator)
}

learn_task_proc :: proc(task: thread.Task) {
    task_data := cast(^Learn_Task)task.data
    network := task_data.network
    data_point := task_data.data_point
    learn_data := learn_data_create(network.layers)
    forward(network, data_point.input, learn_data)
    task_data.cost = update_gradients(network, data_point.expected, learn_data)

    vmem.arena_destroy(&task_data.arena)
}

// train the network on the given batch with the given learn rate
// the learn rate is divided by the training batch so learn rate values should be proportional to batch size
// if num_threads is 0 then no multithreading is used. This is faster for small networks but scales
// badly when layer sizes get large
// Returns the average cost for this training batch
learn :: proc(self: Neural_Network, training_batch: []Data_Point, learn_rate: f32, num_threads := 0) -> f32 {
    // if num_threads is not set then we don't do any thread pool
    total_cost: f32
    if num_threads == 0 {
        arena: vmem.Arena
        learn_data := learn_data_create(self.layers, vmem.arena_allocator(&arena))
        defer vmem.arena_destroy(&arena)
        for data_point in training_batch {
            assert(len(data_point.input) == self.input_size)
            assert(len(data_point.expected) == self.output_size)

            context.allocator = context.temp_allocator
            forward(self, data_point.input, learn_data)
            total_cost += update_gradients(self, data_point.expected, learn_data)
            free_all(context.temp_allocator)
        }
    } else {
        learn_tasks := make([]Learn_Task, len(training_batch))
        defer delete(learn_tasks)

        pool: thread.Pool
        thread.pool_init(&pool, context.allocator, num_threads)
        thread.pool_start(&pool)
        defer thread.pool_destroy(&pool)

        for data_point, i in training_batch {
            assert(len(data_point.input) == self.input_size)
            assert(len(data_point.expected) == self.output_size)

            task := &learn_tasks[i]
            task.network = self
            task.data_point = data_point
            thread.pool_add_task(&pool, vmem.arena_allocator(&task.arena), learn_task_proc, task, i)
        }
        thread.pool_finish(&pool)
        for i in 0..<len(learn_tasks) {
            task, _ := thread.pool_pop_done(&pool)
            task_data := cast(^Learn_Task)task.data
            total_cost += task_data.cost
        }
    }
    apply_gradients(self, learn_rate / f32(len(training_batch)))
    return total_cost / f32(len(training_batch))
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

// Serialization using bill tin cbor
save_to_file :: proc(self: Neural_Network, path: string) -> Error {
    data := cbor.marshal(self) or_return
    defer delete(data)
    os.write_entire_file(path, data) or_return

    return nil
}

load_from_file :: proc(path: string) -> (network: Neural_Network, err: Error) {
    data := os.read_entire_file(path, context.allocator) or_return
    defer delete(data)
    // make sure that we use the arena allocator so that the network
    // can be properly deallocated on deinit
    allocator := vmem.arena_allocator(&network.arena)
    err = cbor.unmarshal(data, &network, allocator=allocator)
    if err != nil {
        vmem.arena_destroy(&network.arena)
        return
    }

    // set the configuration values from the kinds
    // we only serialize kinds since we can't serialize the function pointers in the structs
    network.cost = cost_from_kind(network.cost.kind)
    network.random = random_from_kind(network.random.kind)
    for &layer in network.layers {
        layer.activation = activation_from_kind(layer.activation.kind)
    }

    return
}

// Initialize a neural network with the given layer sizes and config
// The first size will be of the input
init :: proc(self: ^Neural_Network, layer_sizes: []int, config: Config = DEFAULT_CONFIG) {
    config := config
    context.allocator = vmem.arena_allocator(&self.arena)

    self.largest_layer_size = slice.max(layer_sizes)
    self.input_size = layer_sizes[0]
    self.output_size = layer_sizes[len(layer_sizes)-1]

    self.cost = cost_from_kind(config.cost)
    self.random = random_from_kind(config.random)

    layers: [dynamic]Layer
    // Create layers for all but first size since that is the input
    for i in 1..<len(layer_sizes) {
        num_in := layer_sizes[i-1]
        layer: Layer
        activation: Activation
        if i == len(layer_sizes) - 1 {
            activation = activation_from_kind(config.output_activation)
        } else {
            activation = activation_from_kind(config.activation)
        }
        layer_init(&layer, num_in, layer_sizes[i], activation, self.random)
        append(&layers, layer)
    }

    self.layers = layers[:]
}

reset :: proc{
    reset_with_config,
    reset_no_config,
}

// reset the weights and biases also changing the networks config
reset_with_config :: proc(self: ^Neural_Network, config: Config) {
    self.cost = cost_from_kind(config.cost)
    self.random = random_from_kind(config.random)
    for i in 0..<len(self.layers) {
        layer := &self.layers[i]
        if i == len(self.layers) - 1 {
            layer.activation = activation_from_kind(config.output_activation)
        } else {
            layer.activation = activation_from_kind(config.activation)
        }
        for i in 0..<layer.num_out {
            for &weight in layer.weights[i] {
                weight = self.random.function(layer.num_in, layer.num_out, context.random_generator)
            }
            layer.biases[i] = 0
        }
    }
}

// reset the weights and biases
reset_no_config :: proc(self: ^Neural_Network) {
    for &layer in self.layers {
        for i in 0..<layer.num_out {
            for &weight in layer.weights[i] {
                weight = self.random.function(layer.num_in, layer.num_out, context.random_generator)
            }
            layer.biases[i] = 0
        }
    }
}

deinit :: proc(self: ^Neural_Network) {
    vmem.arena_destroy(&self.arena)
    self^ = Neural_Network{}
}
