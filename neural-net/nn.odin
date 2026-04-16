package nn

import "base:runtime"
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

    // weight matrices num_out x num_in
    weights: []f32,
    weight_grads: []f32,
    // the moving averages used for ADAM
    weight_m: []f32,
    weight_v: []f32,

    biases: []f32,
    bias_grads: []f32,
    bias_m: []f32,
    bias_v: []f32,

    disabled: []bool, // used during dropout
    activation: Activation,
}

Neural_Network :: struct {
    layers: []Layer,
    input_size, output_size: int,
    largest_layer_size: int,
    dropout: f32,
    training_iterations: int,
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
    label: i32,
}

Learn_Task :: struct {
    network: Neural_Network,
    data_point: Data_Point,
    cost: f32,
    arena: vmem.Arena,
}

Eval_Task :: struct {
    network: Neural_Network,
    data_point: Data_Point,
    prediction: i32,
}

// shared nil is important if we want to check if the err is nil otherwise we would have to switch
// the type and check if it is the nil of that type
Error :: union #shared_nil {
    os.Error,
    cbor.Marshal_Error,
    cbor.Unmarshal_Error,
}

// This should be used with an arena allocator
learn_data_create :: proc(layers: []Layer, allocator := context.allocator) -> []Learn_Data {
    context.allocator = allocator
    learn_data := make([]Learn_Data, len(layers))
    for i in 0..<len(layers) {
        learn_data[i].inputs = make([]f32, layers[i].num_in)
        learn_data[i].activations = make([]f32, layers[i].num_out)
        learn_data[i].weighted_inputs = make([]f32, layers[i].num_out)
        learn_data[i].node_values = make([]f32, layers[i].num_out)
    }
    return learn_data
}

// Create a batch of data points given inputs and labels.
// This creates the one hot list for expected output
// NOTE: This clones the inputs
batch_create :: proc(inputs: [][]f32, labels: []i32, num_labels: i32, allocator := context.allocator) -> []Data_Point {
    assert(len(inputs) == len(labels))
    context.allocator = allocator
    batch := make([]Data_Point, len(inputs))
    for i in 0..<len(batch) {
        batch[i].input = slice.clone(inputs[i])

        label := labels[i]
        batch[i].label = label

        // one hot
        assert(label < num_labels)
        batch[i].expected = make([]f32, num_labels)
        batch[i].expected[label] = 1.0
    }

    return batch
}

batch_destroy :: proc(batch: []Data_Point, allocator := context.allocator) {
    context.allocator = allocator
    for dp in batch {
        delete(dp.expected)
        delete(dp.input)
    }
    delete(batch)
}

// HOLY MEMORY
layer_init :: proc(self: ^Layer, num_in: int, num_out: int,
    activation: Activation, random: Random) {
    self.num_in = num_in
    self.num_out = num_out
    self.activation = activation

    self.weights = make([]f32, num_out*num_in)
    self.weight_grads = make([]f32, num_out*num_in)
    self.weight_m = make([]f32, num_out*num_in)
    self.weight_v = make([]f32, num_out*num_in)

    self.biases = make([]f32, num_out)
    self.bias_grads = make([]f32, num_out)
    self.bias_m = make([]f32, num_out)
    self.bias_v = make([]f32, num_out)

    self.disabled = make([]bool, num_out)
    for i in 0..<num_out {
        for j in 0..<num_in {
            self.weights[i*num_in + j] = random.function(num_in, num_out, context.random_generator)
        }
    }
}

// update the layer's gradients
layer_update_gradients :: proc(self: ^Layer, layer_learn: Learn_Data) {
    //XXX: We have no lock since it is major performace boost to just embrace the race conditions
    // idk if this is cooked but it mostly works
    for neuron in 0..<self.num_out {
        if self.disabled[neuron] do continue
        for j in 0..<self.num_in {
            dcost_dweight := layer_learn.inputs[j] * layer_learn.node_values[neuron]
            self.weight_grads[neuron*self.num_in + j] += dcost_dweight
        }
        self.bias_grads[neuron] += layer_learn.node_values[neuron]
    }
}

// regularization comes from adding λ/2n * sum(w^2) to the cost function to prevent large weights
// which can lead to overfitting
// beta1 and beta2 are the momentum paramaters for the moving averages used in ADAM to smooth out
// the decent down the cost function
// m_t = beta_1*m_t-1 + (1 - beta_1)dC/dw
// v_t = beta_2*v_t-1 + (1 - beta_2)(dC/dw)^2
UPDATE_EPSILON :: 1e-8
apply_gradients :: proc(self: ^Neural_Network, learn_rate: f32, regularization: f32, beta1: f32, beta2: f32) {
    self.training_iterations += 1

    weight_decay := (1 - regularization*learn_rate)
    beta1_t := 1 - math.pow(beta1, f32(self.training_iterations))
    beta2_t := 1 - math.pow(beta2, f32(self.training_iterations))

    // apply gradients and reset to zero
    for layer in self.layers {
        for i in 0..<len(layer.weights) {
            if layer.disabled[i / layer.num_in] do continue
            grad := layer.weight_grads[i]
            layer.weights[i] *= weight_decay

            // ADAM
            m := beta1*layer.weight_m[i] + (1 - beta1)*grad
            v := beta2*layer.weight_v[i] + (1 - beta2) * (grad*grad)
            m_hat := m / beta1_t
            v_hat := v / beta2_t

            layer.weights[i] += -learn_rate*(m_hat / (math.sqrt(v_hat) + UPDATE_EPSILON))
            layer.weight_m[i] = m
            layer.weight_v[i] = v
            layer.weight_grads[i] = 0
        }

        // update biases
        for i in 0..<layer.num_out {
            if layer.disabled[i] do continue
            grad := layer.bias_grads[i]
            
            // ADAM
            m := beta1*layer.bias_m[i] + (1 - beta1)*grad
            v := beta2*layer.bias_v[i] + (1 - beta2) * (grad*grad)
            m_hat := m / beta1_t
            v_hat := v / beta2_t
            
            layer.biases[i] += -learn_rate*(m_hat / (math.sqrt(v_hat) + UPDATE_EPSILON ))
            layer.bias_m[i] = m
            layer.bias_v[i] = v
            layer.bias_grads[i] = 0.0
        }
    }
}
 
update_gradients :: proc(self: Neural_Network, expected: []f32, learn_data: []Learn_Data) -> f32 {
    // Last layer
    last_layer_learn := learn_data[len(learn_data)-1]
    last_layer := &self.layers[len(self.layers)-1]
    cost := self.cost.function(last_layer_learn.activations, expected)
    
    // calculate node values
    for i in 0..<last_layer.num_out {
        cost_derivative := self.cost.derivative(last_layer_learn.activations[i], expected[i])
        activation_derivative := last_layer.activation.derivative(last_layer_learn.weighted_inputs, i)
        last_layer_learn.node_values[i] = activation_derivative * cost_derivative 
    }
    layer_update_gradients(last_layer, last_layer_learn)

    // Hidden Layers
    old_node_values := last_layer_learn.node_values
    for l := len(self.layers) - 2; l >= 0; l -= 1 {
        layer := &self.layers[l]
        next_layer := self.layers[l+1]
        layer_learn := learn_data[l]

        // calculate node values
        for i in 0..<layer.num_out {
            if layer.disabled[i] do continue

            node_value: f32
            // propogate node values
            for j in 0..<len(old_node_values) {
                // from current layer i into next layer neuron j
                weighted_input_derivative := next_layer.weights[j*next_layer.num_in + i]
                node_value += weighted_input_derivative * old_node_values[j]
            }
            node_value *= layer.activation.derivative(layer_learn.weighted_inputs, i)
            layer_learn.node_values[i] = node_value
        }
        // update the layer's gradients
        layer_update_gradients(layer, layer_learn)

        old_node_values = layer_learn.node_values
    }

    // zero the learn data
    for data in learn_data {
        slice.zero(data.inputs)
        slice.zero(data.activations)
        slice.zero(data.weighted_inputs)
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
    for layer, i in self.layers {
        layer_learn := learn_data[i]
        copy(layer_learn.inputs, layer_input)
        for neuron in 0..<layer.num_out {
            // skip any disabled hidden layers
            if layer.disabled[neuron] {
                continue
            }
            weighted_input := layer.biases[neuron]
            for j in 0..<layer.num_in {
                weighted_input += layer_input[j] * layer.weights[neuron*layer.num_in + j]
            }
            layer_learn.weighted_inputs[neuron] = weighted_input
        }
        for neuron in 0..<layer.num_out {
            // skip any disabled hidden layers
            if layer.disabled[neuron] {
                continue
            }
            activation := layer.activation.function(layer_learn.weighted_inputs, neuron)
            // we need to scale the weights coming out from hidden layers so that the overall
            // sum is the same
            activation = i == len(self.layers) - 1 ? activation : activation / (1 - self.dropout)
            layer_learn.activations[neuron] = activation
        }
        layer_input = layer_learn.activations
    }
}

// runs the inputs throught the neural network and returns the outputs
forward_no_learn :: proc(self: Neural_Network, input: []f32, allocator := context.allocator) -> []f32 {
    assert(len(input) == self.input_size)
    context.allocator = allocator

    layer_input := make([]f32, self.largest_layer_size)
    layer_output := make([]f32, self.largest_layer_size)
    weighted_inputs := make([]f32, self.largest_layer_size)
    defer { delete(layer_input); delete(weighted_inputs) }
    copy(layer_input, input)
    for layer, i in self.layers {
        layer_in := layer_input[:layer.num_in]
        layer_out := layer_output[:layer.num_out]
        for neuron in 0..<layer.num_out {
            weighted_input := layer.biases[neuron]
            for j in 0..<layer.num_in {
                weight := layer.weights[neuron*layer.num_in + j]
                weighted_input += layer_in[j] * weight
            }
            weighted_inputs[neuron] = weighted_input
        }
        for neuron in 0..<layer.num_out {
            layer_out[neuron] = layer.activation.function(weighted_inputs[:layer.num_out], neuron)
        }
        copy(layer_input, layer_out)
    }
    return layer_output[:self.output_size]
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
// regularization is the λ/n term from L2 regularization / weight decay
// Returns the average cost for this training batch
learn :: proc(self: ^Neural_Network, training_batch: []Data_Point, learn_rate: f32,
    regularization: f32 = 0, beta1: f32 = 0.9, beta2: f32 = 0.99, num_threads := 4) -> f32 {
    learn_tasks := make([]Learn_Task, len(training_batch))
    defer delete(learn_tasks)

    // XXX: I feel like the thread pool stuff can be improved a lot. I have tried manually spawning
    // num_threads threads and then slicing the training batch but that didn'y seem to help.
    // I also tried keeping a single thread pool alive during training. It is possible that I made
    // other changes and that these actaully are improvements
    pool: thread.Pool
    thread.pool_init(&pool, context.allocator, num_threads)
    thread.pool_start(&pool)
    defer thread.pool_destroy(&pool)

    // dropout hidden neurons
    for layer in self.layers[:len(self.layers) - 1] {
        for i in 0 ..< layer.num_out {
            layer.disabled[i] = rand.float32() < self.dropout
        }
    }

    for data_point, i in training_batch {
        assert(len(data_point.input) == self.input_size)
        assert(len(data_point.expected) == self.output_size)

        task := &learn_tasks[i]
        task.network = self^
        task.data_point = data_point
        thread.pool_add_task(&pool, vmem.arena_allocator(&task.arena), learn_task_proc, task, i)
    }
    thread.pool_finish(&pool)
    total_cost: f32
    for i in 0..<len(learn_tasks) {
        task, _ := thread.pool_pop_done(&pool)
        task_data := cast(^Learn_Task)task.data
        total_cost += task_data.cost
    }
    apply_gradients(self, learn_rate / f32(len(training_batch)), regularization, beta1, beta2)

    // reset disabled neurons
    for layer in self.layers[:len(self.layers) - 1] {
        slice.zero(layer.disabled)
    }

    return total_cost / f32(len(training_batch))
}

eval_task_proc :: proc(task: thread.Task) {
    task_data := cast(^Eval_Task)task.data
    data_point := task_data.data_point
    network := task_data.network

    output := forward(network, data_point.input)
    defer delete(output)
    predicted := slice.max_index(output)
    task_data.prediction = i32(predicted)
}

// Return the percent of correct predictions from the neural network
evaluate :: proc(self: Neural_Network, testing_batch: []Data_Point, num_threads := 4) -> f32 {
    eval_tasks := make([]Eval_Task, len(testing_batch))
    defer delete(eval_tasks)

    pool: thread.Pool
    thread.pool_init(&pool, context.allocator, num_threads)
    thread.pool_start(&pool)
    defer thread.pool_destroy(&pool)
    for data_point, i in testing_batch {
        assert(len(data_point.input) == self.input_size)
        assert(len(data_point.expected) == self.output_size)
        task := &eval_tasks[i]
        task.network = self
        task.data_point = data_point
        thread.pool_add_task(&pool, runtime.default_allocator(), eval_task_proc, task, i)
    }

    thread.pool_finish(&pool)
    num_correct := 0
    num_inputs := len(testing_batch)
    for i in 0..<len(eval_tasks) {
        task, _ := thread.pool_pop_done(&pool)
        task_data := cast(^Eval_Task)task.data
        if task_data.prediction == task_data.data_point.label {
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
    err = cbor.unmarshal(data, &network, allocator = allocator)
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
init :: proc(self: ^Neural_Network, layer_sizes: []int, dropout: f32 = 0, config: Config = DEFAULT_CONFIG) {
    config := config
    context.allocator = vmem.arena_allocator(&self.arena)

    self.largest_layer_size = slice.max(layer_sizes)
    self.input_size = layer_sizes[0]
    self.output_size = layer_sizes[len(layer_sizes)-1]

    if config.cost == .Cross_Entropy && config.output_activation != .Softmax {
        panic("Cross Entropy must be used with a Softmax output")
    }
    self.dropout = dropout
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
            for j in 0..<layer.num_in {
                layer.weights[i*layer.num_in + j] = self.random.function(layer.num_in,
                    layer.num_out, context.random_generator)
            }
        }
    }
}

// reset the weights and biases
reset_no_config :: proc(self: ^Neural_Network) {
    for &layer in self.layers {
        for i in 0..<layer.num_out {
            for j in 0..<layer.num_in {
                layer.weights[i*layer.num_in + j] = self.random.function(layer.num_in,
                    layer.num_out, context.random_generator)
            }
        }
    }
}

deinit :: proc(self: ^Neural_Network) {
    vmem.arena_destroy(&self.arena)
    self^ = Neural_Network{}
}
