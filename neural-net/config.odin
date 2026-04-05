// Configuration for the neural network
// Optimizer, Cost function, Activation function
package nn

import "core:math"

DEFAULT_CONFIG :: Config{ MEAN_SQUARED_ERROR, SIGMOID, SOFTMAX }

Cost :: struct {
    function: proc(ypred, y: []f32) -> f32,
    derivative: proc(ypred, y: f32) -> f32,
}

Activation :: struct {
    function: proc(inputs: []f32, idx: int) -> f32,
    derivative: proc(inputs: []f32, idx: int) -> f32,
}

Config :: struct {
    cost: Cost,
    activation: Activation,
    output_activation: Activation,
}

// Cost functions

MEAN_SQUARED_ERROR :: Cost{
    function = proc(ypred, y: []f32) -> f32 {
        assert(len(y) == len(ypred))
        cost: f32
        for i in 0..<len(y) {
            diff := ypred[i] - y[i]
            cost += diff*diff
        }
        return 0.5*cost
    },
    derivative = proc(ypred, y: f32) -> f32 {
        return ypred - y
    }
}

// Activations

TANH :: Activation{
    function = proc(inputs: []f32, idx: int) -> f32 {
        return math.tanh(inputs[idx])
    },
    derivative = proc(inputs: []f32, idx: int) -> f32 {
        cosh := math.cosh(inputs[idx])
        return 1.0 / (cosh * cosh)
    }
}

RELU :: Activation{
    function = proc(inputs: []f32, idx: int) -> f32 {
        return max(0, inputs[idx])
    },
    derivative = proc(inputs: []f32, idx: int) -> f32 {
        return inputs[idx] > 0.0 ? 1.0 : 0.0
    }
}

SIGMOID :: Activation{
    function = proc(inputs: []f32, idx: int) -> f32 {
        return 1.0 / (1.0 + math.exp(-inputs[idx]))

    },
    derivative = proc(inputs: []f32, idx: int) -> f32 {
        sig := 1.0 / (1.0 + math.exp(-inputs[idx]))
        return sig * (1.0 - sig)
    }
}

SOFTMAX :: Activation{
    function = proc(inputs: []f32, idx: int) -> f32 {
        sum: f32 = 0
        for input in inputs {
            sum += math.exp(input)
        }
        return math.exp(inputs[idx]) / sum
    },
    derivative = proc(inputs: []f32, idx: int) -> f32 {
        sum: f32 = 0
        for input in inputs {
            sum += math.exp(input)
        }
        x := math.exp(inputs[idx])
        return (x*sum - x*x) / (sum*sum)
    }
}
