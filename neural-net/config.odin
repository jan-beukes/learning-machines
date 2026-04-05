// Configuration for the neural network
// Optimizer, Cost function, Activation function
package nn

import "core:math"

DEFAULT_CONFIG :: Config{ .SGD, MEAN_SQUARED_ERROR, SIGMOID }

Optmizer :: enum {
    SGD,
}

Cost :: struct {
    function: proc(ypred, y: []f32) -> f32,
    derivative: proc(ypred, y: f32) -> f32,
}

Activation :: struct {
    function: proc(x: f32) -> f32,
    derivative: proc(x: f32) -> f32,
}

Config :: struct {
    optimizer: Optmizer,
    cost: Cost,
    activation: Activation,
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

// Leaky ReLu
RELU :: Activation{
    function = proc(x: f32) -> f32 {
        return max(0.1*x, x)
    },
    derivative = proc(x: f32) -> f32 {
        return x > 0.0 ? 1.0 : 0.01
    }
}

SIGMOID :: Activation{
    function = proc(x: f32) -> f32 {
        return 1.0 / (1.0 + math.exp(-x))

    },
    derivative = proc(x: f32) -> f32 {
        sig := 1.0 / (1.0 + math.exp(-x))
        return sig * (1.0 - sig)
    }
}
