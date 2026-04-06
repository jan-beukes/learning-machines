// Configuration for the neural network
// Optimizer, Cost function, Activation function
package nn

import "core:fmt"
import "core:math"
import "core:math/rand"
import "base:runtime"

DEFAULT_CONFIG :: Config{ .Cross_Entropy, .Sigmoid, .Softmax, .Gaussian }

Config :: struct {
    cost: Cost_Kind,
    activation: Activation_Kind,
    output_activation: Activation_Kind,
    random: Random_Kind,
}

Cost_Kind :: enum {
    Cross_Entropy,
    Mean_Squared_Error,
}

Activation_Kind :: enum {
    Sigmoid,
    Softmax,
    ReLu,
    Tanh,
}

Random_Kind :: enum {
    Gaussian,
    Standard_Normal,
}

Cost :: struct {
    kind: Cost_Kind,
    function: proc(ypred, y: []f32) -> f32 `cbor:"-"`,
    derivative: proc(ypred, y: f32) -> f32 `cbor:"-"`,
}

Activation :: struct {
    kind: Activation_Kind,
    function: proc(inputs: []f32, idx: int) -> f32 `cbor:"-"`,
    derivative: proc(inputs: []f32, idx: int) -> f32 `cbor:"-"`,
}

// Some random initializations use num_in and num_out
Random :: struct {
    kind: Random_Kind,
    function: proc(num_in, num_out: int, gen: runtime.Random_Generator) -> f32 `cbor:"-"`,
}

cost_from_kind :: proc(kind: Cost_Kind) -> Cost {
    cost: Cost
    switch kind {
    case .Mean_Squared_Error: cost = MEAN_SQUARED_ERROR
    case .Cross_Entropy: cost = CROSS_ENTROPY
    }
    return cost
}

activation_from_kind :: proc(kind: Activation_Kind) -> Activation {
    activation: Activation
    switch kind {
    case .Sigmoid: activation = SIGMOID
    case .Softmax: activation = SOFTMAX
    case .Tanh: activation = TANH
    case .ReLu: activation = RELU
    }
    return activation
}

random_from_kind :: proc(kind: Random_Kind) -> Random {
    random: Random
    switch kind {
    case .Standard_Normal: random = STANDARD_NORMAL
    case .Gaussian: random = GAUSSIAN
    }
    return random
}

//----Random Initializors---
STANDARD_NORMAL :: Random{
    kind = .Standard_Normal,
    function = proc(num_in, num_out: int, gen: runtime.Random_Generator) -> f32 {
        return rand.float32_normal(0, 1, gen)
    }
}

GAUSSIAN :: Random{
    kind = .Gaussian,
    function = proc(num_in, num_out: int, gen: runtime.Random_Generator) -> f32 {
        return rand.float32_normal(0, 1.0/math.sqrt(f32(num_in)), gen)
    }
}

//----Cost functions----
MEAN_SQUARED_ERROR :: Cost{
    kind = .Mean_Squared_Error,
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

CROSS_ENTROPY :: Cost{
    kind = .Cross_Entropy,
	// NOTE: expected outputs are expected to all be either 0 or 1
    function = proc(pred, expected: []f32) -> f32 {
        assert(len(pred) == len(expected))
        cost: f32
        for i in 0..<len(pred) {
            a := pred[i]
            y := expected[i]
            c := y == 1.0 ? -math.ln(a) : -math.ln(1.0 - a)
            cost += c
        }
        return cost
    },
    derivative = proc(pred, expected: f32) -> f32 {
        a := pred
        y := expected
        if a == 0 || a == 1 {
            return 0
        }
        return (-a + y) / (a * (a - 1.0))
    }

}

//----Activations----
TANH :: Activation{
    kind = .Tanh,
    function = proc(inputs: []f32, idx: int) -> f32 {
        return math.tanh(inputs[idx])
    },
    derivative = proc(inputs: []f32, idx: int) -> f32 {
        cosh := math.cosh(inputs[idx])
        return 1.0 / (cosh * cosh)
    }
}

RELU :: Activation{
    kind = .ReLu,
    function = proc(inputs: []f32, idx: int) -> f32 {
        return max(0, inputs[idx])
    },
    derivative = proc(inputs: []f32, idx: int) -> f32 {
        return inputs[idx] > 0.0 ? 1.0 : 0.0
    }
}

SIGMOID :: Activation{
    kind = .Sigmoid,
    function = proc(inputs: []f32, idx: int) -> f32 {
        return 1.0 / (1.0 + math.exp(-inputs[idx]))

    },
    derivative = proc(inputs: []f32, idx: int) -> f32 {
        sig := 1.0 / (1.0 + math.exp(-inputs[idx]))
        return sig * (1.0 - sig)
    }
}

SOFTMAX :: Activation{
    kind = .Softmax,
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
        result := (x*sum - x*x) / (sum*sum)
        return result
    }
}
