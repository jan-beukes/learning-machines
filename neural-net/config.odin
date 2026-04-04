package nn

DEFAULT_CONFIG :: Config{ .SGD, MEAN_SQUARED_ERROR, RELU }

// Return the value as well as the derivative

Optmizer :: enum {
    SGD,
}

Loss :: struct {
    function: #type proc(y, ypred: []f32) -> f32,
    derivative: #type proc(y, ypred: []f32) -> f32,
}

Activation :: struct {
    function: proc(x: f32) -> f32,
    derivative: proc(x: f32) -> f32,
}

Config :: struct {
    optimizer: Optmizer,
    loss: Loss,
    activation: Activation,
}

// Loss functions

MEAN_SQUARED_ERROR :: Loss{
    function = proc(y, ypred: []f32) -> f32 {
        assert(len(y) == len(ypred))
        loss: f32
        for i in 0..<len(y) {
            diff := y[i] - ypred[i]
            loss += diff*diff
        }
        return loss
    },
    derivative = proc(y, ypred: []f32) -> f32 {
        assert(len(y) == len(ypred))
        d: f32
        for i in 0..<len(y) {
            d += -2*(y[i] - ypred[i])
        }
        return d
    }
}

RELU :: Activation{
    function = proc(x: f32) -> f32 {
        return x > 0.0 ? x : 0.0
    },
    derivative = proc(x: f32) -> f32 {
        return x > 0.0 ? 1.0 : 0.0
    }
}
