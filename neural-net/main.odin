package nn

import "core:fmt"
import "core:os"
import "core:math/rand"
import "core:log"
import "core:slice"
import "core:strings"
import "core:strconv"

main :: proc() {
    mnist_train, mnist_test := load_fashion_mnist("fashion-mnist")
    defer { data_set_destroy(mnist_train); data_set_destroy(mnist_test) }

    model: Neural_Network
    init(&model, {mnist_train.input_size, 50, mnist_train.output_size})

    fmt.println("Training Network...")
    // train the lil boy
    mini_batch_size := 20
    // learn rate is divided by batch size since we only learn once per batch
    learn_rate: f32 = 0.005 * f32(mini_batch_size)
    epochs := 10
    for epoch in 0..<epochs {
        cost: f32
        for i := 0; i < len(mnist_train.data); i += mini_batch_size {
            batch := mnist_train.data[i:i+mini_batch_size]
            cost = learn(model, batch, learn_rate, os.get_processor_core_count())
        }
        fmt.printfln("Epoch(%v) Cost = %v", epoch, cost)
    }
    fmt.println("Accuracy on test set:", evaluate(model, mnist_test.data, os.get_processor_core_count()))
}
