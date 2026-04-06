package nn

import "core:fmt"
import "core:os"
import "core:math/rand"
import "core:log"
import "core:slice"
import "core:strings"
import "core:strconv"

main :: proc() {
    mnist_train, mnist_test := load_mnist("./mnist")
    defer {
        batch_destroy(mnist_train.data)
        batch_destroy(mnist_test.data)
    }

    model, err := load_from_file("./mnist.cbor")
    if err != nil {
        config := Config{ .Cross_Entropy, .Sigmoid, .Softmax, .Gaussian }
        init(&model, {mnist_train.input_size, 30, mnist_train.output_size}, config)

        fmt.println("Training Network...")
        train_set := mnist_train.data[:50_000]
        validation_set := mnist_train.data[50_000:]

        mini_batch_size := 10
        learn_rate: f32 = 0.5
        epochs := 20
        for epoch in 0..<epochs {
            cost: f32
            for i := 0; i < len(train_set); i += mini_batch_size {
                batch := train_set[i:i+mini_batch_size]
                cost = learn(model, batch, learn_rate, os.get_processor_core_count())
            }
            eval := evaluate(model, validation_set, os.get_processor_core_count())
            fmt.printfln("Epoch(%v) Accuracy = %v", epoch + 1, eval)
        }
        //save_to_file(model, "./mnist.cbor")
    }
    fmt.println("Testing...")
    fmt.println("Accuracy on test set:", evaluate(model, mnist_test.data, os.get_processor_core_count()))
}
