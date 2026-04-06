package nn

import "core:fmt"
import "core:os"
import "core:math/rand"
import "core:log"
import "core:slice"
import "core:strings"
import "core:strconv"

load_iris :: proc(path: string) -> ([]Data_Point, []string) {
    content, err := os.read_entire_file(path, allocator=context.temp_allocator)
    s: string = string(content)
    inputs: [dynamic][]f32
    labels: [dynamic]int
    names: [dynamic]string
    for line in strings.split_lines_iterator(&s) {
        items := strings.split(line, ",", allocator = context.temp_allocator)
        row := make([]f32, 4)
        for i in 0..<len(row) {
            ok: bool
            row[i], ok = strconv.parse_f32(items[i])
            if !ok {
                fmt.eprintln("Could not parse iris line:", line)
                os.exit(1)
            }
        }
        append(&inputs, row)
        name := items[4]
        idx, found := slice.linear_search(names[:], name)
        if found {
            append(&labels, idx)
        } else {
            append(&labels, len(names))
            append(&names, strings.clone(name))
        }
    }

    free_all(context.temp_allocator)
    return batch_create(inputs[:], labels[:], len(names)), names[:]
}

main :: proc() {
    context.logger = log.create_console_logger(opt=log.Options{.Level, .Terminal_Color})

    iris, names := load_iris("iris/iris.data")
    defer batch_destroy(iris)

    input_size, output_size := len(iris[0].input), len(iris[0].expected)

    model, err := load_from_file("model.cbor")
    defer deinit(&model)
    if err != nil {
        fmt.printfln("Training new model...")
        init(&model, {input_size, 8, 8, output_size})
        epochs := 500
        learn_rate := 0.008 * f32(len(iris))
        for i in 0..<epochs {
            cost := learn(model, iris, learn_rate)
            if i % 80 == 0 {
                fmt.printfln("Epoch(%v): Cost = %v", i, cost)
            }
        }
        err := save_to_file(model, "model.cbor")
        if err != nil {
            fmt.eprintln("Could not save model:", err)
        }
    }
    fmt.println("Accuracy:", evaluate(model, iris))
}
