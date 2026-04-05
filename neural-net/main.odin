package nn

import "core:fmt"
import "core:os"
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

    input_size, output_size := len(iris[0].input), len(iris[0].expected)

    model: Neural_Network
    init(&model, {input_size, 6, output_size})
    defer deinit(&model)

    epochs := 8000
    for i in 0..<epochs {
        cost := learn(model, iris, 0.2)
        fmt.printfln("Epoch(%v): Cost = %v", i, cost)
    }

    fmt.println("Accuracy:", evaluate(model, iris))
}
