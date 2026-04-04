package nn

import "core:fmt"
import "core:os"
import "core:log"
import "core:slice"
import "core:strings"
import "core:strconv"

DataSet :: struct {
    inputs: [][]f32,
    outputs: []int,
    labels: []string,
}

load_iris :: proc(path: string) -> DataSet {
    content, err := os.read_entire_file(path, allocator=context.allocator)
    defer delete(content)
    s: string = string(content)
    inputs: [dynamic][]f32
    outputs: [dynamic]int
    labels: [dynamic]string
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
        label := items[4]
        idx, found := slice.linear_search(labels[:], label)
        if found {
            append(&outputs, idx)
        } else {
            append(&outputs, len(labels))
            append(&labels, strings.clone(label))
        }
    }

    free_all(context.temp_allocator)
    return DataSet{ inputs[:], outputs[:], labels[:] }
}

main :: proc() {
    context.logger = log.create_console_logger(opt=log.Options{.Level, .Terminal_Color})

    iris := load_iris("iris/iris.data")
    model: Neural_Network

    input_size, output_size := len(iris.inputs[0]), len(iris.labels)
    init(&model, {input_size, 6, output_size})

    output: [3]f32
    forward(model, iris.inputs[0], output[:])
}
