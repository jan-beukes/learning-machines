package nn

import "core:fmt"
import "core:os"
import "core:log"
import "core:slice"
import "core:strings"
import "core:strconv"

Data_Set :: struct {
    inputs: [][]f32,
    labels: []int,
    num_labels: int,
    label_values: []string,
}

load_iris :: proc(path: string) -> Data_Set {
    content, err := os.read_entire_file(path, allocator=context.temp_allocator)
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
    return Data_Set{ inputs[:], outputs[:], len(labels), labels[:] }
}

main :: proc() {
    context.logger = log.create_console_logger(opt=log.Options{.Level, .Terminal_Color})

    iris := load_iris("iris/iris.data")
    model: Neural_Network

    input_size, output_size := len(iris.inputs[0]), iris.num_labels
    init(&model, {input_size, 6, output_size})

    batch := batch_create(iris.inputs, iris.labels, iris.num_labels)
    learn(model, batch, 0.01)
}
