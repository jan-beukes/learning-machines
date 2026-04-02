package nn

import "core:fmt"
import "core:log"
import "core:os"

ValueID :: int

Value :: struct {
    data: f32,
    grad: f32,
    id: ValueID,
    children: [2]ValueID,
    op: rune,
}

@(private="file")
values_map: [dynamic]Value // ValueId -> Value

values_init :: proc() {
    // register custom formatter
    fmt.set_user_formatters(new(map[typeid]fmt.User_Formatter))
    fmt.register_user_formatter(Value, val_user_formatter)
}

values_deinit :: proc() {
    delete(values_map)
}

val :: proc(v: f32) -> Value {
    ret := Value{
        data = v,
        children = {-1, -1},
        id = len(values_map),
    }
    append(&values_map, ret)

    return ret
}

val_add :: proc(a, b: Value) -> Value {
    ret := Value{
        data = a.data + b.data,
        id = len(values_map),
        children = {a.id, b.id},
        op = '+'
    }
    append(&values_map, ret)

    return ret
}

val_mul :: proc(a, b: Value) -> Value {
    ret := Value{
        data = a.data * b.data,
        id = len(values_map),
        children = {a.id, b.id},
        op = '*'
    }
    append(&values_map, ret)

    return ret
}

val_sub :: proc(a, b: Value) -> Value {
    ret := Value{
        data = a.data - b.data,
        id = len(values_map),
        children = {a.id, b.id},
        op = '-'
    }
    append(&values_map, ret)

    return ret
}

// renders to graph.svg
val_draw_dot :: proc(val: Value) {
    Edge :: [2]Value
    edges: [dynamic]Edge
    defer delete(edges)
    for id in 0..<len(values_map) {
        for child_id in values_map[id].children {
            if child_id < 0 do break
            append(&edges, Edge{ values_map[child_id], values_map[id] })
        }
    }

    // create pipe
    r, w, err := os.pipe()
    if err != nil {
        log.error("Pipe")
        return
    }

    outfile: ^os.File
    outfile, err = os.create("graph.svg")
    if err != nil {
        log.error("Could not create file 'graph.svg'")
        return
    }

    p: os.Process
    p, err = os.process_start(os.Process_Desc{
        command = { "dot", "-Tsvg"},
        stdin = r,
        stdout = outfile,
        stderr = os.stderr,
    })

    if err != nil {
        log.error("Could exec dot:", err)
        return
    }

    fmt.fprintln(w, "digraph {")
    fmt.fprintln(w, "  rankdir=LR;")
    for id in 0..<len(values_map) {
        v := values_map[id]
        fmt.fprintfln(w, "  %v[label=\"{{ data: %f }}\",shape=\"record\"];", id, v.data)
        if v.op != 0 {
            op_id := len(values_map) + id + int(v.op)
            fmt.fprintfln(w, "  %v[label=\"%v\"];", op_id, v.op)
            fmt.fprintfln(w, "  %v -> %v;", op_id, id)
        }
    }
    for e in edges {
        op_id := len(values_map) + e.y.id + int(e.y.op)
        fmt.fprintfln(w, "  %v -> %v;", e.x.id, op_id)
    }

    fmt.fprintln(w, "}")
    os.close(w)

    state: os.Process_State
    state, err = os.process_wait(p)
    if err != nil {
        log.error("Waiting for 'dot'")
    }
}

val_user_formatter :: proc(fi: ^fmt.Info, arg: any, verb: rune) -> bool {
    val := arg.(Value)
    data := val.data
    switch verb {
    case 'v':
        fmt.fmt_string(fi, "Value(", verb)
        fmt.fmt_float(fi, f64(data), 8*size_of(data), verb)
        fmt.fmt_string(fi, ")", verb)
    case 'f', 'F', 'g', 'G', 'e', 'E':
        fmt.fmt_float(fi, f64(data), 8*size_of(data), verb)
    case:
        return false
    }
    return true
}
