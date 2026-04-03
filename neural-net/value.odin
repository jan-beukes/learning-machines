package nn

import "core:fmt"
import "core:log"
import mem "core:mem"
import vmem "core:mem/virtual"
import "core:os"

ValueID :: int
void :: struct{}

Value :: struct {
    data: f32,
    grad: f32,
    children: [2]^Value,
    label: string,
    op: rune,
}

@(private="file") values_arena: vmem.Arena
@(private="file") allocator: mem.Allocator

values_init :: proc() {
    // register custom formatter
    fmt.set_user_formatters(new(map[typeid]fmt.User_Formatter))
    fmt.register_user_formatter(^Value, val_user_formatter)
    fmt.register_user_formatter(Value, val_user_formatter)
    allocator = vmem.arena_allocator(&values_arena)
}

values_deinit :: proc() {
    vmem.arena_destroy(&values_arena)
}

val :: proc(v: f32, label := "") -> ^Value {
    ret := new(Value, allocator)
    ret^ = Value{
        data = v,
        label = label,
    }
    return ret
}

val_add :: proc(a, b: ^Value) -> ^Value {
    ret := new(Value, allocator)
    ret^ = Value{
        data = a.data + b.data,
        children = {a, b},
        op = '+'
    }
    return ret
}

val_mul :: proc(a, b: ^Value) -> ^Value {
    ret := new(Value, allocator)
    ret^ = Value{
        data = a.data * b.data,
        children = {a, b},
        op = '*'
    }
    return ret
}

val_sub :: proc(a, b: ^Value) -> ^Value {
    ret := new(Value, allocator)
    ret^ = Value{
        data = a.data - b.data,
        children = {a, b},
        op = '-'
    }
    return ret
}

// renders to graph.svg
val_draw_dot :: proc(val: ^Value) {
    Edge :: [2]^Value
    edges: [dynamic]Edge
    nodes: map[^Value]void
    defer { delete(edges); delete(nodes) }
    build :: proc(v: ^Value, nodes: ^map[^Value]void, edges: ^[dynamic]Edge) {
        if v in nodes {
            return
        }
        nodes[v] = {}
        for child in v.children {
            if child == nil do break
            append(edges, Edge{child, v})
            build(child, nodes, edges)
        }
    }
    build(val, &nodes, &edges)

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
    for v in nodes {
        fmt.fprintfln(w, "  %d[label=\"{{ %s | data: %f }}\",shape=\"record\"];", uintptr(v), v.label, v.data)
        if v.op != 0 {
            op_id := uintptr(v) + uintptr(v.op)
            fmt.fprintfln(w, "  %d[label=\"%v\"];", op_id, v.op)
            fmt.fprintfln(w, "  %d -> %d;", op_id, uintptr(v))
        }
    }
    for e in edges {
        op_id := uintptr(e.y) + uintptr(e.y.op)
        fmt.fprintfln(w, "  %d -> %d;", uintptr(e.x), op_id)
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
    data: f32
    switch v in arg {
    case Value:
        data = arg.(Value).data
    case ^Value:
        data = arg.(^Value).data
    case:
        unreachable()
    }
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
