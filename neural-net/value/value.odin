package value

import "core:fmt"
import "core:log"
import "core:math"
import mem "core:mem"
import vmem "core:mem/virtual"
import "core:os"

ValueID :: int
void :: struct{}

Value :: struct {
    data: f32,
    grad: f32,
 // NOTE: instead of having a closure (We don't get that for free in odin) 
 // that can compute the gradient during back prop we store the local gradient    local_grad: f32,
 // to then be multiplied with                                                                    
    local_grad: f32,
    children: [2]^Value,
    op: string,
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

val :: proc{
    val_new,
    val_new_op,
}

val_new :: proc(v: f32) -> ^Value {
    out := new(Value, allocator)
    out^ = Value{
        data = v,
    }
    return out
}

val_new_op :: proc(v: f32, children: [2]^Value, op: string) -> ^Value {
    out := new(Value, allocator)
    out^ = Value{
        data = v,
        children = children,
        op = op,
    }
    return out
}

val_add :: proc(a, b: ^Value) -> ^Value {
    out := val(a.data + b.data, {a, b}, "+")
    a.local_grad += 1.0
    b.local_grad += 1.0
    return out
}

val_mul :: proc(a, b: ^Value) -> ^Value {
    out := val(a.data * b.data, {a, b}, "*")
    a.local_grad += b.data
    b.local_grad += a.data
    return out
}

val_sub :: proc(a, b: ^Value) -> ^Value {
    out := val(a.data - b.data, {a, b}, "-")
    a.local_grad += 1.0
    b.local_grad += -1.0
    return out
}

val_pow :: proc(a: ^Value, b: f32) -> ^Value {
    op := fmt.aprintf("^(%v)", b, allocator=allocator)
    out := val(math.pow(a.data, b), [2]^Value{a, nil}, op)
    a.local_grad += b * math.pow(a.data, b - 1)
    return out
}

val_relu :: proc(v: ^Value) -> ^Value {
    out := val(v.data < 0 ? 0 : v.data, {v, nil}, "ReLu")
    v.local_grad += out.data > 0 ? 1.0 : 0.0
    return out
}

// back propogate throught the graph, applying chain rule by mutiplying local derivatives
val_backwards :: proc(v: ^Value) {
    q: [dynamic]^Value

    v.grad = 1.0
    append(&q, v)
    for len(q) > 0 {
        node := pop(&q)
        grad := node.grad
        a, b := node.children.x, node.children.y
        if a != nil {
            a.grad = grad * a.local_grad
            append(&q, a)
        }
        if b != nil {
            b.grad = grad * b.local_grad
            append(&q, b)
        }
    }
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

    get_op_id :: proc(op: string) -> uintptr {
        sum: uintptr
        for r in op {
            sum += uintptr(r)
        }
        return sum
    }

    fmt.fprintln(w, "digraph {")
    fmt.fprintln(w, "  rankdir=LR;")
    for v in nodes {
        fmt.fprintfln(w, "  %d[label=\"{{ data: %f | grad: %f }}\",shape=\"record\"];",
            uintptr(v), v.data, v.grad)
        if v.op != "" {
            op_id := uintptr(v) + get_op_id(v.op)
            fmt.fprintfln(w, "  %d[label=\"%v\"];", op_id, v.op)
            fmt.fprintfln(w, "  %d -> %d;", op_id, uintptr(v))
        }
    }
    for e in edges {
        op_id := uintptr(e.y) + get_op_id(e.y.op)
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
