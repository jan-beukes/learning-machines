package value

import "core:fmt"

main :: proc() {
    values_init()
    defer values_deinit()

    a := val(3.0)
    b := val(2.0)
    c := val_add(a, b)
    d := val_pow(c, 2.0)

    out := val_mul(d, val(2.0))
    fmt.printfln("Result: %v", out)

    val_backwards(out)
    val_draw_dot(out)
}
