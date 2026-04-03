package nn

import "core:fmt"
import "core:log"

main :: proc() {
    context.logger = log.create_console_logger(opt=log.Options{.Level, .Terminal_Color})

    values_init()
    defer values_deinit()

    a := val(2.0, label="a")
    b := val(-3.0, label="b")
    c := val(10.0, label="c")
    e := val_mul(a, b); e.label = "e"
    d := val_add(e, c); d.label = "d"
    f := val(-2.0, label="f")

    l := val_mul(d, f); l.label="L"

    fmt.println(l)
    val_draw_dot(l)
}
