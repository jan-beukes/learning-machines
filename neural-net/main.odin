package nn

import "core:fmt"
import "core:log"

main :: proc() {
    context.logger = log.create_console_logger(opt=log.Options{.Level, .Terminal_Color})

    values_init()
    defer values_deinit()

    a := val(2.0)
    b := val(-3.0)
    c := val(10.0)

    d := val_add(val_mul(a, b), c)

    fmt.println(d)
    val_draw_dot(d)
}
