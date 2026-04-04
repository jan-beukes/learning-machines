package nn

import "core:fmt"
import "core:log"

main :: proc() {
    context.logger = log.create_console_logger(opt=log.Options{.Level, .Terminal_Color})
    values_init()
    defer values_deinit()

    a := val(3.0)
    b := val_add(a, a)
    c := val_pow(b, 2)
    val_backwards(c)
    val_draw_dot(c)
}
