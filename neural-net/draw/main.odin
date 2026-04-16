package draw

import "core:fmt"
import "core:os"
import "core:slice"
import rl "vendor:raylib"
import "vendor:raylib/rlgl"

import nn ".."

RES :: nn.MNIST_RES
Prediction :: struct {
    label: int,
    confidence: f32,
}

// TODO: randomly augment the images and train that mf
train_mnist :: proc(model: ^nn.Neural_Network) {
    train_set, test_set := nn.load_mnist("digits-mnist")
    defer {
        nn.batch_destroy(train_set.data)
        nn.batch_destroy(test_set.data)
    }

    // params
    config := nn.Config{ .Cross_Entropy, .Sigmoid, .Softmax, .Gaussian }
    layers := []int{train_set.input_size, 100, train_set.output_size}
    nn.init(model, layers, dropout = 0.3, config = config)
    num_threads := os.get_processor_core_count()
    train_split: f32 = 0.90
    mini_batch_size := 60
    learn_rate: f32 = 0.02
    regularization: f32 = 0.01

    // decay rates for moving averages
    beta1: f32 = 0.9
    beta2: f32 = 0.99

    split_idx := int(train_split*f32(len(train_set.data)))
    train := train_set.data[:split_idx]
    validation := train_set.data[split_idx:]

    for epoch in 0..<100 {
        for i := 0; i + mini_batch_size < len(train); i += mini_batch_size {
            end_idx := min(len(train), i+mini_batch_size)
            batch := train[i:end_idx]
            nn.learn(model, batch, learn_rate, regularization, beta1, beta2, num_threads)
        }
        eval_validation := nn.evaluate(model^, validation, num_threads)
        eval_train := nn.evaluate(model^, train[:len(validation)], num_threads)
        fmt.printfln("Epoch(%v) Validation = %v | Training = %v", epoch+1, eval_validation, eval_train)
    }

    nn.save_to_file(model^, "Digits.cbor")
}

create_render_texture :: proc(width, height: i32) -> rl.RenderTexture {
    target: rl.RenderTexture
    target.id = rlgl.LoadFramebuffer()
    assert(target.id > 0)

    rlgl.EnableFramebuffer(target.id)
    target.texture.id = rlgl.LoadTexture(nil, width, height, i32(rl.PixelFormat.UNCOMPRESSED_R32), 1)
    target.texture.width = width
    target.texture.height = height
    target.texture.format = .UNCOMPRESSED_R32
    target.texture.mipmaps = 1

    attach_type := i32(rlgl.FramebufferAttachType.COLOR_CHANNEL0)
    attach_texture_type := i32(rlgl.FramebufferAttachTextureType.TEXTURE2D)
    rlgl.FramebufferAttach(target.id, target.texture.id, attach_type, attach_texture_type, 0)

    if rlgl.FramebufferComplete(target.id) {
        fmt.println("Float framebuffer created!")
    }

    return target
}

// TODO: fragment shader for input_canvas where each pixel finds it's corresponding box of pixels in
// the original canvas and set's its value to the average

draw_to_canvas :: proc(canvas: rl.RenderTexture, area: rl.Rectangle, brush_size: f32) -> bool {
    was_input := false
    mouse_pos := rl.GetMousePosition()
    rl.BeginTextureMode(canvas)
    if rl.IsKeyPressed(.C) {
        rl.ClearBackground(rl.BLACK)
    }
    if rl.CheckCollisionPointRec(mouse_pos, area) {
        if rl.IsMouseButtonDown(.LEFT) || rl.IsMouseButtonDown(.RIGHT) {
            color := rl.IsMouseButtonDown(.LEFT) ? rl.WHITE : rl.BLACK
            delta := rl.GetMouseDelta()
            start := mouse_pos - delta
            rl.DrawLineEx(start, mouse_pos, 2*brush_size, color)
            rl.DrawCircleV(start, brush_size, color)
            rl.DrawCircleV(mouse_pos, brush_size, color)
            was_input = true
        }
    }
    rl.EndTextureMode()
    return was_input
}

main :: proc() {
    model, err := nn.load_from_file("Digits.cbor")
    if err != nil {
        fmt.println("Training Network")
        train_mnist(&model)
    }

    canvas_size: i32 = 840
    rl.InitWindow(1000, canvas_size, "Neural Network")
    rl.ConfigFlag(.MSAA_4X_HINT)

    font_size: f32 = 50
    font := rl.LoadFontEx("iosevka.ttf", i32(font_size), nil, 0)

    draw_area := rl.Rectangle{ 0, 0, f32(canvas_size), f32(canvas_size) }
    canvas := rl.LoadRenderTexture(canvas_size, canvas_size)
    input_canvas := create_render_texture(RES, RES)
    rl.SetTextureFilter(canvas.texture, .BILINEAR)

    // how often do we classify
    update_ticks := 60
    tick_delta := 1.0 / f64(update_ticks)
    last_tick: f64
    should_classify := true

    brush_size: f32 = 30
    predictions: [10]Prediction
    for i in 0..=9 {
        predictions[i].label = i
    }
    for !rl.WindowShouldClose() {
        current_time := rl.GetTime()
        time_since_tick := current_time - last_tick
        if time_since_tick > tick_delta {
            last_tick = current_time
            texture := input_canvas.texture
            pixels := rlgl.ReadTexturePixels(texture.id, texture.width, texture.height, i32(texture.format))
            defer rl.MemFree(pixels)
            length := texture.width*texture.height
            input := ([^]f32)(pixels)[:length]
            output := nn.forward(model, input)
            for &p in predictions {
                p.confidence = output[p.label]
            }
            slice.sort_by(predictions[:], proc(l, r: Prediction) -> bool {
                return r.confidence < l.confidence
            })
        }

        should_classify = draw_to_canvas(canvas, draw_area, brush_size)

        // Downscale canvas to input texture
        rl.BeginTextureMode(input_canvas)
        src := rl.Rectangle{ 0, 0, f32(canvas.texture.width), f32(canvas.texture.height) }
        dst := rl.Rectangle{ 0, 0, f32(input_canvas.texture.width), f32(input_canvas.texture.height) }
        rl.DrawTexturePro(canvas.texture, src, dst, {}, 0, rl.WHITE)
        rl.EndTextureMode()

        // Draw to screen
        rl.BeginDrawing()
        rl.ClearBackground(rl.BLACK)
        src = rl.Rectangle{ 0, 0, f32(canvas.texture.width), -f32(canvas.texture.height) }
        rl.DrawTexturePro(canvas.texture, src, draw_area, {}, 0, rl.WHITE)
        rl.DrawRectangleLinesEx(draw_area, 10, rl.GRAY)

        // Draw predictions
        step := font_size + 10
        cursor := step
        for p, i in predictions {
            text := rl.TextFormat("%d (%.1f)", p.label, p.confidence)
            color := i == 0 ? rl.GREEN : rl.GRAY
            rl.DrawTextEx(font, text, {f32(canvas_size) + 10, cursor}, font_size, 0, color)
            cursor += font_size + 10
        }

        rl.EndDrawing()
        free_all(context.temp_allocator)
    }
}

