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

INPUT_CANVAS_SHADER: cstring :
`
#version 330 
in vec2 fragTexCoord;
out vec4 outColor;

#define RES 28

uniform sampler2D texture0;

void main() {
    ivec2 size = textureSize(texture0, 0);
    vec2 pos = fragTexCoord;
    float boxSize = RES / float(size.x);
    float pixelSize = boxSize / RES;
    float sum = 0.0;
    int count = 0;
    for (float y = pos.y - boxSize/2.0; y < pos.y + boxSize/2; y += pixelSize) {
        for (float x = pos.x - boxSize/2.0; x < pos.x + boxSize/2.0; x += pixelSize) {
            if (x < 0.0 || x >= RES || y < 0.0 || y >= RES) continue;
            vec2 pixel = vec2(x, y);
            vec4 color = texture(texture0, pixel);
            float avg = (color.r + color.g + color.b) / 3.0;
            sum += avg;
            count++;
        }
    }
    float v = sum/count;
    outColor = vec4(v, v, v, 1.0);
}
`


draw_to_canvas :: proc(canvas: rl.RenderTexture, area: rl.Rectangle, brush_size: f32) -> bool {
    was_input := false
    mouse_pos := rl.GetMousePosition()
    rl.BeginTextureMode(canvas)
    if rl.IsKeyPressed(.C) {
        rl.ClearBackground(rl.BLACK)
    }
    if rl.CheckCollisionPointRec(mouse_pos, area) {
        if rl.IsMouseButtonDown(.LEFT) || rl.IsMouseButtonDown(.RIGHT) {
            delta := rl.GetMouseDelta()
            // This is stupid but mouse pos is not updated until the mouse moves on wayland or at
            // least clicks on X11
            @(static) first_input := true
            if first_input && rl.Vector2Length(delta) > 0.0 {
                first_input = false
                delta = rl.Vector2(0)
            }

            color := rl.IsMouseButtonDown(.LEFT) ? rl.WHITE : rl.BLACK
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
    rl.SetWindowMonitor(0)
    rl.ConfigFlag(.MSAA_4X_HINT)

    font_size: f32 = 50
    font := rl.LoadFontEx("iosevka.ttf", i32(font_size), nil, 0)

    input_canvas_shader := rl.LoadShaderFromMemory(nil, INPUT_CANVAS_SHADER)

    draw_area := rl.Rectangle{ 0, 0, f32(canvas_size), f32(canvas_size) }
    canvas := rl.LoadRenderTexture(canvas_size, canvas_size)
    input_canvas := create_render_texture(RES, RES)
    draw_input_canvas := false

    // how often do we classify
    update_ticks := 60
    tick_delta := 1.0 / f64(update_ticks)
    last_tick: f64
    should_classify := true

    brush_size: f32 = 26
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

        if rl.IsKeyPressed(.SPACE) {
            draw_input_canvas = !draw_input_canvas
        }

        should_classify = draw_to_canvas(canvas, draw_area, brush_size) || should_classify

        // Downscale canvas to input texture
        rl.BeginTextureMode(input_canvas)
        rl.BeginShaderMode(input_canvas_shader)
        src := rl.Rectangle{ 0, 0, f32(canvas.texture.width), f32(canvas.texture.height) }
        dst := rl.Rectangle{ 0, 0, f32(input_canvas.texture.width), f32(input_canvas.texture.height) }
        rl.DrawTexturePro(canvas.texture, src, dst, {}, 0, rl.WHITE)
        rl.EndShaderMode()
        rl.EndTextureMode()

        // Draw to screen
        rl.BeginDrawing()
        rl.ClearBackground(rl.BLACK)
        if draw_input_canvas {
            src = rl.Rectangle{ 0, 0, f32(input_canvas.texture.width), f32(input_canvas.texture.height) }
            rl.DrawTexturePro(input_canvas.texture, src, draw_area, {}, 0, rl.WHITE)
        } else {
            src = rl.Rectangle{ 0, 0, f32(canvas.texture.width), -f32(canvas.texture.height) }
            rl.DrawTexturePro(canvas.texture, src, draw_area, {}, 0, rl.WHITE)
        }
        rl.DrawRectangleLinesEx(draw_area, 10, rl.GRAY)

        // Draw predictions
        step := font_size + 10
        cursor := step
        for p, i in predictions {
            text := rl.TextFormat("%d %2.0f%%", p.label, 100*p.confidence)
            color := i == 0 ? rl.GREEN : rl.GRAY
            rl.DrawTextEx(font, text, {f32(canvas_size) + 10, cursor}, font_size, 0, color)
            cursor += font_size + 10
        }

        rl.EndDrawing()
        free_all(context.temp_allocator)
    }
}

