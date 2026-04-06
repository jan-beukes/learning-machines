package nn

import "core:fmt"
import "core:os"
import "core:math"
import "core:math/rand"
import "core:log"
import "core:slice"
import "core:strings"
import "core:strconv"

import rl "vendor:raylib"

Data_Set_Kind :: enum {
    Digits,
    Fashion,
}

update_input_image_texture :: proc(texture: ^rl.Texture, input: []f32) {
    grayscale := make([]u8, len(input))
    defer delete(grayscale)
    for &pixel, i in grayscale {
        pixel = u8(input[i] * 255.0)
    }
    data := raw_data(grayscale)
    if !rl.IsTextureValid(texture^) {
        image := rl.Image{
            data = data,
            width = MNIST_RES,
            height = MNIST_RES,
            mipmaps = 1,
            format = .UNCOMPRESSED_GRAYSCALE,
        }
        texture^ = rl.LoadTextureFromImage(image)
    } else {
        rl.UpdateTexture(texture^, data)
    }
}

predict :: proc(model: Neural_Network, data_point: Data_Point) -> (i32, f32) {
    output := forward(model, data_point.input, context.temp_allocator)
    prediction: i32 = i32(slice.max_index(output))
    return prediction, output[prediction]
}

run_viewer :: proc(model: Neural_Network, mnist: Data_Set, kind: Data_Set_Kind) {
    resx, resy: i32 = 800, 900
    rl.InitWindow(resx, resy, "Neural Network")
    defer rl.EndDrawing()

    font_size: f32 = 30
    font := rl.LoadFontEx("./iosevka.ttf", i32(font_size), nil, 0)

    image_idx := 0
    texture: rl.Texture
    update_input_image_texture(&texture, mnist.data[image_idx].input)
    prediction, confidence := predict(model, mnist.data[image_idx])
    for !rl.WindowShouldClose() {

        if rl.IsKeyPressed(.RIGHT) {
            image_idx = (image_idx + 1) % len(mnist.data)
            prediction, confidence = predict(model, mnist.data[image_idx])
            update_input_image_texture(&texture, mnist.data[image_idx].input)
        } else if rl.IsKeyPressed(.LEFT) {
            image_idx = math.floor_mod(image_idx - 1, len(mnist.data))
            prediction, confidence = predict(model, mnist.data[image_idx])
            update_input_image_texture(&texture, mnist.data[image_idx].input)
        } else if rl.IsKeyPressed(.SPACE) {
            // find next incorrect prediction
            for i := image_idx + 1; i != image_idx; i = (i + 1) % len(mnist.data) {
                data_point := mnist.data[i]
                prediction, confidence = predict(model, data_point)
                if prediction != data_point.label {
                    image_idx = i
                    break
                }
            }
            update_input_image_texture(&texture, mnist.data[image_idx].input)
        }

        rl.BeginDrawing()
        rl.ClearBackground({ 0x20, 0x20, 0x20, 0xff })
        src := rl.Rectangle{ 0, 0, f32(texture.width), f32(texture.height) }
        dst := rl.Rectangle{ 0, 0, f32(resx), f32(resx) }
        rl.DrawTexturePro(texture, src, dst, {}, 0, rl.WHITE)

        // UI

        label := mnist.data[image_idx].label

        pad: f32 = 10
        text: cstring
        if mnist.classes != nil {
            text = rl.TextFormat("Prediction: %v (%.2f%%)", mnist.classes[prediction], 100*confidence)
        } else {
            text = rl.TextFormat("Prediction: %v (%.2f%%)", prediction, 100*confidence)
        }
        text_width := rl.MeasureTextEx(font, text, font_size, 0).x
        text_pos := rl.Vector2{ f32(resx) - pad - text_width, f32(resy) - pad - font_size }
        rl.DrawTextEx(font, text, text_pos, font_size, 0, rl.RAYWHITE)

        if mnist.classes != nil {
            text = rl.TextFormat("Label: %v", mnist.classes[label])
        } else {
            text = rl.TextFormat("Label: %v", label)
        }
        rl.DrawTextEx(font, text, { pad, f32(resy) - pad - font_size }, font_size, 0, rl.RAYWHITE)

        color: rl.Color
        if prediction == label {
            text = "Correct"
            color = rl.GREEN
        } else {
            text = "Incorrect"
            color = rl.RED
        }
        text_width = rl.MeasureTextEx(font, text, font_size, 0).x
        rl.DrawTextEx(font, text, { 0.5*(f32(resx) - text_width), 10 }, font_size, 0, color)

        rl.EndDrawing()
        
        free_all(context.temp_allocator)
    }
}

main :: proc() {
    context.logger = log.create_console_logger(opt = { .Level, .Terminal_Color })
    data_set_kind: Data_Set_Kind
    data_set_dir := "digits-mnist"
    if len(os.args) > 1 {
        data_set := os.args[1]
        switch data_set {
        case "digits":
            data_set_kind = .Digits
            data_set_dir = "digits-mnist"
        case "fashion":
            data_set_kind = .Fashion
            data_set_dir = "fashion-mnist"
        case:
            fmt.eprintfln("usage: %v <dataset>", os.base(os.args[0]))
            fmt.eprintfln("Supported datasets: digits, fashion")
        }
    }

    // Load chosen data set
    log.info("Loading Dataset")
    train_set, test_set: Data_Set
    if data_set_kind == .Digits || data_set_kind == .Fashion {
        train_set, test_set = load_mnist(data_set_dir)
        if data_set_kind == .Fashion {
            test_set.classes = FASHION_MNIST_CLASSES
        }
    } else {
        unimplemented()
    }
    defer {
        batch_destroy(train_set.data)
        batch_destroy(test_set.data)
    }

    enum_str, _ := fmt.enum_value_to_string(data_set_kind)
    model_path, _ := os.join_filename(enum_str, "cbor", context.temp_allocator)
    model, err := load_from_file(model_path)
    defer deinit(&model)
    if err != nil {
        config := Config{ .Cross_Entropy, .Sigmoid, .Softmax, .Gaussian }
        init(&model, {train_set.input_size, 100, 30, train_set.output_size}, config)

        log.info("Training Network")
        train := train_set.data[:50_000]
        validation := train_set.data[50_000:]

        mini_batch_size := 10
        learn_rate: f32 = 0.5
        regularization := 5.0 / f32(len(train))
        epochs := 30
        for epoch in 0..<epochs {
            cost: f32
            for i := 0; i < len(train); i += mini_batch_size {
                batch := train[i:i+mini_batch_size]
                cost = learn(model, batch, learn_rate, os.get_processor_core_count(), regularization)
            }
            eval := evaluate(model, validation, os.get_processor_core_count())
            log.infof("Epoch(%v) Accuracy = %v", epoch + 1, eval)
        }
        log.info("Testing")
        log.info("Accuracy on test set:", evaluate(model, test_set.data, os.get_processor_core_count()))
        save_to_file(model, model_path)
    }

    run_viewer(model, test_set, data_set_kind)
}
