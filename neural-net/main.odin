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
    Cifar,
}

update_input_image_texture :: proc(texture: ^rl.Texture, input: []f32, kind: Data_Set_Kind) {
    pixels := make([]u8, len(input))
    data := raw_data(pixels)
    defer delete(pixels)

    width, height: i32
    format: rl.PixelFormat
    switch kind {
    case .Digits, .Fashion:
        for &pixel, i in pixels {
            pixel = u8(input[i] * 255.0)
        }
        format = .UNCOMPRESSED_GRAYSCALE
        width, height = MNIST_RES, MNIST_RES
    case .Cifar:
        image_size := CIFAR_RES*CIFAR_RES
        for i in 0..<image_size {
            idx := 3*i
            pixels[idx] = u8(input[i] * 255.0)
            pixels[idx+1] = u8(input[image_size + i] * 255.0)
            pixels[idx+2] = u8(input[2*image_size + i] * 255.0)
        }
        format = .UNCOMPRESSED_R8G8B8
        width, height = CIFAR_RES, CIFAR_RES
    }
    if !rl.IsTextureValid(texture^) {
        image := rl.Image{
            data = data,
            width = width,
            height = height,
            mipmaps = 1,
            format = format,
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

run_viewer :: proc(model: Neural_Network, data_set: Data_Set, kind: Data_Set_Kind) {
    resx, resy: i32 = 800, 900
    rl.SetTraceLogLevel(.ERROR)
    rl.InitWindow(resx, resy, "Neural Network")
    defer rl.EndDrawing()

    font_size: f32 = 30
    font := rl.LoadFontEx("./iosevka.ttf", i32(font_size), nil, 0)

    image_idx := 0
    texture: rl.Texture
    update_input_image_texture(&texture, data_set.data[image_idx].input, kind)
    prediction, confidence := predict(model, data_set.data[image_idx])
    for !rl.WindowShouldClose() {

        if rl.IsKeyPressed(.RIGHT) {
            image_idx = (image_idx + 1) % len(data_set.data)
            prediction, confidence = predict(model, data_set.data[image_idx])
            update_input_image_texture(&texture, data_set.data[image_idx].input, kind)
        } else if rl.IsKeyPressed(.LEFT) {
            image_idx = math.floor_mod(image_idx - 1, len(data_set.data))
            prediction, confidence = predict(model, data_set.data[image_idx])
            update_input_image_texture(&texture, data_set.data[image_idx].input, kind)
        } else if rl.IsKeyPressed(.SPACE) {
            // find next incorrect prediction
            for i := image_idx + 1; i != image_idx; i = (i + 1) % len(data_set.data) {
                data_point := data_set.data[i]
                prediction, confidence = predict(model, data_point)
                if prediction != data_point.label {
                    image_idx = i
                    break
                }
            }
            update_input_image_texture(&texture, data_set.data[image_idx].input, kind)
        }

        rl.BeginDrawing()
        rl.ClearBackground({ 0x20, 0x20, 0x20, 0xff })
        src := rl.Rectangle{ 0, 0, f32(texture.width), f32(texture.height) }
        dst := rl.Rectangle{ 0, 0, f32(resx), f32(resx) }
        rl.DrawTexturePro(texture, src, dst, {}, 0, rl.WHITE)

        // UI
        label := data_set.data[image_idx].label

        pad: f32 = 10
        text: cstring
        if data_set.classes != nil {
            text = rl.TextFormat("Prediction: %v (%.2f%%)", data_set.classes[prediction], 100*confidence)
        } else {
            text = rl.TextFormat("Prediction: %v (%.2f%%)", prediction, 100*confidence)
        }
        text_width := rl.MeasureTextEx(font, text, font_size, 0).x
        text_pos := rl.Vector2{ f32(resx) - pad - text_width, f32(resy) - pad - font_size }
        rl.DrawTextEx(font, text, text_pos, font_size, 0, rl.RAYWHITE)

        if data_set.classes != nil {
            text = rl.TextFormat("Label: %v", data_set.classes[label])
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
        case "cifar":
            data_set_kind = .Cifar
            data_set_dir = "cifar-10"
        case:
            fmt.eprintfln("usage: %v <dataset>", os.base(os.args[0]))
            fmt.eprintfln("Supported datasets: digits, fashion, cifar")
        }
    }

    // Load chosen data set
    log.info("Loading Dataset")
    train_set, test_set: Data_Set
    switch data_set_kind {
    case .Digits, .Fashion:
        train_set, test_set = load_mnist(data_set_dir)
        if data_set_kind == .Fashion {
            test_set.classes = FASHION_MNIST_CLASSES
        }
    case .Cifar:
        train_set, test_set = load_cifar(data_set_dir)
    }
    defer {
        batch_destroy(train_set.data)
        batch_destroy(test_set.data)
        // This should be fine for fashion
        delete(train_set.classes)
    }

    enum_str, _ := fmt.enum_value_to_string(data_set_kind)
    model_path, _ := os.join_filename(enum_str, "cbor", context.temp_allocator)
    model, err := load_from_file(model_path)
    defer deinit(&model)
    if err != nil {
        log.info("Training Network")

        // params
        config := Config{ .Cross_Entropy, .ReLu, .Softmax, .Gaussian }
        layers := []int{train_set.input_size, 100, train_set.output_size}
        init(&model, layers, config)
        num_threads := os.get_processor_core_count()
        train_split: f32 = 0.90
        mini_batch_size := 48
        learn_rate: f32 = 0.05
        regularization: f32 = 0.0005
        epochs := 30

        split_idx := int(train_split*f32(len(train_set.data)))
        train := train_set.data[:split_idx]
        validation := train_set.data[split_idx:]

        for epoch in 0..<epochs {
            cost: f32
            for i := 0; i + mini_batch_size < len(train); i += mini_batch_size {
                batch := train[i:i+mini_batch_size]
                cost = learn(model, batch, num_threads, learn_rate, regularization)
            }
            eval := evaluate(model, validation, num_threads)
            log.infof("Epoch(%v) Accuracy = %v", epoch + 1, eval)
        }
        save_to_file(model, model_path)
    }
    log.info("Testing")
    log.info("Accuracy on test set:", evaluate(model, test_set.data, os.get_processor_core_count()))

    // run_viewer(model, test_set, data_set_kind)
}
