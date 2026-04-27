package nn

import "core:fmt"
import "core:strings"
import "core:strconv"
import "core:bytes"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:os"

MNIST_RES :: 28
CIFAR_RES :: 32
CIFAR_BATCH_IMAGE_COUNT :: 10000

FASHION_MNIST_CLASSES :: []string{
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
}

Data_Set :: struct {
    data: []Data_Point,
    classes: []string,
    input_size, output_size: int,
}

load_iris :: proc(path: string, allocator := context.allocator) -> Data_Set {
    context.allocator = allocator
    content, err := os.read_entire_file(path, allocator)
    defer delete(content)
    s: string = string(content)
    inputs := make([dynamic][]f32, context.temp_allocator)
    labels := make([dynamic]i32, context.temp_allocator)
    names: [dynamic]string
    for line in strings.split_lines_iterator(&s) {
        items := strings.split(line, ",", allocator = context.temp_allocator)
        row := make([]f32, 4)
        for i in 0..<len(row) {
            ok: bool
            row[i], ok = strconv.parse_f32(items[i])
            if !ok {
                fmt.panicf("Could not parse iris line:", line)
            }
        }
        append(&inputs, row)
        name := items[4]
        idx, found := slice.linear_search(names[:], name)
        if found {
            append(&labels, i32(idx))
        } else {
            append(&labels, i32(len(names)))
            append(&names, strings.clone(name))
        }
    }

    batch := batch_create(inputs[:], labels[:], i32(len(names)))

    free_all(context.temp_allocator)
    return Data_Set{
        data = batch,
        classes = names[:],
        input_size = len(batch[0].input),
        output_size = len(batch[0].expected),
    }
}

load_cifar_batch :: proc(batches: ^[dynamic]Data_Point, path: string, num_labels: i32, allocator := context.allocator) {
    data, err := os.read_entire_file(path, allocator)
    defer delete(data)
    if err != nil {
        fmt.panicf("Could not read file '%v'", path)
    }
    r: bytes.Reader
    bytes.reader_init(&r, data)
    for _ in 0..<CIFAR_BATCH_IMAGE_COUNT {
        data_point: Data_Point
        data_point.input = make([]f32, 3*CIFAR_RES*CIFAR_RES)
        data_point.expected = make([]f32, num_labels)
        label, _ := bytes.reader_read_byte(&r)
        data_point.label = i32(label)
        data_point.expected[label] = 1.0
        // store as rgb triples
        for chan in 0..<3 {
            for i in 0..<CIFAR_RES*CIFAR_RES {
                b, _ := bytes.reader_read_byte(&r)
                data_point.input[i*3 + chan] = f32(b) / 255.0
            }
        }
        append(batches, data_point)
    }
}

load_cifar :: proc(dir: string, allocator := context.allocator) -> (Data_Set, Data_Set) {
    context.allocator = allocator

    meta_path, _ := os.join_path({dir, "batches.meta.txt"}, context.temp_allocator)
    content, err := os.read_entire_file(meta_path, context.allocator)
    if err != nil {
        fmt.panicf("Could not read file '%v'", meta_path)
    }
    defer delete(content)
    classes := strings.split_lines(strings.trim_space(string(content)))
    // clone since we are deleting the file content
    for &class in classes {
        class = strings.clone(class)
    }

    num_labels := i32(len(classes))
    train_batch: [dynamic]Data_Point
    
    train_batch_path_fmt, _ := os.join_path({dir, "data_batch_%d.bin"}, context.temp_allocator)
    for i in 1..=5 {
        path := fmt.tprintf(train_batch_path_fmt, i)
        load_cifar_batch(&train_batch, path, num_labels)
    }

    test_batch_path, _ := os.join_path({dir, "test_batch.bin"}, context.temp_allocator)
    test_batch: [dynamic]Data_Point
    load_cifar_batch(&test_batch, test_batch_path, num_labels)

    free_all(context.temp_allocator)
    train_set := Data_Set{ train_batch[:], classes, len(train_batch[0].input), len(train_batch[0].expected) }
    test_set := Data_Set{ test_batch[:], classes, len(test_batch[0].input), len(test_batch[0].expected) }
    return train_set, test_set
}

// returns an array of all the flattened images with float pixels 
load_mnist_images :: proc(path: string, allocator := context.allocator) -> [][]f32 {
    context.allocator = allocator
    data, err := os.read_entire_file(path, allocator)
    defer delete(data)
    if err != nil {
        fmt.panicf("Could not read file '%v'", path)
    }

    r: bytes.Reader
    bytes.reader_init(&r, data)

    header: [4]i32be
    bytes.reader_read_slice(&r, header[:])
    assert(header[0] == i32be(2051), "Magick number must be 2051")
    count := header[1]
    nrows, ncols := header[2], header[3]

    images := make([][]f32, count)
    for i in 0..<count {
        images[i] = make([]f32, nrows*ncols)
        // read each byte and convert to float
        for j in 0..<nrows*ncols {
            b, err := bytes.reader_read_byte(&r)
            assert(err == nil)
            images[i][j] = f32(b) / 255.0
        }
    }
    return images
}

// labels 0..9
load_mnist_labels :: proc(path: string, allocator := context.allocator) -> []i32 {
    context.allocator = allocator
    data, err := os.read_entire_file(path, allocator)
    if err != nil {
        fmt.panicf("Could not open file '%v'", path)
    }
    r: bytes.Reader
    bytes.reader_init(&r, data)

    // magick and label count
    header: [2]i32be
    bytes.reader_read_slice(&r, header[:])
    assert(header[0] == i32be(2049), "Magick number must be 2049")
    count := header[1]

    labels := make([]i32, count)
    for i in 0..<count {
        b, err := bytes.reader_read_byte(&r)
        assert(err == nil)
        labels[i] = i32(b)
    }

    return labels
}

load_mnist :: proc(dir: string, allocator := context.allocator) -> (Data_Set, Data_Set) {
    train_labels_path, _ := os.join_path({dir, "train-labels-idx1-ubyte"}, context.temp_allocator)
    train_images_path, _ := os.join_path({dir, "train-images-idx3-ubyte"}, context.temp_allocator)
    test_labels_path, _ := os.join_path({dir, "t10k-labels-idx1-ubyte"}, context.temp_allocator)
    test_images_path, _ := os.join_path({dir, "t10k-images-idx3-ubyte"}, context.temp_allocator)

    train_labels := load_mnist_labels(train_labels_path)
    train_images := load_mnist_images(train_images_path)
    train_batch := batch_create(train_images, train_labels, 10)
    delete(train_labels); delete(train_images)

    test_labels := load_mnist_labels(test_labels_path)
    test_images := load_mnist_images(test_images_path)
    test_batch := batch_create(test_images, test_labels, 10)
    // delete(test_labels); delete(test_images)

    train_set := Data_Set{ train_batch, nil, len(train_batch[0].input), len(train_batch[0].expected) }
    test_set := Data_Set{ test_batch, nil, len(test_batch[0].input), len(test_batch[0].expected) }

    free_all(context.temp_allocator)
    return train_set, test_set
}

// This too is taken from the goat Sebastian after my idea sucked
sample :: proc(pixels: []f32, u, v: f32, w, h: int) -> f32 {
    u := math.max(0, math.min(1, u))
    v := math.max(0, math.min(1, v))
    x := u * f32(w - 1)
    y := v * f32(h - 1)

    idx_left := int(x)
    idx_top := int(y)
    idx_right := math.min(idx_left + 1, w - 1)
    idx_bottom := math.min(idx_top + 1, h - 1)

    blend_x := x - f32(idx_left)
    blend_y := y - f32(idx_top)

    bl := pixels[idx_bottom*w + idx_left]
    br := pixels[idx_bottom*w + idx_right]
    tl := pixels[idx_top*w + idx_left]
    tr := pixels[idx_top*w + idx_right]

    value_bottom := bl + (br - bl) * blend_x
    value_top := tl + (tr - tl) * blend_x
    value := value_top + (value_bottom - value_top) * blend_y
    return value
}

// expect single channel float pixels
random_process_images :: proc(data: []Data_Point, width: int, height: int) {
    ROTATE_RANGE :: 0.15
    SCALE_RANGE :: 0.15
    OFFSET_AMMOUNT :: 0.8

    process_pixels := make([]f32, len(data[0].input), context.allocator)
    defer delete(process_pixels)
    assert(width * height == len(process_pixels))
    for data_point in data {
        pixels := data_point.input
        scale: f32 = rand.float32_normal(1, SCALE_RANGE)
        rotation: f32 = rand.float32_normal(0, ROTATE_RANGE)

        // get the maximum offset we can go in each direction while keeping image on screen
        // then random lerp between them
        bounds_minx := width
        bounds_maxx := 0
        bounds_miny := height
        bounds_maxy := 0
        for y in 0..<height {
            for x in 0..<width {
                if pixels[y*width + x] > 0 {
                    bounds_minx = math.min(bounds_minx, x)
                    bounds_maxx = math.max(bounds_maxx, x)
                    bounds_miny = math.min(bounds_miny, y)
                    bounds_maxy = math.max(bounds_maxy, y)
                }
            }
        }
        offset_minx := -f32(bounds_minx) / f32(width)
        offset_maxx := f32(width - bounds_maxx) / f32(width)
        offset_miny := -f32(bounds_miny) / f32(height)
        offset_maxy := f32(height - bounds_maxy) / f32(height)
        offset_x := math.lerp(offset_minx, offset_maxx, rand.float32()) * OFFSET_AMMOUNT
        offset_y := math.lerp(offset_miny, offset_maxy, rand.float32()) * OFFSET_AMMOUNT

        for y in 0..<height {
            for x in 0..<width {
                ihat: [2]f32 = { math.cos(rotation), math.sin(rotation) } / scale
                jhat: [2]f32 = { -ihat.y, ihat.x }

                u := f32(x) / (f32(width) - 1)
                v := f32(y) / (f32(height) - 1)
                u = ihat.x * (u - 0.5) + jhat.x * (v - 0.5) + 0.5 - offset_x
                v = ihat.y * (u - 0.5) + jhat.y * (v - 0.5) + 0.5 - offset_y
                pixel_value := sample(pixels, u, v, width, height)
                process_pixels[y*width + x] = math.clamp(pixel_value, 0, 1)
            }
        }
        copy(data_point.input, process_pixels)
        slice.zero(process_pixels)
    }
}
