package nn

import "core:fmt"
import "core:strings"
import "core:strconv"
import "core:bytes"
import "core:slice"
import "core:os"

MNIST_RES :: 28

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
