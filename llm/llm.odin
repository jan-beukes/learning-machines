package llm

import "core:fmt"
import "core:thread"
import "core:os"
import "core:math/rand"

import nn "../neural-net"
import "../bpe"

INPUT_FILE :: "data/shakespeare.txt"

Token :: bpe.Token

Data_Point :: struct {
    input: []Token,
    target: []Token,
}

Model :: struct {
    vocab_size: int,
    embedding: [][]f32,
    embedding_grads: [][]f32,
}

// sample from multinomial distribution in probs
sample_multinomial :: proc(probs: []f32) -> int {
    r := rand.float32_uniform(0, 1.0)
    sum: f32
    for prob, i in probs {
        sum += prob
        if sum > r {
            return i
        }
    }
    panic("invalid probabilites, do not sum to 1.0")
}

// Cross Entropy
cost_function :: proc(logits: [][]f32, targets: []Token) -> f32 {
    cost := 0
    for n in 0..<len(logits) {
        target := targets[n]
        for p, i in logits[n] {
            c := i == target ? -math.ln(p) : -math.ln(1 - p)
            cost += c
        }
    }
    return cost / f32(len(targets))
}

// 
cost_function_derivative :: proc(pred, target: f32) -> f32 {
    a := pred
    y := target
    if a == 0 || a == 1 {
        return 0
    }
    return (-a + y) / (a * (a - 1))
}

init :: proc(self: ^Model, vocab_size: int, allocator := context.allocator) {
    context.allocator = allocator
    self.vocab_size = vocab_size
    self.embedding = make([][]f32, vocab_size)
    self.embedding_grads = make([][]f32, vocab_size)
    for i in 0..<len(self.embedding) {
        self.embedding[i] = make([]f32, vocab_size)
        self.embedding_grads[i] = make([]f32, vocab_size)
        for &entry in self.embedding[i] {
            entry = rand.float32_normal(0, 1)
        }
    }
}

// TODO: will probably need to add train/no_train to simulate zero grad
forward :: proc(self: Model, input: []Token) -> [][]f32 {
    output:= make([][]f32, len(input))
    for token, i in input {
        output[i] = self.embedding[token]
    }
    return output
}

backward :: proc(self: Model, logits: []f32) -> f32 {
        cost := cost_function(logits)

        return cost
}

learn :: proc(self: Model, batch: []Data_Point, learn_rate: f32, num_threads := 4) {
    ouput := make([]f32, )
    for data_point in batch {
        logits := forward(self, data_point.input)
    }
    apply_gradients(self)
}

// get a batch of input, target pairs, each of which is a view into the provided data
get_batches :: proc(data: []Token, context, batch_size: int, allocator := context.allocator) -> ([][]Token, [][]Token) {
    batch: [dynamic]Data_Point
    for i in 0..<batch_size {
        batch_idx := rand.int_range(0, len(data) - context)
        dp: Data_Point
        dp.inputs = data[batch_idx:batch_idx+1]
        dp.target = data[batch_idx+1:batch_idx+1]
        append(&batch, dp)
    }
    return batch[:]
}

main :: proc() {
    tokenizer, ok := bpe.load("tokenizer.cbor")
    if !ok {
        bpe.train(&tokenizer, INPUT_FILE)
    }
    defer bpe.destroy(tokenizer)

    data, err := os.read_entire_file(INPUT_FILE, context.allocator)
    defer delete(data)
    if err != nil {
        fmt.panicf("Could not open '%v': %v", INPUT_FILE, err)
    }

    text := string(data)[:1000]
    tokens := bpe.encode(tokenizer, text)
    defer delete(tokens)

    train_split := int(0.9*f64(len(tokens)))
    train_data := data[:train_split]
    val_data := data[train_split:]

    context := 10
    batch_size := 24
    batch := get_batch(train_data, context, batch_size)

    fmt.println(train_data)
}
