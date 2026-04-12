package llm

import "core:fmt"
import "core:strconv"
import "core:strings"
import "core:thread"
import "core:sync"
import "core:time"
import "core:os"
import "core:slice"
import "core:math"
import "core:math/rand"
import vmem "core:mem/virtual"

import "../bpe"

INPUT_FILE :: "data/shakespeare.txt"
VOCAB_SIZE :: 1000

Token :: bpe.Token

Data_Point :: struct {
    input: []Token,
    target: []Token,
}

Model :: struct {
    vocab_size: int,
    embedding: []f32, // flattened vocab_size*vocab_size
    embedding_grads: []f32, // flattend

    arena: vmem.Arena,
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

// apply softmax to the array, does not allocate
softmax :: proc(xs: []f32) -> []f32 {
    sum: f32
    for &x in xs {
        x = math.exp(x) 
        sum += x
    }
    for &x in xs {
        x /= sum
    }
    return xs
}

// Cross Entropy using Log softmax and log-likelihood so that we can take non [0, 1] inputs
cost_function :: proc(logits: [][]f32, targets: []Token) -> f32 {
    cost: f32
    for n in 0..<len(logits) {
        target := targets[n]
        max_logit := slice.max(logits[n])

        sum_exp: f32
        // safe log(sum(exp(ai)))
        for logit in logits[n] {
            sum_exp += math.exp(logit - max_logit)
        }
        log_sum_exp := max_logit + math.ln(sum_exp)

        cost += log_sum_exp - logits[n][target]
    }
    return cost / f32(len(targets))
}

init :: proc(self: ^Model, vocab_size: int) {
    context.allocator = vmem.arena_allocator(&self.arena)

    self.vocab_size = vocab_size
    n := vocab_size*vocab_size
    self.embedding = make([]f32, n)
    self.embedding_grads = make([]f32, n)

    for i in 0..<n {
        self.embedding[i] = rand.float32_normal(0, 0.1)
    }
}

forward :: proc(self: Model, input: []Token) -> [][]f32 {
    output:= make([][]f32, len(input))
    for token, i in input {
        row_start := int(token)*self.vocab_size
        row_end := row_start + self.vocab_size
        output[i] = self.embedding[row_start:row_end]
    }
    return output
}

backward :: proc(self: Model, logits: [][]f32, input: []Token, targets: []Token) -> f32 {
    cost := cost_function(logits, targets)
    for n in 0..<len(logits) {
        idx := int(input[n])
        target := int(targets[n])
        logit_row := logits[n]

        sum_exp: f32
        max_logit := slice.max(logit_row)
        for logit in logit_row {
            sum_exp += math.exp(logit - max_logit)
        }

        for j in 0..<len(logit_row) {
            prob := math.exp(logit_row[j] - max_logit) / sum_exp
            y: f32 = j == target ? 1.0 : 0.0
            derivative := prob - y
            self.embedding_grads[idx*self.vocab_size + j] += derivative
        }
    }
    return cost
}

apply_gradients :: proc(self: Model, learn_rate: f32) {
    size := self.vocab_size
    for i in 0..<len(self.embedding) {
        self.embedding[i] += -learn_rate*self.embedding_grads[i]
        self.embedding_grads[i] = 0
    }
}

learn :: proc(self: Model, train_batch: []Data_Point, learn_rate: f32, num_threads := 4) -> f32 {
    self := self

    learn_proc :: proc(model: ^Model, batch: []Data_Point, total_cost: ^f32) {
        context.allocator = context.temp_allocator
        for data_point in batch {
            logits := forward(model^, data_point.input)
            total_cost^ += backward(model^, logits, data_point.input, data_point.target)
        }
        free_all(context.temp_allocator)
    }

    threads := make([]^thread.Thread, num_threads)
    costs := make([]f32, num_threads)
    defer { delete(threads); delete(costs) }
    thread_batch_size := int(math.ceil(f32(len(train_batch)) / f32(num_threads)))
    for i in 0..<num_threads {
        start_idx := i * thread_batch_size
        end_idx := min(start_idx + thread_batch_size, len(train_batch))
        batch := train_batch[start_idx:end_idx]
        threads[i] = thread.create_and_start_with_poly_data3(&self, batch, &costs[i], learn_proc, priority=.High)
    }

    total_cost: f32 = 0
    for t, i in threads {
        thread.destroy(t)
        total_cost += costs[i]
    }

    context_len := f32(len(train_batch[0].input))
    scale := 1.0 / (f32(len(train_batch)) * context_len)
    apply_gradients(self, learn_rate * scale)
    return total_cost / f32(len(train_batch))
}

generate :: proc(self: Model, start: Token, max_tokens := 200, allocator := context.allocator) -> []Token {
    token := start
    tokens := make([dynamic]Token, allocator)
    append(&tokens, token)

    context.allocator = context.temp_allocator
    for _ in 0..<max_tokens {
        logits := forward(self, {token})
        probs := softmax(logits[0])
        token = Token(sample_multinomial(probs))
        append(&tokens, token)
    }
    free_all(context.temp_allocator)
    return tokens[:]
}

// get a batch of input, target pairs, each of which is a view into the provided data
get_batch :: proc(data: []Token, max_context, batch_size: int, allocator := context.allocator) -> []Data_Point {
    batch: [dynamic]Data_Point
    for i in 0..<batch_size {
        idx := rand.int_range(0, len(data) - max_context)
        dp: Data_Point
        dp.input = data[idx:idx+max_context]
        dp.target = data[idx+1:idx+max_context+1]
        append(&batch, dp)
    }
    return batch[:]
}

// returns tokens and vocab size
get_tokens :: proc(t: bpe.Tokenizer, file: string) -> []Token {
    tokens: []Token

    tokens_path := fmt.tprintf("%v.tokens", os.stem(file))
    if os.exists(tokens_path) {
        fmt.println("Loading tokens")
        data, err := os.read_entire_file(tokens_path, context.allocator)
        assert(err == nil)

        content := string(data)
        split_idx := strings.index(content, "\n")
        vocab_size, ok := strconv.parse_int(content[:split_idx])
        assert(ok && vocab_size == len(t.vocab))

        toks: [dynamic]Token
        tokens_line := content[split_idx+1:]
        for s in strings.split_iterator(&tokens_line, " ") {
            if len(s) == 0 do continue
            value, ok := strconv.parse_uint(s)
            assert(ok)
            append(&toks, Token(value))
        }
        tokens = toks[:]
    } else {
        data, err := os.read_entire_file(file, context.allocator)
        defer delete(data)
        if err != nil {
            fmt.panicf("Could not open '%v': %v", INPUT_FILE, err)
        }
        fmt.println("Encoding file")
        vocab_size := len(t.vocab)
        tokens = bpe.encode(t, string(data))
        f: ^os.File
        f, err = os.create(tokens_path)
        defer os.close(f)
        if err != nil {
            fmt.panicf("Could not create '%v'", tokens_path)
        }
        fmt.fprintln(f, vocab_size)
        fmt.fprint(f, tokens[0])
        for token in tokens[1:] {
            fmt.fprint(f, "", token)
        }
    }

    return tokens
}

main :: proc() {
    tokenizer, ok := bpe.load("tokenizer.cbor")
    if !ok {
        fmt.println("Could not load 'tokenizer.cbor', training on input file")
        bpe.train(&tokenizer, INPUT_FILE, VOCAB_SIZE)
    }
    defer bpe.destroy(tokenizer)
    vocab_size := len(tokenizer.vocab)

    tokens := get_tokens(tokenizer, INPUT_FILE)
    defer delete(tokens)

    train_split := int(0.9*f64(len(tokens)))
    train_data := tokens[:train_split]
    val_data := tokens[train_split:]
    fmt.println("Training")

    // Hyperparams
    max_context := 8
    batch_size := 24
    learn_rate: f32 = 1
    iterations := 10_000
    num_threads := os.get_processor_core_count()

    model: Model
    init(&model, vocab_size)
    defer vmem.arena_destroy(&model.arena)

    // Train
    {
        context.allocator = context.temp_allocator
        for i in 0..<iterations {
            batch := get_batch(train_data, max_context, batch_size)
            cost := learn(model, batch, learn_rate, num_threads)
            if i % 100 == 0 do fmt.printfln("%v: Cost: %v", i, cost)
            free_all(context.temp_allocator)
        }
    }

    generated := generate(model, Token('\n'))
    text := bpe.decode(tokenizer, generated)
    fmt.println(text)
}
