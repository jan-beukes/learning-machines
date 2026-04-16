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
VOCAB_SIZE :: 2000

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

softmax :: proc(xs: []f32) -> []f32 {
    ret := make([]f32, len(xs))
    max_x := slice.max(xs)
    sum: f32
    for x, i in xs {
        ret[i] = math.exp(x - max_x)
        sum += ret[i]
    }
    for &x in ret do x /= sum
    return ret
}

cost_function :: proc(logits: [][]f32, targets: []Token) -> f32 {
    cost: f32
    for n in 0..<len(logits) {
        target    := targets[n]
        max_logit := slice.max(logits[n])
        sum_exp: f32
        for logit in logits[n] {
            sum_exp += math.exp(logit - max_logit)
        }
        cost += max_logit + math.ln(sum_exp) - logits[n][target]
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

forward :: proc(self: Model, input: []Token, allocator := context.allocator) -> [][]f32 {
    output:= make([][]f32, len(input), allocator)
    for token, i in input {
        row_start := int(token)*self.vocab_size
        row_end := row_start + self.vocab_size
        output[i] = self.embedding[row_start:row_end]
    }
    return output
}

backward_into :: proc(self: Model, logits: [][]f32, input: []Token, targets: []Token, grads: []f32) -> f32 {
    cost := cost_function(logits, targets)
    for n in 0..<len(logits) {
        idx := int(input[n])
        target := int(targets[n])
        logit_row := logits[n]
        max_logit := slice.max(logit_row)
        sum_exp: f32
        for logit in logit_row {
            sum_exp += math.exp(logit - max_logit)
        }
        for j in 0..<len(logit_row) {
            prob := math.exp(logit_row[j] - max_logit) / sum_exp
            y: f32 = j == target ? 1.0 : 0.0
            grads[idx*self.vocab_size + j] += prob - y
        }
    }
    return cost
}

learn :: proc(self: ^Model, train_batch: []Data_Point, learn_rate: f32, num_threads := 4) -> f32 {
    // Why is this not a problem in my neural nets????
    Thread_Data :: struct {
        model: ^Model,
        batch: []Data_Point,
        grads: []f32,
        total_cost: f32,
    }

    learn_proc :: proc(td: ^Thread_Data) {
        num_grads := td.model.vocab_size * td.model.vocab_size
        slice.zero(td.grads)
        for data_point in td.batch {
            logits := forward(td.model^, data_point.input, context.temp_allocator)
            td.total_cost += backward_into(td.model^, logits, data_point.input, data_point.target, td.grads)
        }
        free_all(context.temp_allocator)
    }

    num_grads := self.vocab_size*self.vocab_size
    thread_batch_size := len(train_batch) / num_threads

    threads := make([]^thread.Thread, num_threads, context.temp_allocator)
    thread_data := make([]Thread_Data, num_threads, context.temp_allocator)
    for i in 0..<num_threads {
        thread_data[i].model = self
        thread_data[i].grads = make([]f32, num_grads, context.temp_allocator)
        start := i * thread_batch_size
        end := min(start + thread_batch_size, len(train_batch))
        thread_data[i].batch = train_batch[start:end]
        threads[i] = thread.create_and_start_with_poly_data(&thread_data[i], learn_proc, priority=.High)
    }

    slice.zero(self.embedding_grads)

    total_cost: f32
    remaining := len(train_batch) - thread_batch_size * num_threads
    if remaining > 0 {
        grads := make([]f32, num_grads, context.temp_allocator)
        batch := train_batch[len(train_batch)-remaining:]
        slice.zero(grads)
        for data_point in batch {
            logits := forward(self^, data_point.input, context.temp_allocator)
            total_cost += backward_into(self^, logits, data_point.input, data_point.target, grads)
        }
        for i in 0..<num_grads do self.embedding_grads[i] += grads[i]
    }

    for i in 0..<num_threads {
        thread.destroy(threads[i])
        total_cost += thread_data[i].total_cost
        for j in 0..<num_grads {
            self.embedding_grads[j] += thread_data[i].grads[j]
        }
    }

    scale := 1.0 / f32(len(train_batch))
    for i in 0..<len(self.embedding) {
        self.embedding[i] += -learn_rate * self.embedding_grads[i] * scale
    }

    free_all(context.temp_allocator)
    return total_cost / f32(len(train_batch))
}

generate :: proc(self: Model, start: Token, max_tokens := 200, allocator := context.allocator) -> []Token {
    tokens := make([dynamic]Token, allocator)
    append(&tokens, start)

    context.allocator = context.temp_allocator
    defer free_all(context.temp_allocator)
    token := start
    for _ in 0..<max_tokens {
        logits := forward(self, {token})
        probs := softmax(logits[0])
        token = Token(sample_multinomial(probs))
        append(&tokens, token)
    }
    return tokens[:]
}

// get a batch of input, target pairs, each of which is a view into the provided data
get_batch :: proc(data: []Token, max_context, batch_size: int, allocator := context.allocator) -> []Data_Point {
    context.allocator = allocator
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
        return toks[:]
    }

    data, err := os.read_entire_file(file, context.allocator)
    defer delete(data)
    if err != nil {
        fmt.panicf("Could not open '%v': %v", INPUT_FILE, err)
    }
    fmt.println("Encoding file")
    vocab_size := len(t.vocab)
    tokens := bpe.encode(t, string(data))
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
    num_threads := os.get_processor_core_count()
    max_context := 2
    batch_size := 4*num_threads
    learn_rate: f32 = 40
    iterations := 10_000

    model: Model
    init(&model, vocab_size)
    defer vmem.arena_destroy(&model.arena)

    // Train
    for i in 0..<iterations {
        batch := get_batch(train_data, max_context, batch_size, context.temp_allocator)
        cost := learn(&model, batch, learn_rate, num_threads)
        if i % 10 == 0 do fmt.printfln("%v: Cost: %v", i, cost)
    }

    generated := generate(model, Token('\n'))
    text := bpe.decode(tokenizer, generated)
    fmt.println(text)
}
