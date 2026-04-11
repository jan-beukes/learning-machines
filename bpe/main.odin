package bpe

import "core:fmt"
import "core:log"
import "core:strconv"
import "core:os"
import "core:strings"
import "core:math/rand"

DEFAULT_GEN_MAX :: 100

print_random_tokens :: proc(t: Tokenizer, gen_max: int) {
    sb: strings.Builder
    next: [dynamic]Pair
    defer delete(next)

    token: Token
    for t.freqs[token] == 0 {
        token = rand.uint32_range(0, u32(len(t.vocab)))
    }

    for i in 0 ..< gen_max {
        strings.builder_reset(&sb)
        write_token(&sb, t, token)
        fmt.print(strings.to_string(sb))

        for {
            clear(&next)
            if token < 256 {
                break
            }
            // find candidates for a next token by checking for pairs where this token is the left item
            for pair in t.vocab {
                if pair.x == token && pair.y != 0 {
                    append(&next, pair)
                }
            }
            if len(next) > 0 {
                break
            }
            // try again for the smaller (unmerged) token
            token = t.vocab[token].y
        }
        if len(next) == 0 {
            token = t.merges[Pair{ ' ', rand.uint32_range(0, u32(len(t.vocab))) }]
            for t.freqs[token] == 0 {
                token = rand.uint32_range(0, u32(len(t.vocab)))
            }
            continue
        }

        freqs := make([]uint, len(next))
        sum: uint = 0
        for j in 0..<len(next) {
            freq := t.freqs[t.merges[next[j]]]
            freqs[j] = freq
            sum += freq
        }

        roll := rand.uint_range(0, sum)
        curr: uint = 0
        for j in 0..<len(next) {
            if curr + freqs[j] > roll {
                token = next[j].y
                break
            }
            curr += freqs[j]
        }

    }
}

main :: proc() {
    context.logger = log.create_console_logger(.Debug, log.Options{.Level, .Terminal_Color})

    args := os.args
    if len(args) < 2 {
        fmt.eprintfln("Usage: %v gen|<input> [gen max]", os.base(args[0]))
        os.exit(1)
    }
    input := args[1]

    input_file := "../llm/data/shakespeare.txt"
    tokenizer, ok := load("./tokenizer.cbor")
    if !ok {
        log.infof("Training tokenizer on '%v'", input_file)
        train(&tokenizer, input_file)
    }

    if input == "gen" {
        gen_max := DEFAULT_GEN_MAX
        if len(args) > 2 {
            ok: bool
            gen_max = strconv.parse_int(args[2]) or_else DEFAULT_GEN_MAX
        }
        print_random_tokens(tokenizer, gen_max)
    } else {
        tokens := encode(tokenizer, input)
        fmt.println("Tokens:", tokens)
        output := decode(tokenizer, tokens, show_tokens=true) 
        fmt.println("Decoded:", output)
    }
}

