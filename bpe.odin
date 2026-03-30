package bpe

import "core:fmt"
import "core:strings"
import "core:os"
import "core:slice"
import "core:mem"

Map_Entry :: slice.Map_Entry
Token :: u32 // Token is an index into the vocab of pairs
Pair :: [2]Token

VOCAB_SIZE :: 1000

// State
freq_map: map[Pair]uint
// NOTE: we could also do what the python implementations do and for each new token we map to an entire
// string directly, this would take more memory unless we somehow free strings that get merged
vocab: [dynamic]Pair

find_most_frequent_pair :: proc(tokens: []Token) -> (Pair, uint) {
    if len(freq_map) == 0 {
        for i := 0; i < len(tokens) - 1; i += 1 {
            p := Pair{ tokens[i], tokens[i+1] }
            count := freq_map[p] or_else 0
            freq_map[p] = count + 1
        }
    }

    max: Map_Entry(Pair, uint)
    for pair, freq in freq_map {
        pair := pair
        if freq > max.value || freq == max.value && mem.compare_ptrs(&pair, &max.key, size_of(Pair)) > 0 {
            max = { pair, freq }
        }
    }

    return max.key, max.value
}

// replace all occurances of 'pair' with 'tok'
replace_pair_with_token :: proc(tokens: ^[dynamic]Token, pair: Pair, tok: Token) {
    tokens_in := slice.clone(tokens[:], allocator = context.temp_allocator)
    tokens_out := tokens
    clear(tokens_out)
    for i := 0; i < len(tokens_in); {
        if i + 1 >= len(tokens_in) {
            append(tokens_out, tokens_in[i])
            break
        }
        p := Pair{ tokens_in[i], tokens_in[i+1] }
        if p == pair {
            // decrease the pairs that the tokens made with tokens before and after this pair
            // and increase the new pairs made
            if len(tokens_out) > 0 {
                pair_with_prev := Pair{ tokens_out[len(tokens_out)-1], p.x }
                if freq_map[pair_with_prev] <= 1 {
                    delete_key(&freq_map, pair_with_prev)
                } else {
                    freq_map[pair_with_prev] -= 1
                }
                pair_with_prev.y = tok
                count := freq_map[pair_with_prev] or_else 0
                freq_map[pair_with_prev] = count + 1
            }
            if i + 2 < len(tokens_in) {
                pair_with_next := Pair{ p.y, tokens_in[i+2]}
                if freq_map[pair_with_next] <= 1 {
                    delete_key(&freq_map, pair_with_next)
                } else {
                    freq_map[pair_with_next] -= 1
                }
                pair_with_next.x = tok
                count := freq_map[pair_with_next] or_else 0
                freq_map[pair_with_next] = count + 1
            }

            append(tokens_out, tok)
            i += 2
        } else {
            append(tokens_out, tokens_in[i])
            i += 1
        }
    }
    delete_key(&freq_map, pair)
}

encode :: proc(input: string, allocator := context.allocator) -> []Token {
    text := transmute([]byte)input

    // load initial ascii chars into vocab
    for i in 0..<256 {
        append(&vocab, Pair{ Token(i), 0 })
    }

    tokens: [dynamic]Token
    for b in text {
        append(&tokens, Token(b))
    }

    decrease := 0
    for _ in len(vocab)..<VOCAB_SIZE {
        most, count := find_most_frequent_pair(tokens[:])
        if count == 1 {
            break
        }
        new_tok := Token(len(vocab))
        append(&vocab, most)
        replace_pair_with_token(&tokens, most, new_tok)

        free_all(context.temp_allocator)
    }
    fmt.println(decrease)
    return tokens[:]
}

// recursively decode the token mappings untill we find a token that corresponds to a single
// character
write_token :: proc(sb: ^strings.Builder, tok: Token) {
    pair := vocab[tok]
    if pair.x == tok {
        strings.write_rune(sb, rune(tok))
        return
    }

    write_token(sb, pair.x)
    write_token(sb, pair.y)
}

decode :: proc(tok_ids: []Token) -> string {
    sb: strings.Builder
    strings.builder_init(&sb)
    for tok in tok_ids {
        write_token(&sb, tok)
    }
    return strings.to_string(sb)
}

main :: proc() {
    input_file := "bpe.txt"
    content, err := os.read_entire_file(input_file, context.allocator)
    if err != nil {
        fmt.eprintf("Could not open '%v'\n", input_file)
        os.exit(1)
    }

    input := string(content)
    fmt.println("Input:", input)
    fmt.println("Chars:", len(input))
    tokens := encode(input)
    fmt.println(tokens)
    output := decode(tokens)
    fmt.println("Tokens:", len(tokens))
    fmt.println("Output:", output)
}
