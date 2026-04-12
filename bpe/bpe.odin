package bpe

import "core:fmt"
import "core:log"
import "core:strings"
import "core:os"
import "core:slice"
import "core:encoding/cbor"
import "core:mem"
import "core:math/bits"

Map_Entry :: slice.Map_Entry
Token :: u32 // Token is an index into the vocab of pairs
Pair :: [2]Token

BPE_HEADER :: "BPE67"

freq_map: map[Pair]uint

Tokenizer :: struct {
    // this is only needed for making random token generation better
    // NOTE: we could also do what the python implementations do and for each new token we map to an entire
    // string directly, this would take more memory unless we somehow free strings that get merged
    vocab: [dynamic]Pair,
    merges: map[Pair]Token,
    freqs: [dynamic]uint,
}

find_most_frequent_pair :: proc(t: ^Tokenizer, tokens: []Token) -> (Pair, uint) {
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

// merge all occurances of 'pair' into 'tok' while also updating the tokenizer's freqs
merge_pairs :: proc(t: ^Tokenizer, tokens: ^[dynamic]Token, pair: Pair, tok: Token) {
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
            // if t is nil then don't try to update the tokenizer
            if t != nil {
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

// recursively decode the token mappings untill we find a token that corresponds to a single
// character
write_token :: proc(sb: ^strings.Builder, t: Tokenizer, tok: Token) {
    pair := t.vocab[tok]
    if pair.x == tok {
        strings.write_rune(sb, rune(tok))
        return
    }

    write_token(sb, t, pair.x)
    write_token(sb, t, pair.y)
}


decode :: proc(t: Tokenizer, tok_ids: []Token, show_tokens := false, allocator := context.allocator) -> string {
    sb: strings.Builder
    strings.builder_init(&sb)
    colors := [?]string{"\033[31m", "\033[32m", "\033[34m"}
    color_idx := 0
    for tok in tok_ids {
        if show_tokens {
            strings.write_string(&sb, colors[color_idx])
            color_idx = (color_idx + 1) % len(colors)
        }
        write_token(&sb, t, tok)
    }
    if show_tokens {
        strings.write_string(&sb, "\033[m")
    }
    return strings.to_string(sb)
}

encode :: proc(t: Tokenizer, input: string, allocator := context.allocator) -> []Token {
    text := transmute([]byte)input

    tokens := make([dynamic]Token, allocator=allocator)
    for b in text {
        append(&tokens, Token(b))
    }
    for {
        // Find the min index token to merge
        min_id: Token = bits.U32_MAX
        found_merge := false
        for i := 0; i < len(tokens) - 1; i += 1 {
            p := Pair{tokens[i], tokens[i+1]}
            tok_id, ok := t.merges[p]
            if ok {
                found_merge = true
                min_id = min(min_id, tok_id)
            }
            
        }
        if !found_merge {
            break
        }
        pair := t.vocab[min_id]
        merge_pairs(nil, &tokens, pair, min_id)
    }
    return tokens[:]
}

destroy :: proc(t: Tokenizer) {
    delete(t.vocab)
    delete(t.merges)
    delete(t.freqs)
}

load :: proc(path: string, allocator := context.allocator) -> (t: Tokenizer, ok: bool) {
    data, err := os.read_entire_file(path, allocator)
    if err != nil {
        return
    }

    cbor_err := cbor.unmarshal(data, &t)
    if cbor_err != nil {
        return
    }

    ok = true
    return
}

save :: proc(t: Tokenizer, path: string) {
    data, cbor_err := cbor.marshal_into_bytes(t)
    defer delete(data)
    if cbor_err != nil {
        log.error("Could not marshal tokenizer:", cbor_err)
        return
    }

    err := os.write_entire_file(path, data)
    if err != nil {
        fmt.println("Could not save tokenizer:", os.error_string(err))
    }
    log.infof("Saved tokenizer to '%v'", path)
}

train :: proc(t: ^Tokenizer, input_file: string, vocab_size := 1000, allocator := context.allocator) {
    content, err := os.read_entire_file(input_file, allocator)
    defer delete(content)
    if err != nil {
        fmt.eprintf("Could not open '%v'\n", input_file)
        os.exit(1)
    }
    text := transmute([]byte)content
    // load initial ascii chars into vocab
    for i in 0..<256 {
        p := Pair{ Token(i), 0 }
        append(&t.vocab, p)
        t.merges[p] = Token(i)
    }

    t.freqs = make([dynamic]uint, 256)
    tokens := make([dynamic]Token, allocator=allocator)
    defer delete(tokens) // just for training
    for b in text {
        append(&tokens, Token(b))
        t.freqs[b] += 1
    }

    clear(&freq_map)
    for _ in len(t.vocab)..<vocab_size {
        most, count := find_most_frequent_pair(t, tokens[:])
        if count == 1 {
            fmt.println("DONE")
            break
        }

        new_tok := Token(len(t.vocab))
        append(&t.vocab, most)
        append(&t.freqs, count)
        t.merges[most] = new_tok

        merge_pairs(t, &tokens, most, new_tok)

        free_all(context.temp_allocator)
    }

    save(t^, "tokenizer.cbor")
}
