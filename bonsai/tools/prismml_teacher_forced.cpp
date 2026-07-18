// Pinned llama.cpp teacher-forcing harness for the Bonsai quality gate.
//
// This deliberately depends only on llama.h.  build_prismml_teacher_forced.sh
// extracts that header from the exact sparse llama.cpp source revision paired
// with the installed PrismML runtime, preventing an accidental ABI mismatch.

#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::vector<llama_token> tokens;
    int32_t start = -1;
    uint32_t ctx_size = 2048;
    int32_t threads = 4;
    int32_t top_k = 10;
    int32_t gpu_layers = 99;
    int32_t n_new = 0;
    bool verbose = false;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

template <typename T>
T parse_integer(std::string_view text, const char * name) {
    T value{};
    const char * begin = text.data();
    const char * end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        fail(std::string("invalid ") + name + ": " + std::string(text));
    }
    return value;
}

std::vector<llama_token> parse_tokens(std::string_view text) {
    std::vector<llama_token> result;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const size_t end = comma == std::string_view::npos ? text.size() : comma;
        if (end == begin) {
            fail("--tokens contains an empty token id");
        }
        result.push_back(parse_integer<llama_token>(text.substr(begin, end - begin), "token id"));
        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1;
    }
    return result;
}

Options parse_args(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto value = [&]() -> std::string_view {
            if (++i >= argc) {
                fail(std::string(arg) + " requires a value");
            }
            return argv[i];
        };
        if (arg == "--model") {
            options.model = std::string(value());
        } else if (arg == "--tokens") {
            options.tokens = parse_tokens(value());
        } else if (arg == "--start") {
            options.start = parse_integer<int32_t>(value(), "--start");
        } else if (arg == "--ctx-size") {
            options.ctx_size = parse_integer<uint32_t>(value(), "--ctx-size");
        } else if (arg == "--threads") {
            options.threads = parse_integer<int32_t>(value(), "--threads");
        } else if (arg == "--top-k") {
            options.top_k = parse_integer<int32_t>(value(), "--top-k");
        } else if (arg == "--gpu-layers") {
            options.gpu_layers = parse_integer<int32_t>(value(), "--gpu-layers");
        } else if (arg == "--n-new") {
            options.n_new = parse_integer<int32_t>(value(), "--n-new");
        } else if (arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: prismml_teacher_forced --model FILE --tokens ID,... --start N\n"
                << "       [--ctx-size N] [--threads N] [--top-k N] [--gpu-layers N]\n"
                << "       [--n-new N] [--verbose]  # generation requires --start == token count\n";
            std::exit(0);
        } else {
            fail("unknown argument: " + std::string(arg));
        }
    }
    if (options.model.empty()) {
        fail("--model is required");
    }
    if (options.tokens.empty()) {
        fail("--tokens is required");
    }
    if (options.n_new < 0) {
        fail("--n-new must be non-negative");
    }
    const int32_t token_count = static_cast<int32_t>(options.tokens.size());
    if (options.start <= 0 ||
        (options.n_new == 0 && options.start >= token_count) ||
        (options.n_new > 0 && options.start != token_count)) {
        fail(options.n_new > 0
            ? "with --n-new, --start must equal the number of prompt tokens"
            : "--start must be in [1, number of tokens - 1]");
    }
    if (options.ctx_size < options.tokens.size() + static_cast<size_t>(options.n_new)) {
        fail("--ctx-size is smaller than the requested token sequence");
    }
    if (options.threads <= 0) {
        fail("--threads must be positive");
    }
    if (options.top_k <= 0) {
        fail("--top-k must be positive");
    }
    return options;
}

struct ModelDeleter {
    void operator()(llama_model * model) const { llama_model_free(model); }
};

struct ContextDeleter {
    void operator()(llama_context * context) const { llama_free(context); }
};

struct BackendGuard {
    BackendGuard() { llama_backend_init(); }
    ~BackendGuard() { llama_backend_free(); }
    BackendGuard(const BackendGuard &) = delete;
    BackendGuard & operator=(const BackendGuard &) = delete;
};

void silent_logger(enum ggml_log_level, const char *, void *) {}

struct RankedToken {
    float logit;
    llama_token token;
};

// llama.cpp's greedy choice is the greatest logit, with token id providing a
// stable tie break.  Make the gate deterministic even for exact float ties.
bool better(const RankedToken & a, const RankedToken & b) {
    if (a.logit != b.logit) {
        return a.logit > b.logit;
    }
    return a.token < b.token;
}

std::vector<RankedToken> rank_logits(const float * logits, int32_t n_vocab, int32_t top_k) {
    std::vector<RankedToken> ranked;
    ranked.reserve(static_cast<size_t>(n_vocab));
    for (int32_t token = 0; token < n_vocab; ++token) {
        ranked.push_back({logits[token], token});
    }
    const size_t keep = std::min(static_cast<size_t>(top_k), ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + keep, ranked.end(), better);
    ranked.resize(keep);
    return ranked;
}

int64_t target_rank(const float * logits, int32_t n_vocab, llama_token target) {
    if (target < 0 || target >= n_vocab) {
        return -1;
    }
    const float target_logit = logits[target];
    int64_t rank = 1;
    for (int32_t token = 0; token < n_vocab; ++token) {
        if (logits[token] > target_logit || (logits[token] == target_logit && token < target)) {
            ++rank;
        }
    }
    return rank;
}

void decode_or_fail(llama_context * context, llama_token * tokens, int32_t count) {
    llama_batch batch = llama_batch_get_one(tokens, count);
    const int32_t status = llama_decode(context, batch);
    if (status != 0) {
        fail("llama_decode failed with status " + std::to_string(status));
    }
}

void print_rows(llama_context * context, const llama_vocab * vocab, const Options & options) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::cout << "{\"rows\":[";
    bool first_row = true;
    for (int32_t position = options.start;
         position < static_cast<int32_t>(options.tokens.size()); ++position) {
        float * logits = llama_get_logits_ith(context, -1);
        if (logits == nullptr) {
            fail("llama_get_logits_ith returned null");
        }
        const llama_token target = options.tokens[static_cast<size_t>(position)];
        const std::vector<RankedToken> top = rank_logits(logits, n_vocab, options.top_k);
        if (!first_row) {
            std::cout << ',';
        }
        first_row = false;
        std::cout << "{\"position\":" << position
                  << ",\"target\":" << target
                  << ",\"top1\":" << top.front().token
                  << ",\"targetRank\":" << target_rank(logits, n_vocab, target)
                  << ",\"topK\":[";
        for (size_t i = 0; i < top.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << top[i].token;
        }
        std::cout << "]}";

        // The current logits predict tokens[position].  Feed that ground-truth
        // token to obtain the row predicting tokens[position + 1].
        if (position + 1 < static_cast<int32_t>(options.tokens.size())) {
            llama_token next = target;
            decode_or_fail(context, &next, 1);
        }
    }
    std::cout << "]}\n";
}

void print_generated_rows(llama_context * context, const llama_vocab * vocab, const Options & options) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> generated;
    generated.reserve(static_cast<size_t>(options.n_new));
    std::cout << "{\"rows\":[";
    for (int32_t step = 0; step < options.n_new; ++step) {
        float * logits = llama_get_logits_ith(context, -1);
        if (logits == nullptr) {
            fail("llama_get_logits_ith returned null");
        }
        const std::vector<RankedToken> top = rank_logits(logits, n_vocab, options.top_k);
        const llama_token target = top.front().token;
        generated.push_back(target);
        if (step != 0) {
            std::cout << ',';
        }
        std::cout << "{\"position\":" << options.start + step
                  << ",\"target\":" << target
                  << ",\"top1\":" << target
                  << ",\"targetRank\":1,\"topK\":[";
        for (size_t i = 0; i < top.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << top[i].token;
        }
        std::cout << "]}";
        if (step + 1 < options.n_new) {
            llama_token next = target;
            decode_or_fail(context, &next, 1);
        }
    }
    std::cout << "],\"generatedIds\":[";
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << generated[i];
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_args(argc, argv);
        if (!options.verbose) {
            llama_log_set(silent_logger, nullptr);
        }
        BackendGuard backend;

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.gpu_layers;
        std::unique_ptr<llama_model, ModelDeleter> model(
            llama_model_load_from_file(options.model.c_str(), model_params));
        if (!model) {
            fail("failed to load model: " + options.model);
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = options.ctx_size;
        context_params.n_batch = std::max<uint32_t>(options.ctx_size, options.tokens.size());
        context_params.n_ubatch = std::min<uint32_t>(context_params.n_batch, 512);
        context_params.n_threads = options.threads;
        context_params.n_threads_batch = options.threads;
        context_params.no_perf = true;
        std::unique_ptr<llama_context, ContextDeleter> context(
            llama_init_from_model(model.get(), context_params));
        if (!context) {
            fail("failed to create llama context");
        }

        std::vector<llama_token> prefix(options.tokens.begin(), options.tokens.begin() + options.start);
        decode_or_fail(context.get(), prefix.data(), static_cast<int32_t>(prefix.size()));
        if (options.n_new > 0) {
            print_generated_rows(context.get(), llama_model_get_vocab(model.get()), options);
        } else {
            print_rows(context.get(), llama_model_get_vocab(model.get()), options);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "prismml_teacher_forced: " << error.what() << '\n';
        return 2;
    }
}
