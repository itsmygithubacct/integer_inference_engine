// Persistent, exact libllama tokenizer used by the Bonsai REPL/benchmarks.
// Protocol: one lowercase/uppercase hexadecimal UTF-8 payload per stdin line;
// one JSON token-id array per stdout line.  A line containing only "Q" exits.

#include "llama.h"

#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct ModelDeleter {
    void operator()(llama_model * model) const { llama_model_free(model); }
};

struct BackendGuard {
    BackendGuard() { llama_backend_init(); }
    ~BackendGuard() { llama_backend_free(); }
};

void silent_logger(enum ggml_log_level, const char *, void *) {}

int nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string decode_hex(std::string_view line) {
    if (line.size() % 2 != 0) {
        throw std::runtime_error("odd-length hexadecimal input");
    }
    std::string text(line.size() / 2, '\0');
    for (size_t i = 0; i < text.size(); ++i) {
        const int hi = nibble(line[2 * i]);
        const int lo = nibble(line[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            throw std::runtime_error("non-hexadecimal input");
        }
        text[i] = static_cast<char>((hi << 4) | lo);
    }
    return text;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("input exceeds libllama int32 length");
    }
    std::vector<llama_token> tokens(text.size() + 16);
    int32_t count = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (count == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("token count overflow");
    }
    if (count < 0) {
        tokens.resize(static_cast<size_t>(-count));
        count = llama_tokenize(
            vocab, text.data(), static_cast<int32_t>(text.size()),
            tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    }
    if (count < 0) {
        throw std::runtime_error("libllama tokenization failed");
    }
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

void print_tokens(const std::vector<llama_token> & tokens) {
    std::cout << '[';
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << tokens[i];
    }
    std::cout << "]\n" << std::flush;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path;
        bool verbose = false;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (arg == "--verbose") {
                verbose = true;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "usage: prismml_tokenizer_server --model FILE [--verbose]\n";
                return 0;
            } else {
                throw std::runtime_error("unknown or incomplete argument: " + std::string(arg));
            }
        }
        if (model_path.empty()) {
            throw std::runtime_error("--model is required");
        }
        if (!verbose) llama_log_set(silent_logger, nullptr);
        BackendGuard backend;
        llama_model_params params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, ModelDeleter> model(
            llama_model_load_from_file(model_path.c_str(), params));
        if (!model) {
            throw std::runtime_error("failed to load model vocabulary");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "Q") break;
            try {
                print_tokens(tokenize(vocab, decode_hex(line)));
            } catch (const std::exception & error) {
                std::cout << "{\"error\":\"" << error.what() << "\"}\n" << std::flush;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "prismml_tokenizer_server: " << error.what() << '\n';
        return 2;
    }
}
