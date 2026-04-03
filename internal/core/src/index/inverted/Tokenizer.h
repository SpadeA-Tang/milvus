// Tokenizer interface for text analysis.
//
// Port of tantivy's tokenizer::Tokenizer trait + TokenStream trait.
// Production uses TantivyTokenizer (Rust FFI via milvus::tantivy::Tokenizer).
// SimpleTokenizer provided for testing without Rust dependency.

#pragma once

#include <cctype>
#include <memory>
#include <string>
#include <vector>

namespace milvus::index::inverted {

// Port of tantivy's tokenizer::TokenStream.
class TokenStream {
 public:
    virtual ~TokenStream() = default;
    virtual bool advance() = 0;
    virtual const std::string& token() const = 0;
};

// Port of tantivy's tokenizer::Tokenizer trait.
class Tokenizer {
 public:
    virtual ~Tokenizer() = default;
    virtual std::unique_ptr<TokenStream>
    token_stream(const std::string& text) = 0;
};

// Split on non-alphanumeric boundaries, lowercase.
// For testing only. Production uses Rust FFI tokenizer.
class SimpleTokenStream : public TokenStream {
 public:
    explicit SimpleTokenStream(const std::string& text) {
        std::string current;
        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                current +=
                    static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            } else {
                if (!current.empty()) {
                    tokens_.push_back(std::move(current));
                    current.clear();
                }
            }
        }
        if (!current.empty()) {
            tokens_.push_back(std::move(current));
        }
    }

    bool
    advance() override {
        pos_++;
        return pos_ < tokens_.size();
    }

    const std::string&
    token() const override {
        return tokens_[pos_];
    }

 private:
    std::vector<std::string> tokens_;
    size_t pos_ = static_cast<size_t>(-1);
};

class SimpleTokenizer : public Tokenizer {
 public:
    std::unique_ptr<TokenStream>
    token_stream(const std::string& text) override {
        return std::make_unique<SimpleTokenStream>(text);
    }
};

}  // namespace milvus::index::inverted
