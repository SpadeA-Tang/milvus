// Adapter: bridges milvus::tantivy::Tokenizer (Rust FFI) to our
// inverted::Tokenizer interface.
//
// This file depends on tantivy Rust bindings and can only be compiled
// within the full Milvus build system (not standalone unit tests).

#pragma once

#include <memory>
#include <string>

#include "index/inverted/Tokenizer.h"
#include "tantivy/token-stream.h"
#include "tantivy/tokenizer.h"

namespace milvus::index::inverted {

// Adapts milvus::tantivy::TokenStream → inverted::TokenStream.
class TantivyTokenStreamAdapter : public TokenStream {
 public:
    explicit TantivyTokenStreamAdapter(
        std::unique_ptr<tantivy::TokenStream> inner)
        : inner_(std::move(inner)) {
    }

    bool
    advance() override {
        return inner_->advance();
    }

    const std::string&
    token() const override {
        current_ = inner_->get_token();
        return current_;
    }

 private:
    std::unique_ptr<tantivy::TokenStream> inner_;
    mutable std::string current_;
};

// Adapts milvus::tantivy::Tokenizer → inverted::Tokenizer.
class TantivyTokenizer : public Tokenizer {
 public:
    explicit TantivyTokenizer(const std::string& analyzer_params)
        : inner_(std::string(analyzer_params)) {
    }

    TantivyTokenizer(const std::string& analyzer_params,
                     const std::string& extra_info)
        : inner_(std::string(analyzer_params), std::string(extra_info)) {
    }

    std::unique_ptr<TokenStream>
    token_stream(const std::string& text) override {
        auto stream = inner_.CreateTokenStreamCopyText(text);
        return std::make_unique<TantivyTokenStreamAdapter>(
            std::move(stream));
    }

 private:
    tantivy::Tokenizer inner_;
};

}  // namespace milvus::index::inverted
