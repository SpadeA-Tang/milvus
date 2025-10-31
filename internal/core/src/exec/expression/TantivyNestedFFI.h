// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <cstddef>

extern "C" {

/// Search nested documents using a protobuf-encoded query
///
/// Returns: Error message string if failed, NULL if succeeded
char*
tantivy_search_nested(void* reader_ptr,
                      const uint8_t* query_proto_data,
                      size_t query_proto_len,
                      int64_t** result_row_ids,
                      size_t* result_count);

/// Free the row IDs array returned by tantivy_search_nested
void
tantivy_free_row_ids(int64_t* row_ids, size_t count);

/// Free error message string returned by tantivy functions
void
tantivy_free_error(char* error_msg);

}  // extern "C"
