# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License


add_source_at_current_directory_recursively()
add_library(milvus_segcore OBJECT ${SOURCE_FILES})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CHECK_SORTED ON)
else()
    set(CHECK_SORTED OFF)
endif()

add_definitions(-DCHECK_SORTED=${CHECK_SORTED})