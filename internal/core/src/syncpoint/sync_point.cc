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
//
// Adapted from RocksDB's SyncPoint implementation

#include "syncpoint/sync_point.h"

#include <fcntl.h>

#include "syncpoint/sync_point_impl.h"


#ifdef MILVUS_FAILPOINT
namespace milvus {

SyncPoint*
SyncPoint::GetInstance() {
    static SyncPoint sync_point;
    return &sync_point;
}

SyncPoint::SyncPoint() : impl_(new Data) {
}

SyncPoint::~SyncPoint() {
    delete impl_;
}

void
SyncPoint::LoadDependency(const std::vector<SyncPointPair>& dependencies) {
    impl_->LoadDependency(dependencies);
}

void
SyncPoint::LoadDependencyAndMarkers(
    const std::vector<SyncPointPair>& dependencies,
    const std::vector<SyncPointPair>& markers) {
    impl_->LoadDependencyAndMarkers(dependencies, markers);
}

void
SyncPoint::SetCallBack(const std::string& point,
                       const std::function<void(void*)>& callback) {
    impl_->SetCallBack(point, callback);
}

void
SyncPoint::ClearCallBack(const std::string& point) {
    impl_->ClearCallBack(point);
}

void
SyncPoint::ClearAllCallBacks() {
    impl_->ClearAllCallBacks();
}

void
SyncPoint::EnableProcessing() {
    impl_->EnableProcessing();
}

void
SyncPoint::DisableProcessing() {
    impl_->DisableProcessing();
}

void
SyncPoint::ClearTrace() {
    impl_->ClearTrace();
}

void
SyncPoint::BlockAtPoint(const std::string& point) {
    impl_->BlockAtPoint(point);
}

void
SyncPoint::UnblockPoint(const std::string& point) {
    impl_->UnblockPoint(point);
}

void
SyncPoint::ClearAllBlockedPoints() {
    impl_->ClearAllBlockedPoints();
}

void
SyncPoint::Reset() {
    impl_->Reset();
}

void
SyncPoint::Process(const Slice& point, void* cb_arg) {
    impl_->Process(point, cb_arg);
}

}  // namespace milvus
#endif  // MILVUS_FAILPOINT