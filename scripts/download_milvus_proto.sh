#!/usr/bin/env bash

SCRIPTS_DIR=$(dirname "$0")
THIRD_PARTY_DIR=$SCRIPTS_DIR/../cmake_build/thirdparty
# Get the actual repo and version after replace rules
API_MODULE=$(go list -m github.com/milvus-io/milvus-proto/go-api/v2)
API_VERSION=$(echo $API_MODULE | awk -F' ' '{print $NF}')
# Extract repo URL from replace rule if exists
if [[ $API_MODULE == *"=>"* ]]; then
  REPO_PATH=$(echo $API_MODULE | awk -F'=>' '{print $2}' | awk '{print $1}')
  # Convert go module path to github URL (e.g., github.com/SpadeA-Tang/milvus-proto/go-api/v2 -> https://github.com/SpadeA-Tang/milvus-proto.git)
  REPO_URL="https://$(echo $REPO_PATH | sed 's|/go-api/v2||').git"
else
  REPO_URL="https://github.com/milvus-io/milvus-proto.git"
fi

if [ ! -d "$THIRD_PARTY_DIR/milvus-proto" ]; then
  mkdir -p $THIRD_PARTY_DIR
  pushd $THIRD_PARTY_DIR
  git clone $REPO_URL milvus-proto
  cd milvus-proto
  # try tagged version first
  COMMIT_ID=$(git ls-remote $REPO_URL refs/tags/${API_VERSION} | cut -f 1)
  if [[ -z $COMMIT_ID ]]; then
    # parse commit from pseudo version (eg v0.0.0-20230608062631-c453ef1b870a => c453ef1b870a)
    COMMIT_ID=$(echo $API_VERSION | awk -F'-' '{print $3}')
  fi
  echo "repo: $REPO_URL, version: $API_VERSION, commitID: $COMMIT_ID"
  if [ -z $COMMIT_ID ]; then
      git checkout -b $API_VERSION $API_VERSION
  else
      git reset --hard $COMMIT_ID
  fi
  popd
fi
