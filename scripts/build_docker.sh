#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

docker build -t xgboost:latest - < "$DIR/Dockerfile-xgboost"
docker build -t xgboost-testing:latest - < "$DIR/Dockerfile-testing"
