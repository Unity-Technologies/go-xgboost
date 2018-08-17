#!/bin/bash

set -e

docker run --rm -it -v "$PWD":/go/src/github.com/Applifier/go-xgboost -w /go/src/github.com/Applifier/go-xgboost xgboost-testing:latest $@