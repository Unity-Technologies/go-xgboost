#!/bin/bash

docker build -t xgboost:latest - < ./scripts/Dockerfile-xgboost
docker build -t xgboost-testing:latest - < ./scripts/Dockerfile-testing