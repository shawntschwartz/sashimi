#!/bin/bash
tmpid="$(docker ps -alq)"
docker cp $tmpid:/app/_outputs/. ./_outputs
docker cp $tmpid:/app/_outputs/. ./_outputs
docker cp $tmpid:/app/_models/. ./_models
docker cp $tmpid:/app/_logs/. ./_logs
docker cp $tmpid:/app/sashimi/logs/. ./sashimi/logs