#!/bin/bash
tmpid="$(docker ps -alq)"
docker cp tmpid:/app/_outputs/. ./_outputs