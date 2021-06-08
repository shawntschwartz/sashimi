@ECHO OFF
docker ps -alq > tmp_id.txt
set /p msg= < tmp_id.txt
docker cp "%msg%":/app/_outputs/. ./_outputs
docker cp "%msg%":/app/_models/. ./_models
docker cp "%msg%":/app/_logs/. ./_logs
docker cp "%msg%":/app/sashimi/logs/. ./sashimi/logs
del tmp_id.txt