@ECHO OFF
docker ps -alq > tmp_id.txt
set /p msg= < tmp_id.txt
docker cp "%msg%":/app/_outputs/. ./_outputs
del tmp_id.txt