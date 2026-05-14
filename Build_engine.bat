@echo off
trtexec --onnx=movement_gru_best.onnx --saveEngine=movement_gru_best.trt --fp16
python GRU.py
pause