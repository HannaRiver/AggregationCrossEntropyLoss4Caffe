export PYTHONPATH=./:$PYTHONPATH
export LD_LIBRARY_PATH=./protobuf-3.1.0/build/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./boost-1.58/lib:$LD_LIBRARY_PATH
./model/CRNN_ACE/caffe-lstm-ocr/build/tools/caffe train -gpu 0 \
-solver ./model/CRNN_ACE/model/solver.prototxt \
-weights ./model/CRNN_ACE/model/coc_iter_53000.caffemodel \
2>&1 |tee ./model/CRNN_ACE/logs/try.log
