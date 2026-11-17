Write a Free Pascal program using tch-fpc and `TchJit` that loads the
TorchScript module `tests/jit/models/foo1.pt` (path relative to the
current directory, which is the tch-fpc checkout), calls its forward
method with two 0-dimensional Int64 tensors 42 and 1337, and prints the
Int64 result:

    result = 1421
