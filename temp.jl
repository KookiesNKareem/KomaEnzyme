using CUDA, Enzyme
x = CUDA.ones(100)
f(x) = sum(abs2, x)
grad = gradient(set_runtime_activity(Reverse), f, x)