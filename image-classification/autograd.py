import torch

def p(msg, init_char='\n'):
    print(f"{init_char}{msg}")

p('Create a tensor and track computations')
x = torch.ones(2, 2, requires_grad=True)
p(x, '')

p('Now we perform some tensor computations which will cause another tensor "y" to be created and by default it will inherit the properties of the parent')
y = x + 2
print(y, '')

p('If we look carefully to y.grad_fn, we can verify that it references the function that created this tensor - and add operation')
p(y.grad_fn, '')

p('Moreover, if we further use this tensor with other operators, we can verify that new gradient functions keep being associated to these functions')
z = y * y * 3
p(z.grad_fn, '')

p('Finally, we combine the results of the tensor z in a single scalar value, which we called out')
out = z.mean()
p(f"z: {z}, \nout: {out}", '')

p('By default, when creating tensors the require_grad flag is False, as we can see below')
a = torch.randn(2, 2)
p(a, '')
a = ((a * 3) / (a - 1))
p(a, 'By default, even upon the combination of multiple operations, this tensor is not tracking the operations done to it, as we can verify: ')
p(a.requires_grad, '')

p('However, we can set the flag in place, to state that from this moment on we want to track the computation history.')
a.requires_grad_(True)
print(a.requires_grad)


p('Gradients')
p('Note however, that previous computations are not considered for a, and therefore will not influence the backprop')
b = (a * a).sum()
print(b.grad_fn)

print("Before backprop")
print(x.grad)

out.backward()
print("After backprop")
print(x.grad)



p("Let's take a look into an example of vector-Jacobian product")
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
    print("1")
print(f"In this case, y is no longer a scalar: {y}")

print(f"if we want to take the jacobian product we have to pass a vector an argument to backward()")
v = torch.tensor([0.1, 1.0, 0.00001], dtype=torch.float)
y.backward(v)
print(x.grad)
