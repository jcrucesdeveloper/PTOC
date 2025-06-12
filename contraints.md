## 1. torch.range

Returns a 1-D tensor of size ⌊(end−start)/step⌋+1 with values from start to end with step step. Step is the gap between two values in the tensor.

**Parameters:**

- `start` (`float`, optional): The starting value for the set of points. Default: 0
- `end` (`float`): The ending value for the set of points
- `step` (`float`, optional): The gap between each pair of adjacent points. Default: 1

### Errors

- RuntimeError: step must be nonzero

```python
torch.range(1, 5, 0)
>>> RuntimeError: step must be nonzero
```

- RuntimeError: upper bound and lower bound inconsistent with step sign

```python
torch.range(1, -3, 1)
>>> RuntimeError: upper bound and lower bound inconsistent with step sign

torch.range(1, 3,  -1)
>>> RuntimeError: upper bound and lower bound inconsistent with step sign
```

### Constraints

```
range c1 c2 c3 -> TT_output
- c3 != 0
- (c2 - c1) * c3 > 0
```

## 2. torch.Tensor.size

torch.Size is the result type of a call to torch.Tensor.size(). It describes the size of all dimensions of the original tensor. As a subclass of tuple, it supports common sequence operations like indexing and length.

**Parameters:**
No parameters

### Errors

Operations without parameters no error

### Contraints

...

## 3. torch.nn.Linear

Applies an affine linear transformation to the incoming data: y = x(A^T) + b

This module supports TensorFloat32.

On certain ROCm devices, when using float16 inputs this module will use different precision for backward.

**Parameters:**

- `in_features` (`int`) – size of each input sample
- `out_features` (`int`) – size of each output sample
- `bias` (`bool`) – If set to False, the layer will not learn an additive bias. Default: True

### Errors

- RuntimeError: Trying to create tensor with negative dimension -1: [20, -1]

```python
nn.Linear(-1,20)
>>> RuntimeError: Trying to create tensor with negative dimension -1: [20, -1]

nn.Linear(1,-1)
>>> RuntimeError: Trying to create tensor with negative dimension -1: [-1, 1]
```

### Contraints

linear c1 c2 b -> TT_output

- c1 > 0
- c2 > 0

## 4. torch.reshape

Returns a tensor with the same data and number of elements as input, but with the specified shape. When possible, the returned tensor will be a view of input. Otherwise, it will be a copy. Contiguous inputs and inputs with compatible strides can be reshaped without copying, but you should not depend on the copying vs. viewing behavior.

**Parameters:**

- `input` (`Tensor`): The tensor to be reshaped
- `shape` (`tuple of int`): The new shape

### Errors

- RuntimeError: All other dimensions must be positive integers.

```python
t = torch.zeros(4)
torch.reshape(t, (-2,2))
>>> RuntimeError: invalid shape dimension -2
```

- RuntimeError: Only one dimension can be -1 (PyTorch will infer its value).

```python
t = torch.zeros(4)
torch.reshape(t, (-1,-1))
>>> RuntimeError: only one dimension can be inferred
```

- RuntimeError: The product of the new shape's dimensions should be equal to the product of the original shape's dimensions

```python
t = torch.zeros(4, 2)
torch.reshape(t, (4, 3))
>>> RuntimeError: shape '[4, 3]' is invalid for input of size 8
```

### Constraints

reshape TT_input[ti_1, ... ,t1_n] -> TT_output[to_1, ... to_2]

- for each t in TT_input: d > 0 || d == -1
- count(d == -1 for d in shape) <= 1 # Only one dimension can be -1
- product(TT_input) == product(TT_output) # Total elements must match

## 6. torch.transpose

Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.

**Parameters:**

- `input` (`Tensor`): The input tensor.
- `dim0` (`int`): The first dimension to be transposed
- `dim1` (`int`): The second dimension to be transposed

### Errors

- IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)

```python
x = torch.randn(2, 3)  # Shape (2,3)
torch.transpose(x, 0, 2)  # dim 2 is out of range [-2, 1]
>>> IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

### Constraints

transpose TT_input[ti_1, ... ,ti_n] dim0 dim1 -> TT_output[to_1, ... ,to_n]

- dim0 >= -n && dim0 < n
- dim1 >= -n && dim1 < n

## 5. torch.flatten

Flattens input by reshaping it into a one-dimensional tensor. If start_dim or end_dim are passed, only dimensions starting with start_dim and ending with end_dim are flattened. The order of elements in input is unchanged.

**Parameters:**

- `input` (`Tensor`): The input tensor
- `start_dim` (`int`): The first dimension to flatten
- `end_dim` (`int`): The last dimension to flatten

### Errors

- IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 10)

```python
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
torch.flatten(t, start_dim=10)
>>> IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 10)

torch.flatten(t, start_dim=2,end_dim=3)
>>> IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
```

### Constraints

flatten

- start_dim >= -n && start_dim < n
- end_dim >= -n && end_dim < n
- start_dim <= end_dim

## 6. torch.cat

Concatenates the given sequence of tensors in tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be a 1-D empty tensor with size (0,).

**Parameters:**

- `tensors` (`sequence of Tensors`): Non-empty tensors provided must have the same shape, except in the cat dimension
- `dim` (`int`, optional): The dimension along which the tensors are concatenated

### Errors

- IndexError: Dimensions must be in range [-n, n - 1] where n is the number of dimensions

```python
x = torch.randn(2, 3)  # Shape (2,3)
y = torch.randn(2, 3)  # Shape (2,3)
torch.cat((x, y), dim=2)  # dim 2 is out of range [-2, 1]
>>> IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

- RuntimeError: All tensors must have the same shape except in the concatenating dimension

```python
x = torch.randn(2, 3)  # Shape (2,3)
y = torch.randn(2, 4)  # Shape (2,4)
torch.cat((x, y), dim=0)
>>> RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 3 but got size 4 for tensor number 1 in the list.
```

### Constraints

cat [TT_input_1[ti_1, ... ,ti_n], ... ,TT_input_m[ti_1, ... ,ti_n]] dim -> TT_output[to_1, ... ,to_n]

- dim >= -n && dim < n
- for all i,j in [1..m]:
  - TT_input_i.shape[k] == TT_input_j.shape[k] for all k != dim
  - or (TT_input_i.shape == (0,) && TT_input_j.shape == (0,))

## 7. torch.sum

Returns the sum of all elements in the input tensor.

**Parameters:**

- `input` (`Tensor`): The input tensor.
- `dim` (`int` or `tuple of ints`, optional): The dimension or dimensions to reduce. If None, all dimensions are reduced

### Errors

- IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)

```python
x = torch.randn(2, 3)  # Shape (2,3)
torch.sum(x,2)
```

...

### Constraints

...

## 8. torch.split

Splits the tensor into chunks. Each chunk is a view of the original tensor.

If split_size_or_sections is an integer type, then tensor will be split into equally sized chunks (if possible). Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.

If split_size_or_sections is a list, then tensor will be split into len(split_size_or_sections) chunks with sizes in dim according to split_size_or_sections.

### Errors

- RuntimeError: split_size can only be 0 if dimension size is 0, but got dimension size of 5

```python
x = torch.arange(10).reshape(5, 2)
torch.split(x, 0)
>>> RuntimeError: split expects split_size be non-negative, but got split_size=-1
```

- RuntimeError: split expects split_size be non-negative, but got split_size=-1

```python
x = torch.arange(10).reshape(5, 2)
torch.split(x, -1)
>>> RuntimeError: split expects split_size be non-negative, but got split_size=-1
```

- RuntimeError: split_with_sizes expects split_sizes to sum exactly to 2 (input tensor's size at dimension 1), but got split_sizes=[1, 10]

```python
x = torch.arange(10).reshape(5, 2)

# Without parameter dim
torch.split(x,[1,10], 1)
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 5 (input tensor's size at dimension 0), but got split_sizes=[1, 10]

# With parameter dim
torch.split(x,[1,10], 1)
>>>  RuntimeError: split_with_sizes expects split_sizes to sum exactly to 2 (input tensor's size at dimension 1), but got split_sizes=[1, 10]
```

## 9. torch.unsqueeze

Adds a dimension of size 1 at the specified position.

**Parameters:**

- `input` (`Tensor`): The input tensor.
- `dim` (`int`): The index at which to insert the singleton dimension.

### Errors

- IndexError: Dimension must be in range [-n - 1, n] where n is the number of dimensions

```python
x = torch.tensor([1, 2, 3, 4])  # Shape (4)
torch.unsqueeze(x, 2)  # dim 2 is out of range [-2, 1]
>>> IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

## 10. torch.zeros

Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
**Parameters:**

- `size` (`int...`): A sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.

### Errors

- RuntimeError: Trying to create tensor with negative dimension -3: [-3, 2]

```python
x = torch.zeros(-3,2)
>>> RuntimeError: Trying to create tensor with negative dimension -3: [-3, 2]
```

### Constraints

...

## 11. torch.arange

Returns a 1-D tensor of size ⌈(end-start)/step⌉ with values from the interval [start, end) taken with common difference step beginning from start.

The sequence follows: outᵢ₊₁ = outᵢ + step

Important notes:

- With floating-point dtypes (especially bfloat16), results may have rounding errors
- Some values may not be exactly representable, causing repeated or unexpected rounding
- For precise sequences, use integer dtypes instead of floating-point
- With non-integer step, subtract a small epsilon from end to avoid floating point comparison issues

**Parameters:**

- `start` (`Number`, optional): The starting value for the set of points. Default: 0.
- `end` (`Number`): The ending value for the set of points.
- `step` (`Number`, optional): The gap between each pair of adjacent points. Default: 1.

### Errors

- RuntimeError: upper bound and larger bound inconsistent with step sign

```python
x = torch.arange(-1)
>>>RuntimeError: upper bound and larger bound inconsistent with step sign

x = torch.arange(-1, -3, 1)
>>>RuntimeError: upper bound and larger bound inconsistent with step sign
```

- RuntimeError: step must be nonzero

```python
x. = torch.arange(1,2,0)
>>>RuntimeError: step must be nonzero
```

### Constraints

...

## 13. torch.ones

Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.

**Parameters:**

- `size` (`int...`) - A sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.

### Errors

- RuntimeError: Trying to create tensor with negative dimension -1: [-1]

```python
x = torch.ones(-1)
- RuntimeError: Trying to create tensor with negative dimension -1: [-1]
```

### Constraints

...

## 14. torch.max

Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).

If keepdim is True, the output tensors are of the same size as input except in the dimension dim where they are of size 1. Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensors having 1 fewer dimension than input.

**Parameters**

- `input` (`Tensor`) - The input tensor
- `dim` (`int` or `tuple of ints`, optional) - The dimension or dimensions to reduce. If None, all dimensions are reduced

### Errors

- IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 4)

```python
x = torch.rand(3,2,1)
torch.sum(x,4)
>>> IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 4)
```

## 15. torch.any

Tests if any element in input evaluates to True.

For each row of input in the given dimension dim, returns True if any element in the row evaluate to True and False otherwise.

If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).

**Parameters:**

- `input` (`Tensor`): The input tensor
- `dim` (`int` or `tuple of ints`): The dimension or dimensions to reduce
- `keepdim` (`bool`): Whether the output tensor has dim retained or not

### Errors

- IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)

```python
x = torch.rand(3,2,1)
torch.any(x,3)
>>> IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)

x = torch.rand(3,2,1)
torch.any(x,3)
>>> IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
```

### Constraints

..

## 16. torch.nn.LayerNorm

## 17. torch.squeeze

## 18. torch.expand

## 19. torch.allclose

## 20. torch.nn.Dropout

## 21. torch.permute

## 22. torch.mean

## 23. torch.min

## 24. torch.all

## 25. torch.matmul

## 26. torch.stack

## 27. torch.log

## 28. torch.flatten

## 29. torch.Embedding

## 30. torch.sqrt
