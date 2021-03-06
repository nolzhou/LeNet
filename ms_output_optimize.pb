"?=
?

conv1.weight 

cst1 1Load"-Default/network-WithLossCell/_backbone-LeNet52



B8Default/network-WithLossCell/_backbone-LeNet5/Load-op153
?

data 

1 2Conv2D":Default/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d*
kernel_size??*
mode*
out_channel*!
input_names ?:x?:w*%
pad? ? ? ? *
pad_mode*
format:NCHW**
pad_list? ? ? ? *
groups*(
stride????*
group**
dilation????*
output_names ?
:output2
 


BGDefault/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/Conv2D-op154Rconv2d
?

2 3ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names ?
:output*
input_names
 ?:x2
 


BBDefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op155Rrelu
?

3 4MaxPool"BDefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d*
pad_mode*
output_names ?
:output*-
kernel_size????*
format:NCHW*)
strides????*
input_names
 ?:x2
 


BPDefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/MaxPool-op156Rmax_pool
?

conv2.weight 

cst1 5Load"-Default/network-WithLossCell/_backbone-LeNet52



B8Default/network-WithLossCell/_backbone-LeNet5/Load-op158
?

4 

5 6Conv2D":Default/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d*
kernel_size??*
mode*
out_channel*!
input_names ?:x?:w*%
pad? ? ? ? *
pad_mode*
format:NCHW**
pad_list? ? ? ? *
groups*(
stride????*
group**
dilation????*
output_names ?
:output2
 




BGDefault/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d/Conv2D-op157Rconv2d
?

6 7ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names ?
:output*
input_names
 ?:x2
 




BBDefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op159Rrelu
?

7 8MaxPool"BDefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d*
pad_mode*
output_names ?
:output*-
kernel_size????*
format:NCHW*)
strides????*
input_names
 ?:x2
 


BPDefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/MaxPool-op160Rmax_pool
?

8 

cst2 9Reshape"=Default/network-WithLossCell/_backbone-LeNet5/flatten-Flatten*
output_names ?
:output**
input_names ?
:tensor?	:shape2	
 
?BKDefault/network-WithLossCell/_backbone-LeNet5/flatten-Flatten/Reshape-op161
?


fc1.weight 

cst1 10Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2	
x
?BBDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/Load-op163
?

9 

10 11MatMul"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*
output_names ?
:output*
transpose_a *#
input_names ?:x1?:x2*
transpose_x2*
transpose_x1 *
transpose_b2
 
xBDDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MatMul-op162Rmatmul
?

fc1.bias 

cst1 12Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2
xBBDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/Load-op165
?

11 

12 13BiasAdd"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*
output_names ?
:output*
format:NCHW*!
input_names ?:x?:b2
 
xBEDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op164Rbias_add
?

13 14ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names ?
:output*
input_names
 ?:x2
 
xBBDefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op166Rrelu
?


fc2.weight 

cst1 15Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2
T
xBBDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/Load-op168
?

14 

15 16MatMul"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*
output_names ?
:output*
transpose_a *#
input_names ?:x1?:x2*
transpose_x2*
transpose_x1 *
transpose_b2
 
TBDDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MatMul-op167Rmatmul
?

fc2.bias 

cst1 17Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2
TBBDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/Load-op170
?

16 

17 18BiasAdd"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*
output_names ?
:output*
format:NCHW*!
input_names ?:x?:b2
 
TBEDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op169Rbias_add
?

18 19ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names ?
:output*
input_names
 ?:x2
 
TBBDefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op171Rrelu
?


fc3.weight 

cst1 20Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2


TBBDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/Load-op173
?

19 

20 21MatMul"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*
output_names ?
:output*
transpose_a *#
input_names ?:x1?:x2*
transpose_x2*
transpose_x1 *
transpose_b2
 

BDDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MatMul-op172Rmatmul
?

fc3.bias 

cst1 22Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2

BBDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/Load-op175
?

21 

22 23BiasAdd"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*
output_names ?
:output*
format:NCHW*!
input_names ?:x?:b2
 

BEDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op174Rbias_add
?

23 
	
label 24#SparseSoftmaxCrossEntropyWithLogits"CDefault/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits*
output_names ?
:output*-
input_names ?:features?
:labels*
sens-  ??*
is_grad 2BmDefault/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits-op177Rsparse_softmax_cross_entropy
j

24 

23 
	
label 25	MakeTuple"Default2


BDefault/MakeTuple-op176
?

20 

15 

10 

1 

5 

12 

17 

22 26	MakeTuple"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2D@







BGDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MakeTuple-op179
?

cst1 

26 27UpdateState"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2 BIDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/UpdateState-op180
x

25 

27 28Depend"Default*
side_effect_propagate2


BDefault/Depend-op178306_255_construct_wrapper.1358 
data
 

 
 
label
 
fc3.bias



fc3.weight


T
fc2.bias
T

fc2.weight
T
x(
conv2.weight




fc1.bias
x

fc1.weight	
x
?(
conv1.weight



""
28


*
cst1:U*!
cst2? ??????????