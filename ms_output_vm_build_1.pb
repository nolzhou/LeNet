"áD
|

conv1.weight 

cst1 1Load"-Default/network-WithLossCell/_backbone-LeNet52



Bconv1.weight
ţ

data 

1 2Conv2D":Default/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d*"
IsFeatureMapInputList		  *
kernel_size*
mode*
out_channel*!
input_names :x:w*%
pad    *
pad_mode	:valid*
format:NCHW**
pad_list    *
groups*(
stride*
group**
dilation*
output_names 
:output*
IsFeatureMapOutput2
 


BGDefault/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/Conv2D-op182Rconv2d
Š

2 3ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
input_names
 :x2
 


BBDefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op183Rrelu
Î

3 4MaxPool"BDefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d*"
IsFeatureMapInputList		  *-
kernel_size*)
strides*
input_names
 :x*
pad_mode	:VALID*
output_names 
:output*
IsFeatureMapOutput*
format:NCHW2
 


BPDefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/MaxPool-op184Rmax_pool
|

conv2.weight 

cst1 5Load"-Default/network-WithLossCell/_backbone-LeNet52



Bconv2.weight
ű

4 

5 6Conv2D":Default/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d*"
IsFeatureMapInputList		  *
kernel_size*
mode*
out_channel*!
input_names :x:w*%
pad    *
pad_mode	:valid*
format:NCHW**
pad_list    *
groups*(
stride*
group**
dilation*
output_names 
:output*
IsFeatureMapOutput2
 




BGDefault/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d/Conv2D-op185Rconv2d
Š

6 7ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
input_names
 :x2
 




BBDefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op186Rrelu
Î

7 8MaxPool"BDefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d*"
IsFeatureMapInputList		  *-
kernel_size*)
strides*
input_names
 :x*
pad_mode	:VALID*
output_names 
:output*
IsFeatureMapOutput*
format:NCHW2
 


BPDefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/MaxPool-op187Rmax_pool
ă

8 9Reshape"=Default/network-WithLossCell/_backbone-LeNet5/flatten-Flatten*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *"
shape ˙˙˙˙˙˙˙˙˙**
input_names 
:tensor	:shape2	
 
BKDefault/network-WithLossCell/_backbone-LeNet5/flatten-Flatten/Reshape-op188
|


fc1.weight 

cst1 10Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2	
x
B
fc1.weight
ę

9 

10 11MatMul"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*"
IsFeatureMapInputList		  *!
right_format:DefaultFormat*#
input_names :x1:x2*
transpose_x2*
transpose_b* 
left_format:DefaultFormat*
output_names 
:output*
transpose_a *
IsFeatureMapOutput*
transpose_x1 *
dst_type'˘&2
 
xBDDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MatMul-op189Rmatmul
s

fc1.bias 

cst1 12Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2
xBfc1.bias
Ń

11 

12 13BiasAdd"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
format:NCHW*!
input_names :x:b2
 
xBEDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op190Rbias_add
Ł

13 14ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
input_names
 :x2
 
xBBDefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op191Rrelu
{


fc2.weight 

cst1 15Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2
T
xB
fc2.weight
ë

14 

15 16MatMul"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*"
IsFeatureMapInputList		  *!
right_format:DefaultFormat*#
input_names :x1:x2*
transpose_x2*
transpose_b* 
left_format:DefaultFormat*
output_names 
:output*
transpose_a *
IsFeatureMapOutput*
transpose_x1 *
dst_type'˘&2
 
TBDDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MatMul-op192Rmatmul
s

fc2.bias 

cst1 17Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2
TBfc2.bias
Ń

16 

17 18BiasAdd"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
format:NCHW*!
input_names :x:b2
 
TBEDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op193Rbias_add
Ł

18 19ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
input_names
 :x2
 
TBBDefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op194Rrelu
{


fc3.weight 

cst1 20Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2


TB
fc3.weight
ë

19 

20 21MatMul"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*"
IsFeatureMapInputList		  *!
right_format:DefaultFormat*#
input_names :x1:x2*
transpose_x2*
transpose_b* 
left_format:DefaultFormat*
output_names 
:output*
transpose_a *
IsFeatureMapOutput*
transpose_x1 *
dst_type'˘&2
 

BDDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MatMul-op195Rmatmul
s

fc3.bias 

cst1 22Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2

Bfc3.bias
Ń

21 

22 23BiasAdd"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
format:NCHW*!
input_names :x:b2
 

BEDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op196Rbias_add
Ď

23 
	
label 24#SparseSoftmaxCrossEntropyWithLogits"CDefault/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits*
output_names 
:output*
IsFeatureMapOutput*)
IsFeatureMapInputList	  	 *-
input_names :features
:labels*
sens-  ?*
is_grad 2BmDefault/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits-op197Rsparse_softmax_cross_entropy
j

24 

23 
	
label 25	MakeTuple"Default2


BDefault/MakeTuple-op198

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
BGDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MakeTuple-op199
Š

cst1 

26 27UpdateState"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2 BIDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/UpdateState-op200
x

25 

27 28Depend"Default*
side_effect_propagate2


BDefault/Depend-op201
=

28 29	MakeTuple"Default2"BDefault/MakeTuple-op202kernel_graph_1
fc3.bias


fc2.bias
T
fc1.bias
x(
conv2.weight



(
conv1.weight





fc1.weight	
x


fc2.weight
T
x

fc3.weight


T 
data
 

 
 
label
 "
29"*
cst1:U