"έΚ
|

conv1.weight 

cst1 1Load"-Default/network-WithLossCell/_backbone-LeNet52



Bconv1.weight


inputs0 
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
BFDefault/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/Conv2D-op77Rconv2d
¨
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
BADefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op78Rrelu
Ν
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
BODefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/MaxPool-op79Rmax_pool
|

conv2.weight 

cst1 5Load"-Default/network-WithLossCell/_backbone-LeNet52



Bconv2.weight
ϊ
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
BFDefault/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d/Conv2D-op80Rconv2d
¨
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
BADefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op81Rrelu
Ν
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
BODefault/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/MaxPool-op82Rmax_pool
β

8 9Reshape"=Default/network-WithLossCell/_backbone-LeNet5/flatten-Flatten*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *"
shape ?????????**
input_names 
:tensor	:shape2	
 
BJDefault/network-WithLossCell/_backbone-LeNet5/flatten-Flatten/Reshape-op83
|


fc1.weight 

cst1 10Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2	
x
B
fc1.weight
ι
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
dst_type'’&2
 
xBCDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MatMul-op84Rmatmul
s

fc1.bias 

cst1 12Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2
xBfc1.bias
Π
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
xBDDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op85Rbias_add
’

13 14ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
input_names
 :x2
 
xBADefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op86Rrelu
{


fc2.weight 

cst1 15Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2
T
xB
fc2.weight
κ
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
dst_type'’&2
 
TBCDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MatMul-op87Rmatmul
s

fc2.bias 

cst1 17Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2
TBfc2.bias
Π
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
TBDDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op88Rbias_add
’

18 19ReLU"7Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
input_names
 :x2
 
TBADefault/network-WithLossCell/_backbone-LeNet5/relu-ReLU/ReLU-op89Rrelu
{


fc3.weight 

cst1 20Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2


TB
fc3.weight
κ
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
dst_type'’&2
 

BCDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/MatMul-op90Rmatmul
s

fc3.bias 

cst1 22Load"7Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense2

Bfc3.bias
Π
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
BDDefault/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op91Rbias_add
Π

23 

inputs1 24#SparseSoftmaxCrossEntropyWithLogits"CDefault/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits*
output_names 
:output*
IsFeatureMapOutput*)
IsFeatureMapInputList	  	 *-
input_names :features
:labels*
sens-  ?*
is_grad 2BlDefault/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits-op92Rsparse_softmax_cross_entropy
‘

23 

inputs1 25#SparseSoftmaxCrossEntropyWithLogits"uGradients/Default/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits/gradSparseSoftmaxCrossEntropyWithLogits*
output_names 
:output*
IsFeatureMapOutput*)
IsFeatureMapInputList	  	 *-
input_names :features
:labels*
sens-  ?*
is_grad2
 

BGradients/Default/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits/gradSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits-op93
Θ

25 

24 26Depend"uGradients/Default/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits/gradSparseSoftmaxCrossEntropyWithLogits*
side_effect_propagate2
 

BGradients/Default/network-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits/gradSparseSoftmaxCrossEntropyWithLogits/Depend-op94


26 

20 27MatMul"LGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul*"
IsFeatureMapInputList		  *!
right_format:DefaultFormat*#
input_names :x1:x2*
transpose_x2 *
transpose_b * 
left_format:DefaultFormat*
output_names 
:output*
transpose_a *
IsFeatureMapOutput*
transpose_x1 *
dst_type'’&2
 
TBXGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul/MatMul-op95
κ

27 

19 28ReluGrad"JGradients/Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU/gradReLU*
output_names 
:output*
IsFeatureMapOutput*)
IsFeatureMapInputList	  	 **
input_names :
y_backprop:x2
 
TBXGradients/Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU/gradReLU/ReluGrad-op96


28 

15 29MatMul"LGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul*"
IsFeatureMapInputList		  *!
right_format:DefaultFormat*#
input_names :x1:x2*
transpose_x2 *
transpose_b * 
left_format:DefaultFormat*
output_names 
:output*
transpose_a *
IsFeatureMapOutput*
transpose_x1 *
dst_type'’&2
 
xBXGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul/MatMul-op97
κ

29 

14 30ReluGrad"JGradients/Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU/gradReLU*
output_names 
:output*
IsFeatureMapOutput*)
IsFeatureMapInputList	  	 **
input_names :
y_backprop:x2
 
xBXGradients/Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU/gradReLU/ReluGrad-op98


30 

10 31MatMul"LGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul*"
IsFeatureMapInputList		  *!
right_format:DefaultFormat*#
input_names :x1:x2*
transpose_x2 *
transpose_b * 
left_format:DefaultFormat*
output_names 
:output*
transpose_a *
IsFeatureMapOutput*
transpose_x1 *
dst_type'’&2	
 
BXGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul/MatMul-op99


31 32Reshape"SGradients/Default/network-WithLossCell/_backbone-LeNet5/flatten-Flatten/gradReshape*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *'
shape **
input_names 
:tensor	:shape2
 


BaGradients/Default/network-WithLossCell/_backbone-LeNet5/flatten-Flatten/gradReshape/Reshape-op100
Ή

7 

8 

32 33MaxPoolGrad"XGradients/Default/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/gradMaxPool*0
IsFeatureMapInputList	  	 	 *-
kernel_size*)
strides*<
input_names- :x_origin:
out_origin:grad*
pad_mode	:VALID*
output_names 
:output*
IsFeatureMapOutput*
format:NCHW2
 




BjGradients/Default/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/gradMaxPool/MaxPoolGrad-op101
ς

33 

7 34ReluGrad"JGradients/Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU/gradReLU*
output_names 
:output*
IsFeatureMapOutput*)
IsFeatureMapInputList	  	 **
input_names :
y_backprop:x2
 




BYGradients/Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU/gradReLU/ReluGrad-op102


34 

5 35Conv2DBackpropInput"OGradients/Default/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d/gradConv2D*"
IsFeatureMapInputList		  *
kernel_size*
mode*
out_channel*C
input_names4 :out_backprop
:filter:input_sizes*%
pad    *
pad_mode	:VALID*
format:NCHW**
pad_list    *
groups*(
stride*
group**
dilation*
output_names 
:output*
IsFeatureMapOutput*-
input_sizes 2
 


BiGradients/Default/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d/gradConv2D/Conv2DBackpropInput-op103
Ή

3 

4 

35 36MaxPoolGrad"XGradients/Default/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/gradMaxPool*0
IsFeatureMapInputList	  	 	 *-
kernel_size*)
strides*<
input_names- :x_origin:
out_origin:grad*
pad_mode	:VALID*
output_names 
:output*
IsFeatureMapOutput*
format:NCHW2
 


BjGradients/Default/network-WithLossCell/_backbone-LeNet5/max_pool2d-MaxPool2d/gradMaxPool/MaxPoolGrad-op104
ς

36 

3 37ReluGrad"JGradients/Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU/gradReLU*
output_names 
:output*
IsFeatureMapOutput*)
IsFeatureMapInputList	  	 **
input_names :
y_backprop:x2
 


BYGradients/Default/network-WithLossCell/_backbone-LeNet5/relu-ReLU/gradReLU/ReluGrad-op105


37 

inputs0 38Conv2DBackpropFilter"OGradients/Default/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/gradConv2D*)
IsFeatureMapInputList	  	 *
kernel_size*
mode*
out_channel*C
input_names4 :out_backprop	:input:filter_sizes*%
pad    *.
filter_sizes*
pad_mode	:VALID*
format:NCHW**
pad_list    *
groups*
stride*
group**
dilation*
output_names 
:output*
IsFeatureMapOutput2



BjGradients/Default/network-WithLossCell/_backbone-LeNet5/conv1-Conv2d/gradConv2D/Conv2DBackpropFilter-op106
΅

22 

17 

12 

5 

1 

10 

15 

20 39	MakeTuple"Default2D@







BDefault/MakeTuple-op107
Q

cst1 

39 

23 40UpdateState"Default2 BDefault/UpdateState-op108
κ

26 41BiasAddGrad"MGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradBiasAdd*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
format:NCHW*
input_names :dout2

B_Gradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradBiasAdd/BiasAddGrad-op109
ξ

fc3.bias 

moments.fc3.bias 

learning_rate 

41 

momentum 

40 42ApplyMomentum"Default/optimizer-Momentum*"
IsFeatureMapInputList		 *
side_effect_mem*e
input_namesV :variable:accumulation:learning_rate:gradient:momentum*
output_names 
:output*
IsFeatureMapOutput*
use_nesterov *
use_locking *
gradient_scale-  ?2

B.Default/optimizer-Momentum/ApplyMomentum-op110Ropt
}

40 

42 

21 

27 43UpdateState"Default/optimizer-Momentum2 B,Default/optimizer-Momentum/UpdateState-op111


26 

19 44MatMul"LGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul*)
IsFeatureMapInputList	  	 *!
right_format:DefaultFormat*#
input_names :x1:x2*
transpose_x2 *
transpose_b * 
left_format:DefaultFormat*
output_names 
:output*
transpose_a*
IsFeatureMapOutput*
transpose_x1*
dst_type'’&2


TBYGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul/MatMul-op112
φ


fc3.weight 

moments.fc3.weight 

learning_rate 

44 

momentum 

43 45ApplyMomentum"Default/optimizer-Momentum*"
IsFeatureMapInputList		 *
side_effect_mem*e
input_namesV :variable:accumulation:learning_rate:gradient:momentum*
output_names 
:output*
IsFeatureMapOutput*
use_nesterov *
use_locking *
gradient_scale-  ?2


TB.Default/optimizer-Momentum/ApplyMomentum-op113Ropt
u

43 

45 

18 46UpdateState"Default/optimizer-Momentum2 B,Default/optimizer-Momentum/UpdateState-op114
κ

28 47BiasAddGrad"MGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradBiasAdd*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
format:NCHW*
input_names :dout2
TB_Gradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradBiasAdd/BiasAddGrad-op115
ξ

fc2.bias 

moments.fc2.bias 

learning_rate 

47 

momentum 

46 48ApplyMomentum"Default/optimizer-Momentum*"
IsFeatureMapInputList		 *
side_effect_mem*e
input_namesV :variable:accumulation:learning_rate:gradient:momentum*
output_names 
:output*
IsFeatureMapOutput*
use_nesterov *
use_locking *
gradient_scale-  ?2
TB.Default/optimizer-Momentum/ApplyMomentum-op116Ropt
}

46 

48 

16 

29 49UpdateState"Default/optimizer-Momentum2 B,Default/optimizer-Momentum/UpdateState-op117


28 

14 50MatMul"LGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul*)
IsFeatureMapInputList	  	 *!
right_format:DefaultFormat*#
input_names :x1:x2*
transpose_x2 *
transpose_b * 
left_format:DefaultFormat*
output_names 
:output*
transpose_a*
IsFeatureMapOutput*
transpose_x1*
dst_type'’&2
T
xBYGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul/MatMul-op118
φ


fc2.weight 

moments.fc2.weight 

learning_rate 

50 

momentum 

49 51ApplyMomentum"Default/optimizer-Momentum*"
IsFeatureMapInputList		 *
side_effect_mem*e
input_namesV :variable:accumulation:learning_rate:gradient:momentum*
output_names 
:output*
IsFeatureMapOutput*
use_nesterov *
use_locking *
gradient_scale-  ?2
T
xB.Default/optimizer-Momentum/ApplyMomentum-op119Ropt
u

49 

51 

13 52UpdateState"Default/optimizer-Momentum2 B,Default/optimizer-Momentum/UpdateState-op120
κ

30 53BiasAddGrad"MGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradBiasAdd*
output_names 
:output*
IsFeatureMapOutput*"
IsFeatureMapInputList		  *
format:NCHW*
input_names :dout2
xB_Gradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradBiasAdd/BiasAddGrad-op121
ξ

fc1.bias 

moments.fc1.bias 

learning_rate 

53 

momentum 

52 54ApplyMomentum"Default/optimizer-Momentum*"
IsFeatureMapInputList		 *
side_effect_mem*e
input_namesV :variable:accumulation:learning_rate:gradient:momentum*
output_names 
:output*
IsFeatureMapOutput*
use_nesterov *
use_locking *
gradient_scale-  ?2
xB.Default/optimizer-Momentum/ApplyMomentum-op122Ropt
}

52 

54 

11 

31 55UpdateState"Default/optimizer-Momentum2 B,Default/optimizer-Momentum/UpdateState-op123


30 

9 56MatMul"LGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul*)
IsFeatureMapInputList	  	 *!
right_format:DefaultFormat*#
input_names :x1:x2*
transpose_x2 *
transpose_b * 
left_format:DefaultFormat*
output_names 
:output*
transpose_a*
IsFeatureMapOutput*
transpose_x1*
dst_type'’&2	
x
BYGradients/Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/gradMatMul/MatMul-op124
χ


fc1.weight 

moments.fc1.weight 

learning_rate 

56 

momentum 

55 57ApplyMomentum"Default/optimizer-Momentum*"
IsFeatureMapInputList		 *
side_effect_mem*e
input_namesV :variable:accumulation:learning_rate:gradient:momentum*
output_names 
:output*
IsFeatureMapOutput*
use_nesterov *
use_locking *
gradient_scale-  ?2	
x
B.Default/optimizer-Momentum/ApplyMomentum-op125Ropt
|

55 

57 

6 

35 58UpdateState"Default/optimizer-Momentum2 B,Default/optimizer-Momentum/UpdateState-op126


34 

4 59Conv2DBackpropFilter"OGradients/Default/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d/gradConv2D*)
IsFeatureMapInputList	  	 *
kernel_size*
mode*
out_channel*C
input_names4 :out_backprop	:input:filter_sizes*%
pad    *.
filter_sizes*
pad_mode	:VALID*
format:NCHW**
pad_list    *
groups*
stride*
group**
dilation*
output_names 
:output*
IsFeatureMapOutput2



BjGradients/Default/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d/gradConv2D/Conv2DBackpropFilter-op127


conv2.weight 

moments.conv2.weight 

learning_rate 

59 

momentum 

58 60ApplyMomentum"Default/optimizer-Momentum*"
IsFeatureMapInputList		 *
side_effect_mem*e
input_namesV :variable:accumulation:learning_rate:gradient:momentum*
output_names 
:output*
IsFeatureMapOutput*
use_nesterov *
use_locking *
gradient_scale-  ?2



B.Default/optimizer-Momentum/ApplyMomentum-op128Ropt
t

58 

60 

2 61UpdateState"Default/optimizer-Momentum2 B,Default/optimizer-Momentum/UpdateState-op129


conv1.weight 

moments.conv1.weight 

learning_rate 

38 

momentum 

61 62ApplyMomentum"Default/optimizer-Momentum*"
IsFeatureMapInputList		 *
side_effect_mem*e
input_namesV :variable:accumulation:learning_rate:gradient:momentum*
output_names 
:output*
IsFeatureMapOutput*
use_nesterov *
use_locking *
gradient_scale-  ?2



B.Default/optimizer-Momentum/ApplyMomentum-op130Ropt


cst2 

62 63Depend"Default/optimizer-Momentum*
side_effect_propagate2B'Default/optimizer-Momentum/Depend-op145


cst3 

60 64Depend"Default/optimizer-Momentum*
side_effect_propagate2B'Default/optimizer-Momentum/Depend-op146


cst4 

57 65Depend"Default/optimizer-Momentum*
side_effect_propagate2B'Default/optimizer-Momentum/Depend-op147


cst5 

54 66Depend"Default/optimizer-Momentum*
side_effect_propagate2B'Default/optimizer-Momentum/Depend-op148


cst6 

51 67Depend"Default/optimizer-Momentum*
side_effect_propagate2B'Default/optimizer-Momentum/Depend-op149


cst7 

48 68Depend"Default/optimizer-Momentum*
side_effect_propagate2B'Default/optimizer-Momentum/Depend-op150


cst8 

45 69Depend"Default/optimizer-Momentum*
side_effect_propagate2B'Default/optimizer-Momentum/Depend-op151


cst9 

42 70Depend"Default/optimizer-Momentum*
side_effect_propagate2B'Default/optimizer-Momentum/Depend-op152
½

63 

64 

65 

66 

67 

68 

69 

70 71	MakeTuple"Default/optimizer-Momentum2$ 







B*Default/optimizer-Momentum/MakeTuple-op139
b

24 

71 72Depend"Default*
side_effect_propagate2BDefault/Depend-op140
m

61 

62 73UpdateState"Default/optimizer-Momentum2 B,Default/optimizer-Momentum/UpdateState-op141
b

72 

73 74Depend"Default*
side_effect_propagate2BDefault/Depend-op142
=

74 75	MakeTuple"Default2"BDefault/MakeTuple-op143kernel_graph_0
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
x
fc1.bias
x

fc1.weight	
x
(
conv2.weight



(
conv1.weight



#
inputs0
 

 
 
inputs1
  
moments.fc3.bias


learning_rate
momentum&
moments.fc3.weight


T 
moments.fc2.bias
T&
moments.fc2.weight
T
x 
moments.fc1.bias
x'
moments.fc1.weight	
x
0
moments.conv2.weight



0
moments.conv1.weight



"
75"*
cst1:U*
cst2B*
cst3B*
cst4B*
cst5B*
cst6B*
cst7B*
cst8B*
cst9B