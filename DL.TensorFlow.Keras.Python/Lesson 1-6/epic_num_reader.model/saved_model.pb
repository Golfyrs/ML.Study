��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108��
�
sequential_24/dense_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_namesequential_24/dense_72/kernel
�
1sequential_24/dense_72/kernel/Read/ReadVariableOpReadVariableOpsequential_24/dense_72/kernel* 
_output_shapes
:
��*
dtype0
�
sequential_24/dense_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namesequential_24/dense_72/bias
�
/sequential_24/dense_72/bias/Read/ReadVariableOpReadVariableOpsequential_24/dense_72/bias*
_output_shapes	
:�*
dtype0
�
sequential_24/dense_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*.
shared_namesequential_24/dense_73/kernel
�
1sequential_24/dense_73/kernel/Read/ReadVariableOpReadVariableOpsequential_24/dense_73/kernel* 
_output_shapes
:
��*
dtype0
�
sequential_24/dense_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namesequential_24/dense_73/bias
�
/sequential_24/dense_73/bias/Read/ReadVariableOpReadVariableOpsequential_24/dense_73/bias*
_output_shapes	
:�*
dtype0
�
sequential_24/dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*.
shared_namesequential_24/dense_74/kernel
�
1sequential_24/dense_74/kernel/Read/ReadVariableOpReadVariableOpsequential_24/dense_74/kernel*
_output_shapes
:	�
*
dtype0
�
sequential_24/dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namesequential_24/dense_74/bias
�
/sequential_24/dense_74/bias/Read/ReadVariableOpReadVariableOpsequential_24/dense_74/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
$Adam/sequential_24/dense_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*5
shared_name&$Adam/sequential_24/dense_72/kernel/m
�
8Adam/sequential_24/dense_72/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/sequential_24/dense_72/kernel/m* 
_output_shapes
:
��*
dtype0
�
"Adam/sequential_24/dense_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/sequential_24/dense_72/bias/m
�
6Adam/sequential_24/dense_72/bias/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_24/dense_72/bias/m*
_output_shapes	
:�*
dtype0
�
$Adam/sequential_24/dense_73/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*5
shared_name&$Adam/sequential_24/dense_73/kernel/m
�
8Adam/sequential_24/dense_73/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/sequential_24/dense_73/kernel/m* 
_output_shapes
:
��*
dtype0
�
"Adam/sequential_24/dense_73/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/sequential_24/dense_73/bias/m
�
6Adam/sequential_24/dense_73/bias/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_24/dense_73/bias/m*
_output_shapes	
:�*
dtype0
�
$Adam/sequential_24/dense_74/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*5
shared_name&$Adam/sequential_24/dense_74/kernel/m
�
8Adam/sequential_24/dense_74/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/sequential_24/dense_74/kernel/m*
_output_shapes
:	�
*
dtype0
�
"Adam/sequential_24/dense_74/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/sequential_24/dense_74/bias/m
�
6Adam/sequential_24/dense_74/bias/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_24/dense_74/bias/m*
_output_shapes
:
*
dtype0
�
$Adam/sequential_24/dense_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*5
shared_name&$Adam/sequential_24/dense_72/kernel/v
�
8Adam/sequential_24/dense_72/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/sequential_24/dense_72/kernel/v* 
_output_shapes
:
��*
dtype0
�
"Adam/sequential_24/dense_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/sequential_24/dense_72/bias/v
�
6Adam/sequential_24/dense_72/bias/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_24/dense_72/bias/v*
_output_shapes	
:�*
dtype0
�
$Adam/sequential_24/dense_73/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*5
shared_name&$Adam/sequential_24/dense_73/kernel/v
�
8Adam/sequential_24/dense_73/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/sequential_24/dense_73/kernel/v* 
_output_shapes
:
��*
dtype0
�
"Adam/sequential_24/dense_73/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/sequential_24/dense_73/bias/v
�
6Adam/sequential_24/dense_73/bias/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_24/dense_73/bias/v*
_output_shapes	
:�*
dtype0
�
$Adam/sequential_24/dense_74/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*5
shared_name&$Adam/sequential_24/dense_74/kernel/v
�
8Adam/sequential_24/dense_74/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/sequential_24/dense_74/kernel/v*
_output_shapes
:	�
*
dtype0
�
"Adam/sequential_24/dense_74/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/sequential_24/dense_74/bias/v
�
6Adam/sequential_24/dense_74/bias/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_24/dense_74/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
�%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�%
�
layer-0
layer-1
layer-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratemFmGmHmImJmKvLvMvNvOvPvQ
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
�

&layers
'layer_regularization_losses
	variables
(metrics
regularization_losses
)non_trainable_variables
trainable_variables
 
 
 
 
�

*layers
+layer_regularization_losses
	variables
,metrics
regularization_losses
-non_trainable_variables
trainable_variables
\Z
VARIABLE_VALUEsequential_24/dense_72/kernel)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsequential_24/dense_72/bias'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

.layers
/layer_regularization_losses
	variables
0metrics
regularization_losses
1non_trainable_variables
trainable_variables
\Z
VARIABLE_VALUEsequential_24/dense_73/kernel)layer-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsequential_24/dense_73/bias'layer-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

2layers
3layer_regularization_losses
	variables
4metrics
regularization_losses
5non_trainable_variables
trainable_variables
\Z
VARIABLE_VALUEsequential_24/dense_74/kernel)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsequential_24/dense_74/bias'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

6layers
7layer_regularization_losses
	variables
8metrics
regularization_losses
9non_trainable_variables
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

:0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	;total
	<count
=
_fn_kwargs
>	variables
?regularization_losses
@trainable_variables
A	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1
 
 
�

Blayers
Clayer_regularization_losses
>	variables
Dmetrics
?regularization_losses
Enon_trainable_variables
@trainable_variables
 
 
 

;0
<1
}
VARIABLE_VALUE$Adam/sequential_24/dense_72/kernel/mElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_24/dense_72/bias/mClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/sequential_24/dense_73/kernel/mElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_24/dense_73/bias/mClayer-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/sequential_24/dense_74/kernel/mElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_24/dense_74/bias/mClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/sequential_24/dense_72/kernel/vElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_24/dense_72/bias/vClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/sequential_24/dense_73/kernel/vElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_24/dense_73/bias/vClayer-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE$Adam/sequential_24/dense_74/kernel/vElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential_24/dense_74/bias/vClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential_24/dense_72/kernelsequential_24/dense_72/biassequential_24/dense_73/kernelsequential_24/dense_73/biassequential_24/dense_74/kernelsequential_24/dense_74/bias*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_94649
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1sequential_24/dense_72/kernel/Read/ReadVariableOp/sequential_24/dense_72/bias/Read/ReadVariableOp1sequential_24/dense_73/kernel/Read/ReadVariableOp/sequential_24/dense_73/bias/Read/ReadVariableOp1sequential_24/dense_74/kernel/Read/ReadVariableOp/sequential_24/dense_74/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp8Adam/sequential_24/dense_72/kernel/m/Read/ReadVariableOp6Adam/sequential_24/dense_72/bias/m/Read/ReadVariableOp8Adam/sequential_24/dense_73/kernel/m/Read/ReadVariableOp6Adam/sequential_24/dense_73/bias/m/Read/ReadVariableOp8Adam/sequential_24/dense_74/kernel/m/Read/ReadVariableOp6Adam/sequential_24/dense_74/bias/m/Read/ReadVariableOp8Adam/sequential_24/dense_72/kernel/v/Read/ReadVariableOp6Adam/sequential_24/dense_72/bias/v/Read/ReadVariableOp8Adam/sequential_24/dense_73/kernel/v/Read/ReadVariableOp6Adam/sequential_24/dense_73/bias/v/Read/ReadVariableOp8Adam/sequential_24/dense_74/kernel/v/Read/ReadVariableOp6Adam/sequential_24/dense_74/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_94889
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential_24/dense_72/kernelsequential_24/dense_72/biassequential_24/dense_73/kernelsequential_24/dense_73/biassequential_24/dense_74/kernelsequential_24/dense_74/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount$Adam/sequential_24/dense_72/kernel/m"Adam/sequential_24/dense_72/bias/m$Adam/sequential_24/dense_73/kernel/m"Adam/sequential_24/dense_73/bias/m$Adam/sequential_24/dense_74/kernel/m"Adam/sequential_24/dense_74/bias/m$Adam/sequential_24/dense_72/kernel/v"Adam/sequential_24/dense_72/bias/v$Adam/sequential_24/dense_73/kernel/v"Adam/sequential_24/dense_73/bias/v$Adam/sequential_24/dense_74/kernel/v"Adam/sequential_24/dense_74/bias/v*%
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_94976��
�)
�
 __inference__wrapped_model_94476
input_19
5sequential_24_dense_72_matmul_readvariableop_resource:
6sequential_24_dense_72_biasadd_readvariableop_resource9
5sequential_24_dense_73_matmul_readvariableop_resource:
6sequential_24_dense_73_biasadd_readvariableop_resource9
5sequential_24_dense_74_matmul_readvariableop_resource:
6sequential_24_dense_74_biasadd_readvariableop_resource
identity��-sequential_24/dense_72/BiasAdd/ReadVariableOp�,sequential_24/dense_72/MatMul/ReadVariableOp�-sequential_24/dense_73/BiasAdd/ReadVariableOp�,sequential_24/dense_73/MatMul/ReadVariableOp�-sequential_24/dense_74/BiasAdd/ReadVariableOp�,sequential_24/dense_74/MatMul/ReadVariableOp�
sequential_24/flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2 
sequential_24/flatten_24/Const�
 sequential_24/flatten_24/ReshapeReshapeinput_1'sequential_24/flatten_24/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_24/flatten_24/Reshape�
,sequential_24/dense_72/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_72_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_24/dense_72/MatMul/ReadVariableOp�
sequential_24/dense_72/MatMulMatMul)sequential_24/flatten_24/Reshape:output:04sequential_24/dense_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_24/dense_72/MatMul�
-sequential_24/dense_72/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_24/dense_72/BiasAdd/ReadVariableOp�
sequential_24/dense_72/BiasAddBiasAdd'sequential_24/dense_72/MatMul:product:05sequential_24/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_24/dense_72/BiasAdd�
sequential_24/dense_72/ReluRelu'sequential_24/dense_72/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_24/dense_72/Relu�
,sequential_24/dense_73/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_73_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_24/dense_73/MatMul/ReadVariableOp�
sequential_24/dense_73/MatMulMatMul)sequential_24/dense_72/Relu:activations:04sequential_24/dense_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_24/dense_73/MatMul�
-sequential_24/dense_73/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_24/dense_73/BiasAdd/ReadVariableOp�
sequential_24/dense_73/BiasAddBiasAdd'sequential_24/dense_73/MatMul:product:05sequential_24/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_24/dense_73/BiasAdd�
sequential_24/dense_73/ReluRelu'sequential_24/dense_73/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_24/dense_73/Relu�
,sequential_24/dense_74/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_74_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02.
,sequential_24/dense_74/MatMul/ReadVariableOp�
sequential_24/dense_74/MatMulMatMul)sequential_24/dense_73/Relu:activations:04sequential_24/dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential_24/dense_74/MatMul�
-sequential_24/dense_74/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_74_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_24/dense_74/BiasAdd/ReadVariableOp�
sequential_24/dense_74/BiasAddBiasAdd'sequential_24/dense_74/MatMul:product:05sequential_24/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
sequential_24/dense_74/BiasAdd�
sequential_24/dense_74/SoftmaxSoftmax'sequential_24/dense_74/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2 
sequential_24/dense_74/Softmax�
IdentityIdentity(sequential_24/dense_74/Softmax:softmax:0.^sequential_24/dense_72/BiasAdd/ReadVariableOp-^sequential_24/dense_72/MatMul/ReadVariableOp.^sequential_24/dense_73/BiasAdd/ReadVariableOp-^sequential_24/dense_73/MatMul/ReadVariableOp.^sequential_24/dense_74/BiasAdd/ReadVariableOp-^sequential_24/dense_74/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::2^
-sequential_24/dense_72/BiasAdd/ReadVariableOp-sequential_24/dense_72/BiasAdd/ReadVariableOp2\
,sequential_24/dense_72/MatMul/ReadVariableOp,sequential_24/dense_72/MatMul/ReadVariableOp2^
-sequential_24/dense_73/BiasAdd/ReadVariableOp-sequential_24/dense_73/BiasAdd/ReadVariableOp2\
,sequential_24/dense_73/MatMul/ReadVariableOp,sequential_24/dense_73/MatMul/ReadVariableOp2^
-sequential_24/dense_74/BiasAdd/ReadVariableOp-sequential_24/dense_74/BiasAdd/ReadVariableOp2\
,sequential_24/dense_74/MatMul/ReadVariableOp,sequential_24/dense_74/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
�	
�
C__inference_dense_72_layer_call_and_return_conditional_losses_94505

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
a
E__inference_flatten_24_layer_call_and_return_conditional_losses_94486

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_94649
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_944762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�	
�
-__inference_sequential_24_layer_call_fn_94629
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_24_layer_call_and_return_conditional_losses_946202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
F
*__inference_flatten_24_layer_call_fn_94736

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_24_layer_call_and_return_conditional_losses_944862
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
H__inference_sequential_24_layer_call_and_return_conditional_losses_94564
input_1+
'dense_72_statefulpartitionedcall_args_1+
'dense_72_statefulpartitionedcall_args_2+
'dense_73_statefulpartitionedcall_args_1+
'dense_73_statefulpartitionedcall_args_2+
'dense_74_statefulpartitionedcall_args_1+
'dense_74_statefulpartitionedcall_args_2
identity�� dense_72/StatefulPartitionedCall� dense_73/StatefulPartitionedCall� dense_74/StatefulPartitionedCall�
flatten_24/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_24_layer_call_and_return_conditional_losses_944862
flatten_24/PartitionedCall�
 dense_72/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0'dense_72_statefulpartitionedcall_args_1'dense_72_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_72_layer_call_and_return_conditional_losses_945052"
 dense_72/StatefulPartitionedCall�
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0'dense_73_statefulpartitionedcall_args_1'dense_73_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_73_layer_call_and_return_conditional_losses_945282"
 dense_73/StatefulPartitionedCall�
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0'dense_74_statefulpartitionedcall_args_1'dense_74_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_74_layer_call_and_return_conditional_losses_945512"
 dense_74/StatefulPartitionedCall�
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�	
�
C__inference_dense_72_layer_call_and_return_conditional_losses_94747

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
a
E__inference_flatten_24_layer_call_and_return_conditional_losses_94731

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�	
�
C__inference_dense_74_layer_call_and_return_conditional_losses_94783

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
H__inference_sequential_24_layer_call_and_return_conditional_losses_94578
input_1+
'dense_72_statefulpartitionedcall_args_1+
'dense_72_statefulpartitionedcall_args_2+
'dense_73_statefulpartitionedcall_args_1+
'dense_73_statefulpartitionedcall_args_2+
'dense_74_statefulpartitionedcall_args_1+
'dense_74_statefulpartitionedcall_args_2
identity�� dense_72/StatefulPartitionedCall� dense_73/StatefulPartitionedCall� dense_74/StatefulPartitionedCall�
flatten_24/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_24_layer_call_and_return_conditional_losses_944862
flatten_24/PartitionedCall�
 dense_72/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0'dense_72_statefulpartitionedcall_args_1'dense_72_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_72_layer_call_and_return_conditional_losses_945052"
 dense_72/StatefulPartitionedCall�
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0'dense_73_statefulpartitionedcall_args_1'dense_73_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_73_layer_call_and_return_conditional_losses_945282"
 dense_73/StatefulPartitionedCall�
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0'dense_74_statefulpartitionedcall_args_1'dense_74_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_74_layer_call_and_return_conditional_losses_945512"
 dense_74/StatefulPartitionedCall�
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�k
�
!__inference__traced_restore_94976
file_prefix2
.assignvariableop_sequential_24_dense_72_kernel2
.assignvariableop_1_sequential_24_dense_72_bias4
0assignvariableop_2_sequential_24_dense_73_kernel2
.assignvariableop_3_sequential_24_dense_73_bias4
0assignvariableop_4_sequential_24_dense_74_kernel2
.assignvariableop_5_sequential_24_dense_74_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count<
8assignvariableop_13_adam_sequential_24_dense_72_kernel_m:
6assignvariableop_14_adam_sequential_24_dense_72_bias_m<
8assignvariableop_15_adam_sequential_24_dense_73_kernel_m:
6assignvariableop_16_adam_sequential_24_dense_73_bias_m<
8assignvariableop_17_adam_sequential_24_dense_74_kernel_m:
6assignvariableop_18_adam_sequential_24_dense_74_bias_m<
8assignvariableop_19_adam_sequential_24_dense_72_kernel_v:
6assignvariableop_20_adam_sequential_24_dense_72_bias_v<
8assignvariableop_21_adam_sequential_24_dense_73_kernel_v:
6assignvariableop_22_adam_sequential_24_dense_73_bias_v<
8assignvariableop_23_adam_sequential_24_dense_74_kernel_v:
6assignvariableop_24_adam_sequential_24_dense_74_bias_v
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp.assignvariableop_sequential_24_dense_72_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_sequential_24_dense_72_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_sequential_24_dense_73_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_sequential_24_dense_73_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp0assignvariableop_4_sequential_24_dense_74_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_sequential_24_dense_74_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_sequential_24_dense_72_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_sequential_24_dense_72_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp8assignvariableop_15_adam_sequential_24_dense_73_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adam_sequential_24_dense_73_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_sequential_24_dense_74_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_sequential_24_dense_74_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_sequential_24_dense_72_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_sequential_24_dense_72_bias_vIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp8assignvariableop_21_adam_sequential_24_dense_73_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_sequential_24_dense_73_bias_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp8assignvariableop_23_adam_sequential_24_dense_74_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_sequential_24_dense_74_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25�
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�
�
(__inference_dense_73_layer_call_fn_94772

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_73_layer_call_and_return_conditional_losses_945282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
H__inference_sequential_24_layer_call_and_return_conditional_losses_94703

inputs+
'dense_72_matmul_readvariableop_resource,
(dense_72_biasadd_readvariableop_resource+
'dense_73_matmul_readvariableop_resource,
(dense_73_biasadd_readvariableop_resource+
'dense_74_matmul_readvariableop_resource,
(dense_74_biasadd_readvariableop_resource
identity��dense_72/BiasAdd/ReadVariableOp�dense_72/MatMul/ReadVariableOp�dense_73/BiasAdd/ReadVariableOp�dense_73/MatMul/ReadVariableOp�dense_74/BiasAdd/ReadVariableOp�dense_74/MatMul/ReadVariableOpu
flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
flatten_24/Const�
flatten_24/ReshapeReshapeinputsflatten_24/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_24/Reshape�
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_72/MatMul/ReadVariableOp�
dense_72/MatMulMatMulflatten_24/Reshape:output:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_72/MatMul�
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_72/BiasAdd/ReadVariableOp�
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_72/BiasAddt
dense_72/ReluReludense_72/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_72/Relu�
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_73/MatMul/ReadVariableOp�
dense_73/MatMulMatMuldense_72/Relu:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_73/MatMul�
dense_73/BiasAdd/ReadVariableOpReadVariableOp(dense_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_73/BiasAdd/ReadVariableOp�
dense_73/BiasAddBiasAdddense_73/MatMul:product:0'dense_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_73/BiasAddt
dense_73/ReluReludense_73/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_73/Relu�
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02 
dense_74/MatMul/ReadVariableOp�
dense_74/MatMulMatMuldense_73/Relu:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_74/MatMul�
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_74/BiasAdd/ReadVariableOp�
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_74/BiasAdd|
dense_74/SoftmaxSoftmaxdense_74/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_74/Softmax�
IdentityIdentitydense_74/Softmax:softmax:0 ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp ^dense_73/BiasAdd/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2B
dense_73/BiasAdd/ReadVariableOpdense_73/BiasAdd/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�:
�
__inference__traced_save_94889
file_prefix<
8savev2_sequential_24_dense_72_kernel_read_readvariableop:
6savev2_sequential_24_dense_72_bias_read_readvariableop<
8savev2_sequential_24_dense_73_kernel_read_readvariableop:
6savev2_sequential_24_dense_73_bias_read_readvariableop<
8savev2_sequential_24_dense_74_kernel_read_readvariableop:
6savev2_sequential_24_dense_74_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopC
?savev2_adam_sequential_24_dense_72_kernel_m_read_readvariableopA
=savev2_adam_sequential_24_dense_72_bias_m_read_readvariableopC
?savev2_adam_sequential_24_dense_73_kernel_m_read_readvariableopA
=savev2_adam_sequential_24_dense_73_bias_m_read_readvariableopC
?savev2_adam_sequential_24_dense_74_kernel_m_read_readvariableopA
=savev2_adam_sequential_24_dense_74_bias_m_read_readvariableopC
?savev2_adam_sequential_24_dense_72_kernel_v_read_readvariableopA
=savev2_adam_sequential_24_dense_72_bias_v_read_readvariableopC
?savev2_adam_sequential_24_dense_73_kernel_v_read_readvariableopA
=savev2_adam_sequential_24_dense_73_bias_v_read_readvariableopC
?savev2_adam_sequential_24_dense_74_kernel_v_read_readvariableopA
=savev2_adam_sequential_24_dense_74_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a33458e227b14d21ba96d0e3bffce489/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_sequential_24_dense_72_kernel_read_readvariableop6savev2_sequential_24_dense_72_bias_read_readvariableop8savev2_sequential_24_dense_73_kernel_read_readvariableop6savev2_sequential_24_dense_73_bias_read_readvariableop8savev2_sequential_24_dense_74_kernel_read_readvariableop6savev2_sequential_24_dense_74_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop?savev2_adam_sequential_24_dense_72_kernel_m_read_readvariableop=savev2_adam_sequential_24_dense_72_bias_m_read_readvariableop?savev2_adam_sequential_24_dense_73_kernel_m_read_readvariableop=savev2_adam_sequential_24_dense_73_bias_m_read_readvariableop?savev2_adam_sequential_24_dense_74_kernel_m_read_readvariableop=savev2_adam_sequential_24_dense_74_bias_m_read_readvariableop?savev2_adam_sequential_24_dense_72_kernel_v_read_readvariableop=savev2_adam_sequential_24_dense_72_bias_v_read_readvariableop?savev2_adam_sequential_24_dense_73_kernel_v_read_readvariableop=savev2_adam_sequential_24_dense_73_bias_v_read_readvariableop?savev2_adam_sequential_24_dense_74_kernel_v_read_readvariableop=savev2_adam_sequential_24_dense_74_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *'
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:
��:�:	�
:
: : : : : : : :
��:�:
��:�:	�
:
:
��:�:
��:�:	�
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�	
�
C__inference_dense_73_layer_call_and_return_conditional_losses_94528

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
C__inference_dense_73_layer_call_and_return_conditional_losses_94765

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
-__inference_sequential_24_layer_call_fn_94604
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_24_layer_call_and_return_conditional_losses_945952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
H__inference_sequential_24_layer_call_and_return_conditional_losses_94595

inputs+
'dense_72_statefulpartitionedcall_args_1+
'dense_72_statefulpartitionedcall_args_2+
'dense_73_statefulpartitionedcall_args_1+
'dense_73_statefulpartitionedcall_args_2+
'dense_74_statefulpartitionedcall_args_1+
'dense_74_statefulpartitionedcall_args_2
identity�� dense_72/StatefulPartitionedCall� dense_73/StatefulPartitionedCall� dense_74/StatefulPartitionedCall�
flatten_24/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_24_layer_call_and_return_conditional_losses_944862
flatten_24/PartitionedCall�
 dense_72/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0'dense_72_statefulpartitionedcall_args_1'dense_72_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_72_layer_call_and_return_conditional_losses_945052"
 dense_72/StatefulPartitionedCall�
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0'dense_73_statefulpartitionedcall_args_1'dense_73_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_73_layer_call_and_return_conditional_losses_945282"
 dense_73/StatefulPartitionedCall�
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0'dense_74_statefulpartitionedcall_args_1'dense_74_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_74_layer_call_and_return_conditional_losses_945512"
 dense_74/StatefulPartitionedCall�
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_dense_72_layer_call_fn_94754

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_72_layer_call_and_return_conditional_losses_945052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
-__inference_sequential_24_layer_call_fn_94714

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_24_layer_call_and_return_conditional_losses_945952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
H__inference_sequential_24_layer_call_and_return_conditional_losses_94620

inputs+
'dense_72_statefulpartitionedcall_args_1+
'dense_72_statefulpartitionedcall_args_2+
'dense_73_statefulpartitionedcall_args_1+
'dense_73_statefulpartitionedcall_args_2+
'dense_74_statefulpartitionedcall_args_1+
'dense_74_statefulpartitionedcall_args_2
identity�� dense_72/StatefulPartitionedCall� dense_73/StatefulPartitionedCall� dense_74/StatefulPartitionedCall�
flatten_24/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_flatten_24_layer_call_and_return_conditional_losses_944862
flatten_24/PartitionedCall�
 dense_72/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0'dense_72_statefulpartitionedcall_args_1'dense_72_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_72_layer_call_and_return_conditional_losses_945052"
 dense_72/StatefulPartitionedCall�
 dense_73/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0'dense_73_statefulpartitionedcall_args_1'dense_73_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_73_layer_call_and_return_conditional_losses_945282"
 dense_73/StatefulPartitionedCall�
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0'dense_74_statefulpartitionedcall_args_1'dense_74_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_74_layer_call_and_return_conditional_losses_945512"
 dense_74/StatefulPartitionedCall�
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
-__inference_sequential_24_layer_call_fn_94725

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_24_layer_call_and_return_conditional_losses_946202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
H__inference_sequential_24_layer_call_and_return_conditional_losses_94676

inputs+
'dense_72_matmul_readvariableop_resource,
(dense_72_biasadd_readvariableop_resource+
'dense_73_matmul_readvariableop_resource,
(dense_73_biasadd_readvariableop_resource+
'dense_74_matmul_readvariableop_resource,
(dense_74_biasadd_readvariableop_resource
identity��dense_72/BiasAdd/ReadVariableOp�dense_72/MatMul/ReadVariableOp�dense_73/BiasAdd/ReadVariableOp�dense_73/MatMul/ReadVariableOp�dense_74/BiasAdd/ReadVariableOp�dense_74/MatMul/ReadVariableOpu
flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
flatten_24/Const�
flatten_24/ReshapeReshapeinputsflatten_24/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_24/Reshape�
dense_72/MatMul/ReadVariableOpReadVariableOp'dense_72_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_72/MatMul/ReadVariableOp�
dense_72/MatMulMatMulflatten_24/Reshape:output:0&dense_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_72/MatMul�
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_72/BiasAdd/ReadVariableOp�
dense_72/BiasAddBiasAdddense_72/MatMul:product:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_72/BiasAddt
dense_72/ReluReludense_72/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_72/Relu�
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_73/MatMul/ReadVariableOp�
dense_73/MatMulMatMuldense_72/Relu:activations:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_73/MatMul�
dense_73/BiasAdd/ReadVariableOpReadVariableOp(dense_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_73/BiasAdd/ReadVariableOp�
dense_73/BiasAddBiasAdddense_73/MatMul:product:0'dense_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_73/BiasAddt
dense_73/ReluReludense_73/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_73/Relu�
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02 
dense_74/MatMul/ReadVariableOp�
dense_74/MatMulMatMuldense_73/Relu:activations:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_74/MatMul�
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_74/BiasAdd/ReadVariableOp�
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_74/BiasAdd|
dense_74/SoftmaxSoftmaxdense_74/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_74/Softmax�
IdentityIdentitydense_74/Softmax:softmax:0 ^dense_72/BiasAdd/ReadVariableOp^dense_72/MatMul/ReadVariableOp ^dense_73/BiasAdd/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������::::::2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2@
dense_72/MatMul/ReadVariableOpdense_72/MatMul/ReadVariableOp2B
dense_73/BiasAdd/ReadVariableOpdense_73/BiasAdd/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
C__inference_dense_74_layer_call_and_return_conditional_losses_94551

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_dense_74_layer_call_fn_94790

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_74_layer_call_and_return_conditional_losses_945512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
*R&call_and_return_all_conditional_losses
S__call__
T_default_save_signature"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_24", "layers": [{"class_name": "Flatten", "config": {"name": "flatten_24", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_72", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_73", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_74", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null, 28, 28]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_24", "layers": [{"class_name": "Flatten", "config": {"name": "flatten_24", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_72", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_73", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_74", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null, 28, 28]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
	variables
regularization_losses
trainable_variables
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_24", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*W&call_and_return_all_conditional_losses
X__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_72", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_72", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_73", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_73", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
*[&call_and_return_all_conditional_losses
\__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_74", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_74", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratemFmGmHmImJmKvLvMvNvOvPvQ"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
�

&layers
'layer_regularization_losses
	variables
(metrics
regularization_losses
)non_trainable_variables
trainable_variables
S__call__
T_default_save_signature
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
,
]serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

*layers
+layer_regularization_losses
	variables
,metrics
regularization_losses
-non_trainable_variables
trainable_variables
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
1:/
��2sequential_24/dense_72/kernel
*:(�2sequential_24/dense_72/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

.layers
/layer_regularization_losses
	variables
0metrics
regularization_losses
1non_trainable_variables
trainable_variables
X__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
1:/
��2sequential_24/dense_73/kernel
*:(�2sequential_24/dense_73/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

2layers
3layer_regularization_losses
	variables
4metrics
regularization_losses
5non_trainable_variables
trainable_variables
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
0:.	�
2sequential_24/dense_74/kernel
):'
2sequential_24/dense_74/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

6layers
7layer_regularization_losses
	variables
8metrics
regularization_losses
9non_trainable_variables
trainable_variables
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	;total
	<count
=
_fn_kwargs
>	variables
?regularization_losses
@trainable_variables
A	keras_api
*^&call_and_return_all_conditional_losses
___call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

Blayers
Clayer_regularization_losses
>	variables
Dmetrics
?regularization_losses
Enon_trainable_variables
@trainable_variables
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
6:4
��2$Adam/sequential_24/dense_72/kernel/m
/:-�2"Adam/sequential_24/dense_72/bias/m
6:4
��2$Adam/sequential_24/dense_73/kernel/m
/:-�2"Adam/sequential_24/dense_73/bias/m
5:3	�
2$Adam/sequential_24/dense_74/kernel/m
.:,
2"Adam/sequential_24/dense_74/bias/m
6:4
��2$Adam/sequential_24/dense_72/kernel/v
/:-�2"Adam/sequential_24/dense_72/bias/v
6:4
��2$Adam/sequential_24/dense_73/kernel/v
/:-�2"Adam/sequential_24/dense_73/bias/v
5:3	�
2$Adam/sequential_24/dense_74/kernel/v
.:,
2"Adam/sequential_24/dense_74/bias/v
�2�
H__inference_sequential_24_layer_call_and_return_conditional_losses_94676
H__inference_sequential_24_layer_call_and_return_conditional_losses_94564
H__inference_sequential_24_layer_call_and_return_conditional_losses_94703
H__inference_sequential_24_layer_call_and_return_conditional_losses_94578�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_sequential_24_layer_call_fn_94604
-__inference_sequential_24_layer_call_fn_94725
-__inference_sequential_24_layer_call_fn_94629
-__inference_sequential_24_layer_call_fn_94714�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_94476�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������
�2�
E__inference_flatten_24_layer_call_and_return_conditional_losses_94731�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_flatten_24_layer_call_fn_94736�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_72_layer_call_and_return_conditional_losses_94747�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_72_layer_call_fn_94754�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_73_layer_call_and_return_conditional_losses_94765�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_73_layer_call_fn_94772�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_74_layer_call_and_return_conditional_losses_94783�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_74_layer_call_fn_94790�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2B0
#__inference_signature_wrapper_94649input_1
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
 __inference__wrapped_model_94476s4�1
*�'
%�"
input_1���������
� "3�0
.
output_1"�
output_1���������
�
C__inference_dense_72_layer_call_and_return_conditional_losses_94747^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� }
(__inference_dense_72_layer_call_fn_94754Q0�-
&�#
!�
inputs����������
� "������������
C__inference_dense_73_layer_call_and_return_conditional_losses_94765^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� }
(__inference_dense_73_layer_call_fn_94772Q0�-
&�#
!�
inputs����������
� "������������
C__inference_dense_74_layer_call_and_return_conditional_losses_94783]0�-
&�#
!�
inputs����������
� "%�"
�
0���������

� |
(__inference_dense_74_layer_call_fn_94790P0�-
&�#
!�
inputs����������
� "����������
�
E__inference_flatten_24_layer_call_and_return_conditional_losses_94731]3�0
)�&
$�!
inputs���������
� "&�#
�
0����������
� ~
*__inference_flatten_24_layer_call_fn_94736P3�0
)�&
$�!
inputs���������
� "������������
H__inference_sequential_24_layer_call_and_return_conditional_losses_94564m<�9
2�/
%�"
input_1���������
p

 
� "%�"
�
0���������

� �
H__inference_sequential_24_layer_call_and_return_conditional_losses_94578m<�9
2�/
%�"
input_1���������
p 

 
� "%�"
�
0���������

� �
H__inference_sequential_24_layer_call_and_return_conditional_losses_94676l;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������

� �
H__inference_sequential_24_layer_call_and_return_conditional_losses_94703l;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������

� �
-__inference_sequential_24_layer_call_fn_94604`<�9
2�/
%�"
input_1���������
p

 
� "����������
�
-__inference_sequential_24_layer_call_fn_94629`<�9
2�/
%�"
input_1���������
p 

 
� "����������
�
-__inference_sequential_24_layer_call_fn_94714_;�8
1�.
$�!
inputs���������
p

 
� "����������
�
-__inference_sequential_24_layer_call_fn_94725_;�8
1�.
$�!
inputs���������
p 

 
� "����������
�
#__inference_signature_wrapper_94649~?�<
� 
5�2
0
input_1%�"
input_1���������"3�0
.
output_1"�
output_1���������
