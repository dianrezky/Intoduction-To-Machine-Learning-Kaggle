"�E
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE13333ss�@A3333ss�@a�BTb�A�?i�BTb�A�?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff��@9fffff��@Afffff��@Ifffff��@a�3���"�?i�&���?�Unknown�
}HostMatMul")gradient_tape/sequential_2/dense_4/MatMul(1     \�@9     \�@A     \�@I     \�@a����X��?i�!OcCO�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1�����T�@9�����T�@A�����T�@I�����T�@a�U����?ig},Q��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1�����͊@9�����͊@A�����͊@I�����͊@a^�8�@U�?i�h;� �?�Unknown
�HostRandomUniform";sequential_2/dropout_2/dropout/random_uniform/RandomUniform(1�����$�@9�����$�@A�����$�@I�����$�@a쳍8�Τ?i,��M�?�Unknown
sHost_FusedMatMul"sequential_2/dense_4/Relu(1�����L�@9�����L�@A�����L�@I�����L�@a��x�6(�?i�p�P�?�Unknown
�	HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1333333@9333333@A333333@I333333@a�xN�"՘?i���0��?�Unknown
^
HostGatherV2"GatherV2(133333ce@933333ce@A33333ce@I33333ce@a�ף���?it�`�Z�?�Unknown
HostMatMul"+gradient_tape/sequential_2/dense_5/MatMul_1(1������`@9������`@A������`@I������`@a!�����z?i���k|��?�Unknown
vHost_FusedMatMul"sequential_2/dense_5/BiasAdd(1fffffS@9fffffS@AfffffS@IfffffS@a�N��Hn?ihnFŮ�?�Unknown
iHostWriteSummary"WriteSummary(1     �Q@9     �Q@A     �Q@I     �Q@a�#fWl?i$,ԝ���?�Unknown�
�HostGreaterEqual"+sequential_2/dropout_2/dropout/GreaterEqual(1333333L@9333333L@A333333L@I333333L@a&c��qf?i�Y}E��?�Unknown
�HostReluGrad"+gradient_tape/sequential_2/dense_4/ReluGrad(1����̌J@9����̌J@A����̌J@I����̌J@a(dˬ�!e?i��+g��?�Unknown
}HostMatMul")gradient_tape/sequential_2/dense_5/MatMul(1     �J@9     �J@A     �J@I     �J@a�|�}e?i�]��~�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1�����I@9�����I@A�����I@I�����I@a�U�<�c?i)E��x�?�Unknown
qHostSoftmax"sequential_2/dense_5/Softmax(133333�H@933333�H@A33333�H@I33333�H@a�ts���c?i��M�!3�?�Unknown
uHostCast"#sequential_2/dropout_2/dropout/Cast(1ffffffG@9ffffffG@AffffffG@IffffffG@aq�z'ڟb?ix3uz�E�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������F@9������F@A������F@I������F@ar���/b?i�N�C�W�?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333�@@933333�@@A     �;@I     �;@aᙞ�>�U?iJ�]��b�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1fffff�8@9fffff�8@Afffff�8@Ifffff�8@a*e��{�S?i��G��l�?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff�A@9fffff�A@A�����8@I�����8@a����z.S?i�׿�bv�?�Unknown
�HostBiasAddGrad"6gradient_tape/sequential_2/dense_4/BiasAdd/BiasAddGrad(1������5@9������5@A������5@I������5@a.gq$1Q?i��j�~�?�Unknown
dHostDataset"Iterator::Model(1fffff�Q@9fffff�Q@Afffff�4@Ifffff�4@a�&\w�P?i�#��L��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      1@9      1@A      1@I      1@a��y �K?i9B ���?�Unknown
lHostIteratorGetNext"IteratorGetNext(1      /@9      /@A      /@I      /@a:��b�H?i[ǩ�;��?�Unknown
�HostBiasAddGrad"6gradient_tape/sequential_2/dense_5/BiasAdd/BiasAddGrad(1ffffff*@9ffffff*@Affffff*@Iffffff*@a��߁E?iW?
p|��?�Unknown
sHostMul""sequential_2/dropout_2/dropout/Mul(1ffffff*@9ffffff*@Affffff*@Iffffff*@a��߁E?iS�j7���?�Unknown
XHostEqual"Equal(1      )@9      )@A      )@I      )@aX]J���C?i�	g����?�Unknown
`HostGatherV2"
GatherV2_1(1      )@9      )@A      )@I      )@aX]J���C?i�\c%���?�Unknown
� HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1ffffff'@9ffffff'@Affffff'@Iffffff'@aq�z'ڟB?i8;�X��?�Unknown
e!Host
LogicalAnd"
LogicalAnd(1ffffff&@9ffffff&@Affffff&@Iffffff&@a�(Y	�A?i��/"ͱ�?�Unknown�
Z"HostArgMax"ArgMax(1333333&@9333333&@A333333&@I333333&@aD8�X�A?iP�c�7��?�Unknown
x#HostDataset"#Iterator::Model::ParallelMapV2::Zip(1������S@9������S@Affffff%@Iffffff%@a�v7�WA?i.g^z��?�Unknown
u$HostMul"$sequential_2/dropout_2/dropout/Mul_1(1333333%@9333333%@A333333%@I333333%@at�����@?i��J����?�Unknown
~%HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1������$@9������$@A������$@I������$@a���?�@?i��z���?�Unknown
�&HostMul"2gradient_tape/sequential_2/dropout_2/dropout/Mul_2(1������"@9������"@A������"@I������"@az����=?iLM���?�Unknown
�'HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333"@9333333"@A333333"@I333333"@a�0���<?i)2c(��?�Unknown
[(HostAddV2"Adam/add(1������!@9������!@A������!@I������!@a�?�W'<?i�,N����?�Unknown
V)HostSum"Sum_2(1ffffff!@9ffffff!@Affffff!@Iffffff!@a_a妲;?i��*���?�Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a�S#���6?ig�"2���?�Unknown
�+HostMul"0gradient_tape/sequential_2/dropout_2/dropout/Mul(1������@9������@A������@I������@a�;ٞ�5?i���%���?�Unknown
�,HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1������@9������@A������@I������@a�l��3?i��t�(��?�Unknown
�-HostReadVariableOp"*sequential_2/dense_5/MatMul/ReadVariableOp(1      @9      @A      @I      @a��(�3?i�+����?�Unknown
�.HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9������@A������@I������@a�ʴ`��2?i�B ���?�Unknown
`/HostDivNoNan"
div_no_nan(1������@9������@A������@I������@a[	�{�%2?i6�J�)��?�Unknown
t0HostReadVariableOp"Adam/Cast/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a�(Y	�1?i[�kVd��?�Unknown
[1HostPow"
Adam/Pow_1(1ffffff@9ffffff@Affffff@Iffffff@a �͖<0?i�E�k��?�Unknown
o2HostReadVariableOp"Adam/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a �͖<0?i�,|s��?�Unknown
Y3HostPow"Adam/Pow(1������@9������@A������@I������@a3��*�-?i���NR��?�Unknown
�4HostReadVariableOp"*sequential_2/dense_4/MatMul/ReadVariableOp(1������@9������@A������@I������@a�?�W',?i�5���?�Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@a��%�*?iE�s���?�Unknown
�6HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff@9ffffff@Affffff@Iffffff@a}��$*?i%��%`��?�Unknown
�7HostReadVariableOp"+sequential_2/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a:6�#x)?i������?�Unknown
]8HostCast"Adam/Cast_1(1333333@9333333@A333333@I333333@a�xN�"�(?iq�B����?�Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_4(1������@9������@A������@I������@a=4�0 �&?i��E����?�Unknown
v:HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1333333@9333333@A333333@I333333@aV��f�%?i_3,N��?�Unknown
X;HostCast"Cast_2(1������@9������@A������@I������@a�l��#?if�����?�Unknown
�<HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1      @9      @A      @I      @a��(�#?i�攑���?�Unknown
b=HostDivNoNan"div_no_nan_1(1ffffff@9ffffff@Affffff@Iffffff@a�(Y	�!?i�|%����?�Unknown
v>HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1������@9������@A������@I������@a.gq$1!?i�ç����?�Unknown
T?HostMul"Mul(1      @9      @A      @I      @a��C�,�?i�m����?�Unknown
�@HostReadVariableOp"+sequential_2/dense_4/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@a�Et�*�?iZ�d���?�Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff@9ffffff@Affffff@Iffffff@a�¤!)J?i�֭h���?�Unknown
aBHostIdentity"Identity(1ffffff@9ffffff@Affffff@Iffffff@a�¤!)J?i�������?�Unknown�
XCHostCast"Cast_3(1       @9       @A       @I       @a:6�#x?iX{��?�Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a#�f�!2?i��$A��?�Unknown
wEHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?ao.��`?iO�	���?�Unknown
yFHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a��(�?i�Y��|��?�Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�(Y	�?i]��~��?�Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�Et�*�?i.RT����?�Unknown
vIHostCast"$sparse_categorical_crossentropy/Cast(1333333�?9333333�?A333333�?I333333�?a�Et�*�?i�������?�Unknown*�D
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff��@9fffff��@Afffff��@Ifffff��@a���Nc��?i���Nc��?�Unknown�
}HostMatMul")gradient_tape/sequential_2/dense_4/MatMul(1     \�@9     \�@A     \�@I     \�@aaf�����?i\�#o�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1�����T�@9�����T�@A�����T�@I�����T�@a+h��?igv��Q�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1�����͊@9�����͊@A�����͊@I�����͊@a�g����?i$�H`�a�?�Unknown
�HostRandomUniform";sequential_2/dropout_2/dropout/random_uniform/RandomUniform(1�����$�@9�����$�@A�����$�@I�����$�@aC�Y�?i�}�H��?�Unknown
sHost_FusedMatMul"sequential_2/dense_4/Relu(1�����L�@9�����L�@A�����L�@I�����L�@aJj����?i+˸]t��?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1333333@9333333@A333333@I333333@a�M�p��?i���K���?�Unknown
^HostGatherV2"GatherV2(133333ce@933333ce@A33333ce@I33333ce@ai��gx1�?iC�Y��?�Unknown
	HostMatMul"+gradient_tape/sequential_2/dense_5/MatMul_1(1������`@9������`@A������`@I������`@a�G��E�?i��n��?�Unknown
v
Host_FusedMatMul"sequential_2/dense_5/BiasAdd(1fffffS@9fffffS@AfffffS@IfffffS@a��n���?i�%A�i�?�Unknown
iHostWriteSummary"WriteSummary(1     �Q@9     �Q@A     �Q@I     �Q@ax*R���?iTn~����?�Unknown�
�HostGreaterEqual"+sequential_2/dropout_2/dropout/GreaterEqual(1333333L@9333333L@A333333L@I333333L@aZ��K��~?i�����?�Unknown
�HostReluGrad"+gradient_tape/sequential_2/dense_4/ReluGrad(1����̌J@9����̌J@A����̌J@I����̌J@a�K���|?iArUc@,�?�Unknown
}HostMatMul")gradient_tape/sequential_2/dense_5/MatMul(1     �J@9     �J@A     �J@I     �J@ag�ż|?i@���e�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1�����I@9�����I@A�����I@I�����I@a���8{?i��>(*��?�Unknown
qHostSoftmax"sequential_2/dense_5/Softmax(133333�H@933333�H@A33333�H@I33333�H@aV��z?i�$yJ���?�Unknown
uHostCast"#sequential_2/dropout_2/dropout/Cast(1ffffffG@9ffffffG@AffffffG@IffffffG@a��	)+`y?i�8ˠ|�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������F@9������F@A������F@I������F@a�*f{�x?i6 �6�?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333�@@933333�@@A     �;@I     �;@a�z��b�m?i����S�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1fffff�8@9fffff�8@Afffff�8@Ifffff�8@a���� k?i��ϐ�n�?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff�A@9fffff�A@A�����8@I�����8@a�$Ø"j?iĥh��?�Unknown
�HostBiasAddGrad"6gradient_tape/sequential_2/dense_4/BiasAdd/BiasAddGrad(1������5@9������5@A������5@I������5@a�0	�vlg?i���m��?�Unknown
dHostDataset"Iterator::Model(1fffff�Q@9fffff�Q@Afffff�4@Ifffff�4@a� P,"�f?i��0���?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      1@9      1@A      1@I      1@a��#Ppob?i�"����?�Unknown
lHostIteratorGetNext"IteratorGetNext(1      /@9      /@A      /@I      /@aȴM��`?iYpV��?�Unknown
�HostBiasAddGrad"6gradient_tape/sequential_2/dense_5/BiasAdd/BiasAddGrad(1ffffff*@9ffffff*@Affffff*@Iffffff*@aI浾�\?iL�e����?�Unknown
sHostMul""sequential_2/dropout_2/dropout/Mul(1ffffff*@9ffffff*@Affffff*@Iffffff*@aI浾�\?i?&� ���?�Unknown
XHostEqual"Equal(1      )@9      )@A      )@I      )@a��C�Y[?iȴM��?�Unknown
`HostGatherV2"
GatherV2_1(1      )@9      )@A      )@I      )@a��C�Y[?i�i�z�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1ffffff'@9ffffff'@Affffff'@Iffffff'@a��	)+`Y?i��8���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1ffffff&@9ffffff&@Affffff&@Iffffff&@a��%��JX?i��4��*�?�Unknown�
Z HostArgMax"ArgMax(1333333&@9333333&@A333333&@I333333&@ax�^ X?i �D[�6�?�Unknown
x!HostDataset"#Iterator::Model::ParallelMapV2::Zip(1������S@9������S@Affffff%@Iffffff%@a��A��4W?iR�ӌB�?�Unknown
u"HostMul"$sequential_2/dropout_2/dropout/Mul_1(1333333%@9333333%@A333333%@I333333%@aR�z�j�V?i���N�?�Unknown
~#HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1������$@9������$@A������$@I������$@a�g�@_�V?i���RY�?�Unknown
�$HostMul"2gradient_tape/sequential_2/dropout_2/dropout/Mul_2(1������"@9������"@A������"@I������"@aW?]�+T?iY4B�hc�?�Unknown
�%HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333"@9333333"@A333333"@I333333"@a���X��S?iƛ��Fm�?�Unknown
[&HostAddV2"Adam/add(1������!@9������!@A������!@I������!@a1Dy�S?ih����v�?�Unknown
V'HostSum"Sum_2(1ffffff!@9ffffff!@Affffff!@Iffffff!@a���{�R?iq��A��?�Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a��}"=�N?i�P  ��?�Unknown
�)HostMul"0gradient_tape/sequential_2/dropout_2/dropout/Mul(1������@9������@A������@I������@a�a�%�M?i-)��o��?�Unknown
�*HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1������@9������@A������@I������@atT|��J?iBH��(��?�Unknown
�+HostReadVariableOp"*sequential_2/dense_5/MatMul/ReadVariableOp(1      @9      @A      @I      @a��_��J?i%������?�Unknown
�,HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9������@A������@I������@a'����I?io��y��?�Unknown
`-HostDivNoNan"
div_no_nan(1������@9������@A������@I������@a(^����H?i��h�>��?�Unknown
t.HostReadVariableOp"Adam/Cast/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a��%��JH?i��Q��?�Unknown
[/HostPow"
Adam/Pow_1(1ffffff@9ffffff@Affffff@Iffffff@ag^�SF?i�b�Xٴ�?�Unknown
o0HostReadVariableOp"Adam/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@ag^�SF?i:�-a��?�Unknown
Y1HostPow"Adam/Pow(1������@9������@A������@I������@a�q$�$cD?i#���y��?�Unknown
�2HostReadVariableOp"*sequential_2/dense_4/MatMul/ReadVariableOp(1������@9������@A������@I������@a1Dy�C?it�\w?��?�Unknown
t3HostAssignAddVariableOp"AssignAddVariableOp(1������@9������@A������@I������@aF{\y�7B?i���q���?�Unknown
�4HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff@9ffffff@Affffff@Iffffff@a�����A?i쭩?��?�Unknown
�5HostReadVariableOp"+sequential_2/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a[�?�YA?i|u���?�Unknown
]6HostCast"Adam/Cast_1(1333333@9333333@A333333@I333333@a�M�p��@?iY�Q����?�Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1������@9������@A������@I������@a
A�H;??i�k9���?�Unknown
v8HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1333333@9333333@A333333@I333333@a4��=?i7�h��?�Unknown
X9HostCast"Cast_2(1������@9������@A������@I������@atT|��:?i��/����?�Unknown
�:HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1      @9      @A      @I      @a��_��:?i��Ŏ��?�Unknown
b;HostDivNoNan"div_no_nan_1(1ffffff@9ffffff@Affffff@Iffffff@a��%��J8?ir�����?�Unknown
v<HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1������@9������@A������@I������@a�0	�vl7?i�Xo���?�Unknown
T=HostMul"Mul(1      @9      @A      @I      @a���G�5?i��Tx���?�Unknown
�>HostReadVariableOp"+sequential_2/dense_4/BiasAdd/ReadVariableOp(1333333@9333333@A333333@I333333@aֲ�0�4?i�f�L��?�Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff@9ffffff@Affffff@Iffffff@a�/�3?i��A���?�Unknown
a@HostIdentity"Identity(1ffffff@9ffffff@Affffff@Iffffff@a�/�3?ik���I��?�Unknown�
XAHostCast"Cast_3(1       @9       @A       @I       @a[�?�Y1?ia��t��?�Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?ap�"û{0?i�'�v���?�Unknown
wCHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a_�c��+?iPaD�@��?�Unknown
yDHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a��_��*?iI7���?�Unknown
uEHostReadVariableOp"div_no_nan/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��%��J(?i���e��?�Unknown
wFHostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aֲ�0�$?i�T�ܲ��?�Unknown
vGHostCast"$sparse_categorical_crossentropy/Cast(1333333�?9333333�?A333333�?I333333�?aֲ�0�$?i     �?�Unknown2GPU