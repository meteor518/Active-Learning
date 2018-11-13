## Active Learning

基于`libact库`修改。[libact库](https://libact.readthedocs.io/en/latest/)的使用可以详见文档。

* libact库本身的查询策略，每次只能选最不确定的一个样本，将其函数重写，可以一次查询最不确定的前n个样本。

> 使用如：ask_ids= qs.make_query(n_instances=100)，仅需指定参数n_instances即可，默认为1.

* 添加基于keras网络训练的模型，和基于TensorFlow模型的调用。

> 添加自己的模型将其放在`libact/models/`下。读取keras和TensorFlow的已训练模型，用于查询最不确定样本。对添加模型的测试文件分别为：`test_keras_model.py`,`test_TF_model.py`。

> 如需自己添加模型，可以仿照`models/mymodel_keras.py`，`models/mymodel_TF.py`进行修改。

* 论文[Cost-Effective Active Learning (CEAL) for Deep Image Classification Implementation with keras](https://arxiv.org/pdf/1701.03551)的实现代码.
 > 代码见：`CEAL_keras.py`。后续有时间将其整合到libact库中。
 
* 其他实现Active Learning的库有[modAL](https://modal-python.readthedocs.io/en/latest/),有详细的官方文档，本人尝试也还不错，有兴趣的可以自行了解。
