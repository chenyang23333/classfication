## 本代码仓包含:
* 训练代码 torch_gpu_with_eval.py
* 测试集验证代码 eval_test.py
* 训练日志 my_log.txt 以及paddle上某方案的 original_log.txt 
* 测试集预测结果 resulet.txt
* 训练出来的最佳权重 

### 训练执行命令
python toch_gpu_with_eval.py

### 测试预测命令 
python eval_test.py

### 创新点：
数据集太小，采用数据增强扩充数据集
增大batch size，调小learning rate，以提高泛化性

环境说明：3090+CUDA Version: 11.6+python3.8

精度指标：目测比官方好，具体指标稍后更新



