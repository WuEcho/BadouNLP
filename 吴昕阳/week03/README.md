## Home Work week3

第三周作业：尝试修改nlpdemo，做一个多分类任务，判断特定字符在字符串的第几个位置，使用rnn和交叉熵。

2025.04.07 复习并重新进行了作业的编写，发现并总结问题如下：

## 错误点： 1. 正确答案中去掉了pooling层 
## 错误点： 2. 在使用rnn的时候我使用的是self.calssify 正确的应该是self.rnn = nn.RNN
## 错误点： 3. rnn的输入是(batch_size, seq_len, input_size) 但是我使用的是(batch_size, input_size, seq_len)
##          batch_size是隐藏层的个数，seq_len是输入序列的长度，input_size是每个输入的维度，正确的应该是self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
## 
## 错误点： 4. 损失函数赋值方法错误.应该是self.loss = nn.functional.cross_entropy 而不是self.loss = nn.CrossEntropyLoss
## 错误点： 5. forward函数由于不知道需要去掉pooling层，所以我没有去掉对应的转置操作，squeeze操作
## 错误点： 6. forward函数中对‘  rnn_out, hidden = self.rnn(x)
##                             x = rnn_out[:, -1, :] ’ 这么操作是为什么我也不清楚
## 错误点： 7. 构建正负样本方式错误，这个错误比较离谱，我想的是x结合中的字母是否包含“你我他”并返回序号，然后通过y来判断正负样本
##            实际应该是x结合中的字母是否包含“你我他”并返回序号，然后通过y来通过index来判断正负样本


## 总结：1.基础概念不够熟悉
##      2.基本写法错误

再次复习此处需要注意如下问题;

## 1. 为什么没有使用Pooling层？
## 2. 为什么RNN层命名为self.rnn而不是self.classify或self.layer？
## 3. 为什么用RNN后还要接线性层？
## 4. x = rnn_out[:, -1, :] 的作用








问题答案：
## 1. 为什么没有使用Pooling层？

 代码中原本有池化层（self.pool），但被注释掉了，当前使用的是RNN层。是否使用池化层取决于任务需求和模型设计：

池化层的作用：对序列维度进行压缩（如全局平均池化），提取序列的全局特征。例如，在文本分类中，池化可以将变长序列转换为固定长度的特征向量。
​当前选择RNN的原因：RNN更适合处理序列的时序依赖关系。如果任务是捕捉字符在序列中的位置信息（如字符a的位置），RNN能更好地建模序列的时序逻辑，而池化层可能会丢失位置细节。
如果想对比两种方法，可以取消注释池化层，并注释RNN层，观察性能差异。

## 2. 为什么RNN层命名为self.rnn而不是self.classify或self.layer？
PyTorch模型的层命名通常遵循功能语义化原则：
​**self.rnn**：明确表示这一层是RNN结构，便于理解模型结构。
​**self.classify**：表示分类输出层，用于将特征映射到类别空间。
**self.layer**：过于通用，不利于代码可读性和维护。
例如：
清晰的命名
```
self.rnn = nn.RNN(...)       # RNN层
self.classify = nn.Linear(...)  # 分类层
```

## 不清晰的命名
```
self.layer1 = nn.RNN(...)    # 难以直接看出功能
self.layer2 = nn.Linear(...)
```
## 3. 为什么用RNN后还要接线性层？

RNN层的输出是隐藏状态（hidden states），而分类任务需要将隐藏状态映射到类别空间：
RNN的输出：假设RNN的隐藏层维度是vector_dim，则输出形状为 (batch_size, sequence_length, vector_dim)。
分类需求：需要将vector_dim维的特征转换为(sentence_length + 1)个类别（即字符a的位置可能为0到sentence_length，共sentence_length + 1种可能）。
线性层的作用：通过self.classify将vector_dim维的特征映射到类别空间，公式为：
y=W⋅hlast +b
其中 h last 是RNN最后一个时间步的隐藏状态。

## ​4. x = rnn_out[:, -1, :] 的作用
这一行代码提取了RNN输出的最后一个时间步的隐藏状态：
​**rnn_out的形状**：(batch_size, sequence_length, vector_dim)。
**rnn_out[:, -1, :]**：取所有样本（:）、最后一个时间步（-1）、所有特征维度（:），结果形状为 (batch_size, vector_dim)。
为什么选择最后一个时间步？
在字符位置分类任务中，字符a的位置可能是序列中的任意位置。若使用所有时间步的隐藏状态，需要更复杂的处理（如注意力机制）。直接使用最后一个时间步的隐藏状态是一种简化设计，假设RNN能够通过时序传递捕捉到字符a的位置信息。
 