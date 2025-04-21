

bert_layer_num = 12 #bert 层数
vocab = 21128       #词表大小
embedding_size = 128 #embedding维度
world_length = 12
hidden_size = 256

##
#   embedding  = 文本 embedding + segment embedding + position embedding
#
#   self attention = softmax(Q*K.T/根号(dk)) * V
# 
#   add & layer Normal = x + self attention out put 
# 
#   feed norm =   两个线性层 + 一个gelu激活函数 
# 
#    
# #


embedding = world_length * embedding_size + 2 * embedding_size + 512 * embedding_size

self_attention = world_length * embedding_size * world_length *embedding_size     #句子长度*句子长度

layer= world_length * embedding_size * hidden_size * 3  #两个线形层 一个激活层

all_params = embedding + self_attention + layer


####
word_num = 2   ##词数
max_word_length = 256 
hidden_size_value = 5512
layer_num = 12
 
embedding_params = vocab * embedding_size + word_num *embedding_size + max_word_length *embedding_size + embedding_size + embedding_size

self_attention_params = (embedding_size * embedding_size + embedding_size )*3  #缺了bias的embedding_size

#layer_params = embedding_size * hidden_size_value + embedding_size *hidden_size_value  + embedding_size + embedding_size  +embedding_size

self_attention_output = embedding_size *embedding_size +embedding_size +embedding_size +embedding_size

feed_forward = embedding_size * hidden_size +embedding_size +embedding_size * hidden_size +embedding_size +embedding_size +embedding_size

pole_params = embedding_size *embedding_size +embedding_size

all_params = embedding_params + (self_attention_params + layer_params + feed_forward) * layer_num +pole_params



