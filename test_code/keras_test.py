from keras.layers import Input, Dense
from keras.models import Model

# 输入参数
inputs = Input(shape=(784,))

# 首层初始
x = Dense(64, activation='relu')(inputs)
# 中间层
x = Dense(64, activation='relu')(x)

# 这部分创建了一个包含输入层和三个全连接层的模型
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # 开始训练
