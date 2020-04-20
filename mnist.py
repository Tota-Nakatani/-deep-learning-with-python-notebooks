#part1 ディープラーニングとは何か
#読み物であるため省略

#part2 ニューラルネットワークの数学的要素
#mnist:6万の訓練データ、1万のテストデータ
import keras
keras.__version__

from keras.datasets import mnist
#mnistの読み込み
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#データ型の確認
train_images.shape
len(train_labels)


#====================================================================================
#初めてのニューラルネット

#ニューラルネットワークのアーキテクチャーの構築
from keras import models
from keras import layers

network=models.Sequential()
#Dense:全結合層(FC)
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))


#コンパイル環境の設定(損失関数、optimizer,評価指標)
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',
                metrics=['accuracy'])


#データの前処理
#変形前の訓練データ：unit8(符号なし8ビット整数)、(60000,28,28)、[0.255]
#変形後の訓練データ:float32(32ビット浮動小数点)、(60000,28*28)、[0.,1]
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255


##ラベルの処理
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#訓練:fitメソッド
network.fit(train_images,train_labels,epochs=5,batch_size=128)

#テストの実行
test_loss,test_acc=network.evaluate(test_images,test_labels)
print('test_acc',test_acc)

#ゼロから作るで行なった複雑なcodeがわずか20行ほどの簡潔なコードに!!!


#=================================================================================

#2.2 ニューラルネットワークでのデータ表現
#tenor(テンソル)とは
#数値の入る入れ物(コンテナ)

#スカラー
#数字１つのみ・・・0次元テンソル(スカラーテンソル)である
#ndim=0である
import numpy as np
x=np.array(12)
x
x.ndim

#ベクトル:一次元テンソル
x=np.array([12,3,4,5,3])
x　　#一次元テンソルであり、5次元ベクトル
x.ndim

#行列:2二限テンソル
x=np.array([[1,2,3],[4,5,6]])
x
x.shape
x.ndim

#3次元テンソル
#行列を配列上に並べたもの
#3次元テンソル配列上に並べると、4次元テンソルになる
#ディープラーニングではせいぜい4次元まで

#mnistで確認しよう
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(train_images.ndim)

#4つめのの数字を表示してみよう
digit=train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

#numpyのテンソル操作
#10~100番目を取り出し、(90,28,28)にする
my_slice=train_images[10:100]
my_slice.shape


