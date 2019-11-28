# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.datasets import TupleDataset
import numpy as np
import os
import math
import random
from numpy import random
from PIL import Image
import codecs

uses_device = 0			# GPU#0を使用

# GPU使用時とCPU使用時でデータ形式が変わる
if uses_device >= 0:
	import cupy as cp
	import chainer.cuda
else:
	cp = np

# ベクトルから画像を生成するNN
class DCGAN_Generator_NN(chainer.Chain):

	def __init__(self):
		# 重みデータの初期値を指定する
		w = chainer.initializers.Normal(scale=0.02, dtype=None)
		super(DCGAN_Generator_NN, self).__init__()
		with self.init_scope():
			#ヒント画像
			self.vgg16 = L.VGG16Layers()

			self.v16x2c0=L.Convolution2D(512, 512, 4, 2, 1, initialW=w)
			self.v16x2c1=L.Convolution2D(512, 512, 3, 1, 1, initialW=w)

			self.bnv16x2c0=L.BatchNormalization(512)
			self.bnv16x2c1=L.BatchNormalization(512)

			#vgg16
			self.v16c0=L.Convolution2D(512, 512, 4, 2, 1, initialW=w)
			self.v16c1=L.Convolution2D(512, 512, 3, 1, 1, initialW=w)

			self.bnv16c0=L.BatchNormalization(512)
			self.bnv16c1=L.BatchNormalization(512)

			#U-Net
			self.c0=L.Convolution2D(3, 32, 3, 1, 1, initialW=w)
			self.c1=L.Convolution2D(32, 64, 4, 2, 1, initialW=w)
			self.c2=L.Convolution2D(64, 64, 3, 1, 1, initialW=w)
			self.c3=L.Convolution2D(64, 128, 4, 2, 1, initialW=w)
			self.c4=L.Convolution2D(128, 128, 3, 1, 1, initialW=w)
			self.c5=L.Convolution2D(128, 256, 4, 2, 1, initialW=w)
			self.c6=L.Convolution2D(256, 256, 3, 1, 1, initialW=w)
			self.c7=L.Convolution2D(256, 512, 4, 2, 1, initialW=w)
			self.c8=L.Convolution2D(512, 512, 3, 1, 1, initialW=w)

			self.r0=L.Convolution2D(2048, 512, 3, 1, 1, initialW=w)
			self.r1=L.Convolution2D(1024, 512, 3, 1, 1, initialW=w)
			self.r2=L.Convolution2D(1024, 512, 3, 1, 1, initialW=w)
			self.r3=L.Convolution2D(1024, 512, 3, 1, 1, initialW=w)

			self.dc8=L.Deconvolution2D(1024, 512, 4, 2, 1, initialW=w)
			self.dc7=L.Convolution2D(512, 256, 3, 1, 1, initialW=w)
			self.dc6=L.Deconvolution2D(512, 256, 4, 2, 1, initialW=w)
			self.dc5=L.Convolution2D(256, 128, 3, 1, 1, initialW=w)
			self.dc4=L.Deconvolution2D(256, 128, 4, 2, 1, initialW=w)
			self.dc3=L.Convolution2D(128, 64, 3, 1, 1, initialW=w)
			self.dc2=L.Deconvolution2D(128, 64, 4, 2, 1, initialW=w)
			self.dc1=L.Convolution2D(64, 32, 3, 1, 1, initialW=w)
			self.dc0=L.Convolution2D(64, 3, 3, 1, 1, initialW=w)

			self.bnc0=L.BatchNormalization(32)
			self.bnc1=L.BatchNormalization(64)
			self.bnc2=L.BatchNormalization(64)
			self.bnc3=L.BatchNormalization(128)
			self.bnc4=L.BatchNormalization(128)
			self.bnc5=L.BatchNormalization(256)
			self.bnc6=L.BatchNormalization(256)
			self.bnc7=L.BatchNormalization(512)
			self.bnc8=L.BatchNormalization(512)

			self.bnr0=L.BatchNormalization(512)
			self.bnr1=L.BatchNormalization(512)
			self.bnr2=L.BatchNormalization(512)
			self.bnr3=L.BatchNormalization(512)

			self.bnd8=L.BatchNormalization(512)
			self.bnd7=L.BatchNormalization(256)
			self.bnd6=L.BatchNormalization(256)
			self.bnd5=L.BatchNormalization(128)
			self.bnd4=L.BatchNormalization(128)
			self.bnd3=L.BatchNormalization(64)
			self.bnd2=L.BatchNormalization(64)
			self.bnd1=L.BatchNormalization(32)

	def __call__(self, x1,x2):
		#ヒント画像
		v16x2e0 = F.relu(self.vgg16(x2, layers=['conv4_3'])['conv4_3'])
		v16x2e1 = F.relu(self.bnv16x2c0(self.v16x2c0(v16x2e0)))
		v16x2e2 = F.relu(self.bnv16x2c1(self.v16x2c1(v16x2e1)))

		#vgg16
		v16e0 = F.relu(self.vgg16(x1, layers=['conv4_3'])['conv4_3'])
		v16e1 = F.relu(self.bnv16c0(self.v16c0(v16e0)))
		v16e2 = F.relu(self.bnv16c1(self.v16c1(v16e1)))

		#U-Net
		e0 = F.relu(self.bnc0(self.c0(x1)))
		e1 = F.relu(self.bnc1(self.c1(e0)))
		e2 = F.relu(self.bnc2(self.c2(e1)))
		del e1
		e3 = F.relu(self.bnc3(self.c3(e2)))
		e4 = F.relu(self.bnc4(self.c4(e3)))
		del e3
		e5 = F.relu(self.bnc5(self.c5(e4)))
		e6 = F.relu(self.bnc6(self.c6(e5)))
		del e5
		e7 = F.relu(self.bnc7(self.c7(e6)))
		e8 = F.relu(self.bnc8(self.c8(e7)))

		rs0 = F.relu(self.bnr0(self.r0(F.concat([e7, e8, v16e2, v16x2e2]))))
		rs1 = F.relu(self.bnr1(self.r1(F.concat([e8, rs0]))))
		rs2 = F.relu(self.bnr2(self.r2(F.concat([rs0, rs1]))))
		rs3 = F.relu(self.bnr3(self.r3(F.concat([rs1, rs2]))))

		d8 = F.relu(self.bnd8(self.dc8(F.concat([e8, rs3]))))
		del e7, e8
		d7 = F.relu(self.bnd7(self.dc7(d8)))
		del d8
		d6 = F.relu(self.bnd6(self.dc6(F.concat([e6, d7]))))
		del d7, e6
		d5 = F.relu(self.bnd5(self.dc5(d6)))
		del d6
		d4 = F.relu(self.bnd4(self.dc4(F.concat([e4, d5]))))
		del d5, e4
		d3 = F.relu(self.bnd3(self.dc3(d4)))
		del d4
		d2 = F.relu(self.bnd2(self.dc2(F.concat([e2, d3]))))
		del d3, e2
		d1 = F.relu(self.bnd1(self.dc1(d2)))
		del d2
		d0 = F.sigmoid(self.dc0(F.concat([e0, d1])))
		
		return d0	# 結果を返すのみ

# ニューラルネットワークを作成
model = DCGAN_Generator_NN()

if uses_device >= 0:
	# GPUを使う
	chainer.cuda.get_device_from_id(0).use()
	chainer.cuda.check_cuda_available()
	# GPU用データ形式に変換
	model.to_gpu()

# 学習結果を読み込む
chainer.serializers.load_hdf5( 'gan-gen.hdf5', model )


# 画像を生成する

listdataset1 = []
listdataset2 = []

fs = os.listdir('/home/nagalab/soutarou/dcgan/images')
fs.sort()

for fn in fs:
	#print(fn)
	# 画像を読み込んで128×128ピクセルにリサイズ
	img = Image.open('/home/nagalab/soutarou/dcgan/images/' + fn).convert('RGB').resize((128, 128))

	if 'png' in fn:
		# 画素データを0〜1の領域にする
		hpix1 = np.array(img, dtype=np.float32) / 255.0
		hpix1 = hpix1.transpose(2,0,1)
		listdataset1.append(hpix1)
	else:
		# 画素データを0〜1の領域にする
		hpix2 = np.array(img, dtype=np.float32) / 255.0
		hpix2= hpix2.transpose(2,0,1)
		listdataset2.append(hpix2)

#random.shuffle(listdataset2)

# 配列に追加
tupledataset1 = tuple(listdataset1)
tupledataset2 = tuple(listdataset2)
	
tupledataset1 = cp.array(tupledataset1, dtype=cp.float32)
tupledataset2 = cp.array(tupledataset2, dtype=cp.float32)

result = model(tupledataset1,tupledataset2)
for i in range(10):
	data = np.zeros((128, 128, 3), dtype=np.uint8)
	dst = result.data[i] * 255.0
	if uses_device >= 0:
		dst = chainer.cuda.to_cpu(dst)
	data[:,:,0] = dst[0]
	data[:,:,1] = dst[1]
	data[:,:,2] = dst[2]
	himg = Image.fromarray(data, 'RGB')
	himg.save('gen-'+str(i)+'.png')
	