import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers ,reporter
from chainer.training import extensions
from chainer.datasets import TupleDataset
import numpy as np
import os
import math
from numpy import random
from PIL import Image

batch_size = 10			# バッチサイズ10
uses_device = 0			# GPU#0を使用

# GPU使用時とCPU使用時でデータ形式が変わる
if uses_device >= 0:
	import cupy as cp
	import chainer.cuda
else:
	cp = np

class VGG(chainer.Chain):

    def __init__(self):
        super(VGG, self).__init__()

        with self.init_scope():
        	self.base = L.VGG16Layers()

			self.c0=L.Convolution2D(512, 512, 4, 2, 1, initialW=w)
			self.c1=L.Convolution2D(512, 512, 3, 1, 1, initialW=w)

			self.bnc0=L.BatchNormalization(512)
			self.bnc1=L.BatchNormalization(512)

    def __call__(self, x):
        h = F.relu(self.bace(x, layers=['conv4_3'])['conv4_3'])
		h = F.relu(self.bnc0(self.v16c0(h)))
		h = F.relu(self.bnc1(self.v16c1(h)))

        return h

# ベクトルから画像を生成するNN
class DCGAN_Generator_NN(chainer.Chain):

	def __init__(self):
		# 重みデータの初期値を指定する
		w = chainer.initializers.Normal(scale=0.02, dtype=None)
		# 全ての層を定義する
		super(DCGAN_Generator_NN, self).__init__()
		with self.init_scope():
			#ヒント画像
			self.vgg16 = VGG()

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
		v16x2e2 = self.vgg16(x2)

		#vgg16
		v16e2 = self.vgg16(x1)

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

# 画像を確認するNN
class DCGAN_Discriminator_NN(chainer.Chain):

	def __init__(self):
		# 重みデータの初期値を指定する
		w = chainer.initializers.Normal(scale=0.02, dtype=None)
		super(DCGAN_Discriminator_NN, self).__init__()
		# 全ての層を定義する
		with self.init_scope():
			self.c0_0 = L.Convolution2D(3, 8, 3, 1, 1, initialW=w)
			self.c0_1 = L.Convolution2D(8, 16, 4, 2, 1, initialW=w)
			self.c1_0 = L.Convolution2D(16, 16, 3, 1, 1, initialW=w)
			self.c1_1 = L.Convolution2D(16, 32, 4, 2, 1, initialW=w)
			self.c2_0 = L.Convolution2D(32, 32, 3, 1, 1, initialW=w)
			self.c2_1 = L.Convolution2D(32, 64, 4, 2, 1, initialW=w)
			self.c3_0 = L.Convolution2D(64, 64, 3, 1, 1, initialW=w)
			self.l4 = L.Linear(128 * 128, 1, initialW=w)
			self.bn0_1 = L.BatchNormalization(16, use_gamma=False)
			self.bn1_0 = L.BatchNormalization(16, use_gamma=False)
			self.bn1_1 = L.BatchNormalization(32, use_gamma=False)
			self.bn2_0 = L.BatchNormalization(32, use_gamma=False)
			self.bn2_1 = L.BatchNormalization(64, use_gamma=False)
			self.bn3_0 = L.BatchNormalization(64, use_gamma=False)

	def __call__(self, x):
		h = F.leaky_relu(self.c0_0(x))
		h = F.dropout(F.leaky_relu(self.bn0_1(self.c0_1(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn1_0(self.c1_0(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn1_1(self.c1_1(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn2_0(self.c2_0(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn2_1(self.c2_1(h))),ratio=0.2)
		h = F.dropout(F.leaky_relu(self.bn3_0(self.c3_0(h))),ratio=0.2)
		return self.l4(h)	# 結果を返すのみ

# カスタムUpdaterのクラス
class DCGANUpdater(training.StandardUpdater):

	def __init__(self, train_iter, optimizer, device):
		super(DCGANUpdater, self).__init__(
			train_iter,
			optimizer,
			device=device
		)
	
	# 画像認識側の損失関数
	def loss_dis(self, dis, y_fake, y_real):
		batchsize = len(y_fake)
		L1 = F.sum(F.softplus(-y_real)) / batchsize
		L2 = F.sum(F.softplus(y_fake)) / batchsize
		loss = L1 + L2
		reporter.report({'dis_loss':loss})
		return loss

	# 画像生成側の損失関数
	def loss_gen(self, gen, y_fake):
		batchsize = len(y_fake)
		loss = F.sum(F.softplus(-y_fake)) / batchsize
		reporter.report({'gen_loss':loss})
		return loss

	def update_core(self):
		# Iteratorからバッチ分のデータを取得
		batch = self.get_iterator('main').next()
		src = self.converter(batch, self.device)
		
		# Optimizerを取得
		optimizer_gen = self.get_optimizer('opt_gen')
		optimizer_dis = self.get_optimizer('opt_dis')
		# ニューラルネットワークのモデルを取得
		gen = optimizer_gen.target
		dis = optimizer_dis.target

		# 画像を生成して認識と教師データから認識
		x_fake = gen(src[0],src[1])		# 線画からの生成結果
		y_fake = dis(x_fake)	# 線画から生成したものの識別結果
		y_real = dis(src[1])		# 着色画像の識別結果

		# ニューラルネットワークを学習
		optimizer_dis.update(self.loss_dis, dis, y_fake, y_real)
		optimizer_gen.update(self.loss_gen, gen, y_fake)
		optimizer_gen.update(self.loss_gen, gen, y_fake)
		optimizer_gen.update(self.loss_gen, gen, y_fake)
		
# ニューラルネットワークを作成
model_gen = DCGAN_Generator_NN()
model_dis = DCGAN_Discriminator_NN()

if uses_device >= 0:
	# GPUを使う
	chainer.cuda.get_device_from_id(0).use()
	chainer.cuda.check_cuda_available()
	# GPU用データ形式に変換
	model_gen.to_gpu()
	model_dis.to_gpu()

chainer.serializers.load_hdf5( 'gan-gen.hdf5', model_gen )
chainer.serializers.load_hdf5( 'gan-dis.hdf5', model_dis )

listdataset1 = []
listdataset2 = []

fs = os.listdir('/home/nagalab/soutarou/dcgan/images')
fs.sort()

for fn in fs:
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

# 配列に追加
tupledataset1 = tuple(listdataset1)
tupledataset2 = tuple(listdataset2)
dataset = TupleDataset(tupledataset1, tupledataset2)

# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(dataset, batch_size, shuffle=False)

# 誤差逆伝播法アルゴリズムを選択する
optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_gen.setup(model_gen)
model_gen.vgg16.disable_update()
optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_dis.setup(model_dis)

# デバイスを選択してTrainerを作成する
updater = DCGANUpdater(train_iter, \
		{'opt_gen':optimizer_gen, 'opt_dis':optimizer_dis}, \
		device=uses_device)
trainer = training.Trainer(updater, (50000, 'epoch'), out="result")
# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.LogReport(trigger=(500, 'epoch'), log_name='log'))
trainer.extend(extensions.PlotReport(['dis_loss', 'gen_loss'], x_key='epoch', file_name='loss.png'))

# 中間結果を保存する
n_save = 0
@chainer.training.make_extension(trigger=(100, 'epoch'))
def save_model(trainer):
	# NNのデータを保存
	global n_save
	n_save = n_save+1
	chainer.serializers.save_hdf5( 'gan-gen-'+str(n_save)+'.hdf5', model_gen )
	chainer.serializers.save_hdf5( 'gan-dis-'+str(n_save)+'.hdf5', model_dis )
trainer.extend(save_model)

# 機械学習を実行する
trainer.run()

# 学習結果を保存する
chainer.serializers.save_hdf5( 'gan-gen.hdf5', model_gen )
chainer.serializers.save_hdf5( 'gan-dis.hdf5', model_dis )
