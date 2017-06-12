# 現状

* 論文を読んで構造などの詳細を調べている
  * 3.1 - 3.4前半まで読んだ

# TODO

* 論文から必要な式を探す
  * 3*3 convolution kernelsの元論文を読む
  * no explicit poolingの元論文を読む
* Chainerで↑を実現する方法の調査
* 実装

# メモ

## 使われているらしい技術

* batch normalization
  * 最終層以外のすべての層に使う
  * chainerのでok
* ADADELTA
  * chianerのでok
  * ここをAdamにしたらどうなるか気になる
* 3*3 convolution kernels ( https://arxiv.org/pdf/1409.1556.pdf )
  * なにこれ?
  * 多分ただのフィルタサイズで、3*3が良いという話...?
  * 元論文を読んだ方が良い...
* up convolutions ( https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf )
  * 元論文ではupsamplingと言われている
  * Deconvolutionとはやってることが違う
  * 自分で実装する
* no explicit pooling ( https://arxiv.org/pdf/1412.6806.pdf )
  * なにこれ?

## その他

* 損失関数
  * 重み付き二乗平均誤差、重みの出し方は論文に書いてある通り
