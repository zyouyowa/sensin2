# 現状

* 論文を読んで構造などの詳細を調べている
  * -3.1 - 3.4前半まで読んだ-
* 256*256のデータセットを作成
    * ゴミ(真っ白のものなど)が混ざっているので撤去する必要あり
* Deconvolution、2乗平均誤差を使用して論文に書かれた構造を実装
    * バッチサイズ10にしたらGPUのメモリが足りないというエラーが出た
        * 一度に並列計算する数を指定できれば...
    * dataset 128枚, epoch 30, batch size 3でやった結果がscreenshotsのsengaka2
        * これで多分10分くらいかかった

# TODO

* 論文から必要な式を探す
  * 3*3 convolution kernelsの元論文を読む
  * no explicit poolingの元論文を読む
* Chainerで↑を実現する方法の調査
  * up convolution
* 実装
  * Deconvolutionを使って実装
    * -データセット256*256のものを用意-

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
  * プーリングをしない


## その他

* 損失関数
  * 重み付き二乗平均誤差、重みの出し方は論文に書いてある通り
