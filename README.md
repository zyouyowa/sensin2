# 現状

* 論文を読んで構造などの詳細を調べている

# TODO

* 論文から必要な式を探す
* Chainerで↑を実現する方法の調査
* 実装

# メモ

## 使われているらしい技術

* batch normalization
  * どこに入れてるのか明記されてない...?
  * やること的にはどの層にくっついていてもおかしくない
* ADADELTA
  * 書くだけ
  * ここをAdamにしたらどうなるか気になる
* 3*3 convolution kernels
  * なにこれ?
  * 多分ただのフィルタサイズで、3*3が良いという話...?
  * 元論文を読んだ方が良い...
* up convolutions ( https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf )
  * 元論文ではupsamplingと言われている
  * Deconvolutionとはやってることが違う
  * 自分で実装する
* no explicit pooling
  * なにこれ?

## その他

* 損失関数
  * 重み付き二乗平均誤差、重みの出し方は論文に書いてある通り
