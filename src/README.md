# 手順

1. 元画像データをimgs/colorsに入れておく or `python download_images.py`をする
2. `python make_dataset.py`でデータセットを作る
3. `python make_pathces.py`でデータセットからパッチを作成
-4. `python rename_dataset.py`でデータセットの名前をindexにする-
4. `python devide_patches.py`でパッチを教師データ、テストデータに分ける

# メモ

* "./imgs"などの"."はsensin2ディレクトリのこと
