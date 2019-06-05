# Softmax Regression
## ```data```
```kadai1```のフォルダのコードは```data```フォルダの```1```フォルダのデータを使う。
```kadai2```のフォルダのコードは```data```フォルダの```2```フォルダのデータを使う。

## ```kadai1```
#### 課題1を解くためのコードは```kadai1```フォルダに入れておく。
- preprocessing.py: feature1~feature4とクラス（A, B, C）の情報をnumpy.ndarrayの形に整理する。
- classifier.py: Softmax Regressionの実装。
- train.py: classifier.pyからfit()のメソッドを呼び出し、train.csvを用いて訓練を行う。費用関数でlossを計算した結果が出力される。
- eval.py: 訓練ずみのモデルを使ってtest.csvを用いて検証。Accuracy, Precision, Recall, F1指標で計算したモデルの性能結果が出力される。

開発・実行環境はJetBrainのPycharmを使用した。
train.pyを実行すると```data```フォルダの```1```フォルダの```train.csv```ファイルで訓練を行い、当フォルダに```kadai1.model```でモデルをセーブする。
```eval.py```を実行すると```data```フォルダの```2```フォルダの```test.csv```ファイルで検証を行う。

## ```kadai2```
#### 課題2を解くためのコードは```kadai2```フォルダに入れておく。
行う上での環境と実行手順は、以上でのべた```kadai1```と同様である。