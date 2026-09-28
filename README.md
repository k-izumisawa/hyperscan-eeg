# hyperscan_eeg

2者間脳波同期（ハイパースキャニング）実験の前処理・脳間同期指標算出パッケージ。

`src/hyperscan_eeg/` は特定の実験デザインに依存しない汎用ライブラリ、
`configs/` が個々の実験固有の値（被験者・ペア対応・条件・マーカーコード等）を
注入する設定ファイル、という構成で分離している。**別の実験に転用する場合は
`configs/` に新しい設定ファイルを追加するだけでよく、`src/hyperscan_eeg/` は
変更不要。**

## セットアップ

```bash
pip install -r requirements.txt
pip install -e .
```

## ディレクトリ構成

```
src/hyperscan_eeg/   汎用ライブラリ本体（実験固有の値を持たない）
  config.py            設定スキーマ（データクラス定義のみ）
  presets.py           ハードウェア／一般的なEEG研究慣習に基づく再利用可能な値
                        （EMOTIV Flex Saline 32ch電極配置、標準5周波数帯域 等）
  io.py                生EEG(BDF)読込・デジタイザモンタージュ生成・マーカー抽出
  preprocessing.py     区間切り出し・区間分割(SegmentPlan)・フィルタ・ICA
  phase.py             2者間エポック結合（位相計算の前段）
  connectivity.py      帯域別の脳間同期指標（PLV/PSI等）算出
  network.py           ペア別行列からhyperbrain構築・閾値処理・グラフ指標算出
  visualization.py     ヒートマップ・棒グラフ
  pipeline.py          上記を束ねる高水準オーケストレーション関数
  cli.py               CLIエントリポイント（--config-module で実験設定を切替可能）

configs/              実験固有の設定（ここが「各自で用意する」部分）
  gattai_hyperscan_study.py   本実験（合体ゲーム課題）の具体的な値
                               （被験者・ペア対応・条件・マーカーコード・
                                 ゲーム状態の区間分割定義 等）

scripts/
  run_preprocessing.py  前処理実行スクリプト（configsを読み込んで実行）
  run_analysis.py       同期指標算出実行スクリプト（同上）
  run_network_analysis.py  保存済みconnectivity CSVのネットワーク解析

data/                 raw / digitizer / preprocessed / results / figures
markerdata/           補正イベントCSV（例: sub04_silent.csv、存在する場合のみ優先）
```

## 補正イベントCSV

イベントに異常があるBDFだけ、`markerdata/subNN_condition.csv` を配置する。
CSVが存在すれば `latency`（BDF開始からの秒）と `marker_value`（イベントID）
からイベントを作成し、存在しなければBDF内Annotationsを使用する。補正イベントは
前処理済みFIFのAnnotationsにも保存される。解析対象外のマーカーは、設定された
開始・終了・境界マーカーに基づいて除外される。

## 実行

```bash
python scripts/run_preprocessing.py
python scripts/run_analysis.py
```

3名の全ペアについて保存済みconnectivity CSVから96ノードのhyperbrain
ネットワークを構築し、上位10%の結合に対する二値グラフ指標を算出する場合:

```bash
python scripts/run_network_analysis.py --condition speaking --segment gattai_ato --band alpha
```

`data/network_results/<condition>/<segment>/` に、完全隣接行列、閾値後隣接行列、
ネットワーク指標CSV、ネットワーク図PNGを保存する。既定の閾値は絶対強度の
90パーセンタイルであり、PLVのような非負指標に加えて符号付きPSIにも対応する。
Global/local efficiency、clustering、path lengthは閾値後の**二値トポロジー**に
対して算出し、path lengthは非連結時に最大連結成分を対象とする。

既存PLV結果から発表向けに、状態・帯域別のchannel-level network、
3名をノードとしたparticipant-level network、network metrics比較図を一括生成する場合:

```bash
python scripts/run_network_report.py --condition speaking \
    --segments gattai_mae gattai_ato
```

`data/network_reports/<condition>/` に図と集計CSVを保存する。channel-level図は
既定で上位2%のedgeに絞り、participant-level図は被験者ペア間の全channel pair
平均PLVをedge weightとして状態間で同一尺度・同一node配置により描画する。
network metricsは過度に疎な図示用graphとは分け、既存解析と同じ上位10%
（p90）の二値トポロジーから算出する。

前処理済み波形を全被験者・全条件について順番に確認する場合:

```bash
python scripts/review_preprocessed.py
```

表示ウィンドウを閉じると次の被験者へ進む。被験者・条件や表示時間幅を
限定する場合は、例えば
`python scripts/review_preprocessed.py --subjects 4 5 6 --conditions silent --duration 30`
のように指定する。

実験固有パラメータ（被験者・ペア対応・条件・マーカーコード・ゲーム状態の
区間分割）は `configs/gattai_hyperscan_study.py` を編集する。
同期指標の種類（PLV/PSI等）や区間分割の有無など、実行そのものに関わる
制御パラメータは各スクリプト冒頭の `# ==== 実行制御パラメータ ====`
ブロックを編集する。

## 別の実験への転用

1. `configs/gattai_hyperscan_study.py` を複製し、新しいファイル名で保存する
   （例: `configs/my_other_study.py`）。
2. `ExperimentDesign`（被験者・実施したペア・条件・ゲーム状態の区間分割）、
   `MarkerConfig`（マーカーコード）、`MontageConfig`（電極配置）を
   自分の実験の値に書き換える。
   - `pairs` は総当たりを仮定せず、実際に実施したペアを列挙する
     （総当たりの場合のみ `all_subject_pairs()` を利用してよい）。
   - `condition_segments` は `SegmentPlan(labels=..., boundary_markers=...)` で
     境界マーカーとラベルを対応付ける。状態数が増減しても、この定義を
     差し替えるだけで対応できる。
3. `scripts/*.py` のインポート元を新しい設定ファイルに変更する。
