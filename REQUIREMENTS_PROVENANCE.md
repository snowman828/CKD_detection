# requirements.txt 溯源

每个包均由 `scripts/*.py` 的实际 import 反推：

- `Pillow==12.3.0   ← import 命中`
- `PySocks（未取到版本）← import 命中`
- `PyYAML==6.0.3   ← import 命中`
- `lifelines==0.30.3   ← import 命中`
- `matplotlib==3.11.1   ← import 命中`
- `pandas==2.3.3   ← import 命中`
- `scikit-learn==1.9.0   ← import 命中`
- `scipy==1.17.1   ← import 命中`
- `seaborn（未取到版本）← import 命中`
- `xgboost==3.2.0   ← import 命中`

版本号取自本机科学环境（`uv pip list --python D:/hermes/scienv/Scripts/python.exe`）。
