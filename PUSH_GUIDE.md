# 推送到 GitHub —— 操作指引（中文）

> 本目录是一个**已初始化并提交**的本地 git 仓库，对应远程仓库 **https://github.com/snowman828/CKD_detection**（公开）。

## 1. 创建远程仓库
登录 GitHub，新建一个**空**仓库（不要勾选 README / .gitignore / license，保持空仓）。命名建议 `macki-dry`；
组织/用户名由你决定（本课题的仓库：**snowman828/CKD_detection**）。

## 2. 关联并推送
```bash
cd "G:\项目文件\CKD多模态AI早诊_本项目专属归档\09_无湿实验执行\repo_public_push_ready"
git remote add origin https://github.com/snowman828/CKD_detection     # https://github.com/snowman828/CKD_detection
git branch -M main
git push -u origin main
```

## 3. 推送后必做（三件事，缺一不可）
1. **回填 URL**：把本目录 `README.md` 与 `CITATION.cff` 中所有 `https://github.com/snowman828/CKD_detection` 占位替换为真实地址，然后 `git commit -am "add repository URL" && git push`。
2. **同步稿件声明**：三篇投稿件的 Code availability / Data Sharing Statement 段当前写的是
   "not publicly available but may be made available to qualified researchers on reasonable request"
   （npjDM 官方第 3 式）。**一旦仓库公开，必须改为公开仓库表述并附真实 URL**（注意：CJASN 对机器学习论文
   明确要求脚本 + 训练模型 + 源数据在公开网站可得，公开仓库同时满足 CJASN 该条）。
3. **验证**：浏览器打开仓库地址，确认 31 个脚本、README、LICENSE、CITATION.cff 可见；再确认 data 目录未被提交
   （`git ls-files | grep -c parquet` 应为 0）。

## 4. 本仓库未包含（有意为之）
- 参与者级数据文件（`*.parquet`，已 gitignore）—— 由公开数据脚本重建
- 已训练模型二进制 —— 由脚本重新拟合（README 已写明）
- 投稿稿件正文、内部质检文档 —— 不属于代码仓范畴
