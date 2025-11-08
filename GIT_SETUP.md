# Git 仓库设置指南

本文档说明如何将项目代码推送到 GitHub 仓库。

## 步骤 1: 初始化 Git 仓库

在项目根目录下执行：

```bash
git init
```

## 步骤 2: 添加文件到暂存区

```bash
# 添加所有文件（.gitignore 会自动排除不需要的文件）
git add .

# 查看将要提交的文件
git status
```

## 步骤 3: 创建首次提交

```bash
git commit -m "Initial commit: Transformer implementation for EN-DE translation"
```

## 步骤 4: 在 GitHub 上创建仓库

1. 登录 GitHub (https://github.com)
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - Repository name: `transformer-en-de-translation` (或你喜欢的名称)
   - Description: `Encoder-Decoder Transformer implementation for English-German machine translation`
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为本地已有）
4. 点击 "Create repository"

## 步骤 5: 连接本地仓库到 GitHub

GitHub 创建仓库后会显示命令，类似：

```bash
git remote add origin https://github.com/YOUR_USERNAME/transformer-en-de-translation.git
git branch -M main
git push -u origin main
```

**注意**: 将 `YOUR_USERNAME` 替换为你的 GitHub 用户名，将 `transformer-en-de-translation` 替换为你创建的仓库名。

## 步骤 6: 推送代码

```bash
git push -u origin main
```

如果使用 `master` 分支：

```bash
git branch -M master
git push -u origin master
```

## 步骤 7: 验证

在浏览器中访问你的 GitHub 仓库，确认以下文件/目录已上传：

- ✅ `src/` 目录
- ✅ `requirements.txt`
- ✅ `README.md`
- ✅ `scripts/run.sh`
- ✅ `results/` 目录（如果包含训练结果）
- ✅ `.gitignore`

## 后续更新

如果修改了代码，使用以下命令更新：

```bash
# 查看修改
git status

# 添加修改的文件
git add .

# 提交
git commit -m "描述你的修改"

# 推送到 GitHub
git push
```

## 注意事项

1. **大文件**: `.gitignore` 已配置忽略大文件（如模型检查点 `.pt` 文件），但 `results/` 目录中的图片和 CSV 文件也会被忽略。如果需要上传结果图片，可以：
   - 手动添加: `git add -f results/training_curves.png`
   - 或修改 `.gitignore` 允许特定文件

2. **敏感信息**: 确保没有提交包含 API keys、密码等敏感信息的文件

3. **LaTeX 编译文件**: `.gitignore` 已排除 `.aux`, `.log`, `.out` 等临时文件，但 `report.pdf` 和 `report.tex` 会被包含（如果需要）

## 在报告中引用

在 LaTeX 报告的"可复现性"部分，可以添加：

```latex
\textbf{GitHub 仓库}: \url{https://github.com/YOUR_USERNAME/transformer-en-de-translation}
```


