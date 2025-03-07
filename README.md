# Meteora LP池子分析工具

这个项目包含用于分析Meteora流动性池子的脚本，可以帮助用户找到最佳的LP投资机会。

## 功能

- 从Meteora API获取所有池子数据
- 分析池子的TVL、交易量、费用等指标
- 筛选高质量的LP池子
- 评分和排名池子，找出最佳投资机会

## 文件说明

- `meteora_pool_analyzer_v3.py`: 主要的池子分析脚本，包含数据获取、筛选和评分功能
- `meteora_comprehensive_analysis.py`: 综合分析脚本，提供更详细的池子分析
- `data/pool_history.db`: 池子历史数据库，存储历史分析结果
- `requirements.txt`: 项目依赖

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本分析

运行以下命令进行基本的池子分析：

```bash
python meteora_pool_analyzer_v3.py
```

这将获取所有池子数据，筛选高质量池子，并生成分析结果。

### 综合分析

运行以下命令进行更详细的综合分析：

```bash
python meteora_comprehensive_analysis.py
```

这将提供更多维度的池子分析，包括历史表现、风险评估等。

## 输出

分析结果将保存为CSV文件，包含以下信息：

- 池子地址和名称
- TVL和交易量
- 费用和APY
- 综合评分和排名

## 注意事项

- 脚本需要互联网连接以从Meteora API获取数据
- 分析结果仅供参考，不构成投资建议
- 请根据自己的风险偏好和投资策略做出决策 