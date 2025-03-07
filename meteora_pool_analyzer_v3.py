import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import urllib3
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# 禁用SSL验证警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PoolAnalysisConfig:
    """池子分析配置"""
    min_tvl: float = 1000  # 最小TVL
    min_daily_volume: float = 100  # 最小日交易量
    risk_preference: str = 'moderate'  # 风险偏好: conservative, moderate, aggressive
    market_volatility: float = 0.3  # 市场波动率
    simulation_rounds: int = 1000  # Monte Carlo模拟轮数
    min_pool_allocation: float = 0.05  # 最小池子分配比例
    max_pool_allocation: float = 0.3  # 最大池子分配比例

class MeteoraDLMMAnalyzer:
    def __init__(self):
        self.base_url = "https://dlmm-api.meteora.ag"
        self.session = self._create_session()
        
    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.verify = False
        return session

    def _make_request(self, url):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"请求失败: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {str(e)}")
            return None

    def _parse_pool_data(self, data):
        if not data or not isinstance(data, dict):
            return None
        try:
            # 打印第一个池子的原始数据，帮助调试
            if not hasattr(self, '_debug_printed'):
                logger.info(f"示例池子数据: {json.dumps(data, indent=2)}")
                self._debug_printed = True

            # 尝试不同的字段名并验证数据合理性
            volume_24h = float(data.get('trade_volume_24h', 0))
            tvl = float(data.get('liquidity', 0))
            fees_24h = float(data.get('fees_24h', 0))
            
            # 验证APY数据合理性（不超过1000000%）
            def validate_apy(value):
                try:
                    apy = float(value)
                    return min(apy, 1000000) if apy > 0 else 0
                except (ValueError, TypeError):
                    return 0
            
            apr = validate_apy(data.get('apr', 0))
            apy = validate_apy(data.get('apy', 0))
            farm_apr = validate_apy(data.get('farm_apr', 0))
            farm_apy = validate_apy(data.get('farm_apy', 0))
            
            # 验证费率合理性（0-100%）
            def validate_fee(value):
                try:
                    fee = float(value)
                    return min(max(fee, 0), 100)
                except (ValueError, TypeError):
                    return 0
            
            base_fee = validate_fee(data.get('base_fee_percentage', 0))
            max_fee = validate_fee(data.get('max_fee_percentage', 0))

            return {
                'address': data.get('address', ''),
                'name': data.get('name', ''),
                'token_x': data.get('token_x', {}),
                'token_y': data.get('token_y', {}),
                'reserve_x': float(data.get('reserve_x_amount', 0)),
                'reserve_y': float(data.get('reserve_y_amount', 0)),
                'volume_24h': volume_24h,
                'tvl': tvl,
                'fees_24h': fees_24h,
                'apr': apr,
                'apy': apy,
                'farm_apr': farm_apr,
                'farm_apy': farm_apy,
                'bin_step': float(data.get('bin_step', 0)),
                'base_fee': base_fee,
                'max_fee': max_fee,
                'current_price': float(data.get('current_price', 0))
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"数据解析失败: {str(e)}, 数据: {data}")
            return None

    def get_all_pools(self):
        url = f"{self.base_url}/pair/all"
        response_data = self._make_request(url)
        
        if not response_data:
            logger.error("无法获取池子数据")
            return pd.DataFrame()

        logger.info(f"获取到 {len(response_data)} 个池子")
        
        pools = []
        for pool_data in response_data:
            parsed_pool = self._parse_pool_data(pool_data)
            if parsed_pool:
                pools.append(parsed_pool)

        df = pd.DataFrame(pools)
        if not df.empty:
            logger.info(f"数据统计:\nTVL范围: {df['tvl'].min():.2f} - {df['tvl'].max():.2f}\n交易量范围: {df['volume_24h'].min():.2f} - {df['volume_24h'].max():.2f}")
            logger.info(f"TVL > 1000的池子数: {len(df[df['tvl'] >= 1000])}")
            logger.info(f"交易量 > 100的池子数: {len(df[df['volume_24h'] >= 100])}")
        
        return df

    def analyze_trading_patterns(self, pool_address):
        url = f"{self.base_url}/pair/{pool_address}/trades"
        response_data = self._make_request(url)
        
        if not response_data:
            return {
                'avg_trade_size': 0,
                'trade_frequency': 0,
                'price_impact': 0
            }

        try:
            trades = response_data.get('trades', [])
            if not trades:
                return {
                    'avg_trade_size': 0,
                    'trade_frequency': 0,
                    'price_impact': 0
                }

            trade_sizes = [float(trade.get('amount', 0)) for trade in trades]
            return {
                'avg_trade_size': sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0,
                'trade_frequency': len(trades) / 24,  # trades per hour
                'price_impact': self._calculate_price_impact(trades)
            }
        except Exception as e:
            logger.warning(f"获取交易模式数据失败: {str(e)}")
            return {
                'avg_trade_size': 0,
                'trade_frequency': 0,
                'price_impact': 0
            }

    def analyze_fee_patterns(self, pool_address):
        url = f"{self.base_url}/pair/{pool_address}/fees"
        response_data = self._make_request(url)
        
        if not response_data:
            return {
                'avg_fee_rate': 0,
                'fee_volatility': 0,
                'total_fees_24h': 0
            }

        try:
            fees = response_data.get('fees', [])
            if not fees:
                return {
                    'avg_fee_rate': 0,
                    'fee_volatility': 0,
                    'total_fees_24h': 0
                }

            fee_rates = [float(fee.get('rate', 0)) for fee in fees]
            return {
                'avg_fee_rate': sum(fee_rates) / len(fee_rates) if fee_rates else 0,
                'fee_volatility': self._calculate_volatility(fee_rates),
                'total_fees_24h': sum(float(fee.get('amount', 0)) for fee in fees)
            }
        except Exception as e:
            logger.warning(f"获取费用模式数据失败: {str(e)}")
            return {
                'avg_fee_rate': 0,
                'fee_volatility': 0,
                'total_fees_24h': 0
            }

    def _calculate_price_impact(self, trades):
        if not trades:
            return 0
        try:
            price_changes = []
            for i in range(1, len(trades)):
                prev_price = float(trades[i-1].get('price', 0))
                curr_price = float(trades[i].get('price', 0))
                if prev_price > 0:
                    price_change = abs(curr_price - prev_price) / prev_price
                    price_changes.append(price_change)
            return sum(price_changes) / len(price_changes) if price_changes else 0
        except Exception:
            return 0

    def _calculate_volatility(self, values):
        if not values:
            return 0
        try:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return (variance ** 0.5) if variance > 0 else 0
        except Exception:
            return 0

    def analyze_pools(self):
        # 获取所有池子
        pools_df = self.get_all_pools()
        if pools_df.empty:
            logger.error("没有找到池子数据")
            return pd.DataFrame()

        # 先进行 TVL 和交易量筛选
        logger.info("筛选高质量池子...")
        filtered_pools = self.filter_high_quality_pools(pools_df)
        logger.info(f"找到 {len(filtered_pools)} 个符合条件的池子")

        if filtered_pools.empty:
            logger.warning("没有找到符合条件的池子")
            return pd.DataFrame()

        # 计算额外指标
        filtered_pools['fee_apy'] = filtered_pools.apply(
            lambda x: min((x['fees_24h'] * 365 * 100 / x['tvl']) if x['tvl'] > 0 else 0, 1000000), 
            axis=1
        )
        filtered_pools['total_apy'] = filtered_pools.apply(
            lambda x: min(x['apy'] + x['farm_apy'] + x['fee_apy'], 1000000),
            axis=1
        )
        filtered_pools['activity_ratio'] = filtered_pools['volume_24h'] / filtered_pools['tvl']
        
        # 计算风险指标
        filtered_pools['price_volatility'] = filtered_pools['bin_step'] / 100  # bin步长越大，价格波动性越大
        filtered_pools['liquidity_risk'] = 1 / np.log10(filtered_pools['tvl'] + 1)  # TVL越小，流动性风险越大
        
        # 计算综合得分 (追求高收益但风险可控)
        filtered_pools['score'] = (
            # 收益因素 (70%)
            0.7 * (
                0.5 * np.log10(filtered_pools['total_apy'] + 1) +  # 总APY (50%)
                0.3 * np.log10(filtered_pools['fee_apy'] + 1) +    # 费用APY (30%)
                0.2 * np.log10(filtered_pools['activity_ratio'] + 1)  # 活跃度 (20%)
            ) -
            # 风险因素 (30%)
            0.3 * (
                0.5 * filtered_pools['price_volatility'] +  # 价格波动风险 (50%)
                0.3 * filtered_pools['liquidity_risk'] +    # 流动性风险 (30%)
                0.2 * (filtered_pools['base_fee'] / 100)    # 费率风险 (20%)
            )
        )
        
        # 按得分排序
        result = filtered_pools.sort_values('score', ascending=False)
        
        # 显示统计信息
        logger.info("\n收益率统计:")
        logger.info(f"基础APY范围: {result['apy'].min():.2f}% - {result['apy'].max():.2f}%")
        logger.info(f"农场APY范围: {result['farm_apy'].min():.2f}% - {result['farm_apy'].max():.2f}%")
        logger.info(f"费用APY范围: {result['fee_apy'].min():.2f}% - {result['fee_apy'].max():.2f}%")
        logger.info(f"总APY范围: {result['total_apy'].min():.2f}% - {result['total_apy'].max():.2f}%")
        
        return result

    def filter_high_quality_pools(self, df, min_tvl=10000, min_volume=1000, top_n=100):
        """筛选高质量池子
        
        Args:
            df: 原始数据
            min_tvl: 最小TVL
            min_volume: 最小24h交易量
            top_n: 保留前N个池子
        """
        # 基础筛选
        filtered = df[
            (df['tvl'] >= min_tvl) & 
            (df['volume_24h'] >= min_volume)
        ]
        
        # 按照活跃度（交易量/TVL比率）排序
        filtered['activity_ratio'] = filtered['volume_24h'] / filtered['tvl']
        filtered = filtered.sort_values('activity_ratio', ascending=False)
        
        # 只保留前N个池子
        result = filtered.head(top_n)
        
        logger.info(f"筛选条件: TVL >= {min_tvl}, 24h交易量 >= {min_volume}")
        logger.info(f"初步筛选后的池子数量: {len(filtered)}")
        logger.info(f"最终保留池子数量: {len(result)}")
        
        if not result.empty:
            logger.info(f"筛选后池子的TVL范围: {result['tvl'].min():.2f} - {result['tvl'].max():.2f}")
            logger.info(f"筛选后池子的交易量范围: {result['volume_24h'].min():.2f} - {result['volume_24h'].max():.2f}")
            logger.info(f"筛选后池子的活跃度范围: {result['activity_ratio'].min():.4f} - {result['activity_ratio'].max():.4f}")
        
        return result

def main():
    try:
        logger.info("开始分析池子...")
        analyzer = MeteoraDLMMAnalyzer()
        
        # Get and analyze filtered pools
        results = analyzer.analyze_pools()
        
        if not results.empty:
            # Save results
            results.to_csv('top_100_active_pools.csv', index=False)
            logger.info(f"分析完成. 结果已保存到 top_100_active_pools.csv")
            
            # 显示前10个最佳投资池子
            top_10 = results.head(10)
            logger.info("\n前10个最佳LP投资池子 (高收益+风险可控):")
            for i, (_, pool) in enumerate(top_10.iterrows(), 1):
                logger.info(f"#{i} {pool['name']}")
                logger.info(f"地址: {pool['address']}")
                logger.info(f"综合得分: {pool['score']:.4f}")
                logger.info(f"TVL: ${pool['tvl']:,.2f}")
                logger.info(f"24h交易量: ${pool['volume_24h']:,.2f}")
                logger.info(f"总APY: {pool['total_apy']:.2f}%")
                logger.info(f"基础APY: {pool['apy']:.2f}%")
                logger.info(f"农场APY: {pool['farm_apy']:.2f}%")
                logger.info(f"费用APY: {pool['fee_apy']:.2f}%")
                logger.info(f"活跃度: {pool['activity_ratio']:.2f}")
                logger.info(f"基础费率: {pool['base_fee']:.2f}%")
                logger.info(f"最大费率: {pool['max_fee']:.2f}%")
                logger.info(f"Bin步长: {pool['bin_step']}")
                logger.info("---")
        else:
            logger.warning("没有找到符合条件的池子，未生成分析报告")
        
    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 