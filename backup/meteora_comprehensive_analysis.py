import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import json

# API基础URL
API_BASE = "https://dlmm-api.meteora.ag"

def get_all_pools():
    """从API获取所有池子的数据"""
    try:
        print("正在从API获取所有池子数据...")
        response = requests.get(f"{API_BASE}/pair/all")
        if response.status_code != 200:
            print(f"获取池子数据失败，状态码: {response.status_code}")
            return []
        
        pools = response.json()
        print(f"成功获取{len(pools)}个池子的数据")
        return pools
    except Exception as e:
        print(f"获取池子数据时出错: {str(e)}")
        return []

def process_pool_data(pools):
    """处理池子数据，提取关键信息"""
    processed_data = []
    
    for pool in pools:
        # 提取基本信息
        address = pool.get('address', '')
        name = pool.get('name', '')
        mint_x = pool.get('mint_x', '')
        mint_y = pool.get('mint_y', '')
        bin_step = pool.get('bin_step', 0)
        base_fee_percentage = pool.get('base_fee_percentage', 0)
        max_fee_percentage = pool.get('max_fee_percentage', 0)
        protocol_fee_percentage = pool.get('protocol_fee_percentage', 0)
        
        # 提取流动性和价格信息
        reserve_x_amount = pool.get('reserve_x_amount', 0)
        reserve_y_amount = pool.get('reserve_y_amount', 0)
        liquidity = pool.get('liquidity', 0)
        current_price = pool.get('current_price', 0)
        
        # 提取交易量和费用信息
        volume_24h = pool.get('trade_volume_24h', 0)
        fees_24h = pool.get('fees_24h', 0)
        cumulative_volume = pool.get('cumulative_trade_volume', 0)
        cumulative_fees = pool.get('cumulative_fee_volume', 0)
        
        # 提取APR/APY信息
        apr = pool.get('apr', 0)
        apy = pool.get('apy', 0)
        farm_apr = pool.get('farm_apr', 0)
        farm_apy = pool.get('farm_apy', 0)
        
        # 提取时间序列数据
        volume_data = pool.get('volume', {})
        fees_data = pool.get('fees', {})
        fee_tvl_ratio_data = pool.get('fee_tvl_ratio', {})
        
        # 计算TVL (Total Value Locked)
        # 这里需要获取代币价格，但API没有提供，所以我们使用当前价格作为估计
        tvl_in_y = reserve_y_amount + (reserve_x_amount * current_price) if current_price > 0 else 0
        
        # 计算费用率
        fee_rate = (fees_24h / volume_24h * 100) if volume_24h > 0 else 0
        
        # 计算费用/TVL比率
        fee_tvl_ratio = (fees_24h / tvl_in_y * 100) if tvl_in_y > 0 else 0
        
        # 计算估计APR (基于24小时费用)
        estimated_apr = (fees_24h * 365 / tvl_in_y * 100) if tvl_in_y > 0 else 0
        
        # 计算交易量稳定性 (使用可用的时间序列数据)
        volume_values = [
            volume_data.get('min_30', 0),
            volume_data.get('hour_1', 0),
            volume_data.get('hour_2', 0),
            volume_data.get('hour_4', 0),
            volume_data.get('hour_12', 0),
            volume_data.get('hour_24', 0)
        ]
        
        # 过滤掉0值
        volume_values = [v for v in volume_values if v > 0]
        volume_stability = 0
        if len(volume_values) > 1:
            avg_volume = sum(volume_values) / len(volume_values)
            volume_cv = np.std(volume_values) / avg_volume if avg_volume > 0 else float('inf')
            volume_stability = 1 / (1 + volume_cv)  # 值接近1表示高稳定性
        
        # 计算无常损失风险
        price_volatility = 1.0  # 默认为高波动性
        normalized_bin_step = bin_step / 100  # 将基点转换为百分比
        il_risk = price_volatility / (normalized_bin_step * 2) if normalized_bin_step > 0 else price_volatility * 2
        il_risk = min(1.0, max(0.0, il_risk))  # 确保在0-1范围内
        
        # 计算风险调整后的APR
        risk_adjusted_apr = estimated_apr * (1 - il_risk * 0.5)
        
        # 计算综合评分
        final_score = risk_adjusted_apr * 0.5  # 风险调整后收益占50%
        
        processed_data.append({
            'address': address,
            'name': name,
            'mint_x': mint_x,
            'mint_y': mint_y,
            'bin_step': bin_step,
            'base_fee_percentage': base_fee_percentage,
            'max_fee_percentage': max_fee_percentage,
            'protocol_fee_percentage': protocol_fee_percentage,
            'reserve_x_amount': reserve_x_amount,
            'reserve_y_amount': reserve_y_amount,
            'liquidity': liquidity,
            'current_price': current_price,
            'tvl': tvl_in_y,
            'volume_24h': volume_24h,
            'fees_24h': fees_24h,
            'fee_rate': fee_rate,
            'fee_tvl_ratio': fee_tvl_ratio,
            'cumulative_volume': cumulative_volume,
            'cumulative_fees': cumulative_fees,
            'apr': apr,
            'apy': apy,
            'farm_apr': farm_apr,
            'farm_apy': farm_apy,
            'volume_stability': volume_stability,
            'il_risk': il_risk,
            'estimated_apr': estimated_apr,
            'risk_adjusted_apr': risk_adjusted_apr,
            'final_score': final_score
        })
    
    return processed_data

def filter_high_activity_pools(pools_df, min_tvl=10000, min_volume=10000, min_fee_tvl_ratio=0.001):
    """筛选高活跃度的池子"""
    filtered_df = pools_df[
        (pools_df['tvl'] >= min_tvl) & 
        (pools_df['volume_24h'] >= min_volume) & 
        (pools_df['fee_tvl_ratio'] >= min_fee_tvl_ratio)
    ]
    
    # 按最终评分排序
    filtered_df = filtered_df.sort_values('final_score', ascending=False).reset_index(drop=True)
    
    return filtered_df

def identify_best_pools_per_pair(pools_df):
    """识别每个交易对中最佳的池子"""
    # 按交易对名称分组，然后选择每组中评分最高的池子
    best_pools = pools_df.loc[pools_df.groupby('name')['final_score'].idxmax()]
    
    # 按评分排序
    best_pools = best_pools.sort_values('final_score', ascending=False).reset_index(drop=True)
    
    return best_pools

def main():
    # 获取所有池子数据
    all_pools = get_all_pools()
    
    if not all_pools:
        print("没有获取到池子数据，退出程序")
        return
    
    # 处理池子数据
    processed_data = process_pool_data(all_pools)
    
    if not processed_data:
        print("处理池子数据失败，退出程序")
        return
    
    # 转换为DataFrame
    pools_df = pd.DataFrame(processed_data)
    
    # 保存所有池子数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_pools_file = f"meteora_all_pools_{timestamp}.csv"
    pools_df.to_csv(all_pools_file, index=False)
    print(f"所有池子数据已保存到 {all_pools_file}")
    
    # 筛选高活跃度池子
    high_activity_pools = filter_high_activity_pools(pools_df)
    high_activity_file = f"meteora_high_activity_pools_{timestamp}.csv"
    high_activity_pools.to_csv(high_activity_file, index=False)
    print(f"高活跃度池子数据已保存到 {high_activity_file}，共{len(high_activity_pools)}个池子")
    
    # 识别每个交易对中最佳的池子
    best_pools = identify_best_pools_per_pair(high_activity_pools)
    best_pools_file = f"meteora_best_pools_{timestamp}.csv"
    best_pools.to_csv(best_pools_file, index=False)
    print(f"最佳池子数据已保存到 {best_pools_file}，共{len(best_pools)}个交易对")
    
    # 显示前10个最佳池子
    print("\n前10个最佳LP投资机会:")
    top_pools = best_pools.head(10)
    for i, (_, pool) in enumerate(top_pools.iterrows()):
        print(f"{i+1}. {pool['name']}: 风险调整APR {pool['risk_adjusted_apr']:.2f}%, 无常损失风险 {pool['il_risk']:.2f}, 评分 {pool['final_score']:.2f}")

if __name__ == "__main__":
    main() 