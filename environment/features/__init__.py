"""
ICT 特徵工程模塊

提供所有 Inner Circle Trader (ICT) 策略相關的技術分析特徵

模塊：
- market_structure: 市場結構（BOS/ChoCh）
- order_blocks: Order Blocks 檢測
- fvg: Fair Value Gaps 檢測
- liquidity: 流動性區域檢測
- volume: 成交量與價格分析
- multi_timeframe: 多時間框架趨勢
- feature_aggregator: 特徵整合器（主要接口）

作者：PPO Trading Team
日期：2026-01-14
"""

from .market_structure import MarketStructure
from .order_blocks import OrderBlockDetector
from .fvg import FVGDetector
from .liquidity import LiquidityDetector
from .volume import VolumeAnalyzer
from .multi_timeframe import MultiTimeframeAnalyzer
from .feature_aggregator import FeatureAggregator

__all__ = [
    'MarketStructure',
    'OrderBlockDetector',
    'FVGDetector',
    'LiquidityDetector',
    'VolumeAnalyzer',
    'MultiTimeframeAnalyzer',
    'FeatureAggregator'
]

__version__ = '1.0.0'
