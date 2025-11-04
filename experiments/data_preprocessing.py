"""
Speed Dating 数据预处理模块

功能:
1. 加载 Speed Dating 数据集
2. 过滤高质量样本
3. 提取关键特征
4. 生成训练/测试配对
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

class SpeedDatingDataProcessor:
    def __init__(self, data_path: str):
        """
        初始化数据处理器
        
        Args:
            data_path: Speed Dating Data.csv 的路径
        """
        self.data_path = data_path
        self.df = None
        self.clean_df = None
        self.pairs = []
        
    def load_data(self):
        """加载数据"""
        print("📂 Loading Speed Dating dataset...")
        self.df = pd.read_csv(self.data_path, encoding='latin1')
        print(f"✅ Loaded {len(self.df)} records")
        print(f"   Columns: {self.df.shape[1]}")
        print(f"   Unique participants: {self.df['iid'].nunique()}")
        return self.df
    
    def filter_quality_samples(self):
        """
        过滤高质量样本（中高覆盖率版本 >50%）
        
        使用所有中等和高覆盖率字段（>50%），最大化数据完整性
        
        必需字段分组:
        1. 核心信息: demographics, background, dating behavior
        2. Time 1 完整数据: preferences (self/opposite/same), self-ratings, others' perception
        3. 兴趣爱好: 至少12个有效值 (70%)
        4. Scorecard: 完整评分
        5. Ground truth: 决策和匹配结果
        """
        print("\n🔍 Filtering quality samples (Medium-High coverage >50%)...")
        print("=" * 70)
        
        df = self.df.copy()
        initial_count = len(df)
        
        # 1. 核心人口统计学（98-100% coverage）
        demographics = ['age', 'gender', 'field_cd', 'career_c', 'race']
        df = df.dropna(subset=demographics)
        print(f"   ✅ Demographics (5 fields): {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 2. 背景态度（98-99% coverage）
        background = ['imprace', 'imprelig', 'goal', 'date', 'go_out']
        df = df.dropna(subset=background)
        print(f"   ✅ Background & behavior: {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 3. 期望（只要 exphappy，不要 expnum 因为只有21.5%）
        df = df.dropna(subset=['exphappy'])
        print(f"   ✅ Expectations (exphappy): {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 4. 自我择偶偏好（99% coverage）
        preferences_self = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
        df = df.dropna(subset=preferences_self)
        print(f"   ✅ Preferences (self): {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 5. 对异性择偶观的预测（99% coverage）
        preferences_opposite = ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1']
        df = df.dropna(subset=preferences_opposite)
        print(f"   ✅ Preferences (opposite sex): {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 6. 对同性择偶观的预测（77% coverage - 中等）
        preferences_same = ['attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1']
        df = df.dropna(subset=preferences_same)
        print(f"   ⚠️  Preferences (same sex): {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 7. 自我评价（99% coverage）
        self_ratings = ['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']
        df = df.dropna(subset=self_ratings)
        print(f"   ✅ Self-ratings: {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 8. 他人眼中的自己（58.6% coverage - 中等）
        others_perception = ['attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1']
        df = df.dropna(subset=others_perception)
        print(f"   ⚠️  Others' perception: {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 9. 兴趣爱好至少12个有效值（降低到70%，更宽松）
        interests = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
                    'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
                    'movies', 'concerts', 'music', 'shopping', 'yoga']
        df['valid_interests'] = df[interests].notna().sum(axis=1)
        df = df[df['valid_interests'] >= 12]  # 12/17 = 70%
        print(f"   ✅ Interests (≥12/17): {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 10. Scorecard 完整（包括 shar，87% coverage 是瓶颈）
        scorecard = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like']
        df = df.dropna(subset=scorecard)
        print(f"   ⚠️  Scorecard (7 fields): {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        # 11. 必须有决策和匹配结果（100% coverage）
        df = df.dropna(subset=['dec', 'match'])
        print(f"   ✅ Ground truth: {len(df):,} / {initial_count:,} ({len(df)/initial_count*100:.1f}%)")
        
        print("=" * 70)
        
        self.clean_df = df
        print(f"\n🎉 Final clean dataset: {len(df):,} records ({len(df)/initial_count*100:.1f}%)")
        print(f"   Unique participants: {df['iid'].nunique()}")
        print(f"   Average features per person: {len(df.columns)} columns")
        
        return df
    
    def extract_pairs(self, n_matched: int = 50, n_unmatched: int = 50):
        """
        提取配对样本
        
        Args:
            n_matched: 目标匹配对数
            n_unmatched: 目标非匹配对数
        
        Returns:
            pairs: List[Dict] 配对信息
        """
        print(f"\n🎯 Extracting {n_matched} matched + {n_unmatched} unmatched pairs...")
        
        df = self.clean_df
        pairs = []
        
        # 获取所有互相评价的记录
        # 对于每个 (iid, partner) 对，找到对应的 (partner, iid) 记录
        matched_pairs = []
        unmatched_pairs = []
        
        processed = set()
        
        for idx, row in df.iterrows():
            iid1 = row['iid']
            pid2 = row['pid']  # partner的iid
            
            # 避免重复处理
            pair_key = tuple(sorted([iid1, pid2]))
            if pair_key in processed:
                continue
            
            # 找到对方的记录
            partner_row = df[(df['iid'] == pid2) & (df['pid'] == iid1)]
            
            if len(partner_row) == 0:
                continue
            
            partner_row = partner_row.iloc[0]
            
            # 提取配对信息（全面版）
            pair_info = {
                'pair_id': f"pair_{len(pairs)+1:03d}",
                'person1': {
                    'iid': int(iid1),
                    'gender': int(row['gender']),
                    'age': int(row['age']),
                    'field_cd': int(row['field_cd']) if pd.notna(row['field_cd']) else None,
                    'career_c': int(row['career_c']) if pd.notna(row['career_c']) else None,
                    'race': int(row['race']) if pd.notna(row['race']) else None,
                    'imprace': int(row['imprace']) if pd.notna(row['imprace']) else None,
                    'imprelig': int(row['imprelig']) if pd.notna(row['imprelig']) else None,
                    'goal': int(row['goal']) if pd.notna(row['goal']) else None,
                    'date': int(row['date']) if pd.notna(row['date']) else None,
                    'go_out': int(row['go_out']) if pd.notna(row['go_out']) else None,
                    'data': row.to_dict()
                },
                'person2': {
                    'iid': int(pid2),
                    'gender': int(partner_row['gender']),
                    'age': int(partner_row['age']),
                    'field_cd': int(partner_row['field_cd']) if pd.notna(partner_row['field_cd']) else None,
                    'career_c': int(partner_row['career_c']) if pd.notna(partner_row['career_c']) else None,
                    'race': int(partner_row['race']) if pd.notna(partner_row['race']) else None,
                    'imprace': int(partner_row['imprace']) if pd.notna(partner_row['imprace']) else None,
                    'imprelig': int(partner_row['imprelig']) if pd.notna(partner_row['imprelig']) else None,
                    'goal': int(partner_row['goal']) if pd.notna(partner_row['goal']) else None,
                    'date': int(partner_row['date']) if pd.notna(partner_row['date']) else None,
                    'go_out': int(partner_row['go_out']) if pd.notna(partner_row['go_out']) else None,
                    'data': partner_row.to_dict()
                },
                'ground_truth': {
                    'person1_dec': int(row['dec']),
                    'person2_dec': int(partner_row['dec']),
                    'match': int(row['match']),
                    'person1_ratings': {
                        'attr': float(row['attr']),
                        'sinc': float(row['sinc']),
                        'intel': float(row['intel']),
                        'fun': float(row['fun']),
                        'amb': float(row['amb']),
                        'shar': float(row['shar']),
                        'like': float(row['like'])
                    },
                    'person2_ratings': {
                        'attr': float(partner_row['attr']),
                        'sinc': float(partner_row['sinc']),
                        'intel': float(partner_row['intel']),
                        'fun': float(partner_row['fun']),
                        'amb': float(partner_row['amb']),
                        'shar': float(partner_row['shar']),
                        'like': float(partner_row['like'])
                    }
                }
            }
            
            # 分类
            if pair_info['ground_truth']['match'] == 1:
                matched_pairs.append(pair_info)
            else:
                unmatched_pairs.append(pair_info)
            
            processed.add(pair_key)
        
        print(f"   Found {len(matched_pairs)} matched pairs")
        print(f"   Found {len(unmatched_pairs)} unmatched pairs")
        
        # 采样
        import random
        random.seed(42)
        
        selected_matched = random.sample(matched_pairs, min(n_matched, len(matched_pairs)))
        selected_unmatched = random.sample(unmatched_pairs, min(n_unmatched, len(unmatched_pairs)))
        
        self.pairs = selected_matched + selected_unmatched
        
        print(f"\n✅ Selected {len(self.pairs)} pairs:")
        print(f"   - Matched: {len(selected_matched)}")
        print(f"   - Unmatched: {len(selected_unmatched)}")
        
        return self.pairs
    
    def save_processed_data(self, output_dir: str = "results"):
        """保存处理后的数据"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存清洗后的数据集
        clean_csv_path = output_path / "clean_dataset.csv"
        self.clean_df.to_csv(clean_csv_path, index=False)
        print(f"\n💾 Saved clean dataset to {clean_csv_path}")
        
        # 保存配对信息
        pairs_json_path = output_path / "processed_pairs.json"
        with open(pairs_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.pairs, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved pairs to {pairs_json_path}")
        
        # 保存统计信息
        stats = {
            'total_records': len(self.df),
            'clean_records': len(self.clean_df),
            'unique_participants': int(self.clean_df['iid'].nunique()),
            'total_pairs': len(self.pairs),
            'matched_pairs': sum(1 for p in self.pairs if p['ground_truth']['match'] == 1),
            'unmatched_pairs': sum(1 for p in self.pairs if p['ground_truth']['match'] == 0),
            'age_distribution': self.clean_df['age'].describe().to_dict(),
            'gender_distribution': self.clean_df['gender'].value_counts().to_dict()
        }
        
        stats_path = output_path / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"💾 Saved statistics to {stats_path}")
        
        return output_path


def main():
    """主函数"""
    print("🎭 Speed Dating Data Preprocessing")
    print("=" * 50)
    
    # 初始化处理器
    data_path = "Speed Dating Data.csv"
    processor = SpeedDatingDataProcessor(data_path)
    
    # 加载数据
    processor.load_data()
    
    # 过滤高质量样本
    processor.filter_quality_samples()
    
    # 提取配对
    processor.extract_pairs(n_matched=50, n_unmatched=50)
    
    # 保存结果
    output_dir = processor.save_processed_data()
    
    print("\n" + "=" * 50)
    print("✅ Data preprocessing completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("\n🎯 Next steps:")
    print("   1. Run persona_generator.py to create Persona prompts")
    print("   2. Run speed_dating_simulator.py for Scenario 1")
    print("   3. Run critical_events_engine.py for Scenario 2")


if __name__ == "__main__":
    main()
