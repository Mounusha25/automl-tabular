"""Leaderboard management for model comparison."""

import pandas as pd
from typing import List, Dict
from automl_tabular.models.search import TrialResult


class Leaderboard:
    """Manages and displays model performance leaderboard."""
    
    def __init__(self, results: List[TrialResult]):
        """
        Initialize leaderboard with trial results.
        
        Args:
            results: List of TrialResult objects
        """
        self.results = results
        self.df = None
        self._build_dataframe()
    
    def _build_dataframe(self) -> None:
        """Build pandas DataFrame from results."""
        if not self.results:
            self.df = pd.DataFrame()
            return
        
        data = []
        for i, result in enumerate(self.results, 1):
            row = {
                'rank': i,
                'model': result.model_name,
                'score': round(result.score, 4),
                'train_time_sec': round(result.train_time, 2),
                'params': str(result.params)
            }
            data.append(row)
        
        self.df = pd.DataFrame(data)
    
    def get_top_k(self, k: int = 5) -> pd.DataFrame:
        """
        Get top K models.
        
        Args:
            k: Number of top models
            
        Returns:
            DataFrame with top K models
        """
        return self.df.head(k)
    
    def get_best_model(self) -> TrialResult:
        """
        Get the best performing model.
        
        Returns:
            Best TrialResult
        """
        if not self.results:
            return None
        return self.results[0]
    
    def get_top_contenders(self, tolerance: float = 0.005) -> list:
        """
        Get all models within tolerance of the best model.
        
        Args:
            tolerance: Performance difference threshold (e.g., 0.005 = 0.5%)
            
        Returns:
            List of TrialResult objects within tolerance
        """
        if not self.results:
            return []
        
        best = self.results[0]
        best_metric = best.score
        
        contenders = [
            model for model in self.results
            if abs(best_metric - model.score) <= tolerance
        ]
        
        return contenders
    
    def to_dict(self, tolerance: float = 0.005) -> List[Dict]:
        """
        Convert leaderboard to list of dictionaries.
        
        Args:
            tolerance: Tolerance for marking top contenders
            
        Returns:
            List of dictionaries for each model
        """
        records = self.df.to_dict(orient='records')
        
        # Mark top contenders (exclude the best model itself)
        if records:
            best_score = records[0]['score']
            for record in records:
                diff = abs(best_score - record['score'])
                # Mark as contender only if it's not the best (rank 1) and within tolerance
                record['is_top_contender'] = (record['rank'] > 1) and (diff <= tolerance)
        
        return records
    
    def display(self, top_k: int = 10) -> None:
        """
        Display leaderboard.
        
        Args:
            top_k: Number of top models to display
        """
        if self.df.empty:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("MODEL LEADERBOARD")
        print("="*80)
        
        # Display top K
        display_df = self.get_top_k(top_k)[['rank', 'model', 'score', 'train_time_sec']]
        print(display_df.to_string(index=False))
        print("="*80)
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary stats
        """
        if self.df.empty:
            return {}
        
        return {
            'total_models_tried': len(self.results),
            'best_score': self.df['score'].max(),
            'best_model': self.df.iloc[0]['model'],
            'total_time': self.df['train_time_sec'].sum(),
            'avg_time_per_model': self.df['train_time_sec'].mean()
        }
    
    def get_model_family_summary(self) -> List[Dict]:
        """
        Get summary statistics grouped by model family.
        
        Returns:
            List of dictionaries with per-family statistics
        """
        if self.df.empty:
            return []
        
        summary = []
        for model_name in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model_name]
            summary.append({
                'model_family': model_name,
                'best_score': model_data['score'].max(),
                'mean_score': model_data['score'].mean(),
                'num_trials': len(model_data),
                'avg_time': model_data['train_time_sec'].mean()
            })
        
        # Sort by best score descending
        summary.sort(key=lambda x: x['best_score'], reverse=True)
        return summary


__all__ = ["Leaderboard"]
