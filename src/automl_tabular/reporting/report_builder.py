"""Report generation utilities."""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
import markdown


class ReportBuilder:
    """Builds HTML reports from AutoML results."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report builder.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Add markdown filter
        self.env.filters['markdown'] = lambda text: markdown.markdown(text)
    
    def build_report(
        self,
        context: Dict,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Build HTML report from context.
        
        Args:
            context: Dictionary with all report data
            output_filename: Optional custom filename
            
        Returns:
            Path to generated report
        """
        # Add timestamp if not present
        if 'timestamp' not in context:
            context['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add run ID if not present
        if 'run_id' not in context:
            context['run_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load template
        template = self.env.get_template('report_template.html')
        
        # Render HTML
        html_content = template.render(**context)
        
        # Generate output filename
        if output_filename is None:
            output_filename = f"automl_report_{context['run_id']}.html"
        
        output_path = self.output_dir / output_filename
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def prepare_context(
        self,
        data_info: Dict,
        problem_type: str,
        target_column: str,
        leaderboard_data: List[Dict],
        best_model_info: Dict,
        feature_importance_df = None,
        plots: Dict[str, str] = None,
        warnings: List[str] = None,
        model_summary: str = "",
        feature_importance_summary: str = "",
        recommendations: str = "",
        model_path: str = "",
        selection_info: Dict = None,
        top_contenders: List[Dict] = None,
        model_family_summary: List[Dict] = None
    ) -> Dict:
        """
        Prepare context dictionary for report template.
        
        Args:
            data_info: Dataset information
            problem_type: 'classification' or 'regression'
            target_column: Target column name
            leaderboard_data: List of model results
            best_model_info: Best model details
            feature_importance_df: Feature importance DataFrame
            plots: Dictionary of plot paths
            warnings: List of warning messages
            model_summary: Model summary text
            feature_importance_summary: Feature importance text
            recommendations: Recommendations text
            model_path: Path to saved model
            selection_info: Model selection information (tolerance, strategy, etc.)
            top_contenders: List of top contender models within tolerance
            
        Returns:
            Context dictionary for template
        """
        plots = plots or {}
        warnings = warnings or []
        selection_info = selection_info or {'is_tie': False}
        top_contenders = top_contenders or []
        model_family_summary = model_family_summary or []
        
        # Convert plot paths to relative paths for HTML
        relative_plots = {}
        for key, plot_path in plots.items():
            if plot_path:
                # Convert to relative path from report directory
                plot_path_obj = Path(plot_path)
                relative_plots[key] = f"plots/{plot_path_obj.name}"
        
        # Count number of model families
        num_model_families = len(set(row['model'] for row in leaderboard_data))
        
        context = {
            # Dataset info
            'n_rows': data_info.get('n_rows', 0),
            'n_features': data_info.get('n_columns', 0) - 1,  # Exclude target
            'target_column': target_column,
            'warnings': warnings,
            
            # Problem info
            'problem_type': problem_type,
            
            # Model info
            'best_model_name': best_model_info.get('model_name', 'Unknown'),
            'metric_name': best_model_info.get('metric_name', 'score'),
            'metric_value': best_model_info.get('metric_value', 0.0),
            'total_models_tried': len(leaderboard_data),
            'num_model_families': num_model_families,
            
            # Model selection
            'selection_info': selection_info,
            'top_contenders': top_contenders,
            'top_contender_names': [c['model_name'] for c in top_contenders] if top_contenders else [],
            'num_top_contenders': sum(1 for row in leaderboard_data if row.get('is_top_contender', False)),
            
            # Leaderboard
            'leaderboard': leaderboard_data,
            'model_family_summary': model_family_summary,
            
            # Summaries
            'model_summary': self._markdown_to_html(model_summary),
            'feature_importance_summary': self._markdown_to_html(feature_importance_summary),
            'recommendations': self._markdown_to_html(recommendations),
            
            # Plots (using relative paths)
            'feature_importance_plot': relative_plots.get('feature_importance'),
            'target_distribution_plot': relative_plots.get('target_distribution'),
            'model_comparison_plot': relative_plots.get('model_comparison'),
            'missing_data_plot': relative_plots.get('missing_data'),
            'confusion_matrix_plot': relative_plots.get('confusion_matrix'),
            'additional_plots': [],
            
            # Model path
            'model_path': model_path,
        }
        
        return context
    
    def _markdown_to_html(self, text: str) -> str:
        """Convert markdown text to HTML."""
        if not text:
            return ""
        
        # Simple markdown conversion
        html = text.replace('\n\n', '</p><p>')
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        html = html.replace('##', '<h3>').replace('\n', '</h3>', 1)
        
        # Handle lists
        lines = text.split('\n')
        in_list = False
        result = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    result.append('<ul>')
                    in_list = True
                result.append(f'<li>{line.strip()[2:]}</li>')
            else:
                if in_list:
                    result.append('</ul>')
                    in_list = False
                
                # Handle headings
                if line.startswith('##'):
                    result.append(f'<h3>{line.replace("##", "").strip()}</h3>')
                elif line.startswith('###'):
                    result.append(f'<h4>{line.replace("###", "").strip()}</h4>')
                elif line.strip():
                    # Handle bold
                    line = line.replace('**', '<strong>', 1)
                    line = line.replace('**', '</strong>', 1)
                    result.append(f'<p>{line}</p>')
                else:
                    result.append('<br>')
        
        if in_list:
            result.append('</ul>')
        
        return '\n'.join(result)


__all__ = ["ReportBuilder"]
