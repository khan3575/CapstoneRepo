"""
Configuration Management Module
Loads and provides access to centralized configuration from config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Configuration manager for BraTS GNN Segmentation project.
    Loads settings from config.yaml and provides easy access to paths and parameters.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml. If None, searches in project root.
        """
        # Find project root (directory containing config.yaml)
        if config_path is None:
            # Start from this file's directory and search upwards
            current_dir = Path(__file__).parent
            while current_dir != current_dir.parent:
                config_file = current_dir / 'config.yaml'
                if config_file.exists():
                    config_path = str(config_file)
                    break
                current_dir = current_dir.parent

            if config_path is None:
                raise FileNotFoundError(
                    "config.yaml not found. Please create it in project root."
                )

        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent

        # Load configuration
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Resolve all paths to absolute paths
        self._resolve_paths()

    def _resolve_paths(self):
        """Convert all relative paths to absolute paths."""
        # Project root is the reference for all relative paths
        if 'paths' not in self._config:
            return

        for section, paths in self._config['paths'].items():
            if isinstance(paths, dict):
                for key, path in paths.items():
                    if isinstance(path, str) and not Path(path).is_absolute():
                        # Convert to absolute path
                        self._config['paths'][section][key] = str(
                            self.project_root / path
                        )

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'model.gnn.hidden_channels')
            default: Default value if key not found

        Returns:
            Configuration value

        Examples:
            >>> config.get('model.gnn.hidden_channels')  # Returns 256
            >>> config.get('paths.data.graphs')  # Returns '/path/to/data/graphs'
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """
        Get top-level configuration section.

        Args:
            key: Top-level configuration key

        Returns:
            Configuration section

        Examples:
            >>> config['model']  # Returns entire model configuration
            >>> config['paths']  # Returns entire paths configuration
        """
        return self._config.get(key, {})

    # =============================================================================
    # Convenience Properties for Common Paths
    # =============================================================================

    @property
    def paths(self) -> Dict[str, Any]:
        """Get all paths configuration."""
        return self._config.get('paths', {})

    # Raw Data Paths
    @property
    def brats_2021_raw(self) -> str:
        """Path to raw BraTS 2021 dataset."""
        return self.get('paths.brats_2021_raw', '')

    @property
    def brats_2023_raw(self) -> str:
        """Path to raw BraTS 2023 dataset."""
        return self.get('paths.brats_2023_raw', '')

    # Processed Data Paths
    @property
    def data_root(self) -> str:
        """Path to data root directory."""
        return self.get('paths.data.root', '')

    @property
    def data_organized(self) -> str:
        """Path to organized data directory."""
        return self.get('paths.data.organized', '')

    @property
    def data_preprocessed(self) -> str:
        """Path to preprocessed data directory."""
        return self.get('paths.data.preprocessed', '')

    @property
    def data_graphs(self) -> str:
        """Path to graph data directory."""
        return self.get('paths.data.graphs', '')

    @property
    def data_cv_folds(self) -> str:
        """Path to cross-validation folds directory."""
        return self.get('paths.data.cv_folds', '')

    # Checkpoint Paths
    @property
    def checkpoints_root(self) -> str:
        """Path to checkpoints root directory."""
        return self.get('paths.checkpoints.root', '')

    @property
    def checkpoints_binary(self) -> str:
        """Path to binary training checkpoints."""
        return self.get('paths.checkpoints.binary_training', '')

    @property
    def checkpoints_multiclass(self) -> str:
        """Path to multiclass training checkpoints."""
        return self.get('paths.checkpoints.multiclass_training', '')

    @property
    def checkpoints_baseline(self) -> str:
        """Path to baseline model checkpoints."""
        return self.get('paths.checkpoints.baseline_models', '')

    # Results Paths
    @property
    def results_root(self) -> str:
        """Path to research results root directory."""
        return self.get('paths.results.root', '')

    @property
    def results_ensemble(self) -> str:
        """Path to ensemble results directory."""
        return self.get('paths.results.ensemble', '')

    @property
    def results_ablation(self) -> str:
        """Path to ablation study results directory."""
        return self.get('paths.results.ablation', '')

    @property
    def results_baseline_comparison(self) -> str:
        """Path to baseline comparison results directory."""
        return self.get('paths.results.baseline_comparison', '')

    @property
    def results_speed_benchmark(self) -> str:
        """Path to speed benchmark results directory."""
        return self.get('paths.results.speed_benchmark', '')

    @property
    def results_qualitative(self) -> str:
        """Path to qualitative examples directory."""
        return self.get('paths.results.qualitative_examples', '')

    # Visualization Paths
    @property
    def visualizations_root(self) -> str:
        """Path to visualizations root directory."""
        return self.get('paths.visualizations.root', '')

    @property
    def visualizations_qualitative(self) -> str:
        """Path to qualitative visualizations directory."""
        return self.get('paths.visualizations.qualitative', '')

    # Logs Paths
    @property
    def logs_root(self) -> str:
        """Path to logs root directory."""
        return self.get('paths.logs.root', '')

    # =============================================================================
    # Model Configuration Properties
    # =============================================================================

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get('model', {})

    @property
    def gnn_config(self) -> Dict[str, Any]:
        """Get GNN architecture configuration."""
        return self.get('model.gnn', {})

    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('model.training', {})

    @property
    def cv_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration."""
        return self.get('model.cross_validation', {})

    # =============================================================================
    # Data Processing Configuration
    # =============================================================================

    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config.get('preprocessing', {})

    @property
    def graph_construction_config(self) -> Dict[str, Any]:
        """Get graph construction configuration."""
        return self._config.get('graph_construction', {})

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        dirs_to_create = [
            self.data_root,
            self.data_organized,
            self.data_preprocessed,
            self.data_graphs,
            self.data_cv_folds,
            self.checkpoints_root,
            self.checkpoints_binary,
            self.checkpoints_multiclass,
            self.checkpoints_baseline,
            self.results_root,
            self.results_ensemble,
            self.results_ablation,
            self.results_baseline_comparison,
            self.results_speed_benchmark,
            self.results_qualitative,
            self.visualizations_root,
            self.visualizations_qualitative,
            self.logs_root,
        ]

        for directory in dirs_to_create:
            if directory:  # Only create if path is not empty
                Path(directory).mkdir(parents=True, exist_ok=True)

        print(f"✅ All necessary directories created/verified")

    def print_config(self):
        """Print configuration summary."""
        print("=" * 80)
        print(f"📋 Configuration: {self.config_path}")
        print("=" * 80)
        print(f"Project Root: {self.project_root}")
        print(f"\n🗂️  Data Paths:")
        print(f"  BraTS 2021 Raw: {self.brats_2021_raw}")
        print(f"  Preprocessed: {self.data_preprocessed}")
        print(f"  Graphs: {self.data_graphs}")
        print(f"  CV Folds: {self.data_cv_folds}")
        print(f"\n💾 Model Paths:")
        print(f"  Binary Checkpoints: {self.checkpoints_binary}")
        print(f"  Results: {self.results_root}")
        print(f"\n🎛️  Model Config:")
        print(f"  Architecture: {self.get('model.gnn.architecture')}")
        print(f"  Hidden Channels: {self.get('model.gnn.hidden_channels')}")
        print(f"  Num Layers: {self.get('model.gnn.num_layers')}")
        print(f"  Batch Size: {self.get('model.training.batch_size')}")
        print(f"  Learning Rate: {self.get('model.training.learning_rate')}")
        print("=" * 80)

    def save(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.

        Args:
            output_path: Path to save config. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path

        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

        print(f"✅ Configuration saved to {output_path}")


# =============================================================================
# Global Configuration Instance
# =============================================================================

# Create global config instance (singleton pattern)
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None, reload: bool = False) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to config.yaml. If None, auto-detects.
        reload: Force reload configuration from file.

    Returns:
        Config instance

    Examples:
        >>> from config import get_config
        >>> config = get_config()
        >>> graphs_path = config.data_graphs
    """
    global _global_config

    if _global_config is None or reload:
        _global_config = Config(config_path)

    return _global_config


# =============================================================================
# Command-line Testing
# =============================================================================

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    config.print_config()

    # Test path access
    print(f"\n🧪 Testing path access:")
    print(f"  Graphs directory: {config.data_graphs}")
    print(f"  Checkpoint directory: {config.checkpoints_binary}")
    print(f"  Batch size: {config.get('model.training.batch_size')}")

    # Create directories
    print(f"\n📁 Creating directories...")
    config.ensure_directories()
    print(f"\n✅ Configuration module working correctly!")
