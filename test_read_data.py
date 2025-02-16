import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def test_read_file(filepath, shape=(512, 768, 3, 94), dtype=np.float32):
    """
    Test reading a single data file
    Args:
        filepath: Path to the data file
        shape: Expected data shape (x, y, var, t)
        dtype: Data type for reading (np.float32 or np.float64)
    """
    try:
        # Read binary file with specified data type
        data = np.fromfile(filepath, dtype=dtype)
        print(f"Successfully read file: {filepath}")
        print(f"Using dtype: {dtype}")
        print(f"Raw data shape: {data.shape}")
        print(f"Total elements: {data.size}")
        print(f"Expected elements: {np.prod(shape)}")
        
        # Check if total elements match expected size
        expected_size = np.prod(shape)
        if data.size != expected_size:
            print(f"Warning: Data size mismatch!")
            print(f"Expected {expected_size} elements, but got {data.size} elements")
            return None
        
        # Reshape data (using Fortran order)
        data = data.reshape(shape, order='F')
        print(f"\nReshaped data:")
        print(f"Shape: {data.shape}")
        
        # Basic statistics
        print(f"\nBasic statistics:")
        print(f"Number of NaN values: {np.isnan(data).sum()}")
        print(f"Min value (excluding NaN): {np.nanmin(data)}")
        print(f"Max value (excluding NaN): {np.nanmax(data)}")
        print(f"Mean value (excluding NaN): {np.nanmean(data)}")
        
        # Create visualization directory if it doesn't exist
        vis_dir = "test_visualization"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure name based on dtype and timestamp
        fig_name = f"variables_dtype_{dtype.__name__}_{timestamp}.png"
        fig_path = os.path.join(vis_dir, fig_name)
        
        # Visualization check
        # Show three variables at first time point
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(data[:, :, i, 0])
            plt.colorbar()
            plt.title(f'Variable {i+1}, Time 0')
        plt.tight_layout()
        
        # Save figure and close it
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {fig_path}")
        
        # Check values at specific position
        print("\nSample values at position [256, 384] (middle point):")
        for i in range(3):
            print(f"Variable {i+1}: {data[256, 384, i, 0]}")
        
        return data
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def visualize_time_samples(data: np.ndarray, num_samples: int = 10) -> None:
    """
    Visualize data variables at multiple time points
    Args:
        data: Data array with shape (x, y, var, t)
        num_samples: Number of time points to sample
    """
    # Create visualization directory if it doesn't exist
    vis_dir = "test_visualization"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate time indices to sample
    total_times = data.shape[3]
    time_indices = np.linspace(0, total_times-1, num_samples, dtype=int)
    
    # Create visualization for each sampled time point
    for t_idx in time_indices:
        # Create figure name
        fig_name = f"variables_time_{t_idx:03d}_{timestamp}.png"
        fig_path = os.path.join(vis_dir, fig_name)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(data[:, :, i, t_idx])
            plt.colorbar()
            plt.title(f'Variable {i+1}, Time {t_idx}')
        plt.tight_layout()
        
        # Save and close figure
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for time {t_idx} to: {fig_path}")

def analyze_data_distribution(data: np.ndarray) -> None:
    """
    Analyze data distribution and statistical properties
    Args:
        data: Data array with shape (x, y, var, t)
    """
    print("\n=== Data Distribution Analysis ===")
    
    # Basic statistics for each variable
    for var_idx in range(data.shape[2]):
        var_data = data[:, :, var_idx, :]
        
        # Create mask for valid data (not inf and not nan)
        valid_mask = ~np.isinf(var_data) & ~np.isnan(var_data)
        valid_data = var_data[valid_mask]
        
        print(f"\nVariable {var_idx + 1} Statistics:")
        print(f"Shape: {var_data.shape}")
        print(f"Valid data points: {valid_data.size:,} / {var_data.size:,} ({valid_data.size/var_data.size*100:.2f}%)")
        print(f"Mean: {np.mean(valid_data):.6f}")
        print(f"Std: {np.std(valid_data):.6f}")
        print(f"Min: {np.min(valid_data):.6f}")
        print(f"Max: {np.max(valid_data):.6f}")
        print(f"Median: {np.median(valid_data):.6f}")
        print(f"Skewness: {np.mean(((valid_data - np.mean(valid_data)) / np.std(valid_data)) ** 3):.6f}")
        print(f"Kurtosis: {np.mean(((valid_data - np.mean(valid_data)) / np.std(valid_data)) ** 4):.6f}")
        print(f"NaN count: {np.isnan(var_data).sum():,}")
        print(f"Inf count: {np.isinf(var_data).sum():,}")
        
        # Calculate percentiles for valid data
        percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        perc_values = np.percentile(valid_data, percentiles)
        print("\nPercentiles:")
        for p, v in zip(percentiles, perc_values):
            print(f"{p}th percentile: {v:.6f}")
        
        # Create visualization directory if it doesn't exist
        vis_dir = "test_visualization"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot histogram with valid data only
        plt.figure(figsize=(15, 5))
        
        # Overall distribution
        plt.subplot(1, 3, 1)
        plt.hist(valid_data, bins=100, density=True)
        plt.title(f'Variable {var_idx + 1} Distribution\n(Valid Data Only)')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        # Time series of mean and std (for valid data)
        plt.subplot(1, 3, 2)
        time_means = []
        time_stds = []
        for t in range(var_data.shape[-1]):
            time_slice = var_data[:, :, t]
            valid_slice = time_slice[~np.isinf(time_slice) & ~np.isnan(time_slice)]
            if valid_slice.size > 0:
                time_means.append(np.mean(valid_slice))
                time_stds.append(np.std(valid_slice))
            else:
                time_means.append(np.nan)
                time_stds.append(np.nan)
                
        time_means = np.array(time_means)
        time_stds = np.array(time_stds)
        
        plt.plot(time_means, label='Mean')
        plt.plot(time_means + time_stds, '--', label='+1 Std', alpha=0.5)
        plt.plot(time_means - time_stds, '--', label='-1 Std', alpha=0.5)
        plt.title(f'Variable {var_idx + 1} Time Evolution\n(Valid Data Only)')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        
        # Spatial pattern (mean over time)
        plt.subplot(1, 3, 3)
        spatial_data = var_data.copy()
        spatial_data[~valid_mask] = np.nan
        spatial_mean = np.nanmean(spatial_data, axis=-1)
        plt.imshow(spatial_mean)
        plt.colorbar()
        plt.title(f'Variable {var_idx + 1} Mean Spatial Pattern\n(Valid Data Only)')
        
        plt.tight_layout()
        
        # Save and close figure
        fig_path = os.path.join(vis_dir, f"distribution_var_{var_idx+1}_{timestamp}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nDistribution plot saved to: {fig_path}")

if __name__ == "__main__":
    test_file = "data/dcape/tpe20050702nor.dat"
    
    # Test with float32
    print("\n=== Testing with float32 ===")
    data_f32 = test_read_file(test_file, dtype=np.float32)
    
    # Test with float64
    print("\n=== Testing with float64 ===")
    data_f64 = test_read_file(test_file, dtype=np.float64)
    
    if data_f32 is not None or data_f64 is not None:
        print("\nData reading test completed!")
        
        # Compare data if both reads were successful
        if data_f32 is not None and data_f64 is not None:
            print("\nComparing float32 and float64 data:")
            diff = np.abs(data_f32 - data_f64)
            print(f"Maximum absolute difference: {np.nanmax(diff)}")
            print(f"Mean absolute difference: {np.nanmean(diff)}")
        
        # If data reading is successful, perform time sample visualization
        if data_f32 is not None:
            print("\nGenerating time sample visualizations...")
            visualize_time_samples(data_f32)
            
            print("\nAnalyzing data distribution...")
            analyze_data_distribution(data_f32) 