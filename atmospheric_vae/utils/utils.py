import torch
import sys
import subprocess
import os
from typing import List, Tuple

class Utils:
    @staticmethod
    def get_gpu_memory_usage() -> List[Tuple[int, float, float]]:
        """
        Get memory usage for each GPU
        Returns:
            List of tuples (gpu_id, total_memory, used_memory) in GB
        """
        try:
            # Run nvidia-smi command
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.total,memory.used', 
                 '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            
            # Parse the output
            gpu_memory = []
            for line in output.strip().split('\n'):
                gpu_id, total_memory, used_memory = map(float, line.split(','))
                gpu_memory.append((int(gpu_id), total_memory/1024, used_memory/1024))
            return gpu_memory
            
        except Exception as e:
            print(f"Error getting GPU memory usage: {str(e)}")
            return []

    @staticmethod
    def get_free_gpu(memory_threshold: float = 0.9) -> int:
        """
        Find the first GPU with memory usage below threshold
        Args:
            memory_threshold: Maximum memory usage ratio to consider GPU as free
        Returns:
            GPU ID or -1 if no free GPU found
        """
        gpu_memory = Utils.get_gpu_memory_usage()
        
        for gpu_id, total, used in gpu_memory:
            if used/total < memory_threshold:
                return gpu_id
        return -1

    @staticmethod
    def test_pytorch_gpu(force_gpu_id: int = None) -> None:
        """
        Test PyTorch GPU availability and functionality
        Args:
            force_gpu_id: Force using specific GPU, if None will use first free GPU
        """
        print("=== PyTorch GPU Availability Test ===")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            # Display CUDA version
            print(f"CUDA version: {torch.version.cuda}")
            
            # Display GPU memory usage
            print("\nGPU Memory Usage:")
            for gpu_id, total, used in Utils.get_gpu_memory_usage():
                print(f"GPU {gpu_id}: {used:.2f}GB / {total:.2f}GB ({(used/total)*100:.1f}%)")
            
            # Select GPU
            if force_gpu_id is not None:
                gpu_id = force_gpu_id
            else:
                gpu_id = Utils.get_free_gpu()
            
            if gpu_id >= 0:
                try:
                    # Set GPU
                    torch.cuda.set_device(gpu_id)
                    print(f"\nUsing GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                    
                    # Perform simple GPU computation test
                    print("\nPerforming simple GPU computation test...")
                    
                    # Create tensors on GPU
                    x = torch.rand(1000, 1000).cuda()
                    y = torch.rand(1000, 1000).cuda()
                    
                    # Perform matrix multiplication
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    z = torch.matmul(x, y)
                    end_time.record()
                    
                    # Wait for GPU operations to complete
                    torch.cuda.synchronize()
                    
                    # Calculate execution time
                    print(f"Matrix multiplication time: {start_time.elapsed_time(end_time):.2f} ms")
                    print("GPU computation test successful!")
                    
                except Exception as e:
                    print(f"GPU computation test failed: {str(e)}")
            else:
                print("\nWarning: No free GPU available!")
                print("All GPUs are heavily utilized")
        else:
            print("\nWarning: No available GPU detected!")
            print("System will use CPU for computations")
            
            # Display CPU test
            print("\nCPU test:")
            x = torch.rand(1000, 1000)
            y = torch.rand(1000, 1000)
            z = torch.matmul(x, y)
            print("CPU matrix multiplication test successful!")

    @staticmethod
    def get_device(force_gpu_id: int = None) -> torch.device:
        """
        Get available computation device
        Args:
            force_gpu_id: Force using specific GPU, if None will use first free GPU
        Returns:
            torch.device: Returns GPU if available, otherwise CPU
        """
        if torch.cuda.is_available():
            if force_gpu_id is not None:
                gpu_id = force_gpu_id
            else:
                gpu_id = Utils.get_free_gpu()
            
            if gpu_id >= 0:
                torch.cuda.set_device(gpu_id)
                return torch.device(f'cuda:{gpu_id}')
        
        return torch.device('cpu')

if __name__ == "__main__":
    # Test GPU availability
    Utils.test_pytorch_gpu()
