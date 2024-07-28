import psutil
import time

def monitor_memory(output_file, interval=10):
    """
    Monitor memory usage and write results to a file every `interval` seconds.

    Args:
        output_file (str): Path to the output file where memory usage will be logged.
        interval (int): Time in seconds between each memory usage check.
    """
    with open(output_file, 'a') as file:
        while True:
            # Get memory usage information
            memory_info = psutil.virtual_memory()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            memory_usage = f"Timestamp: {timestamp}, Total: {memory_info.total / (1024 ** 3):.2f} GB, " \
                           f"Available: {memory_info.available / (1024 ** 3):.2f} GB, " \
                           f"Used: {memory_info.used / (1024 ** 3):.2f} GB, " \
                           f"Percent: {memory_info.percent}%"
            
            # Write memory usage to file
            file.write(memory_usage + '\n')
            file.flush()  # Ensure data is written to file
            
            # Wait for the next interval
            time.sleep(interval)

if __name__ == "__main__":
    output_file = "memory_usage.txt"  # Change this to your desired file path
    monitor_memory(output_file, interval=10)
